

'''
Note: This is script is to run the 10% seed comparision for personalized GSSL. 
It is different from 10pct_eval.py, as we here only feed the classifier target's data only
while in 10pct_eval.py, we feed the classifier all the data.

'''
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import shutil
import os 
from sklearn.utils import class_weight

from LR_check import plot_aggregate_auc 

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from new_helper import set_output_dir
from src import compare_pipelines as cp

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from new_helper import bootstrap_auc

CLF_EPOCHS, CLF_PATIENCE                = 200, 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, nargs="+")
    p.add_argument("--fruit", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--task", default="fruit", choices=["fruit", "bp"])
    p.add_argument("--train-frac", type=float, default=0.10)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    return p.parse_args()


def _parse_user_ids(user_args: list[str]) -> list[str]:
    users = []
    for token in user_args:
        parts = [x.strip() for x in str(token).split(",") if x.strip()]
        users.extend(parts)
    return list(dict.fromkeys(users))


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _safe_sample_df(df: pd.DataFrame, train_frac: float | None, seed: int) -> pd.DataFrame:
    """Sample a fraction of a dataframe safely with optional stratification."""
    df = df.reset_index(drop=True)

    if train_frac is None or train_frac >= 1.0:
        return df

    y = df["state_val"]
    stratify_y = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None

    df_sampled, _ = train_test_split(
        df,
        train_size=train_frac,
        stratify=stratify_y,
        random_state=seed,
    )
    return df_sampled.reset_index(drop=True)

def _prepare_global_splits(args, requested_users: list[str]):
    all_splits = {}
    all_negatives = {}

    if args.task == "bp":
        bp_root = Path("DATA/Cardiomate/hp")

        def _bp_load_signal_data(user_dir):
            pid = cp._bp_pid_from_user_dir(user_dir)
            hr_df, st_df, _, _ = cp._bp_load_all(pid)
            return hr_df, st_df

        def _bp_load_label_data(user_dir, fruit, scenario):
            pid = cp._bp_pid_from_user_dir(user_dir)
            _, _, pos_df, neg_df = cp._bp_load_all(pid)
            if scenario == "None":
                return neg_df.copy()
            return pos_df.copy()

        ## reroute the signal and label loading functions 
        cp.BASE_DATA_DIR = bp_root
        cp.load_signal_data = _bp_load_signal_data
        cp.load_label_data = _bp_load_label_data

        bp_users = []
        for p in sorted(bp_root.glob("hp*")):
            try:
                pid = cp._bp_pid_from_user_dir(p)
            except Exception:
                continue
            base = bp_root / f"hp{pid}"
            if not (base / f"hp{pid}_hr.csv").exists():
                continue
            if not (base / f"hp{pid}_steps.csv").exists():
                continue
            if not (base / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
                continue
            bp_users.append(pid)

        for pid in bp_users:
            hr_df, st_df, pos_df, neg_df = cp._bp_load_all(pid)
            uid_key = str(pid)
            all_negatives[uid_key] = neg_df
            try:
                tr_u, val_u, te_u = cp.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue
            all_splits[uid_key] = (tr_u, val_u, te_u)

    else:
        for u, pairs in cp.ALLOWED_SCENARIOS.items():
            if (args.fruit, args.scenario) not in pairs:
                continue

            hr_df, st_df = cp.load_signal_data(Path(cp.BASE_DATA_DIR) / u)
            pos_df = cp.load_label_data(Path(cp.BASE_DATA_DIR) / u, args.fruit, args.scenario)
            orig_neg = cp.load_label_data(Path(cp.BASE_DATA_DIR) / u, args.fruit, "None")

            if len(orig_neg) < len(pos_df):
                extra = cp.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

            uid_key = str(u)
            all_negatives[uid_key] = neg_df
            try:
                tr_u, val_u, te_u = cp.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {u}: {e}")
                continue
            all_splits[uid_key] = (tr_u, val_u, te_u)

    missing = [u for u in requested_users if u not in all_splits]
    if missing:
        print(f"Skipping users with no valid splits: {missing}")

    return all_splits, all_negatives

def _prepare_data(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    shared_enc_root: Path,
    all_splits: dict,
    all_negatives: dict,
    sample_mode: str = "original",
    train_frac: float | None = None,
    seed: int = 42,
):
    '''
    Prepare encoded data for a single user
    '''
    base_data_dir = Path(cp.BASE_DATA_DIR)

    out_dir = user_root / "p_global_ssl"
    models_d = out_dir / "models_saved"
    results_d = out_dir / "results"
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    cp.configure_task_mode(task="bp")

    enc_hr, enc_st, enc_src = cp._ensure_global_encoders(
        shared_enc_root, fruit, scenario, all_splits
    )

    for fpath in Path(enc_src).glob("*"):
        if fpath.suffix == ".keras":
            shutil.copy2(fpath, models_d / fpath.name)
        elif fpath.suffix == ".png":
            shutil.copy2(fpath, results_d / fpath.name)

    train_info = {}
    val_info = {}

    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = cp.load_signal_data(base_data_dir / u)
        pos_df = cp.load_label_data(base_data_dir / u, fruit, scenario)
        orig_neg = cp.load_label_data(base_data_dir / u, fruit, "None")
        if len(orig_neg) < len(pos_df):
            extra = cp.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_tr = cp.collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = cp.collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

        train_info[str(u)] = {"days": tr_days.tolist(), "df": df_tr}
        val_info[str(u)] = {"days": val_days.tolist(), "df": df_val}

    sample_summary = cp.sample_train_info(
        train_info,
        mode=sample_mode,
        random_state=42,
    )

    tr_days_u, val_days_u, te_days_u = all_splits[str(uid)]
    hr_df_u, st_df_u = cp.load_signal_data(base_data_dir / uid)
    pos_df_u = cp.load_label_data(base_data_dir / uid, fruit, scenario)
    neg_df_u = all_negatives[str(uid)]
    if len(neg_df_u) < len(pos_df_u):
        # Keep behavior identical to 10pct_p_GSSL.py:
        # replace with a fully-derived negative set when original negatives are short.
        neg_df_u = cp.derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))

    df_te = cp.collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)

    cp.write_split_details(
        results_d,
        "p_global_ssl",
        train_info,
        val_info,
        (str(uid), te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary,
    )

    def encode(df: pd.DataFrame):
        hr_seq = np.stack(df["hr_seq"])[..., None]
        st_seq = np.stack(df["st_seq"])[..., None]
        H = enc_hr.predict(hr_seq, verbose=0)
        S = enc_st.predict(st_seq, verbose=0)
        return H, S

    df_tr = train_info[str(uid)]["df"].reset_index(drop=True)

    if train_frac is not None and train_frac < 1.0:
        y = df_tr["state_val"]
        stratify_y = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None

        df_tr, _ = train_test_split(
            df_tr,
            train_size=train_frac,
            stratify=stratify_y,
            random_state=seed,
        )
        df_tr = df_tr.reset_index(drop=True)

    num_train = len(df_tr)

    df_val_u = val_info[str(uid)]["df"].reset_index(drop=True)

    H_tr, S_tr = encode(df_tr)
    H_val, S_val = encode(df_val_u)
    H_te, S_te = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1).astype("float32")
    y_tr = df_tr["state_val"].values.astype("float32")

    X_val = np.concatenate([H_val, S_val], axis=1).astype("float32")
    y_val = df_val_u["state_val"].values.astype("float32")

    X_te = np.concatenate([H_te, S_te], axis=1).astype("float32")
    y_te = df_te["state_val"].values.astype("float32")

    return X_tr, y_tr, X_val, y_val, X_te, y_te, num_train, out_dir, results_d

def _train_and_eval_classifier(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    out_dir: Path,
    results_d: Path,
    uid: str,
    tag: str,
):
    """Train classifier, pick threshold on train, evaluate on test."""
    cp.reset_seeds()

    clf = Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_tr.shape[1],), kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    es = EarlyStopping(
        monitor="val_loss",
        patience=CLF_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
    )

    cw_vals = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    hist = clf.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        # class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2,
    )

    cp.plot_clf_losses(
        hist.history["loss"],
        hist.history["val_loss"],
        out_dir,
        f"global_ssl_clf_{tag}",
    )

    best_thr = cp.select_threshold_train(clf, X_tr, y_tr)
    (results_d / f"selected_threshold_{tag}.txt").write_text(f"{best_thr:.4f}\n")

    probs_te = clf.predict(X_te, verbose=0).ravel()
    np.save(results_d / f"test_probs_{uid}_{tag}.npy", probs_te)
    np.save(results_d / f"test_labels_{uid}_{tag}.npy", y_te)

    df_boot, auc_mean, auc_std = cp.bootstrap_threshold_metrics(
        y_te,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42,
    )

    df_boot.to_csv(results_d / f"bootstrap_metrics_{tag}.csv", index=False)

    return {
        "df_boot": df_boot,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "best_thr": best_thr,
        "num_train": len(X_tr),
        "X_tr": X_tr,
        "y_tr": y_tr,
        "probs_te": probs_te,
    }

def run_personal_global_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    shared_enc_root: Path,
    all_splits: dict,
    all_negatives: dict,
    sample_mode: str = "original",
    train_frac: float | None = None,
    seed: int = 42,
    compute_aggregate: bool = True,
):
    """
    Always computes both:
      - original: train on target user's own train split
      - aggregate_10pct: train on sampled train split from all users

    Both are evaluated on the same target user's test set.
    """
    print(f"\n>> Personal-Global-SSL ({fruit}_{scenario})")

    uid = str(uid)
    cp.reset_seeds = lambda s=seed: _set_seed(s)
    out_dir = user_root / "p_global_ssl"
    results_d = out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # Target user's original train/val/test
    (
        X_tr_orig,
        y_tr_orig,
        X_val_orig,
        y_val_orig,
        X_te,
        y_te,
        num_train_orig,
        out_dir,
        results_d,
    ) = _prepare_data(
        uid=uid,
        fruit=fruit,
        scenario=scenario,
        user_root=user_root,
        shared_enc_root=shared_enc_root,
        all_splits=all_splits,
        all_negatives=all_negatives,
        sample_mode=sample_mode,
        train_frac=train_frac,
        seed=seed,
    )

    aggregate_res = None
    if compute_aggregate:
        # Aggregate sampled train/val data across all users
        X_tr_list, y_tr_list = [], []
        X_val_list, y_val_list = [], []
        num_train_agg = 0

        for u in all_splits:
            (
                X_tr_u,
                y_tr_u,
                X_val_u,
                y_val_u,
                _,
                _,
                num_train_u,
                _,
                _,
            ) = _prepare_data(
                uid=str(u),
                fruit=fruit,
                scenario=scenario,
                user_root=user_root,
                shared_enc_root=shared_enc_root,
                all_splits=all_splits,
                all_negatives=all_negatives,
                sample_mode=sample_mode,
                train_frac=train_frac,
                seed=seed,
            )

            X_tr_list.append(X_tr_u)
            y_tr_list.append(y_tr_u)
            X_val_list.append(X_val_u)
            y_val_list.append(y_val_u)
            num_train_agg += num_train_u

        X_tr_agg = np.concatenate(X_tr_list, axis=0).astype("float32")
        y_tr_agg = np.concatenate(y_tr_list, axis=0).astype("float32")
        X_val_agg = np.concatenate(X_val_list, axis=0).astype("float32")
        y_val_agg = np.concatenate(y_val_list, axis=0).astype("float32")

    # Train/eval original
    original_res = _train_and_eval_classifier(
        X_tr=X_tr_orig,
        y_tr=y_tr_orig,
        X_val=X_val_orig,
        y_val=y_val_orig,
        X_te=X_te,
        y_te=y_te,
        out_dir=out_dir,
        results_d=results_d,
        uid=uid,
        tag="original",
    )
    original_res["num_train"] = num_train_orig
    original_res["X_tr"] = X_tr_orig
    original_res["y_tr"] = y_tr_orig

    # Train/eval aggregated
    if compute_aggregate:
        aggregate_res = _train_and_eval_classifier(
            X_tr=X_tr_agg,
            y_tr=y_tr_agg,
            X_val=X_val_agg,
            y_val=y_val_agg,
            X_te=X_te,
            y_te=y_te,
            out_dir=out_dir,
            results_d=results_d,
            uid=uid,
            tag="aggregate_10pct",
        )
        aggregate_res["num_train"] = num_train_agg
        aggregate_res["X_tr"] = X_tr_agg
        aggregate_res["y_tr"] = y_tr_agg

# --- Return only AUC stats ---
    return (
        original_res["auc_mean"],
        original_res["auc_std"],
        aggregate_res["auc_mean"] if aggregate_res is not None else np.nan,
        aggregate_res["auc_std"] if aggregate_res is not None else np.nan,
        original_res["num_train"],
    )
    
def plot_aggregate_10pct(df_seed_results: pd.DataFrame, top_out: Path, seed_to_plot: int, out_path: Path):
    '''
    plot the 10% split aggregate across users for a given seed
    '''
    df = df_seed_results.copy()
    df = df[df["seed"] == seed_to_plot].copy()
    if df.empty:
        print(f"No rows found for seed={seed_to_plot}")
        return

    df = df.sort_values("user").reset_index(drop=True)
    x = np.arange(len(df))

    plt.figure(figsize=(8, 4))
    plt.scatter(x, df["agg_auc_mean"], s=70)
    for i, row in df.iterrows():
        plt.text(i, row["agg_auc_mean"] + 0.002, f"user {row['user']}", ha="center", fontsize=8)

    plt.xticks(x, [str(u) for u in df["user"]], rotation=45, ha="right")
    plt.ylabel("Test AUC mean")
    plt.xlabel("User")
    plt.title(f"Aggregate 10% AUC Across Users (seed={seed_to_plot})")
    plt.tight_layout()

    # out_path = top_out / f"aggregate_10pct_seed_{seed_to_plot}_{args.fruit}_{args.scenario}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    
def plot_aggregate_full_data(df_full_results, top_out, out_path):
        
            if df_full_results.empty:
                return

            df = df_full_results.copy()
            df = df.dropna(subset=["full_auc"]).sort_values("user")
            if df.empty:
                return

            # Pooled AUC (only this needs computing, per-user AUCs already in df)
            all_true = np.concatenate(df["y_true"].values)
            all_pred = np.concatenate(df["y_pred"].values)
            # pooled_auc = roc_auc_score(all_true, all_pred)
            pooled_auc, _, _ = bootstrap_auc(all_true, all_pred)
            x = np.arange(len(df))

            plt.figure(figsize=(8, 4))
            plt.scatter(x, df["full_auc"].values, s=70, color="tab:purple")
            for i, row in df.reset_index(drop=True).iterrows():
                plt.text(i, row["full_auc"] + 0.002, f"user {row['user']}", ha="center", fontsize=8)

            plt.axhline(pooled_auc, color="black", linestyle="--", label=f"Pooled AUC = {pooled_auc:.3f}")

            plt.xticks(x, [str(u) for u in df["user"]], rotation=45, ha="right")
            plt.ylabel("Full-data test AUC")
            plt.xlabel("User")
            plt.title("Aggregate Full-data AUC Across Users")
            plt.legend()
            plt.tight_layout()
            
            # out_path = top_out / f"aggregate_full_data_auc_{args.fruit}_{args.scenario}.png"
            plt.savefig(out_path)
            print(f"Saved: {out_path} | Pooled AUC = {pooled_auc:.3f}")
          

def main():
    args = parse_args()
    requested_users = _parse_user_ids(args.user)
    if not requested_users:
        raise RuntimeError("No valid users provided to --user")

    pool = "p_global_ssl"
    top_out = Path(set_output_dir(pool))
    shared_enc_root = top_out / "_global_encoders"
    all_splits, all_negatives = _prepare_global_splits(args, requested_users)

    orig_sample_train_info = cp.sample_train_info
    orig_reset_seeds = cp.reset_seeds

    def plot_seed_comparision(df_seed_results, user_root):
        tmp = df_seed_results.copy()
        tmp["auc_mean_num"] = pd.to_numeric(tmp["auc_test_mean"], errors="coerce")

        if (tmp["seed"] == "full_data").sum() == 0:
            print(f"Skipping seed plot for {user_root}: missing full_data row")
            return

        seed_df = tmp[tmp["seed"] != "full_data"].copy()
        seed_df["seed"] = seed_df["seed"].astype(str)

        auc_full = float(
            tmp.loc[tmp["seed"] == "full_data", "auc_mean_num"].dropna().iloc[0]
        )
        x = np.arange(len(seed_df))

        plt.figure(figsize=(7, 4))
        plt.scatter(x, seed_df["auc_mean_num"], s=70, color="tab:blue")
        for i, row in seed_df.reset_index(drop=True).iterrows():
            plt.text(
                i,
                row["auc_mean_num"] + 0.002,
                f"seed {row['seed']}",
                ha="center",
                fontsize=9,
            )
        plt.axhline(
            auc_full,
            color="red",
            linestyle="--",
            label=f"Full-data AUC = {auc_full:.3f}",
        )
        plt.xticks(x, [f"seed {s}" for s in seed_df["seed"]])
        plt.ylabel("Test AUC mean")
        plt.title("Per-seed 10% AUC vs Full-data AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(user_root / "seed_comparison.png")
        plt.close()

    try:
        cp.configure_task_mode(task="bp", bp_input="raw")

        aggregate_full_rows = []
        aggregate_10pct_rows = []
        first_seed = args.seeds[0]

        for uid in requested_users:
            uid = str(uid)
            if uid not in all_splits or uid not in all_negatives:
                print(f"Skipping user {uid}: no valid split/negatives")
                continue

            user_root = top_out / uid / f"{args.fruit}_{args.scenario}" / "fixed_seed_10pct"
            user_root.mkdir(parents=True, exist_ok=True)

            results_df = []

            for seed in args.seeds:
                cp.reset_seeds = lambda s=seed: _set_seed(s)
                run_aggregate_for_seed = seed == first_seed

                single_auc_mean, single_auc_std, aggregate_mean_auc, aggregate_std_auc, single_num_train  = run_personal_global_ssl(
                    uid=uid,
                    fruit=str(args.fruit),
                    scenario=str(args.scenario),
                    user_root=user_root,
                    shared_enc_root=shared_enc_root,
                    all_splits=all_splits,
                    all_negatives=all_negatives,
                    sample_mode="original",
                    train_frac=args.train_frac,
                    seed=seed,
                    compute_aggregate=run_aggregate_for_seed,
                )

                results_df.append({
                    "seed": seed,
                    "auc_test_mean": single_auc_mean,
                    "auc_test_std": single_auc_std,
                })

                if run_aggregate_for_seed:
                    aggregate_10pct_rows.append({
                        "user": uid,
                        "seed": seed,
                        "agg_auc_mean": aggregate_mean_auc,
                        "agg_auc_std": aggregate_std_auc,
                        "orig_auc_mean": single_auc_mean,
                        "orig_auc_std": single_auc_std,
                        "num_train": single_num_train,
                    })

            out_df = pd.DataFrame(results_df)

            full_seed = 42
            cp.reset_seeds = lambda s=full_seed: _set_seed(s)

            full_auc_mean, full_auc_std, _, _, full_num_train = run_personal_global_ssl(
                uid=uid,
                fruit=str(args.fruit),
                scenario=str(args.scenario),
                user_root=user_root,
                shared_enc_root=shared_enc_root,
                all_splits=all_splits,
                all_negatives=all_negatives,
                sample_mode="original",
                train_frac=1.0,
                seed=full_seed,
                compute_aggregate=False,
            )

            out_df_full = pd.DataFrame([{
                "seed": "full_data",
                "auc_test_mean": full_auc_mean,
                "auc_test_std": full_auc_std,
                "num_train": full_num_train,
            }])

            out_df = pd.concat([out_df, out_df_full], ignore_index=True)

            out_csv = user_root / "auc_10pct_fixed_seed.csv"
            out_df.to_csv(out_csv, index=False)
            print(out_df.to_string(index=False))
            print(f"Saved: {out_csv}")

            plot_seed_comparision(out_df, user_root)

            aggregate_full_rows.append({
                "user": uid,
                "full_auc": full_auc_mean,
                "full_auc_std": full_auc_std,
                "y_true": np.load(
                    user_root / "p_global_ssl" / "results" / f"test_labels_{uid}_original.npy"
                ),
                "y_pred": np.load(
                    user_root / "p_global_ssl" / "results" / f"test_probs_{uid}_original.npy"
                ),
            })

        aggregate_full_df = pd.DataFrame(aggregate_full_rows)
        aggregate_10pct_df = pd.DataFrame(aggregate_10pct_rows)
        breakpoint()

        plot_aggregate_full_data(
            aggregate_full_df,
            top_out,
            out_path=top_out / f"aggregate_full_data_auc_BP_spike.png",
        )

        seed_to_plot = first_seed
        plot_aggregate_10pct(
            aggregate_10pct_df,
            top_out,
            seed_to_plot=seed_to_plot,
            out_path=top_out / f"aggregate_10pct_seed_{seed_to_plot}_{args.fruit}_{args.scenario}.png",
        )

    finally:
        cp.sample_train_info = orig_sample_train_info
        cp.reset_seeds = orig_reset_seeds


if __name__ == "__main__":
    main()
