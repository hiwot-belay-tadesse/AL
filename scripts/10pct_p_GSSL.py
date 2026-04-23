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
    uid,
    fruit,
    scenario,
    user_root,
    shared_enc_root,
    all_splits,
    neg_df_u,
    sample_mode="original",
    train_frac=None,
    seed=42,
):
    BASE_DATA_DIR = cp.BASE_DATA_DIR  # Ensure this is set for downstream loading
    # Directories
    out_dir   = user_root / 'p_global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    #setting BASE_DATA_DIR to bp data dir for loading raw data and
    cp.configure_task_mode(task="bp",)
    # 1) Load or train shared encoders
    enc_hr, enc_st, enc_src = cp._ensure_global_encoders(
        shared_enc_root, fruit, scenario, all_splits
    )
    for fpath in Path(enc_src).glob('*'):
        if fpath.suffix == '.keras':
            shutil.copy2(fpath, models_d / fpath.name)
        elif fpath.suffix == '.png':
            shutil.copy2(fpath, results_d / fpath.name)

    # 2) Build per-user train_info and val_info
    train_info = {}
    val_info   = {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = cp.load_signal_data(Path(cp.BASE_DATA_DIR) / u)
        pos_df       = cp.load_label_data(Path(cp.BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg     = cp.load_label_data(Path(cp.BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra  = cp.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_tr  = cp.collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = cp.collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    # 3) Apply sampling across users
    sample_summary = cp.sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 4) Build this user's test windows
    tr_days_u, val_days_u, te_days_u = all_splits[uid]
    hr_df_u, st_df_u = cp.load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u        = cp.load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    if len(neg_df_u) < len(pos_df_u):
        neg_df_u = cp.derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
    df_te           = cp.collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)

    # 5) Write split_details.txt
    cp.write_split_details(
        results_d,
        "p_global_ssl",
        train_info,
        val_info,
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Encode and build X/y
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        H = enc_hr.predict(hr_seq, verbose=0)
        S = enc_st.predict(st_seq, verbose=0)
        return H, S

    # df_all_tr    = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)
    ## only train on the target user's data
    df_tr = train_info[uid]["df"]
    if train_frac is not None and train_frac < 1.0:
        df_tr, _ = train_test_split(
                df_tr,
                train_size=train_frac,
                stratify=df_tr["state_val"],
                random_state=seed)
        df_tr = df_tr.reset_index(drop=True)
    else:
         df_tr = df_tr.reset_index(drop=True)  # full data when None or >=1.0
    num_train = len(df_tr)

    H_tr, S_tr   = encode(df_tr)
    df_val_u     = val_info[uid]["df"] ## here we only use target's val
    
    # df_val_all   = pd.concat([v["df"] for v in val_info.values()], ignore_index=True)

    # H_val, S_val = encode(df_val_u)  
    H_val, S_val = encode(df_val_u)
    H_te, S_te   = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1).astype('float32')
    y_tr = df_tr['state_val'].values.astype('float32')
    X_val = np.concatenate([H_val, S_val], axis=1).astype('float32')
    # y_val = df_val_u['state_val'].values.astype('float32')
    y_val = df_val_u['state_val'].values.astype('float32')
    X_te  = np.concatenate([H_te, S_te], axis=1).astype('float32')
    y_te  = df_te['state_val'].values.astype('float32')

    
    return  X_tr, y_tr, X_val, y_val, X_te, y_te, num_train

def run_personal_global_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    shared_enc_root: Path,
    all_splits: dict,
    neg_df_u: pd.DataFrame,
    sample_mode: str = "original",
    train_frac: float = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, float, float, int]:
    
    print(f"\n>> Personal-Global-SSL ({fruit}_{scenario})")
    BASE_DATA_DIR = cp.BASE_DATA_DIR  # Ensure this is set for downstream loading
    # Directories
    out_dir   = user_root / 'p_global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    #setting BASE_DATA_DIR to bp data dir for loading raw data and
    cp.configure_task_mode(task="bp",)
    # 1) Load or train shared encoders
    enc_hr, enc_st, enc_src = cp._ensure_global_encoders(
        shared_enc_root, fruit, scenario, all_splits
    )
    for fpath in Path(enc_src).glob('*'):
        if fpath.suffix == '.keras':
            shutil.copy2(fpath, models_d / fpath.name)
        elif fpath.suffix == '.png':
            shutil.copy2(fpath, results_d / fpath.name)

    # 2) Build per-user train_info and val_info
    train_info = {}
    val_info   = {}
    for u, (tr_days, val_days, _) in all_splits.items():
        hr_df, st_df = cp.load_signal_data(Path(cp.BASE_DATA_DIR) / u)
        pos_df       = cp.load_label_data(Path(cp.BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg     = cp.load_label_data(Path(cp.BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra  = cp.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_tr  = cp.collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = cp.collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    # 3) Apply sampling across users
    sample_summary = cp.sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 4) Build this user's test windows
    tr_days_u, val_days_u, te_days_u = all_splits[uid]
    hr_df_u, st_df_u = cp.load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u        = cp.load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    if len(neg_df_u) < len(pos_df_u):
        neg_df_u = cp.derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
    df_te           = cp.collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)

    # 5) Write split_details.txt
    cp.write_split_details(
        results_d,
        "p_global_ssl",
        train_info,
        val_info,
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Encode and build X/y
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        H = enc_hr.predict(hr_seq, verbose=0)
        S = enc_st.predict(st_seq, verbose=0)
        return H, S

    # df_all_tr    = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)
    ## only train on the target user's data
    df_tr = train_info[uid]["df"]

    
    if train_frac is not None and train_frac < 1.0:
        df_tr, _ = train_test_split(
                df_tr,
                train_size=train_frac,
                stratify=df_tr["state_val"],
                random_state=seed)
        df_tr = df_tr.reset_index(drop=True)
    else:
         df_tr = df_tr.reset_index(drop=True)  # full data when None or >=1.0
    num_train = len(df_tr)

    H_tr, S_tr   = encode(df_tr)
    df_val_u     = val_info[uid]["df"] ## here we only use target's val
    
    # df_val_all   = pd.concat([v["df"] for v in val_info.values()], ignore_index=True)

    # H_val, S_val = encode(df_val_u)  
    H_val, S_val = encode(df_val_u)
    H_te, S_te   = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1).astype('float32')
    y_tr = df_tr['state_val'].values.astype('float32')
    X_val = np.concatenate([H_val, S_val], axis=1).astype('float32')
    # y_val = df_val_u['state_val'].values.astype('float32')
    y_val = df_val_u['state_val'].values.astype('float32')
    X_te  = np.concatenate([H_te, S_te], axis=1).astype('float32')
    y_te  = df_te['state_val'].values.astype('float32')

    # 7) Train classifier
    cp.reset_seeds()  # Ensure deterministic model initialization
    # clf = Sequential([
    #             layers.Input(shape=X_tr.shape[1:]),
    #             layers.Dense(16, activation='relu', kernel_regularizer=l2(1e-3)),
    #             layers.Dropout(0.3),
    #             layers.Dense(1, activation='sigmoid')
    #         ])
    clf = Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_tr.shape[1],), kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(0.5),
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy', metrics=['accuracy'])

    es  = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE,
                                restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    cw_vals     = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    hist = clf.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        # class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2
    )
    ## plot loss curve for each split

    cp.plot_clf_losses(hist.history['loss'], hist.history['val_loss'], out_dir, 'global_ssl_clf_{train_frac}'.format(train_frac=str(train_frac)))

    # 8) Threshold selection on the FULL training set (instead of validation)
    best_thr = cp.select_threshold_train(clf, X_tr, y_tr)
    (results_d / "selected_threshold.txt").write_text(f"{best_thr:.4f}\n")

    # 9) Bootstrap & plot on TEST
    probs_te = clf.predict(X_te, verbose=0).ravel()
    np.save(results_d / f"test_probs_{uid}.npy", probs_te)
    np.save(results_d / f"test_labels_{uid}.npy", y_te)
    print(f"saved in {results_d / f'test_probs_{uid}.npy'} and {results_d / f'test_labels_{uid}.npy'} ")

    df_boot, auc_mean, auc_std = cp.bootstrap_threshold_metrics(
        y_te,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    
    probs_train = clf.predict(X_tr, verbose=0).ravel()
    probs_val = clf.predict(X_val, verbose=0).ravel()

    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)

    return df_boot, auc_mean, auc_std, num_train, X_tr, y_tr

def main():
    args = parse_args()
    requested_users = _parse_user_ids(args.user)
    if not requested_users:
        raise RuntimeError("No valid users provided to --user")
    # _set_seed(args.seed)

    pool = "p_global_ssl"
    top_out = Path(set_output_dir(pool))
    shared_enc_root = top_out / "_global_encoders"
    all_splits, all_negatives = _prepare_global_splits(args, requested_users)

    orig_sample_train_info = cp.sample_train_info
    orig_reset_seeds = cp.reset_seeds

    def plot_seed_comparision(df_seed_results, user_root):
        tmp = df_seed_results.copy()
        tmp["auc_mean_num"] = pd.to_numeric(
            tmp["auc_test_mean"],
            errors="coerce",
        )
        if (tmp["seed"] == "full_data").sum() == 0:
            print(f"Skipping seed plot for {user_root}: missing full_data row")
            return
        seed_df = tmp[tmp["seed"] != "full_data"].copy()

        seed_df["seed"] = seed_df["seed"].astype(str)
        auc_full = float(tmp.loc[tmp["seed"] == "full_data", "auc_mean_num"].dropna().iloc[0])
        x = np.arange(len(seed_df))

        plt.figure(figsize=(7, 4))
        plt.scatter(x, seed_df["auc_mean_num"], s=70, color="tab:blue")
        for i, row in seed_df.reset_index(drop=True).iterrows():
            plt.text(i, row["auc_mean_num"] + 0.002, f"seed {row['seed']}", ha="center", fontsize=9)
        plt.axhline(auc_full, color="red", linestyle="--", label=f"Full-data AUC = {auc_full:.3f}")
        plt.xticks(x, [f"seed {s}" for s in seed_df["seed"]])
        plt.ylabel("Test AUC mean")
        plt.title("Per-seed 10% AUC vs Full-data AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(user_root / "seed_comparison.png")

    def plot_aggregate_10pct(df_10_pct, top):
        pass 
        

    def plot_aggregate_full_data(df_full_results, top_out):
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
        out_path = top_out / f"aggregate_full_data_auc_{args.fruit}_{args.scenario}.png"
        plt.savefig(out_path)
        print(f"Saved: {out_path} | Pooled AUC = {pooled_auc:.3f}")
    
    # def plot_aggregate_full_data(df_full_results, top_out):
    #     if df_full_results.empty:
    #         return

    #     df = df_full_results.sort_values("user")
    #     overall_mean = df["full_auc"].mean()
    #     x = np.arange(len(df))

    #     plt.figure(figsize=(8, 4))
    #     plt.scatter(x, df["full_auc"], s=70, color="tab:purple")
    #     for i, row in df.reset_index(drop=True).iterrows():
    #         plt.text(i, row["full_auc"] + 0.002, f"user {row['user']}", ha="center", fontsize=8)
    #     plt.axhline(overall_mean, color="black", linestyle="--", label=f"Mean full-data AUC = {overall_mean:.3f}")
    #     plt.xticks(x, [str(u) for u in df["user"]], rotation=45, ha="right")
    #     plt.ylabel("Full-data test AUC")
    #     plt.xlabel("User")
    #     plt.title("Aggregate Full-data AUC Across Users")
    #     plt.legend()
    #     plt.tight_layout()
    #     out_path = top_out / f"aggregate_full_data_auc_{args.fruit}_{args.scenario}.png"
    #     plt.savefig(out_path)
    #     print(f"Saved: {out_path}")



    try:
        ##setting BASE_DATA_DIR to bp data dir for loading raw data and
        cp.configure_task_mode(task="bp", bp_input="raw")
        aggregate_full_rows = []
        for uid in requested_users:
            uid = str(uid)
            if uid not in all_splits or uid not in all_negatives:
                print(f"Skipping user {uid}: no valid split/negatives")
                continue

            user_root = top_out / uid / f"{args.fruit}_{args.scenario}" / "fixed_seed_10pct"
            user_root.mkdir(parents=True, exist_ok=True)
            neg_df_u = all_negatives[uid]

            results_df = []
            for seed in args.seeds:
                cp.reset_seeds = lambda: _set_seed(seed)

                _, auc_mean, auc_std, num_train, X_tr_10pct, y_tr_10pct = run_personal_global_ssl(
                    uid=uid,
                    fruit=str(args.fruit),
                    scenario=str(args.scenario),
                    user_root=user_root,
                    shared_enc_root=shared_enc_root,
                    all_splits=all_splits,
                    neg_df_u=neg_df_u,
                    sample_mode="original",
                    train_frac=args.train_frac,
                    seed=seed,
                )
                results_df.append({
                    "seed": seed,
                    "num_train": num_train,
                    "auc_test_mean": auc_mean,
                    "auc_test_std": auc_std,
                    "X_train": X_tr_10pct,
                    "y_train": y_tr_10pct,
                })

            out_df = pd.DataFrame(results_df)

            _, auc_mean_full, auc_std_full, num_train_full, X_tr_full, y_tr_full = run_personal_global_ssl(
                uid=uid,
                fruit=str(args.fruit),
                scenario=str(args.scenario),
                user_root=user_root,
                shared_enc_root=shared_enc_root,
                all_splits=all_splits,
                neg_df_u=neg_df_u,
                sample_mode="original",
                seed=42,
            )
            out_df_full = pd.DataFrame([{
                "seed": "full_data",
                "num_train": num_train_full,
                "auc_test_mean": auc_mean_full,
                "auc_test_std": auc_std_full,
                "X_train": X_tr_full,
                "y_train": y_tr_full,
            }])
            out_df = pd.concat([out_df, out_df_full], ignore_index=True)

            out_csv = user_root / "auc_10pct_fixed_seed.csv"
            out_df.to_csv(out_csv, index=False)
            print(out_df.to_string(index=False))
            print(f"Saved: {out_csv}")
            plot_seed_comparision(out_df, user_root)

            aggregate_full_rows.append({
                "user": uid,
                "full_auc": auc_mean_full,
                "full_auc_std": auc_std_full,
                "y_true": np.load(user_root / "p_global_ssl" / "results" / f"test_labels_{uid}.npy"),
                "y_pred": np.load(user_root / "p_global_ssl" / "results" / f"test_probs_{uid}.npy"),
            })

        aggregate_full_df = pd.DataFrame(aggregate_full_rows)
        breakpoint()
        plot_aggregate_full_data(aggregate_full_df, top_out)
    finally:
        cp.sample_train_info = orig_sample_train_info
        cp.reset_seeds = orig_reset_seeds

if __name__ == "__main__":
    main()
