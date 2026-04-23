
''''
script to run the 10% seed comparision for the global ssl model where the initial 10% is composed of
target user's data and the rest of the users data. 
To run: 
python scripts/10pct_eval.py \
  --user 10,15,16,17,18,20,22,23,24,25,26,30,31,32,33,34,35,36,39,40 \
  --fruit BP \
  --scenario spike \
  --task bp \
  --train-frac 0.10 \
'''
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from new_helper import set_output_dir, bootstrap_auc
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
import utility
import pct_utils

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


def run_global_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    shared_enc_root: Path,
    all_splits: dict,
    neg_df_u: pd.DataFrame,
    sample_mode: str = "original",
    train_frac: int=None, 
    seed: int = 42,):
    print(f"\n>> Global-SSL ({fruit}_{scenario})")
    BASE_DATA_DIR = cp.BASE_DATA_DIR  # Ensure this is set for downstream loading
    # Directories
    out_dir   = user_root / 'global_ssl'
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
        loop_uid = str(u)
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
        df_tr["user_id"] = loop_uid
        df_val["user_id"] = loop_uid
        
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
        "global_ssl",
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

    df_all_tr    = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)
    breakpoint()
    pre_hash, pre_meta = utility.split_fingerprint(df_all_tr)
    print(
        f"[presplit_fingerprint] hash={pre_hash} "
        f"rows={pre_meta['rows']} time_col={pre_meta['time_col']}"
    )
    (results_d / "presplit_df_all_tr_fingerprint.txt").write_text(
        "\n".join(
            [
                "source=scripts/10pct_eval.py",
                f"rows={pre_meta['rows']}",
                f"time_col={pre_meta['time_col']}",
                f"sha256={pre_hash}",
                "",
            ]
        )
    )

    print("first 5 rows user 25:", df_all_tr[df_all_tr["user_id"] == "25"][["state_val", "user_id"]].head())
    print("first 5 rows user 15:", df_all_tr[df_all_tr["user_id"] == "15"][["state_val", "user_id"]].head())
    if train_frac is not None and train_frac < 1.0:
        df_all_tr, _ = utility.make_labeled_unlabeled_with_target_quota(df_all_tr, target_uid=uid, unlabeled_frac=0.9, seed=seed)
        # df_all_tr, _ = train_test_split(
                # df_all_tr,
                # train_size=train_frac,
                # stratify=df_all_tr["state_val"],
                # random_state=seed)
        df_all_tr = df_all_tr.reset_index(drop=True)
    else:
         df_all_tr = df_all_tr.reset_index(drop=True)  # full data when None or >=1.0
    split_hash, split_meta = utility.split_fingerprint(df_all_tr)
    run_tag = "full_data" if (train_frac is None or float(train_frac) >= 1.0) else f"frac{int(round(float(train_frac) * 100))}pct_seed{seed}"
    print(
        f"[split_fingerprint] seed={seed} hash={split_hash} "
        f"rows={split_meta['rows']} time_col={split_meta['time_col']}"
    )
    (results_d / f"labeled_split_fingerprint_{run_tag}.txt").write_text(
        "\n".join(
            [
                "source=scripts/10pct_eval.py",
                f"seed={seed}",
                f"rows={split_meta['rows']}",
                f"time_col={split_meta['time_col']}",
                f"sha256={split_hash}",
                "",
            ]
        )
    )
    breakpoint()
    num_train = len(df_all_tr)

    H_tr, S_tr   = encode(df_all_tr)
    # df_val_u     = val_info[uid]["df"]
    df_val_all   = pd.concat([v["df"] for v in val_info.values()], ignore_index=True)

    # H_val, S_val = encode(df_val_u)  
    H_val, S_val = encode(df_val_all)
    H_te, S_te   = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1).astype('float32')
    y_tr = df_all_tr['state_val'].values.astype('float32')
    X_val = np.concatenate([H_val, S_val], axis=1).astype('float32')
    # y_val = df_val_u['state_val'].values.astype('float32')
    y_val = df_val_all['state_val'].values.astype('float32')
    X_te  = np.concatenate([H_te, S_te], axis=1).astype('float32')
    y_te  = df_te['state_val'].values.astype('float32')
    
    # 7) Train classifier
    cp.reset_seeds()  # Ensure deterministic model initialization
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
    is_full_data = (train_frac is None) or (float(train_frac) >= 1.0)
    if is_full_data:
        run_tag = "full_data"
    else:
        run_tag = f"frac{int(round(float(train_frac) * 100))}pct_seed{seed}"

    tagged_probs_path = results_d / f"test_probs_{uid}_{run_tag}.npy"
    tagged_labels_path = results_d / f"test_labels_{uid}_{run_tag}.npy"
    np.save(tagged_probs_path, probs_te)
    np.save(tagged_labels_path, y_te)

    # Keep legacy filenames for compatibility with older consumers.
    np.save(results_d / f"test_probs_{uid}.npy", probs_te)
    np.save(results_d / f"test_labels_{uid}.npy", y_te)
    print(f"Saved tagged outputs: {tagged_probs_path} and {tagged_labels_path}")

    auc_mean, auc_std, valid_frac = bootstrap_auc(
        y_te,
        probs_te,
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42,
    )
    
    probs_train = clf.predict(X_tr, verbose=0).ravel()
    probs_val = clf.predict(X_val, verbose=0).ravel()

    df_boot = pd.DataFrame(
        [
            {
                "auc": auc_mean,
                "auc_std": auc_std,
                "valid_frac": valid_frac,
                "threshold": best_thr,
            }
        ]
    )
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)

    return df_boot, auc_mean, auc_std, num_train

def main():
    args = parse_args()
    requested_users = _parse_user_ids(args.user)
    
    if not requested_users:
        raise RuntimeError("No valid users provided to --user")
    # _set_seed(args.seed)

    pool = "global"
    top_out = Path(set_output_dir(pool))
    shared_enc_root = top_out / "_global_encoders"
    all_splits, all_negatives = pct_utils._prepare_global_splits(args, requested_users)

    orig_sample_train_info = cp.sample_train_info
    orig_reset_seeds = cp.reset_seeds

    try:
        ##setting BASE_DATA_DIR to bp data dir for loading raw data and
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
            neg_df_u = all_negatives[uid]

            results_df = []
            for seed in args.seeds:
                cp.reset_seeds = lambda: _set_seed(seed)

                _, auc_mean, auc_std, num_train = run_global_ssl(
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
                })
                if seed == first_seed:
                    aggregate_10pct_rows.append({
                        "user": uid,
                        "user_root": str(user_root),
                        "seed": seed,
                        "agg_auc_mean": auc_mean,
                        "agg_auc_std": auc_std,
                        "num_train": num_train,
                    })

            out_df = pd.DataFrame(results_df)

            _, auc_mean_full, auc_std_full, num_train_full = run_global_ssl(
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
            }])
            out_df = pd.concat([out_df, out_df_full], ignore_index=True)

            out_csv = user_root / "auc_10pct_fixed_seed.csv"
            out_df.to_csv(out_csv, index=False)
            print(out_df.to_string(index=False))
            print(f"Saved: {out_csv}")
            pct_utils.plot_seed_comparision(out_df, user_root)

            aggregate_full_rows.append({
                "user": uid,
                "full_auc": auc_mean_full,
                "full_auc_std": auc_std_full,
                "y_true": np.load(
                    user_root / "global_ssl" / "results" / f"test_labels_{uid}_full_data.npy"
                ),
                "y_pred": np.load(
                    user_root / "global_ssl" / "results" / f"test_probs_{uid}_full_data.npy"
                ),

            })

        aggregate_full_df = pd.DataFrame(aggregate_full_rows)
        aggregate_10pct_df = pd.DataFrame(aggregate_10pct_rows)
        seed_run_tag = f"frac{int(round(float(args.train_frac) * 100))}pct_seed{first_seed}"

        pct_utils.plot_aggregate_10pct(
            aggregate_10pct_df,
            seed_to_plot=first_seed,
            run_tag=seed_run_tag,
            out_path=top_out / f"aggregate_10pct_seed_{first_seed}_{args.fruit}_{args.scenario}.png",
        )
        pct_utils.plot_aggregate_full_data(aggregate_full_df, top_out, top_out / f"aggregate_full_data_auc_BP_spike.png")
    finally:
        cp.sample_train_info = orig_sample_train_info
        cp.reset_seeds = orig_reset_seeds

if __name__ == "__main__":
    main()
