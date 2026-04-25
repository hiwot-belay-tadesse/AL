#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import shutil
import os 
from sklearn.utils import class_weight 

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
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

CLF_EPOCHS, CLF_PATIENCE                = 200, 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--fruit", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--task", default="fruit", choices=["fruit", "bp"])
    p.add_argument("--train-frac", type=float, default=0.10)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])

    return p.parse_args()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _prepare_personal_split(args):
    uid = str(args.user)

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

        hr_df, st_df, pos_df, neg_df = cp._bp_load_all(uid)
        tr_u, val_u, te_u = cp.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        return tr_u, val_u, te_u, neg_df

    else:
        pairs = cp.ALLOWED_SCENARIOS.get(uid, [])
        if (args.fruit, args.scenario) not in pairs:
            raise RuntimeError(f"User {uid} has no data for {args.fruit}/{args.scenario}")

        hr_df, st_df = cp.load_signal_data(Path(cp.BASE_DATA_DIR) / uid)
        pos_df = cp.load_label_data(Path(cp.BASE_DATA_DIR) / uid, args.fruit, args.scenario)
        orig_neg = cp.load_label_data(Path(cp.BASE_DATA_DIR) / uid, args.fruit, "None")
        if len(orig_neg) < len(pos_df):
            extra = cp.derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        tr_u, val_u, te_u = cp.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        return tr_u, val_u, te_u, neg_df

def run_personal_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    tr_days_u: np.ndarray,
    val_days_u: np.ndarray,
    te_days_u: np.ndarray,
    neg_df_u: pd.DataFrame,
    sample_mode: str = "original",
    train_frac: int=None, 
    seed:int=None,
):
    print(f"\n>> Personal-SSL ({fruit}_{scenario})")

    # Directories
    out_dir   = user_root / 'personal_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)
    
    # 1) Load signals & labels
    hr_df, st_df  = cp.load_signal_data(Path(cp.BASE_DATA_DIR) / uid)
    pos_df = cp.load_label_data(Path(cp.BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df = neg_df_u

    # 2) Train or load SSL encoders
    enc_hr = cp._train_or_load_encoder(models_d / 'hr_encoder.keras',
                                    'hr', hr_df, tr_days_u, results_d)
    enc_st = cp._train_or_load_encoder(models_d / 'steps_encoder.keras',
                                    'steps', st_df, tr_days_u, results_d)

    # 3) Window & label for TRAIN/VAL/TEST
    df_tr  = cp.collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
    df_val = cp.collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
    df_te  = cp.collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)
    num_train_full = len(df_tr)
    # 4) Sampling summary
    train_info = {uid: {"days": tr_days_u.tolist(), "df": df_tr.copy()}}
    sample_summary = cp.sample_train_info(train_info, mode=sample_mode, random_state=42)


    # 6) Encode into feature vectors
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        return enc_hr.predict(hr_seq, verbose=0), enc_st.predict(st_seq, verbose=0)



    if train_frac is not None and 0 < train_frac < 1.0:
        df_tr, _ = train_test_split(df_tr, 
                train_size=train_frac, stratify=df_tr['state_val'], random_state=seed)
        df_tr = df_tr.reset_index(drop=True)
    else:
        df_tr = df_tr.reset_index(drop=True)

    H_tr, S_tr   = encode(df_tr)
    H_val, S_val = encode(df_val)              # NEW: encode validation set
    H_te, S_te   = encode(df_te)
    X_train = np.concatenate([H_tr, S_tr], axis=1)
    y_train = df_tr['state_val'].values
    X_val   = np.concatenate([H_val, S_val], axis=1)   # NEW: build X_val
    y_val   = df_val['state_val'].values               # NEW: build y_val
    X_test  = np.concatenate([H_te, S_te], axis=1)
    y_test  = df_te['state_val'].values

    # 7) Build & train classifier with explicit val_data and balanced weights
    cp.reset_seeds()  # Ensure deterministic model initialization
    clf = Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(0.5),
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    clf.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE, restore_best_weights=True, verbose=1)

    # NEW: compute balanced class weights
    cw_vals = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    
    hist = clf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),   # REVISED
        epochs=CLF_EPOCHS,
        batch_size=16,
        # class_weight=class_weight,        # REVISED
        callbacks=[es],
        verbose=2
    )
    
    cp.plot_clf_losses(hist.history['loss'], hist.history['val_loss'], results_d, 'personal_ssl_clf_loss')

    # 8) Threshold scan on the FULL training set
    best_thr = cp.select_threshold_train(clf, X_train, y_train)
    (results_d / "selected_threshold.txt").write_text(f"{best_thr:.4f}\n")

    # 9) Bootstrap & plot on TEST
    probs_te = clf.predict(X_test, verbose=0).ravel()
    ##saving test probs and labels for later use in aggregation
    np.save(results_d / f"test_probs_{uid}.npy", probs_te)
    np.save(results_d / f"test_labels_{uid}.npy", y_test)
    print(f"saved in {results_d / f'test_probs_{uid}.npy'} and {results_d / f'test_labels_{uid}.npy'} ")

    df_boot, auc_m, auc_s = cp.bootstrap_threshold_metrics(
        y_test,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    # probs_train = clf.predict(X_train, verbose=0).ravel()
    # probs_val = clf.predict(X_val, verbose=0).ravel()

    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)

    return df_boot, auc_m, auc_s, num_train_full



def main():
    args = parse_args()
    # _set_seed(args.seed)

    pool = "personal"
    top_out = Path(set_output_dir(pool))

    user_root = top_out / str(args.user) / f"{args.fruit}_{args.scenario}" / "fixed_seed_10pct"
    user_root.mkdir(parents=True, exist_ok=True)

    tr_days_u, val_days_u, te_days_u, neg_df_u = _prepare_personal_split(args)
    all_splits = {str(args.user): (tr_days_u, val_days_u, te_days_u)}

    orig_sample_train_info = cp.sample_train_info
    orig_reset_seeds = cp.reset_seeds

    try:
        ##setting BASE_DATA_DIR to bp data dir for loading raw data and
        cp.configure_task_mode(task="bp", bp_input="raw") 
        results_df = []
        for seed in args.seeds:
            
            cp.reset_seeds = lambda: _set_seed(seed)

            _, auc_mean, auc_std, num_train = run_personal_ssl(
                uid=str(args.user),
                fruit=str(args.fruit),
                scenario=str(args.scenario),
                user_root=user_root,
                tr_days_u=tr_days_u,
                val_days_u=val_days_u,
                te_days_u=te_days_u,
                neg_df_u=neg_df_u,
                train_frac=args.train_frac,
                seed=seed,
        )
            results_df.append({
                "seed": seed,
                "num_train": num_train,
                "auc_test_mean": auc_mean,
                "auc_test_std": auc_std,
                # "auc_train": auc_train,
                # "auc_val": auc_val,
            })
    finally:
        cp.sample_train_info = orig_sample_train_info
        cp.reset_seeds = orig_reset_seeds

    out_df = pd.DataFrame(results_df)

#     ##full data results 
    _, auc_mean_full, auc_std_full, num_train_full = run_personal_ssl(
                uid=str(args.user),
                fruit=str(args.fruit),
                scenario=str(args.scenario),
                user_root=user_root,
                tr_days_u=tr_days_u,
                val_days_u=val_days_u,
                te_days_u=te_days_u,
                neg_df_u=neg_df_u,
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

    def plot_seed_comparision(df_seed_results, user_root):
        tmp = df_seed_results.copy()
        tmp["auc_mean_num"] = pd.to_numeric(
            tmp["auc_test_mean"],
            errors="coerce",
        )
        seed_df = tmp[tmp["seed"] != "full_data"].copy()
        seed_df["seed"] = seed_df["seed"].astype(str)
        auc_100 = float(tmp.loc[tmp["seed"] == "full_data", "auc_mean_num"].dropna().iloc[0])

        x = np.arange(len(seed_df))

        plt.figure(figsize=(7, 4))
        plt.scatter(x, seed_df["auc_mean_num"], s=70, color="tab:blue")
        for i, row in seed_df.reset_index(drop=True).iterrows():
            plt.text(i, row["auc_mean_num"] + 0.002, f"seed {row['seed']}", ha="center", fontsize=9)

        plt.axhline(auc_100, color="red", linestyle="--", label=f"100% AUC = {auc_100:.3f}")
        plt.xticks(x, [f"seed {s}" for s in seed_df["seed"]])
        plt.ylabel("Test AUC mean")
        plt.title("Per-seed 10% AUC vs 100% AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(user_root / "seed_comparison.png")
    
    plot_seed_comparision(out_df, user_root) 


if __name__ == "__main__":
    main()
