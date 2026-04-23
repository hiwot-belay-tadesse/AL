
from pathlib import Path
import argparse
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import shutil
import os 
from sklearn.utils import class_weight

# from LR_check import plot_aggregate_auc 

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

CLF_EPOCHS, CLF_PATIENCE                = 200, 15



def plot_aggregate_full_data(df_full_results, top_out, out_path):
    '''
    plots the aggregate of 100% AUC across users, 
    with a point for each user and a line for the pooled AUC.
    '''

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
        
def plot_aggregate_10pct(
    df_seed_results: pd.DataFrame,
    seed_to_plot: int,
    run_tag: str,
    out_path: Path,
):
    """
    Plot accross users 10% AUC for a single fixed seed
    """
    df = df_seed_results.copy()
    df = df[df["seed"] == seed_to_plot].copy()
    if df.empty:
        print(f"No rows found for seed={seed_to_plot}; skipping aggregate 10% plot.")
        return

    df = df.sort_values("user").reset_index(drop=True)
    x = np.arange(len(df))

    plt.figure(figsize=(8, 4))
    plt.scatter(x, df["agg_auc_mean"], s=70)
    for i, row in df.iterrows():
        plt.text(i, row["agg_auc_mean"] + 0.002, f"user {row['user']}", ha="center", fontsize=8)

    # Pooled AUC across all users for this fixed seed, loaded from seed-tagged files.
    all_true_parts = []
    all_pred_parts = []
    for _, row in df.iterrows():
        uid = str(row["user"])
        user_root_i = Path(row["user_root"])
        labels_path = user_root_i / "global_ssl" / "results" / f"test_labels_{uid}_{run_tag}.npy"
        probs_path = user_root_i / "global_ssl" / "results" / f"test_probs_{uid}_{run_tag}.npy"
        if labels_path.exists() and probs_path.exists():
            all_true_parts.append(np.load(labels_path))
            all_pred_parts.append(np.load(probs_path))

    if all_true_parts and all_pred_parts:
        all_true = np.concatenate(all_true_parts)
        all_pred = np.concatenate(all_pred_parts)
        pooled_auc, _, _ = bootstrap_auc(all_true, all_pred)
        plt.axhline(
            pooled_auc,
            color="black",
            linestyle="--",
            label=f"Pooled AUC = {pooled_auc:.3f}",
        )

    plt.xticks(x, [str(u) for u in df["user"]], rotation=45, ha="right")
    plt.ylabel("Test AUC mean")
    plt.xlabel("User")
    plt.title(f"Aggregate 10% AUC Across Users (seed={seed_to_plot})")
    if all_true_parts and all_pred_parts:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

def plot_seed_comparision(df_seed_results, user_root):
    '''
    plots the per-seed AUC results for the 10% data vs the full-data AUC, for a single user.
    
    '''
    tmp = df_seed_results.copy()
    tmp["auc_mean_num"] = pd.to_numeric(
        tmp["auc_test_mean"],
        errors="coerce",
    )
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

def _prepare_global_splits(args, requested_users: list[str]):
    '''
    Prepares all_splits and all_negatives  for the Global_SSL pipeline, 
    which are used for both the Global_SSL and P_Global_SSL pipelines. 
    '''
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
            all_negatives[pid] = neg_df
            try:
                tr_u, val_u, te_u = cp.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue
            all_splits[pid] = (tr_u, val_u, te_u)

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

            all_negatives[u] = neg_df
            try:
                tr_u, val_u, te_u = cp.ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {u}: {e}")
                continue
            all_splits[u] = (tr_u, val_u, te_u)

    missing = [u for u in requested_users if u not in all_splits]
    if missing:
        print(f"Skipping users with no valid splits: {missing}")

    return all_splits, all_negatives