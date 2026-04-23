import json
import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
import pickle

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from sklearn.calibration import calibration_curve
import tensorflow as tf


# from utility import stratified_split_min, run_al
import preprocess
from preprocess import prepare_data
import src.compare_pipelines as compare_pipelines
from src.chart_utils import bootstrap_threshold_metrics, plot_thresholds

import uq_utility
import utility
import random
import numpy as np
import tensorflow as tf  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


DRYRUN = True


OUTPUT_DIR = 'AUC_test'

argparser = argparse.ArgumentParser()
argparser.add_argument("--user", type=str, default="ID21", help="User ID")
argparser.add_argument(
    "--users",
    type=str,
    default="ID10, ID11, ID12, ID13, ID20, ID21, ID27",
    help="Comma-separated list of user IDs (overrides --user if provided)",
)
argparser.add_argument("--pool", type=str, default="global_supervised", help="Data pool")
argparser.add_argument("--fruit", type=str, default="Nectarine", help=" Fruit type")            
argparser.add_argument("--scenario", type=str, default="Crave", help="Scenario")
argparser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
argparser.add_argument("--unlabeled_frac", type=float, default=0.7, help="Fraction of unlabeled data")
args = argparser.parse_args()

def _parse_users(users_str, fallback_user):
    users_str = users_str.strip()
    if not users_str:
        return [fallback_user]
    return [u.strip() for u in users_str.split(",") if u.strip()]

users = _parse_users(args.users, args.user)

pool = args.pool
fruit = args.fruit
scenario = args.scenario
dropout_rate = float(args.dropout_rate)
unlabeled_frac = float(args.unlabeled_frac)

# T = [30]
T = [50]
K = [15]
Budget = [None]


aq_f = "uncertainty"

BATCH_SSL, SSL_EPOCHS = 32, 100
# CLF_EPOCHS, CLF_PATIENCE = 500, 20
CLF_EPOCHS, CLF_PATIENCE = 500, 15

# Top‑level paths
# Use OUTPUT_DIR to match submit_batch.py
top_out = Path(OUTPUT_DIR) 
shared_enc_root = top_out / "_global_encoders"
shared_cnn_root = top_out / "global_cnns"




def _has_min_test_windows(y_test):
    if y_test is None or len(y_test) < 2:
        return False
    classes = np.unique(y_test)
    return classes.size >= 2


def _filter_zero_hr_seq(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0 or "hr_seq" not in df.columns:
        return df
    mask = df["hr_seq"].apply(lambda seq: not np.all(np.asarray(seq) == 0))
    return df[mask].copy()


def _compute_auc_for_user(user_id):
    user_args = argparse.Namespace(**vars(args))
    user_args.user = user_id

    try:
        (
            df_tr,
            df_all_tr,
            df_val,
            df_te,
            enc_hr,
            enc_st,
            user_root,
            all_splits,
            models_d,
            results_d,
        ) = prepare_data(
            args=user_args,
            top_out=top_out,
            shared_enc_root=shared_enc_root,
            batch_ssl=BATCH_SSL,
            ssl_epochs=SSL_EPOCHS,
            pool=pool,
        )
    except (Exception, SystemExit):
        print(
            f"Skipping user {user_id}: Unable to find a TEST split with ≥2 windows and ≥1 positives & negatives after 10 tries"
        )
        return None

    # Remove signals with constant-zero hr_seq
    # df_tr = _filter_zero_hr_seq(df_tr)
    # df_val = _filter_zero_hr_seq(df_val)
    # df_te = _filter_zero_hr_seq(df_te)

    X_test = np.stack([
        np.vstack([h, s]).T
        for h, s in zip(df_te["hr_seq"], df_te["st_seq"])
    ])
    y_test = df_te["state_val"].values

    if not _has_min_test_windows(y_test):
        print(
            f"Skipping user {user_id}: Unable to find a TEST split with ≥2 windows and ≥1 positives & negatives after 10 tries"
        )
        return None 

    gs_init_model = None
    if pool == "global_supervised":
        global _shared_gs_model, _shared_gs_src_dir, _shared_gs_uid
        if _shared_gs_model is None:
            gs_init_model, gs_src_dir = compare_pipelines.ensure_global_supervised(
                shared_cnn_root, fruit, scenario, all_splits, user_id
            )
            _shared_gs_model = gs_init_model
            _shared_gs_src_dir = gs_src_dir
            _shared_gs_uid = user_id
            print(f"Trained shared global_supervised model using val split from {user_id}")
        else:
            gs_init_model = _shared_gs_model
            gs_src_dir = _shared_gs_src_dir
       
    if gs_init_model is None:
        print(f"Skipping user {user_id}: global_supervised model unavailable.")
        return None

    # Representation from penultimate layer (concatenated CNN+LSTM features)
    rep_model = tf.keras.Model(gs_init_model.input, gs_init_model.layers[-2].output)
    Z_test_rep = rep_model.predict(X_test, verbose=0)
    rep_dir = top_out / "representations"
    rep_dir.mkdir(parents=True, exist_ok=True)
    np.save(rep_dir / f"{user_id}_Z_test_rep.npy", Z_test_rep)

    df_train = df_all_tr if df_all_tr is not None else df_tr
    X_train = np.stack([
        np.vstack([h, s]).T
        for h, s in zip(df_train["hr_seq"], df_train["st_seq"])
    ])
    y_train = df_train["state_val"].values

    best_thr = compare_pipelines.select_threshold_train(gs_init_model, X_train, y_train)
    (results_d / "selected_threshold.txt").write_text(f"{best_thr:.4f}\n")

    probs_te = gs_init_model.predict(X_test, verbose=0).ravel()
    df_boot, test_auc_m, test_auc_s = bootstrap_threshold_metrics(
        y_test,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(
        y_test,
        probs_te,
        str(results_d),
        f"{user_id} {fruit}_{scenario} (global_supervised)",
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000
    )
    print(f"{user_id}: TestAUC={test_auc_m:.3f}±{test_auc_s:.3f}")
    counts_str = f"{len(df_tr)}/{len(df_val)}/{len(df_te)}"
    return test_auc_m,test_auc_s, counts_str


results = []

# cache one shared global_supervised model per run
_shared_gs_model = None
_shared_gs_src_dir = None
_shared_gs_uid = None

for user_id in users:
    result = _compute_auc_for_user(user_id)
    if result is not None:
        auc_value, test_auc_s, counts_str = result
        results.append({"user_id": user_id, "auc": auc_value, "auc_std": test_auc_s, "tr/val/te": counts_str})

results_df = pd.DataFrame(results)

print(results_df)

# Load saved representations and run PCA for visualization
rep_dir = top_out / "representations"
rep_users = []
rep_means = []
for user_id in users:
    rep_path = rep_dir / f"{user_id}_Z_test_rep.npy"
    if not rep_path.exists():
        continue
    Z = np.load(rep_path)
    rep_users.append(user_id)
    rep_means.append(Z.mean(axis=0))

if len(rep_means) >= 2:
    X = np.stack(rep_means, axis=0)
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], s=60)
    for (x, y), uid in zip(X2, rep_users):
        plt.text(x, y, uid, fontsize=9, ha="left", va="bottom")
    plt.title("PCA of test-set representations (mean per user)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(rep_dir / "pca_test_rep_means.png", dpi=150)
    plt.close()
elif len(rep_means) == 1:
    print("PCA skipped: only one user representation found.")
else:
    print("PCA skipped: no saved representations found.")
