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
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf


# from utility import stratified_split_min, run_al
import preprocess
from preprocess import prepare_data
from src.classifier_utils import BASE_DATA_DIR, load_signal_data, load_label_data
from src.compare_pipelines import derive_negative_labels, collect_windows

import uq_utility
import utility
import random
import numpy as np
import tensorflow as tf  
import pandas as pd 

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from src import compare_pipelines as compare_pipelines

DRYRUN = True


# OUTPUT_DIR = 'output_64units'
OUTPUT_DIR = 'Full_GS'
# dropout_rate = 0.3
# UNLABELED_FRAC = 0.9 

pa = argparse.ArgumentParser()
pa.add_argument("--user", default= "ID27" )
pa.add_argument("--pool", default="global_supervised")
pa.add_argument("--users", default="all", help="Comma-separated user IDs or 'all'")
pa.add_argument("--sample_mode", default="original")
pa.add_argument("--unlabeled_frac", default=0.7)
pa.add_argument("--dropout_rate", default=0.1)
# pa.add_argument("--T", default=30)
# pa.add_argument("--K", default=10)
# pa.add_argument("--Budget", default=10)
pa.add_argument("--results_subdir", default="results")

args, _ = pa.parse_known_args()

# user = args.user
# fruit = args.fruit
# scenario = args.scenario
unlabeled_frac = [float(args.unlabeled_frac)]
# unlabeled_frac = [0.7]

dropout_rate = [float(args.dropout_rate)]
# T = [30]
T = [100]
K = [40]
Budget = [None]

# Budget = args.Budget
RESULTS_SUBDIR = args.results_subdir
# aq_f = args.aq_f
aq_f = ["uncertainty", "random"]

BATCH_SSL, SSL_EPOCHS = 32, 100
# CLF_EPOCHS, CLF_PATIENCE = 500, 20
CLF_EPOCHS, CLF_PATIENCE = 500, 15

# Top‑level paths
# Use OUTPUT_DIR to match submit_batch.py
top_out = Path(OUTPUT_DIR) 
shared_enc_root = top_out / "_global_encoders"
shared_cnn_root = top_out / "global_cnns"

# Seed everything
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def reset_seeds(seed=42):
    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
rows = []
model_cache = {}

if args.users.strip().lower() == "all":
    target_users = list(preprocess.ALLOWED_SCENARIOS.keys())
else:
    target_users = [u.strip() for u in args.users.split(",") if u.strip()]

for user in target_users:
    scenarios = preprocess.ALLOWED_SCENARIOS.get(user, [])
    if not scenarios:
        print(f"Skipping user {user}: no scenarios in ALLOWED_SCENARIOS.")
        continue
    for fruit, scenario in scenarios:
        args.user = user
        args.fruit = fruit
        args.scenario = scenario
        try:
            prep = prepare_data(
                args=args,
                top_out=top_out,
                shared_enc_root=shared_enc_root,
                shared_cnn_root=shared_cnn_root,
                batch_ssl=BATCH_SSL,
                ssl_epochs=SSL_EPOCHS,
                pool=args.pool,
            )
        except (RuntimeError, ValueError, SystemExit) as e:
            print(f"Skipping {user} {fruit}/{scenario}: {e}")
            continue
        if prep is None:
            print(f"Skipping {user} {fruit}/{scenario}: prepare_data returned no data.")
            continue

        df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives = prep

        if args.pool != "global_supervised":
            print(f"Skipping {user} {fruit}/{scenario}: pool {args.pool} not supported here.")
            continue

        # Match ensure_global_supervised: validation uses the target user's val split
        cache_key = (fruit, scenario, user)
        if cache_key not in model_cache:
            if df_all_tr is None or len(df_all_tr) == 0:
                print(f"Skipping {user} {fruit}/{scenario}: empty pooled training set.")
                continue

            # Build pooled train/val from all_splits (matches run_global_supervised)
            train_info = {}
            for u, (tr_days, val_days, _) in all_splits.items():
                hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
                pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
                orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, "None")
                if len(orig_neg) < len(pos_df):
                    extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                    neg_df = pd.concat([orig_neg, extra], ignore_index=True)
                else:
                    neg_df = orig_neg

                df_tr_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
                train_info[u] = {"days": tr_days.tolist(), "df": df_tr_u}

            df_train_all = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)
            if len(df_train_all) == 0:
                print(f"Skipping {user} {fruit}/{scenario}: empty pooled train windows.")
                continue

            # Build target user's validation windows (per-user validation)
            tr_days_u, val_days_u, _ = all_splits[user]
            hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / user)
            pos_df_u = load_label_data(Path(BASE_DATA_DIR) / user, fruit, scenario)
            orig_neg_u = load_label_data(Path(BASE_DATA_DIR) / user, fruit, "None")
            if orig_neg_u.empty or len(orig_neg_u) < len(pos_df_u):
                neg_df_u = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
            else:
                neg_df_u = orig_neg_u
            df_val_u = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)

            Z_val_u, y_val_u = uq_utility._build_XY_from_windows(df_val_u)
            if len(Z_val_u) == 0:
                print(f"Skipping {user} {fruit}/{scenario}: empty user val windows.")
                continue

            try:
                _, _, model = uq_utility.train_base_init_on_labeled(
                    df_train_all,
                    Z_val_u, y_val_u,
                    uq_utility.build_global_cnn_lstm,
                    epochs=CLF_EPOCHS,
                    patience=CLF_PATIENCE,
                    verbose=0,
                )
            except RuntimeError as e:
                print(f"Skipping {user} {fruit}/{scenario}: {e}")
                continue

            Z_train_all, y_train_all = uq_utility._build_XY_from_windows(df_train_all)
            probs_train_all = model.predict(Z_train_all, verbose=0).ravel()
            auc_m_train, auc_s_train, _ = utility.bootstrap_auc(y_train_all, probs_train_all)

            model_cache[cache_key] = {
                "model": model,
                "auc_m_train": auc_m_train,
                "auc_s_train": auc_s_train,
                "train_count": len(df_train_all),
            }

        model = model_cache[cache_key]["model"]

        Z_te, y_te = uq_utility._build_XY_from_windows(df_te)
        if len(Z_te) == 0:
            print(f"Skipping {user} {fruit}/{scenario}: empty test windows.")
            continue

        probs_te = model.predict(Z_te, verbose=0).ravel()
        auc_m, auc_s, _ = utility.bootstrap_auc(y_te, probs_te)
        ll = log_loss(y_te, probs_te)

        rows.append({
            "user_id": user,
            "fruit": fruit,
            "scenario": scenario,
            "pool": args.pool,
            "train": model_cache[cache_key]["train_count"],
            "test": len(df_te),
            "auc_mean": auc_m,
            "auc_std": auc_s,
            "auc_mean_train": model_cache[cache_key]["auc_m_train"],
            "auc_std_train": model_cache[cache_key]["auc_s_train"],
            "log_loss": ll,
        })

df_table = pd.DataFrame(rows)
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
df_table.to_csv(out_dir / "df_table.csv", index=False)
print(f"Saved results to {out_dir / 'df_table.csv'}")
