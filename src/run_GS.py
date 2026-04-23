#!/usr/bin/env python3
"""
run_GS.py
=========
Run ONLY the Global-Supervised pipeline from compare_pipelines.py.

This script preprocesses only what Global-Supervised needs:
  - day-level train/val/test splits for all eligible users
  - negatives for the target user (for its TEST set)
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os 

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from src.compare_pipelines import (
    ALLOWED_SCENARIOS,
    BASE_DATA_DIR,
    load_signal_data,
    load_label_data,
    derive_negative_labels,
    ensure_train_val_test_days,
    ensure_global_supervised,
    run_global_supervised,
)


def main() -> int:
    pa = argparse.ArgumentParser()
    pa.add_argument("--user")
    pa.add_argument(
        "--users",
        nargs="+",
        help="Run Global-Supervised for a list of users.",
    )
    pa.add_argument("--fruit", required=True)
    pa.add_argument("--scenario", required=True)
    pa.add_argument("--output-dir", default="ALL_results")
    pa.add_argument(
        "--sample-mode",
        choices=["original", "undersample", "oversample"],
        default="original",
        help="How to balance classes in TRAIN/VAL: keep original, undersample negs, or oversample pos.",
    )
    args = pa.parse_args()

    if not args.user and not args.users:
        pa.error("Provide --user or --users.")
    if args.user and args.users:
        pa.error("Use only one of --user or --users.")

    # Seed everything for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Prepare output locations
    top_out = Path(args.output_dir)
    shared_cnn_root = top_out / "global_cnns"

    # Preprocess only what Global-Supervised needs:
    # - splits for all eligible users
    # - negatives for target users (for their TEST sets)
    all_splits = {}
    neg_by_user = {}

    for u, pairs in ALLOWED_SCENARIOS.items():
        if (args.fruit, args.scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")

        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        neg_by_user[u] = neg_df

        try:
            tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as e:
            print(f"Skipping user {u}: {e}")
            continue

        all_splits[u] = (tr_u, val_u, te_u)

    targets = [args.user] if args.user else args.users
    if not targets:
        print("No valid targets after filtering.")
        return 0

    # Force the shared CNN to be trained using the first user's VAL set,
    # matching compare_pipelines.py behavior for a single-user run.
    primary_uid = targets[0]
    shared_model_dir = shared_cnn_root / f"{args.fruit}_{args.scenario}"
    model_path = shared_model_dir / "cnn_classifier.keras"
    if model_path.exists():
        model_path.unlink()
    # Train (or re-train) the shared CNN once using primary user's VAL set.
    ensure_global_supervised(shared_cnn_root, args.fruit, args.scenario, all_splits, primary_uid)

    df = pd.DataFrame(columns=[
        "User",
        "AUC_GS_Mean",
        "AUC_GS_STD",
        "AUC_GS_Train",
        "AUC_GS_Val",
    ])
    for uid in targets:
        if uid not in all_splits:
            print(f"Skipping user {uid}: no data for {args.fruit}/{args.scenario}.")
            continue
        neg_df_u = neg_by_user.get(uid)
        if neg_df_u is None:
            print(f"Skipping user {uid}: negatives could not be prepared.")
            continue

        user_root = top_out / uid / f"{args.fruit}_{args.scenario}"
        user_root.mkdir(parents=True, exist_ok=True)

        # Run only the Global-Supervised pipeline
        df_gs, auc_gs_m, auc_gs_s, auc_gs_train, auc_gs_val  =run_global_supervised(
            args.fruit,
            args.scenario,
            uid,
            user_root,
            all_splits,
            shared_cnn_root,
            neg_df_u=neg_df_u,
            sample_mode=args.sample_mode,
        )
        df = pd.concat([df, pd.DataFrame({
            "User": [uid],
            "AUC_GS_Mean": [auc_gs_m],
            "AUC_GS_STD": [auc_gs_s],
            "AUC_GS_Train": [auc_gs_train],
            "AUC_GS_Val": [auc_gs_val],
        })], ignore_index=True)

    print(df)

    if len(df):
        df.to_csv(top_out / "global_supervised_summary.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
