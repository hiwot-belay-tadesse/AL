#!/usr/bin/env python3
"""
compare_pipelines.py
====================

Now:
  - One deterministic train/test split per user (seeded at top).
  - Global-SSL & Global-Supervised each train only on train-day windows.
  - One shared CNN per (fruit,scenario) saved under `global_cnns/...`
    (like encoders under `_global_encoders/...`).
  - Shared CNN and encoders get copied into each user’s folder.

Optional class balancing (--sample-mode):
  - Modes: original, undersample, oversample (default=original).
  - Undersample: “round_robin_undersample” drops excess negatives
    across users in a round‐robin fashion until total_neg == total_pos.
  - Oversample: “round_robin_oversample” duplicates positive windows
    (with small Gaussian jitter on their hr_seq/st_seq) in turn
    until total_pos == total_neg.
  - All original vs. new counts (per‐user and global) plus number of
    added/removed samples are logged in each pipeline’s split_details.txt:
      – Global pipelines show a “GLOBAL …” summary and per‐user breakdown.
      – Personal‐SSL appends a “POST-SAMPLING SUMMARY” block.
  - To use: pass `--sample-mode undersample` or `--sample-mode oversample`
    when invoking the script; omit or use `original` to leave data untouched.
"""

import argparse, warnings, os, shutil, sys, random
from pathlib import Path
import os

from sklearn.utils import class_weight 

os.environ["KERAS_BACKEND"] = "tensorflow"

import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from sklearn.metrics import confusion_matrix

# ─── BP task helpers ───────────────────────────────────────────────────────
import re

BP_MODE = False
FORCE_RETRAIN = True


def configure_task_mode(task: str, bp_input: str = "raw"):
    global BASE_DATA_DIR, load_signal_data, load_label_data, BP_MODE

    if task == "bp":
        BP_MODE = True
        BASE_DATA_DIR = "DATA/Cardiomate/"

        if bp_input == "raw":
            def _bp_load_signal_data(user_dir):
                pid = _bp_pid_from_user_dir(user_dir)
                hr_df, st_df, _, _ = _bp_load_all(pid)
                return hr_df, st_df

            def _bp_load_label_data(user_dir, fruit, scenario):
                pid = _bp_pid_from_user_dir(user_dir)
                _, _, pos_df, neg_df = _bp_load_all(pid)
                return neg_df.copy() if scenario == "None" else pos_df.copy()

            load_signal_data = _bp_load_signal_data
            load_label_data = _bp_load_label_data
    else:
        BP_MODE = False
        # optionally reset BASE_DATA_DIR/loaders to fruit defaults


def _bp_pid_from_user_dir(user_dir):
    name = Path(user_dir).name
    m = re.search(r"\d+", name)
    if not m:
        raise ValueError(f"Could not infer participant id from '{user_dir}'")
    return m.group(0)

def _bp_load_signal_df(path, value_col="value"):
    '''
    loads raw hr, steps and bp labels
    '''
    df = pd.read_csv(path)
    if "time" not in df.columns:
        if "datetime_local" in df.columns:
            df = df.rename(columns={"datetime_local": "time"})
        else:
            raise ValueError(f"Signal file {path} missing 'time' column")
    if value_col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError(f"Signal file {path} missing numeric value column")
        value_col = num_cols[0]
    df = df[["time", value_col]].rename(columns={value_col: "value"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df.sort_values("time", inplace=True)
    df = df.groupby("time", as_index=False)["value"].mean()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["value"] = df["value"].ffill()
    df.set_index("time", inplace=True)
    ##Reindex to a 1-minute grid and forward fill gaps
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")
    df = df.reindex(idx, method="ffill")
    return df

def _bp_load_all(pid, sys_thresh=130, dia_thresh=80):
    '''
    given user id, it loads hp{pid}_hr.csv, hp{pid}_steps.csv & blood_pressure_readings_ID{pid}_cleaned.csv
    and create labels based on sys_threshold abd dia_threshold
    '''
    base = Path("DATA/Cardiomate/hp") / f"hp{pid}"
    file_hr = base / f"hp{pid}_hr.csv"
    file_steps = base / f"hp{pid}_steps.csv"
    file_bp = base / f"blood_pressure_readings_ID{pid}_cleaned.csv"
    if not (file_hr.exists() and file_steps.exists() and file_bp.exists()):
        raise FileNotFoundError(f"Missing BP files under {base}")

    hr_df = _bp_load_signal_df(file_hr, value_col="value")
    st_df = _bp_load_signal_df(file_steps, value_col="value")
    st_df = st_df.fillna(0)

    df_bp = pd.read_csv(file_bp)
    if "datetime_local" not in df_bp.columns:
        raise ValueError(f"BP file {file_bp} missing 'datetime_local'")
    df_bp["datetime_local"] = pd.to_datetime(df_bp["datetime_local"]).dt.tz_localize(None)
    if "systolic" not in df_bp.columns or "diastolic" not in df_bp.columns:
        raise ValueError(f"BP file {file_bp} missing 'systolic'/'diastolic'")
    df_bp["BP_spike"] = (
        (df_bp["systolic"] > sys_thresh) | (df_bp["diastolic"] > dia_thresh)
    ).astype(int)
    df_bp = df_bp.dropna(subset=["datetime_local"]).sort_values("datetime_local")
    # ── First day removal ──────────────────────────────────────
    # first_bp = df_bp["datetime_local"].dt.date.min()
    # first_hr = hr_df.index.date.min()
    # cutoff   = min(first_bp, first_hr)

    # df_bp  = df_bp[df_bp["datetime_local"].dt.date   > cutoff]
    # hr_df  = hr_df[hr_df.index.date                  > cutoff]
    # st_df  = st_df[st_df.index.date                  > cutoff]

    pos_df = df_bp[df_bp["BP_spike"] == 1][["datetime_local"]].copy()
    neg_df = df_bp[df_bp["BP_spike"] == 0][["datetime_local"]].copy()
    ## rename the time column to hawaii_createdat_time to match what collect_windows expect
    pos_df = pos_df.rename(columns={"datetime_local": "hawaii_createdat_time"})
    neg_df = neg_df.rename(columns={"datetime_local": "hawaii_createdat_time"})

    return hr_df, st_df, pos_df.reset_index(drop=True), neg_df.reset_index(drop=True)

def _bp_load_processed_data(pid):
    in_file = Path("processed") / f"hp{pid}" / "processed_bp_prediction_data.csv"
    if not in_file.exists():
        raise FileNotFoundError(f"Missing processed BP file: {in_file}")

    df = pd.read_csv(in_file)
    if "datetime_local" not in df.columns:
        raise ValueError(f"{in_file} missing 'datetime_local'")
    if "BP_spike" not in df.columns:
        raise ValueError(f"{in_file} missing 'BP_spike'")

    df["datetime_local"] = pd.to_datetime(df["datetime_local"]).dt.tz_localize(None)
    df = df.sort_values("datetime_local").reset_index(drop=True)
    df["state_val"] = df["BP_spike"].astype(int)
    return df

def _bp_feature_cols_from_processed(df):
    drop_cols = {
        "id", "user_id", "reading_id",
        "datetime_local", "datetime", "created_at", "time",
        "device_type", "data_type_hr", "data_type_steps",
        "BP_spike", "state_val", "systolic", "diastolic",
        "time_since_last_BP_spike", 
        "BP_spike_lag_1", "BP_spike_lag_3", "BP_spike_lag_5"
        "user_id"
        
    }
    drop_window_tokens = ("10min", "30min", "60min")
    return [
        c for c in df.columns
        if c not in drop_cols
        and "stress" not in c.lower()
        and not any(tok in c for tok in drop_window_tokens)
    ]
# ─── Seed everything for split reproducibility ─────────────────────────────
import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Additional TensorFlow determinism
tf.config.experimental.enable_op_determinism()

def reset_seeds():
    """Reset all random seeds for maximum reproducibility"""
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

# ─── project helpers ───────────────────────────────────────────────────────
from src.classifier_utils import (
    BASE_DATA_DIR, ALLOWED_SCENARIOS,
    load_signal_data, load_label_data, process_label_window
)
from src.signal_utils import (
    WINDOW_SIZE, STEP_SIZE, create_windows,
    build_simclr_encoder, create_projection_head, train_simclr
)
from src.chart_utils import (
    bootstrap_threshold_metrics, plot_thresholds, plot_ssl_losses
)

# ─── Hyperparameters ──────────────────────────────────────────────────────
# GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 5 ###New for BP SPIKE prediciton
GS_EPOCHS, GS_BATCH, GS_LR, GS_PATIENCE = 200, 32, 1e-3, 7 ###New for Banaware prediciton

BATCH_SSL, SSL_EPOCHS                   = 32, 100
CLF_EPOCHS, CLF_PATIENCE                = 200, 15
WINDOW_LEN                              = WINDOW_SIZE


# ─── Plot helpers ─────────────────────────────────────────────────────────
def plot_clf_losses(train, val, out_dir, fname):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(train, label="Train")
    plt.plot(val,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Binary CE")
    plt.title(fname.replace('_', ' ').title())
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.png")
    plt.close()

# New helper: sample timestamps where no event occurs
def derive_negative_labels(hr_df, pos_df, n_samples,
                          window_hours=1, random_state=42):
    """
    Sample up to n_samples timestamps from hr_df.index such that
    no pos_df event occurs within ±window_hours of any sampled time.
    """
    # round to minute
    all_times = pd.DatetimeIndex(hr_df.index.round('min').unique()).sort_values()
    event_times = pd.DatetimeIndex(
        pos_df['hawaii_createdat_time'].dt.round('min').unique()
    ).sort_values()

    # Build mask of valid times
    valid = []
    w = pd.Timedelta(hours=window_hours)
    for t in all_times:
        # check no event within ±w
        if not ((event_times >= (t - w)) & (event_times <= (t + w))).any():
            valid.append(t)

    if not valid:
        return pd.DataFrame(columns=['hawaii_createdat_time'])

    k       = min(len(valid), n_samples)
    sampled = pd.to_datetime(np.random.default_rng(random_state).choice(valid, size=k, replace=False))
    return pd.DataFrame({'hawaii_createdat_time': sampled})

# ─── Split / window helpers ────────────────────────────────────────────────
def _train_test_days_by_samples(pos_df, neg_df, hr_df, st_df):
    events = pd.concat([pos_df, neg_df])
    if events.empty:
        return np.array([]), np.array([])

    days = list(np.sort(events['hawaii_createdat_time'].dt.date.unique()))
    counts = []
    for d in days:
        p = pos_df[pos_df['hawaii_createdat_time'].dt.date == d]
        n = neg_df[neg_df['hawaii_createdat_time'].dt.date == d]
        rows = pd.concat([
            process_label_window(p, hr_df, st_df, 1),
            process_label_window(n, hr_df, st_df, 0)
        ])
        counts.append(len(rows))

    total = sum(counts)
    if total == 0:
        random.shuffle(days)
        cut = int(round(0.75 * len(days)))
        return np.array(days[:cut]), np.array(days[cut:])

    day_counts = list(zip(days, counts))
    random.shuffle(day_counts)

    target = 0.25 * total
    te_days, cum = [], 0
    for day, cnt in day_counts:
        te_days.append(day)
        cum += cnt
        if cum >= target:
            break

    tr_days = [d for d, _ in day_counts if d not in te_days]
    tr_days.sort(); te_days.sort()
    return np.array(tr_days), np.array(te_days)

# ─── New day‐split helper: 60/20/20 stratified days ────────────────────────
from sklearn.model_selection import train_test_split

# ─── Hyperparameters ──────────────────────────────────────────────────────
MIN_TEST_WINDOWS      = 2       # 
MIN_SAMPLES_PER_CLASS = 2       # new: require ≥1 pos & ≥1 neg in TEST


def undersample_negatives(X, y, random_state=42):
    """
    Down-sample negatives to match positives.
    Returns (X_balanced, y_balanced).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos   = len(pos_idx)
    if n_pos == 0 or len(neg_idx) <= n_pos:
        return X, y
    rng = np.random.default_rng(random_state)
    neg_keep = rng.choice(neg_idx, size=n_pos, replace=False)
    keep = np.concatenate([pos_idx, neg_keep])
    rng.shuffle(keep)
    return X[keep], y[keep]

def oversample_positives(X, y, random_state=42, noise_scale=0.05):
    """
    Up-sample positives with Gaussian jitter to match negatives.
    Returns (X_balanced, y_balanced, n_added).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos   = len(pos_idx)
    n_neg   = len(neg_idx)
    if n_pos == 0 or n_pos >= n_neg:
        return X, y, 0

    rng      = np.random.default_rng(random_state)
    picks    = rng.choice(pos_idx, size=(n_neg - n_pos), replace=True)
    X_pos    = X[picks]
    std_dev  = X_pos.std(axis=0, ddof=1)
    noise    = rng.normal(0, std_dev * noise_scale, size=X_pos.shape)
    X_new    = X_pos + noise
    y_new    = np.ones(len(X_new), dtype=y.dtype)

    Xb = np.vstack([X, X_new])
    yb = np.concatenate([y, y_new])
    return Xb, yb, len(X_new)

def round_robin_undersample(train_info, random_state=42):
    # train_info[u] is a dict {'days':…, 'df': DataFrame}
    total_pos = sum(int(info["df"]["state_val"].sum()) for info in train_info.values())
    total_neg = sum(int((info["df"]["state_val"]==0).sum()) for info in train_info.values())
    to_remove = total_neg - total_pos
    if to_remove <= 0:
        return
    rng   = np.random.default_rng(random_state)
    users = list(train_info.keys())
    removed = 0
    idx = 0
    while removed < to_remove and users:
        u = users[idx % len(users)]
        df = train_info[u]["df"]
        negs = df.index[df["state_val"] == 0].tolist()
        if not negs:
            users.remove(u)
        else:
            drop = rng.choice(negs)
            train_info[u]["df"] = df.drop(drop).reset_index(drop=True)
            removed += 1
        idx += 1

def round_robin_oversample(train_info, random_state=42, noise_scale=0.05):
    """
    Evenly add synthetic positives across users until pos == neg globally.
    Modifies train_info[u]["df"] in place.
    Returns total added.
    """
    rng = np.random.default_rng(random_state)

    # 1) Compute how many to add
    total_pos = sum(int(info["df"]["state_val"].sum()) for info in train_info.values())
    total_neg = sum(int((info["df"]["state_val"] == 0).sum()) for info in train_info.values())
    to_add = total_neg - total_pos
    if to_add <= 0:
        return 0

    # Helper to ensure seq → 1D float array
    def _flatten_to_float(seq):
        return np.asarray(seq, dtype=float)

    users = list(train_info.keys())
    added = 0
    idx = 0

    # 2) Round-robin over users
    while added < to_add and users:
        u = users[idx % len(users)]
        df = train_info[u]["df"]

        # pick by positional index
        pos_idx = np.flatnonzero(df["state_val"].values == 1)
        if pos_idx.size == 0:
            users.remove(u)
        else:
            # sample one positive
            pick = rng.choice(pos_idx)
            row = df.iloc[pick].copy()  # always a Series

            # flatten and jitter
            hr = _flatten_to_float(row["hr_seq"])
            st = _flatten_to_float(row["st_seq"])
            hr_std, st_std = hr.std(ddof=1), st.std(ddof=1)
            hr += rng.normal(0, hr_std * noise_scale, size=hr.shape)
            st += rng.normal(0, st_std * noise_scale, size=st.shape)

            # write back
            row["hr_seq"] = hr.tolist()
            row["st_seq"] = st.tolist()

            # append the synthetic row
            train_info[u]["df"] = pd.concat([df, row.to_frame().T], ignore_index=True)
            added += 1

        idx += 1

    return added

def sample_train_info(train_info, mode, random_state=42):
    """
    Apply sampling to train_info and return:
      {
        "global": {orig_pos, orig_neg, new_pos, new_neg, added, removed},
        "per_user": {
           uid: {orig_pos, orig_neg, new_pos, new_neg}, ...
        }
      }
    """
    # capture originals
    per_user = {}
    for u, info in train_info.items():
        df = info["df"]
        per_user[u] = {
            "orig_pos": int(df["state_val"].sum()),
            "orig_neg": int((df["state_val"]==0).sum())
        }
    g = per_user  # alias

    orig_pos = sum(u["orig_pos"] for u in g.values())
    orig_neg = sum(u["orig_neg"] for u in g.values())

    added = removed = 0
    if mode == "undersample":
        round_robin_undersample(train_info, random_state)
    elif mode == "oversample":
        added = round_robin_oversample(train_info, random_state)
    # recompute per-user new
    for u, info in train_info.items():
        df = info["df"]
        per_user[u].update({
            "new_pos": int(df["state_val"].sum()),
            "new_neg": int((df["state_val"]==0).sum())
        })

    new_pos = sum(u["new_pos"] for u in per_user.values())
    new_neg = sum(u["new_neg"] for u in per_user.values())
    removed = orig_neg - new_neg

    return {
        "global": {
          "orig_pos": orig_pos, "orig_neg": orig_neg,
          "new_pos":  new_pos,  "new_neg":  new_neg,
          "added":    added,    "removed": removed
        },
        "per_user": per_user
    }

#=====================================================================
# Helper 1: Day Split without Validation
#=====================================================================

def ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df, df_processed=None, input_df='raw'):
    """
    1) Split off ~20% of days as TEST (no strat).
    2) From remaining ~80%, stratify into 75/25 → TRAIN (~60%) / VAL (~20%).
    3) Retry until TEST has ≥MIN_TEST_WINDOWS total windows AND
       at least MIN_SAMPLES_PER_CLASS positives AND negatives.
    """
    if input_df == "raw":
        days = np.array(sorted(
            pd.concat([pos_df, neg_df])["hawaii_createdat_time"]
            .dt.date.unique()
        ))
        # Precompute window counts per day for stratification
        day_counts = {
            d: _count_windows(
                df_p=pos_df,
                df_n=neg_df,
                hr_df=hr_df,
                st_df=st_df,
                days=[d],
                input_df="raw",
            )
            for d in days
        }
    elif input_df == "processed":
        if df_processed is None:
            raise ValueError("processed mode requires df_processed")
        if "datetime_local" not in df_processed.columns:
            raise ValueError("df_processed must contain 'datetime_local'")

        df_processed = df_processed.copy()
        df_processed["datetime_local"] = pd.to_datetime(
            df_processed["datetime_local"]
        ).dt.tz_localize(None)

        days = np.array(sorted(df_processed["datetime_local"].dt.date.unique()))
        # Precompute row counts per day for stratification
        day_counts = {
            d: _count_windows(
                days=[d],
                df_processed=df_processed,
                input_df="processed",
                df_p=None,
                df_n=None,
                hr_df=None,
                st_df=None,
            )
            for d in days
        }
    else:
        raise ValueError(f"Unknown input_df: {input_df}")

    # Use a deterministic approach - try different test_size values instead of random_state
    test_sizes = [0.2, 0.18, 0.22, 0.16, 0.24, 0.15, 0.25, 0.17, 0.19, 0.21]

    for i, test_size in enumerate(test_sizes):
        # 1) carve off TEST - always use same random_state
        trval_days, test_days = train_test_split(
            days, test_size=test_size, random_state=42
        )
        # check total windows
        n_test = sum(day_counts[d] for d in test_days)
        if n_test < MIN_TEST_WINDOWS:
            continue

        # check per-class windows in TEST
        if input_df == "raw":
            df_te = collect_windows(pos_df, neg_df, hr_df, st_df, test_days)
            y_te = df_te["state_val"]
        else:
            df_te = collect_rows(df_processed, test_days)
            if "state_val" in df_te.columns:
                y_te = df_te["state_val"]
            elif "BP_spike" in df_te.columns:
                y_te = df_te["BP_spike"]
            else:
                raise ValueError("Processed rows must contain 'state_val' or 'BP_spike'")

        pos_te  = y_te.sum()
        neg_te  = len(df_te) - pos_te
        if pos_te < MIN_SAMPLES_PER_CLASS or neg_te < MIN_SAMPLES_PER_CLASS:
            continue

        # 2) stratify TRAIN/VAL on remaining days - always use same random_state
        trval_counts = [day_counts[d] for d in trval_days]
        try:
            train_days, val_days = train_test_split(
                trval_days,
                test_size=0.25,               # 0.25 of the 80% → 20% overall
                stratify=trval_counts,
                random_state=42  # Fixed random state
            )
        except ValueError:
            train_days, val_days = train_test_split(
                trval_days,
                test_size=0.25,
                random_state=42  # Fixed random state
            )

        # final sanity check: VAL must have at least MIN_TEST_WINDOWS as well
        n_val = sum(day_counts[d] for d in val_days)
        if n_val < MIN_TEST_WINDOWS:
            continue

        return np.array(train_days), np.array(val_days), np.array(test_days)

    raise RuntimeError(
        f"Unable to find a TEST split with ≥{MIN_TEST_WINDOWS} windows "
        f"and ≥{MIN_SAMPLES_PER_CLASS} positives & negatives after 10 tries"
    )


#=====================================================================
# Helper 2: Threshold Selection on Full Training Set
#=====================================================================
from sklearn.metrics import confusion_matrix

def select_threshold_train(model, X, y,
                              thresholds=np.arange(0.0, 1.0001, 0.01)):
    """
    Pick the threshold that maximizes 0.7*TPR + 0.3*Specificity.
    Enhanced for deterministic results.
    """
    y_true = y.astype(int)
    probs  = model.predict(X, verbose=0).ravel()

    best_score = -1.0
    best_thr   = 0.5
    
    # Sort thresholds for deterministic tie-breaking
    thresholds = np.sort(thresholds)

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        # require both classes present
        if len(np.unique(preds)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
        
        # Add small epsilon for numerical stability
        eps = 1e-10
        tpr  = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp + eps) if (tn + fp) > 0 else 0.0

        score = 0.7 * tpr + 0.3 * spec
        # Use strict > for consistent tie-breaking (favor smaller thresholds)
        if score > best_score:
            best_score = score
            best_thr   = thr

    return float(best_thr)

def ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df, df_processed=None, input_df='raw'):
    """
    1) Split off ~20% of days as TEST (no strat).
    2) From remaining ~80%, stratify into 75/25 → TRAIN (~60%) / VAL (~20%).
    3) Retry until TEST has ≥MIN_TEST_WINDOWS total windows AND
       at least MIN_SAMPLES_PER_CLASS positives AND negatives.
    """
    if input_df == "raw":
        days = np.array(sorted(
            pd.concat([pos_df, neg_df])["hawaii_createdat_time"]
            .dt.date.unique()
        ))
        # Precompute window counts per day for stratification
        day_counts = {
            d: _count_windows(
                df_p=pos_df,
                df_n=neg_df,
                hr_df=hr_df,
                st_df=st_df,
                days=[d],
                input_df="raw",
            )
            for d in days
        }
    elif input_df == "processed":
        if df_processed is None:
            raise ValueError("processed mode requires df_processed")
        if "datetime_local" not in df_processed.columns:
            raise ValueError("df_processed must contain 'datetime_local'")

        df_processed = df_processed.copy()
        df_processed["datetime_local"] = pd.to_datetime(
            df_processed["datetime_local"]
        ).dt.tz_localize(None)

        days = np.array(sorted(df_processed["datetime_local"].dt.date.unique()))
        # Precompute row counts per day for stratification
        day_counts = {
            d: _count_windows(
                days=[d],
                df_processed=df_processed,
                input_df="processed",
                df_p=None,
                df_n=None,
                hr_df=None,
                st_df=None,
            )
            for d in days
        }
    else:
        raise ValueError(f"Unknown input_df: {input_df}")

    test_sizes = [0.2, 0.18, 0.22, 0.16, 0.24, 0.15, 0.25, 0.17, 0.19, 0.21]
    for i, test_size in enumerate(test_sizes):
        # 1) carve off TEST - always use same random_state
        trval_days, test_days = train_test_split(
            days, test_size=test_size, random_state=42
        )
        # check total windows
        n_test = sum(day_counts[d] for d in test_days)
        if n_test < MIN_TEST_WINDOWS:
            continue

        # check per-class windows in TEST
        df_te   = collect_windows(pos_df, neg_df, hr_df, st_df, test_days)
        pos_te  = df_te['state_val'].sum()
        neg_te  = len(df_te) - pos_te
        if pos_te < MIN_SAMPLES_PER_CLASS or neg_te < MIN_SAMPLES_PER_CLASS:
            continue

        # 2) stratify TRAIN/VAL on remaining days - always use same random_state
        trval_counts = [day_counts[d] for d in trval_days]
        try:
            train_days, val_days = train_test_split(
                trval_days,
                test_size=0.25,               # 0.25 of the 80% → 20% overall
                stratify=trval_counts,
                random_state=42  # Fixed random state
            )
        except ValueError:
            train_days, val_days = train_test_split(
                trval_days,
                test_size=0.25,
                random_state=42  # Fixed random state
            )

        # final sanity check: VAL must have at least MIN_TEST_WINDOWS as well
        n_val = sum(day_counts[d] for d in val_days)
        if n_val < MIN_TEST_WINDOWS:
            continue

        return np.array(train_days), np.array(val_days), np.array(test_days)

    raise RuntimeError(
        f"Unable to find a TEST split with ≥{MIN_TEST_WINDOWS} windows "
        f"and ≥{MIN_SAMPLES_PER_CLASS} positives & negatives after 10 tries"
    )

def safe_auc(y_true, probs):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:

        return np.nan

    return roc_auc_score(y_true, probs)

def collect_windows(df_p, df_n, hr_df, st_df, days):
    p = df_p[df_p['hawaii_createdat_time'].dt.date.isin(days)]
    n = df_n[df_n['hawaii_createdat_time'].dt.date.isin(days)]
    return pd.concat([
        process_label_window(p, hr_df, st_df, 1),
        process_label_window(n, hr_df, st_df, 0)
    ])

def collect_rows(df_processed, days):
    """
    Collect processed rows whose datetime_local date is in `days`.
    """
    df = df_processed.copy()
    df["datetime_local"] = pd.to_datetime(df["datetime_local"]).dt.tz_localize(None)

    days_norm = pd.to_datetime(days).normalize()
    row_days = df["datetime_local"].dt.normalize()

    return df[row_days.isin(days_norm)].copy()

def _count_windows(df_p, df_n, hr_df, st_df, days, df_processed=None,input_df='raw'):
    if input_df == 'raw':
        return len(collect_windows(df_p, df_n, hr_df, st_df, days))
    elif input_df == 'processed':
        return len(collect_rows(df_processed, days))

def _write_skip_file(root: Path, train_days, test_days, n_tr, n_te):
    root.mkdir(parents=True, exist_ok=True)
    msg = [
        "SKIPPED – Not enough windows for benchmarking.\n",
        f"Train days ({len(train_days)}): {list(train_days)}",
        f"Test  days ({len(test_days)}): {list(test_days)}",
        f"Train windows: {n_tr}",
        f"Test  windows: {n_te}",
        "",
        "Minimum required: 2 train windows AND 2 test windows."
    ]
    (root / "not_enough_data.txt").write_text("\n".join(msg))
    print(">> SKIPPED – see not_enough_data.txt for details")


def write_split_details(
    results_dir, pipeline, train_info, val_info, test_info,
    sample_mode="original", sample_summary=None
):
    """
    Writes split_details.txt with sampling mode, global & per-user stats,
    then the TRAIN/VAL/TEST breakdown.
    """
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    uid, te_days, df_te = test_info
    df_val = val_info[uid]["df"]; val_days = val_info[uid]["days"]

    # helper
    def cnt(df):
        p = int(df["state_val"].sum())
        n = len(df)-p
        return p, n

    with open(results_dir/"split_details.txt","w") as f:
        f.write(f"Pipeline: {pipeline}\n")
        f.write(f"Sampling mode: {sample_mode}\n")
        if sample_summary:
            G = sample_summary["global"]
            f.write(
              f"GLOBAL  ORIG +{G['orig_pos']}/-{G['orig_neg']}  "
              f"→ USED +{G['new_pos']}/-{G['new_neg']}\n"
            )
            if G["added"]:   f.write(f"Synthetic positives added: {G['added']}\n")
            if G["removed"]: f.write(f"Negatives removed: {G['removed']}\n")
            f.write("\nUSER-LEVEL SAMPLING:\n")
            for u, stats in sample_summary["per_user"].items():
                f.write(
                  f"  {u}: ORIG +{stats['orig_pos']}/-{stats['orig_neg']}  "
                  f"→ USED +{stats['new_pos']}/-{stats['new_neg']}\n"
                )
            f.write("\n")

        # TRAIN
        f.write("=== TRAINING DAYS (used for model fitting) ===\n")
        for u, info in train_info.items():
            p,n = cnt(info["df"])
            f.write(f"User {u} TRAIN days: {info['days']}\n")
            f.write(f"   windows={len(info['df'])}  (+={p}, -={n})\n\n")

        # VAL
        p_va,n_va = cnt(df_val)
        f.write("=== VALIDATION DAYS (only target user) ===\n")
        f.write(f"User {uid} VAL days: {val_days}\n")
        f.write(f"   windows={len(df_val)}  (+={p_va}, -={n_va})\n\n")

        # TEST
        p_te,n_te = cnt(df_te)
        f.write("=== TEST DAYS (only target user) ===\n")
        f.write(f"User {uid} TEST days: {te_days}\n")
        f.write(f"   windows={len(df_te)}  (+={p_te}, -={n_te})\n")

# ─── Helper to train/load per-user SSL encoder ─────────────────────────────
def _train_or_load_encoder(path, dtype, df, train_days, results_dir):
    if path.exists():
        enc = load_model(path)
        enc.trainable = False
        return enc

    mask = np.isin(df.index.date, train_days)
    vals = StandardScaler().fit_transform(df.loc[mask, 'value'].values.reshape(-1,1))
    segs = create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32')
    if not len(segs):
        raise RuntimeError(f"No {dtype} segments to train encoder.")

    n, idx = len(segs), np.random.permutation(len(segs))
    tr, va = segs[idx[:int(0.8*n)]], segs[idx[int(0.8*n):]]

    enc = build_simclr_encoder(WINDOW_SIZE)
    head = create_projection_head()
    tr_l, va_l = train_simclr(enc, head, tr, va,
                              batch_size=BATCH_SSL, epochs=SSL_EPOCHS)
    enc.save(path)
    enc.trainable = False
    plot_ssl_losses(tr_l, va_l, results_dir, encoder_name=f"{dtype}_ssl")
    return enc

# ─── Shared SSL encoders (train only on train-days) ───────────────────────
def _ensure_global_encoders(shared_root, fruit, scenario, all_splits):
    sdir = Path(shared_root) / f"{fruit}_{scenario}"
    sdir.mkdir(parents=True, exist_ok=True)
    paths = {
        'hr':    sdir / 'hr_encoder.keras',
        'steps': sdir / 'steps_encoder.keras'
    }
    if all(p.exists() for p in paths.values()):
        hr = load_model(paths['hr'])
        hr.trainable = False
        st = load_model(paths['steps'])
        st.trainable = False
        return hr, st, sdir

    losses = {}
    for dtype in ['hr', 'steps']:
        bank = []
        if BP_MODE:
            user_iter = all_splits.keys()
        else:
            user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (fruit, scenario) in pairs]

        for u in user_iter:

            # now unpack train/val/test days
            tr_days_u, _val_days_u, _te_days_u = all_splits.get(u, ([], [], []))
            if len(tr_days_u) == 0:
                continue

            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            df = hr_df if dtype == 'hr' else st_df

            # only use train days here
            mask = np.isin(df.index.date, tr_days_u)
            vals = StandardScaler()\
                   .fit_transform(df.loc[mask, 'value'].values.reshape(-1, 1))

            if len(vals) < WINDOW_SIZE:
                continue

            bank.append(create_windows(vals, WINDOW_SIZE, STEP_SIZE).astype('float32'))

        if not bank:
            raise RuntimeError(f"No train-day segments for global {dtype} SSL!")

        segs = np.concatenate(bank, axis=0)
        n, idx = len(segs), np.random.permutation(len(segs))
        tr, va = segs[idx[: int(0.8 * n)]], segs[idx[int(0.8 * n) :]]

        enc = build_simclr_encoder(WINDOW_SIZE)
        head = create_projection_head()
        tr_l, va_l = train_simclr(enc, head, tr, va,
                                  batch_size=BATCH_SSL, epochs=SSL_EPOCHS)

        enc.save(paths[dtype])
        enc.trainable = False
        losses[dtype] = (tr_l, va_l)

    # plot losses for both modalities
    plot_ssl_losses(*losses['hr'],    sdir, encoder_name="global_hr")
    plot_ssl_losses(*losses['steps'], sdir, encoder_name="global_steps")

    hr = load_model(paths['hr']);    hr.trainable = False
    st = load_model(paths['steps']); st.trainable = False
    return hr, st, sdir

# ─── Shared CNN for Global-Supervised ─────────────────────────────────────
def ensure_global_supervised(shared_cnn_root, fruit, scenario, all_splits, uid,): #sample_mode):
    """
    Train (or load) a single global CNN on ALL users' train-day windows,
    validating only on the target user's validation windows.

    Returns:
        m     : the trained Keras model
        sdir  : the directory where the model was saved/loaded
    """
    sdir = Path(shared_cnn_root) / f"{fruit}_{scenario}" #/{sample_mode}"
    sdir.mkdir(parents=True, exist_ok=True)
    model_path = sdir / 'cnn_classifier.keras'
    if model_path.exists():
        m = load_model(model_path)
        return m, sdir

    # 1) Gather all users' TRAIN windows
    X_list, y_list = [], []
    if BP_MODE:
        user_iter = all_splits.keys()
    else:
        user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (fruit, scenario) in pairs]

    for u in user_iter:
        tr_days_u, _, _ = all_splits.get(u, ([], [], []))
        if len(tr_days_u) == 0:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        ## skip creating synthetic negative labels since bp dataset has explict neg labels
        if BP_MODE:
            neg_df = orig_neg
        else:
            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

        df_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        for h_seq, s_seq, label in zip(df_u['hr_seq'], df_u['st_seq'], df_u['state_val']):
            X_list.append(np.vstack([h_seq, s_seq]).T)
            y_list.append(label)

    if not X_list:
        raise RuntimeError("No train windows for global-supervised!")

    X = np.stack(X_list)
    y = np.array(y_list)

    ###Augmentation trial
    def augment_batch(X, y, noise_std=0.01, scale_std=0.05):
        X_aug = X.copy()
        noise = np.random.normal(0, noise_std, size=X_aug.shape)
        scale = np.random.normal(1.0, scale_std, size=(X_aug.shape[0], 1, 1))
        X_aug = X_aug * scale + noise
        return X_aug, y

    # 2) Build the target user's VAL set
    tr_days_u, val_days_u, _ = all_splits[uid]
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, 'None')
    if neg_df_u.empty or len(neg_df_u) < len(pos_df_u):
        neg_df_u = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))

    df_val_u = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)

    def build_XY(df):
        X = np.stack([np.vstack([h,s]).T for h,s in zip(df['hr_seq'], df['st_seq'])])
        return X, df['state_val'].values

    X_val_u, y_val_u = build_XY(df_val_u)

    # 3) Build & compile the model
    inp = layers.Input(shape=(WINDOW_SIZE, 2))
    x = layers.Conv1D(64, 8, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])
    x = layers.Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(8, activation='relu', kernel_regularizer=l2(1e-4))(se) #[Hiwot added this]
    se = layers.Dense(128, activation='sigmoid')(se)
    se = layers.Dense(128, activation='sigmoid', kernel_regularizer=l2(1e-4))(se)

    se = layers.Reshape((1, 128))(se)
    x = layers.Multiply()([x, se])
    x = layers.Multiply()([x])

    x = layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(128)(inp)
    lstm_out = layers.Dropout(0.5)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation='sigmoid')(combined)

    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # 4) Compute class weights
    classes = np.unique(y)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}

    # 5) Callbacks
    es = EarlyStopping(monitor='val_loss', patience=GS_PATIENCE, restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # 6) SINGLE TRAINING pass
    hist = m.fit(
        X, y,
        validation_data=(X_val_u, y_val_u),
        batch_size=GS_BATCH,
        epochs=GS_EPOCHS,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2
    )

    # 7) Save & plot
    plot_clf_losses(hist.history['loss'], hist.history['val_loss'], sdir, 'global_supervised_loss')
    m.save(model_path)
    return m, sdir

def build_XY(df):
    X = np.stack([np.vstack([h,s]).T for h,s in zip(df['hr_seq'], df['st_seq'])])
    return X, df['state_val'].values

def build_mlp_classifier(input_dim, lr=1e-3):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn_baseline(shared_cnn_root,fruit, scenario, all_splits, uid,): #sample_mode):
    
    ###Note the following hyperparameters work for cardiomate dataset. 
        ## base_filter=(16, 32) eventhough we only use the 16 units layer only
        ## weight_decay=1e-3 
        ## dropout_conv=0.2
        ## lr=1e-3
        ## no class weights when fitting and batch size of 32 and batchnorm stays
    base_filters=(16, 32)
    weight_decay=1e-3
    dropout_conv=0.2
    lr=1e-4
    sdir = Path(shared_cnn_root) / f"{fruit}_{scenario}" #/{sample_mode}"
    sdir.mkdir(parents=True, exist_ok=True)
    model_path = sdir / 'cnn_classifier.keras'
    if model_path.exists() and not FORCE_RETRAIN:
        m = load_model(model_path)
        return m, sdir
    
    X_list, y_list = [], []
  
    if BP_MODE:
        user_iter = all_splits.keys()
    else:
        user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (fruit, scenario) in pairs]

    for u in user_iter:
        tr_days_u, _, _ = all_splits.get(u, ([], [], []))
        if len(tr_days_u) == 0:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        ## skip creating synthetic negative labels since bp dataset has explict neg labels
        if BP_MODE:
            neg_df = orig_neg
        else:
            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

        df_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        for h_seq, s_seq, label in zip(df_u['hr_seq'], df_u['st_seq'], df_u['state_val']):
            X_list.append(np.vstack([h_seq, s_seq]).T)
            y_list.append(label)

    if not X_list:
        raise RuntimeError("No train windows for global-supervised!")

    X = np.stack(X_list)
    y = np.array(y_list)
    # 2) Build the target user's VAL set
    
    # tr_days_u, val_days_u, _ = all_splits[uid]
    X_val_list, y_val_list = [], []
    for u in user_iter:
        _, val_days_u, _ = all_splits.get(u, ([], [], []))
        if len(val_days_u) == 0:
            continue
        hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df_u = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg_u = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        
        if BP_MODE:
            neg_df_u = orig_neg_u
        else:
            if len(orig_neg_u) < len(pos_df_u):
                extra_u = derive_negative_labels(
                    hr_df_u, pos_df_u, len(pos_df_u) - len(orig_neg_u)
                )
                neg_df_u = pd.concat([orig_neg_u, extra_u], ignore_index=True)
            else:
                neg_df_u = orig_neg_u

        df_val = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)

        if len(df_val) == 0:
            continue
        X_val_u, y_val_u = build_XY(df_val)
        X_val_list.append(X_val_u)
        y_val_list.append(y_val_u)


    if not X_val_list:
        raise RuntimeError("No validation windows built for pooled validation set.")

    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val_all = np.concatenate(y_val_list, axis=0)
   

    inp = layers.Input(shape=(WINDOW_SIZE, 2))
    x = layers.Conv1D(base_filters[0], kernel_size=8, padding='same',
                      activation='relu', kernel_regularizer=l2(weight_decay))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout_conv)(x)
    x = layers.GlobalAveragePooling1D()(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)
    opt = Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # 4) Compute class weights
    classes = np.unique(y)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}

      # 5) Callbacks
    es = EarlyStopping(monitor='val_loss',
                       patience=7,
                       min_delta=2e-3,
                       restore_best_weights=True, verbose=1)
    lr = ReduceLROnPlateau(monitor='val_loss', 
                           factor=0.5, 
                           patience=10, min_lr=1e-6, verbose=1)
    history = model.fit(
        X, y,
        validation_data=(X_val_all, y_val_all),
        batch_size=GS_BATCH,
        # class_weight=class_weight,
        # sample_weight=sw_train,
        epochs=GS_EPOCHS,
        verbose=2,
        callbacks=[lr, es]

    )
    plot_clf_losses(history.history['loss'], history.history['val_loss'], sdir, 'cnn_baseline_loss')
    model.save(model_path)
    train_eval = model.evaluate(X, y, verbose=0)
    val_eval   = model.evaluate(X_val_all, y_val_all, verbose=0)
    print("Train(eval) loss/acc:", train_eval)
    print("Val(eval)   loss/acc:", val_eval)
    

    return model, sdir

def ensure_global_supervised_new(shared_cnn_root, fruit, scenario, all_splits, uid,): #sample_mode):
    """
    Train (or load) a single global CNN on ALL users' train-day windows,
    validating on a pooled validation set across users.
    Only users with at least one validation day are included.

    Returns:
        m     : the trained Keras model
        sdir  : the directory where the model was saved/loaded
    """
    sdir = Path(shared_cnn_root) / f"{fruit}_{scenario}" #/{sample_mode}"
    sdir.mkdir(parents=True, exist_ok=True)
    model_path = sdir / 'cnn_classifier.keras'
    if model_path.exists():
        m = load_model(model_path)
        return m, sdir

    # 1) Gather all users' TRAIN windows
    X_list, y_list = [], []
    if BP_MODE:
        user_iter = all_splits.keys()
    else:
        user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (fruit, scenario) in pairs]

    for u in user_iter:
        tr_days_u, _, _ = all_splits.get(u, ([], [], []))
        if len(tr_days_u) == 0:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        ## skip creating synthetic negative labels since bp dataset has explict neg labels
        if BP_MODE:
            neg_df = orig_neg
        else:
            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

        df_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        for h_seq, s_seq, label in zip(df_u['hr_seq'], df_u['st_seq'], df_u['state_val']):
            X_list.append(np.vstack([h_seq, s_seq]).T)
            y_list.append(label)

    if not X_list:
        raise RuntimeError("No train windows for global-supervised!")

    X = np.stack(X_list)
    y = np.array(y_list)

    ##NEW: Build pooled VAL set across users (require at least 1 val day per user)
    ##NEW: cap and balance per user so large users/classes do not dominate val_loss
    VAL_MAX_PER_USER = 30
    X_val_list, y_val_list = [], []
    for u in user_iter:
        _, val_days_u, _ = all_splits.get(u, ([], [], []))
        ##NEW: skip users that do not have at least one validation day
        if len(val_days_u) < 1:
            continue

        hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df_u = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg_u = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if BP_MODE:
            neg_df_u = orig_neg_u
        else:
            if len(orig_neg_u) < len(pos_df_u):
                extra_u = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u) - len(orig_neg_u))
                neg_df_u = pd.concat([orig_neg_u, extra_u], ignore_index=True)
            else:
                neg_df_u = orig_neg_u

        df_val_u = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)
        if len(df_val_u) == 0:
            continue

        # NEW: keep all windows for small-val users; cap only high-val users
        if len(df_val_u) > VAL_MAX_PER_USER:
            pos_val_u = df_val_u[df_val_u["state_val"] == 1]
            neg_val_u = df_val_u[df_val_u["state_val"] == 0]

            # NEW: class-aware capped sampling for users above the cap
            if len(pos_val_u) > 0 and len(neg_val_u) > 0:
                k = min(len(pos_val_u), len(neg_val_u), VAL_MAX_PER_USER // 2)
                df_val_u = pd.concat([
                    pos_val_u.sample(n=k, random_state=42),
                    neg_val_u.sample(n=k, random_state=42),
                ], ignore_index=True)
            else:
                df_val_u = df_val_u.sample(n=VAL_MAX_PER_USER, random_state=42)

        for h_seq, s_seq, label in zip(df_val_u['hr_seq'], df_val_u['st_seq'], df_val_u['state_val']):
            X_val_list.append(np.vstack([h_seq, s_seq]).T)
            y_val_list.append(label)

    ##NEW: fail fast if pooled global validation has no windows
    if not X_val_list:
        raise RuntimeError("No validation windows for pooled global-supervised validation set!")

    X_val = np.stack(X_val_list)
    y_val = np.array(y_val_list)

    # 3) Build & compile the model
    inp = layers.Input(shape=(WINDOW_SIZE, 2))
    ##NEW: reduced model capacity + stronger regularization for pooled global validation
    x = layers.Conv1D(32, 8, padding='same', activation='relu', kernel_regularizer=l2(3e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(32, activation='sigmoid')(se)
    se = layers.Reshape((1, 32))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l2(3e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(3e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(64)(inp)
    lstm_out = layers.Dropout(0.6)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation='sigmoid')(combined)

    m = Model(inputs=inp, outputs=out)
    m.compile(
        optimizer=Adam(3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=['accuracy']
    )

    # 4) Compute class weights
    classes = np.unique(y)
    if len(classes) == 2:
        cw_vals = compute_class_weight('balanced', classes=classes, y=y)
        class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}
    else:
        class_weight = {0:1, 1:1}

    # 5) Callbacks
    es = EarlyStopping(
        monitor='val_loss',
        patience=min(5, GS_PATIENCE),
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1
    )
    lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # 6) SINGLE TRAINING pass
    hist = m.fit(
        X, y,
        validation_data=(X_val, y_val),  ##NEW: pooled global validation set
        batch_size=GS_BATCH,
        epochs=GS_EPOCHS,
        class_weight=class_weight,
        callbacks=[es, lr_cb],
        verbose=2
    )

    # 7) Save & plot
    plot_clf_losses(hist.history['loss'], hist.history['val_loss'], sdir, 'global_cnn_lstm_loss')
    m.save(model_path)
    return m, sdir

# ─── Pipeline #1: Global-Supervised with Train-Set Threshold ──────────────
def run_global_supervised(
    fruit: str,
    scenario: str,
    uid: str,
    user_root: Path,
    all_splits: dict,
    shared_cnn_root: Path,
    neg_df_u: pd.DataFrame,
    sample_mode: str ,
    input_df: str = "raw",
):
    print(f"\n>> Global-Supervised ({fruit}_{scenario})")

    # Directories
    out_dir   = user_root / 'global_supervised' / input_df
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)
    if input_df == "raw":
        # 1) Train or load the shared CNN (validating inside ensure_global_supervised)
        model, src_dir = ensure_global_supervised(
            shared_cnn_root, fruit, scenario, all_splits, uid, input_df=input_df,
        )

        # Copy model files and plots
        for fpath in Path(src_dir).glob('*'):
            if fpath.suffix == '.keras':
                shutil.copy2(fpath, models_d / fpath.name)
            elif fpath.suffix == '.png':
                shutil.copy2(fpath, results_d / fpath.name)
    elif input_df == "processed":
        print(">> Skipping global CNN training for processed features.")
    else:
        raise ValueError(f"Unknown input_df: {input_df}")

    # 2) Build per-user train_info and val_info (for split_details)
    train_info = {}
    val_info   = {}
    for u, (tr_days, val_days, _) in all_splits.items():
        if input_df == "raw":
            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            pos_df   = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
            orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
            if len(orig_neg) < len(pos_df):
                extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

            df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
            df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        else:
            df_processed = _bp_load_processed_data(u)
            df_tr = collect_rows(df_processed, tr_days)
            df_val = collect_rows(df_processed, val_days)

        train_info[u] = {"days": tr_days.tolist(), "df": df_tr}
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

        ##Hiwot adding user_id for breakpoint debugging
        # df_tr['user_id'] = u
        # df_val['user_id'] = u
            
    # 3) Apply sampling if requested
    sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 4) Prepare TEST windows for the target user
    tr_days_u, val_days_u, te_days_u = all_splits[uid]
    if input_df == "raw":
        hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
        pos_df_u = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
        if neg_df_u.empty or len(neg_df_u) < len(pos_df_u):
            neg_df_u = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
        df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
    else:
        df_processed_u = _bp_load_processed_data(uid)
        df_te = collect_rows(df_processed_u, te_days_u)
    df_te['user_id'] = uid  # for breakpoint debugging
    
    # 5) Write split_details.txt
    write_split_details(
        results_d,
        "global_supervised",
        train_info,
        val_info,
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Build X_train, y_train from train_info
    df_train = pd.concat([info["df"] for info in train_info.values()])
    if uid not in train_info or uid not in val_info:
        raise RuntimeError(f"Missing split info for user {uid}")
    df_val_u = val_info[uid]["df"]
    if input_df == "raw":
        X_train = np.stack([
            np.vstack([h, s]).T
            for h, s in zip(df_train["hr_seq"], df_train["st_seq"])
        ])
        y_train = df_train["state_val"].values

        X_test = np.stack([
            np.vstack([h, s]).T
            for h, s in zip(df_te["hr_seq"], df_te["st_seq"])
        ])
        y_test = df_te["state_val"].values

        X_val_u = np.stack([
            np.vstack([h, s]).T
            for h, s in zip(df_val_u["hr_seq"], df_val_u["st_seq"])
        ])
        y_val_u = df_val_u["state_val"].values
    else:
        # Processed mode: validation set should include all users' validation rows.
        df_val_all = pd.concat([info["df"] for info in val_info.values()], ignore_index=True)
        feature_cols = _bp_feature_cols_from_processed(df_train)


        if not feature_cols:
            raise RuntimeError("No processed feature columns available for MLP.")

        scaler = StandardScaler()
        
        for d in (df_train, df_val_all, df_te):
            d[feature_cols] = d[feature_cols].apply(pd.to_numeric, errors="coerce")
            d[feature_cols] = d[feature_cols].replace([np.inf, -np.inf], np.nan)
        # impute from train only [Note: nan's are replaced by train's median check on this later ]
        # train_med = df_train[feature_cols].median(numeric_only=True)
        # df_train[feature_cols] = df_train[feature_cols].fillna(train_med).fillna(0.0)
        # df_val_all[feature_cols] = df_val_all[feature_cols].fillna(train_med).fillna(0.0)
        # df_te[feature_cols]    = df_te[feature_cols].fillna(train_med).fillna(0.0)
        
        # X_train = scaler.fit_transform(df_train[feature_cols].astype(float).values).astype("float32")
        # X_val_u = scaler.transform(df_val_all[feature_cols].astype(float).values).astype("float32")
        # X_test = scaler.transform(df_te[feature_cols].astype(float).values).astype("float32")
        import new_helper
        X_train, y_train, feature_cols,  train_median, scaler = new_helper.build_XY_from_processed(df_train, fit=True)

        # transform val and test with train scaler
        X_val_u, y_val_u, *_ = new_helper.build_XY_from_processed(
            df_val_all,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,          # ← use train scaler
        )
        X_test, y_test, *_ = new_helper.build_XY_from_processed(
            df_te,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,          # ← use train scaler
        )

        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        # X_train, y_train = adasyn.fit_resample(X_train, y_train)
        # y_train = df_train["state_val"].values.astype("float32")
        # y_val_u = df_val_all["state_val"].values.astype("float32")
        # y_test = df_te["state_val"].values.astype("float32")

        from matplotlib import pyplot as plt
        import seaborn as sns
        corr = pd.DataFrame(X_train, columns=feature_cols).corr().abs()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', center=0, 
                    xticklabels=True, yticklabels=True)
        plt.title('Feature correlation matrix')
        plt.tight_layout()
        plt.savefig(results_d / "feature_correlation_matrix.png")
        
        # take upper triangle only — avoid duplicate pairs
        # upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # drop one feature from every pair above threshold
        # to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
        # print(f"Dropping {len(to_drop)} features due to high correlation: {to_drop}")
        # drop_idx = [feature_cols.index(f) for f in to_drop if f in feature_cols]

        # X_train = np.delete(X_train, drop_idx, axis=1)
        # X_val_u = np.delete(X_val_u, drop_idx, axis=1)
        # X_test = np.delete(X_test, drop_idx, axis=1)
        
        # feature_cols_low_corr = [f for f in feature_cols if f not in to_drop]
        # print(feature_cols_low_corr)  # what did threshold 0.85 keep?


        reset_seeds()
        model = build_mlp_classifier(X_train.shape[1], lr=1e-3)
        
        classes = np.unique(y_train)
        if len(classes) == 2:
            cw_vals = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
        else:
            class_weight = {0: 1.0, 1: 1.0}

        es = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-3, restore_best_weights=True, verbose=1)
        lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val_u, y_val_u),
            batch_size=GS_BATCH,
            epochs=GS_EPOCHS,
            class_weight=class_weight,
            callbacks=[es, lr_cb],
            verbose=2,
        )
        probs_te = model.predict(X_test)

        best_thr = select_threshold_train(model, X_train, y_train)
        df_boot, auc_m, auc_s = bootstrap_threshold_metrics(
            y_test,
            probs_te,
            thresholds=np.array([best_thr]),
            sample_frac=0.7,
            n_iters=1000,
            rng_seed=42
        )
        probs_train = model.predict(X_train)
        probs_val = model.predict(X_val_u)
        auc_train = safe_auc(y_train, probs_train)
        auc_val = safe_auc(y_val_u, probs_val)
        print(f"Full-data train AUC = {auc_train:.4f}")
        print(f"Full-data val AUC = {auc_val:.4f}")
        print(f"Full-data test AUC = {auc_m:.4f} ± {auc_s:.4f}")
        best_idx = int(np.argmin(hist.history["val_loss"]))
        val_loss = float(hist.history["val_loss"][best_idx])
        train_loss = float(hist.history["loss"][best_idx])

        print(f"Train loss: {train_loss:.4f}  Val loss: {val_loss:.4f}  gap: {val_loss - train_loss:.4f}")
        print(f"Train AUC:  {auc_train:.4f}  Val AUC:  {auc_val:.4f}  gap: {auc_train - auc_val:.4f}")

        
        shared_mlp_dir = Path(shared_cnn_root) / f"{fruit}_{scenario}_{input_df}"
        shared_mlp_dir.mkdir(parents=True, exist_ok=True)
        mlp_path = shared_mlp_dir / "mlp_classifier.keras"
        model.save(mlp_path)
        plot_clf_losses(hist.history["loss"], hist.history["val_loss"], shared_mlp_dir, "global_mlp_loss")
        shutil.copy2(mlp_path, models_d / mlp_path.name)
        loss_plot = shared_mlp_dir / "global_mlp_loss.png"
        if loss_plot.exists():
            shutil.copy2(loss_plot, results_d / loss_plot.name)

    # 8) Threshold selection on the FULL training set
    best_thr = select_threshold_train(model, X_train, y_train)
    (results_d / "selected_threshold.txt").write_text(f"{best_thr:.4f}\n")

    # 9) Bootstrap & plot on TEST
    probs_te = model.predict(X_test, verbose=0).ravel()
    
    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(
        y_test,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    probs_train = model.predict(X_train, verbose=0).ravel() 
    probs_val = model.predict(X_val_u, verbose=0).ravel() 
    auc_train = safe_auc(y_train, probs_train)
    auc_val = safe_auc(y_val_u, probs_val)
    print(f"Full-data train AUC = {auc_train:.4f}")
    print(f"Full-data val AUC = {auc_val:.4f}")
    print(f"Full-data test AUC = {auc_m:.4f} ± {auc_s:.4f}")
    
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)
    plot_thresholds(
        y_test,
        probs_te,
        str(results_d),
        f"{uid} {fruit}_{scenario} (global_supervised)",
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000
    )

    return df_boot, auc_m, auc_s, auc_train, auc_val


def run_personal_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    tr_days_u: np.ndarray,
    val_days_u: np.ndarray,
    te_days_u: np.ndarray,
    neg_df_u: pd.DataFrame,
    sample_mode: str = "original"
):
    print(f"\n>> Personal-SSL ({fruit}_{scenario})")

    # Directories
    out_dir   = user_root / 'personal_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    # 1) Load signals & labels
    hr_df, st_df  = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    neg_df = neg_df_u

    # 2) Train or load SSL encoders
    enc_hr = _train_or_load_encoder(models_d / 'hr_encoder.keras',
                                    'hr', hr_df, tr_days_u, results_d)
    enc_st = _train_or_load_encoder(models_d / 'steps_encoder.keras',
                                    'steps', st_df, tr_days_u, results_d)

    # 3) Window & label for TRAIN/VAL/TEST
    df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
    df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
    df_te  = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)

    # 4) Sampling summary
    train_info = {uid: {"days": tr_days_u.tolist(), "df": df_tr.copy()}}
    sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 5) Write split details
    write_split_details(
        results_d,
        "personal_ssl",
        train_info,
        {uid: {"days": val_days_u.tolist(), "df": df_val}},
        (uid, te_days_u.tolist(), df_te),
        sample_mode=sample_mode,
        sample_summary=sample_summary
    )

    # 6) Encode into feature vectors
    def encode(df):
        hr_seq = np.stack(df['hr_seq'])[..., None]
        st_seq = np.stack(df['st_seq'])[..., None]
        return enc_hr.predict(hr_seq, verbose=0), enc_st.predict(st_seq, verbose=0)

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
    reset_seeds()  # Ensure deterministic model initialization
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
        class_weight=class_weight,        # REVISED
        callbacks=[es],
        verbose=2
    )
    plot_clf_losses(hist.history['loss'], hist.history['val_loss'], results_d, 'personal_ssl_clf_loss')

    # 8) Threshold scan on the FULL training set
    best_thr = select_threshold_train(clf, X_train, y_train)
    (results_d / "selected_threshold.txt").write_text(f"{best_thr:.4f}\n")

    # 9) Bootstrap & plot on TEST
    probs_te = clf.predict(X_test, verbose=0).ravel()
    ##saving test probs and labels for later use in aggregation
    np.save(results_d / f"test_probs_{uid}.npy", probs_te)
    np.save(results_d / f"test_labels_{uid}.npy", y_test)
    print(f"saved in {results_d / f'test_probs_{uid}.npy'} and {results_d / f'test_labels_{uid}.npy'} ")

    df_boot, auc_m, auc_s = bootstrap_threshold_metrics(
        y_test,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    probs_train = clf.predict(X_train, verbose=0).ravel()
    probs_val = clf.predict(X_val, verbose=0).ravel()
    auc_train = safe_auc(y_train, probs_train)
    auc_val = safe_auc(y_val, probs_val)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)

    plot_thresholds(
        y_test,
        probs_te,
        str(results_d),
        f"{uid} {fruit}_{scenario} (personal_ssl)",
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000
    )

    return df_boot, auc_m, auc_s, auc_train, auc_val

def run_global_ssl(
    uid: str,
    fruit: str,
    scenario: str,
    user_root: Path,
    shared_enc_root: Path,
    all_splits: dict,
    neg_df_u: pd.DataFrame,
    sample_mode: str = "original"
):
    print(f"\n>> Global-SSL ({fruit}_{scenario})")

    # Directories
    out_dir   = user_root / 'global_ssl'
    models_d  = out_dir / 'models_saved'
    results_d = out_dir / 'results'
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    #setting BASE_DATA_DIR to bp data dir for loading raw data and
    configure_task_mode(task="bp",)
 # 1) Load or train shared encoders
    enc_hr, enc_st, enc_src = _ensure_global_encoders(
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
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, fruit, scenario)
        orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, fruit, 'None')
        if len(orig_neg) < len(pos_df):
            extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_tr['user_id'] = u  # include user_id for breakpoint debugging
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

        train_info[u] = {"days": tr_days.tolist(),  "df": df_tr,}  # include user_id for breakpoint debugging
        val_info[u]   = {"days": val_days.tolist(), "df": df_val}

    # 3) Apply sampling across users
    sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)

    # 4) Build this user's test windows
    tr_days_u, val_days_u, te_days_u = all_splits[uid]
    hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / uid)
    pos_df_u        = load_label_data(Path(BASE_DATA_DIR) / uid, fruit, scenario)
    if len(neg_df_u) < len(pos_df_u):
        neg_df_u = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
    df_te           = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)

    # 5) Write split_details.txt
    write_split_details(
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
    breakpoint()  # check that df_all_tr has user_id column and correct number of rows
    H_tr, S_tr   = encode(df_all_tr)
    df_val_u     = val_info[uid]["df"]
    H_val, S_val = encode(df_val_u)
    H_te, S_te   = encode(df_te)

    X_tr = np.concatenate([H_tr, S_tr], axis=1).astype('float32')
    y_tr = df_all_tr['state_val'].values.astype('float32')
    X_val = np.concatenate([H_val, S_val], axis=1).astype('float32')
    y_val = df_val_u['state_val'].values.astype('float32')
    X_te  = np.concatenate([H_te, S_te], axis=1).astype('float32')
    y_te  = df_te['state_val'].values.astype('float32')

    # 7) Train classifier
    reset_seeds()  # Ensure deterministic model initialization
    clf = Sequential([
        Dense(64, activation='relu', input_shape=(X_tr.shape[1],), kernel_regularizer=l2(0.01)),
        BatchNormalization(), Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    es          = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE,
                                restore_best_weights=True, verbose=1)
    cw_vals     = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    hist = clf.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=CLF_EPOCHS,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[es],
        verbose=2
    )
    plot_clf_losses(hist.history['loss'], hist.history['val_loss'], results_d, 'global_ssl_clf_loss')

    # 8) Threshold selection on the FULL training set (instead of validation)
    best_thr = select_threshold_train(clf, X_tr, y_tr)
    (results_d / "selected_threshold.txt").write_text(f"{best_thr:.4f}\n")

    # 9) Bootstrap & plot on TEST
    probs_te = clf.predict(X_te, verbose=0).ravel()
    np.save(results_d / f"test_probs_{uid}.npy", probs_te)
    np.save(results_d / f"test_labels_{uid}.npy", y_te)
    print(f"saved in {results_d / f'test_probs_{uid}.npy'} and {results_d / f'test_labels_{uid}.npy'} ")

    df_boot, auc_mean, auc_std = bootstrap_threshold_metrics(
        y_te,
        probs_te,
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000,
        rng_seed=42
    )
    
    probs_train = clf.predict(X_tr, verbose=0).ravel()
    probs_val = clf.predict(X_val, verbose=0).ravel()
    auc_train = safe_auc(y_tr, probs_train)
    auc_val = safe_auc(y_val, probs_val)
    df_boot.to_csv(results_d / "bootstrap_metrics.csv", index=False)

    plot_thresholds(
        y_te,
        probs_te,
        str(results_d),
        f"{uid} {fruit}_{scenario} (global_ssl)",
        thresholds=np.array([best_thr]),
        sample_frac=0.7,
        n_iters=1000
    )
    
    
    return df_boot, auc_mean, auc_std, auc_train, auc_val
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    pa = argparse.ArgumentParser()
    pa.add_argument(
        "--task",
        choices=["fruit", "bp"],
        default="fruit",
        help="Use fruit/scenario labels (default) or BP_spike labels from hp data."
    )
   
    pa.add_argument("--user",       required=False)
    pa.add_argument("--participant-id", required=False, help="BP task participant id, e.g., 31")
    pa.add_argument("--fruit",      required=False)
    pa.add_argument("--scenario",   required=False)
    pa.add_argument("--output-dir", default="BP_SPIKE_PRED")
    pa.add_argument(
        "--sample-mode",
        choices=["original", "undersample", "oversample"],
        default="original",
        help="How to balance classes in TRAIN/VAL: keep original, undersample negs, or oversample pos."
    )
    pa.add_argument(
        "--results-subdir",
        default="results",
        help="Name of the subdirectory under each pipeline where results (CSVs, plots, split_details) go."
    )
    pa.add_argument(
        "--pipelines",
        nargs="+",
        choices=["global_supervised", "personal_ssl", "global_ssl"],
        # default=["global_supervised", "personal_ssl", "global_ssl"],
        default=["personal_ssl"],
        help="Which pipelines to run. Defaults to all three."
    )
    pa.add_argument(
        "--force_retrain",
        action="store_true",
        help="Ignore saved models/encoders and retrain all artifacts in this run."
    )
    pa.add_argument(
        "--bp_input", 
        choices=["raw", "processed"],
        default="processed",
    )
    args = pa.parse_args()

    FORCE_RETRAIN = args.force_retrain

    # Task validation and defaults
    if args.task == "fruit":
        if not (args.user and args.fruit and args.scenario):
            raise SystemExit("For --task fruit, require --user --fruit --scenario.")
    else:
        if not (args.user or args.participant_id):
            raise SystemExit("For --task bp, require --user or --participant-id.")
        if not args.user:
            args.user = str(args.participant_id)
        if not args.fruit:
            args.fruit = "BP"
        if not args.scenario:
            args.scenario = "spike"

        BP_MODE = True

    # override the global default
    global RESULTS_SUBDIR
    RESULTS_SUBDIR = args.results_subdir

    # prepare top‐level and per‐user directories
    top_out         = Path(args.output_dir)
    user_root       = top_out / args.user / f"{args.fruit}_{args.scenario}"
    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"
    user_root.mkdir(parents=True, exist_ok=True)

    # seed everything
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # ─── build day‐level splits and negatives for all users ─────────────────
    all_splits   = {}
    all_negatives = {}     # ← NEW: store per-user neg_df
  

    if args.task == "bp":
        BASE_DATA_DIR = "DATA/Cardiomate/"

        if args.bp_input == "raw":
            # BP-mode adapter: route signal loading to hp/hp{pid} raw files.
            def _bp_load_signal_data(user_dir):
                pid = _bp_pid_from_user_dir(user_dir)
                hr_df, st_df, _, _ = _bp_load_all(pid)
                
                return hr_df, st_df
     
            ## Loads label for BP spike prediciton
            def _bp_load_label_data(user_dir, fruit, scenario):
                pid = _bp_pid_from_user_dir(user_dir)
                _, _, pos_df, neg_df = _bp_load_all(pid)
                if scenario == "None":
                    return neg_df.copy()
                return pos_df.copy()

            # override imported helpers for BP mode
            load_signal_data = _bp_load_signal_data
            load_label_data = _bp_load_label_data
        elif args.bp_input == "processed":
            
            def load_processed_data(pid):
                in_file = os.path.join('processed', f'hp{pid}', 'processed_bp_prediction_data.csv')
                processed_data = pd.read_csv(in_file)
                processed_data = processed_data.sort_values("datetime_local").reset_index(drop=True)
                
                return processed_data
            def exract_features_from_processed_data(processed_data):
                '''
                keeps only derived features and target and drops others
                '''
                features = processed_data.drop(columns=['id', 'user_id', 'reading_id', 'device_type', 'data_type_hr', 'data_type_steps'])
                target = processed_data['BP_spike']
                return features, target


        # Discover participants that have all required raw BP inputs (hr, steps, BP readings).
        bp_root = Path(BASE_DATA_DIR)
        candidate_dirs = sorted((bp_root / "hp").glob("hp*")) if (bp_root / "hp").exists() else sorted(bp_root.glob("hp*"))
        bp_users = []
        for p in candidate_dirs:
            m = re.search(r"\d+", p.name)
            if not m:
                continue
            pid = m.group(0)
            base = p
            if not (base / f"hp{pid}_hr.csv").exists():
                continue
            if not (base / f"hp{pid}_steps.csv").exists():
                continue
            if not (base / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
                continue
            bp_users.append(pid)

        if not bp_users:
            raise SystemExit(f"No BP participants found under {bp_root} (or {bp_root / 'hp'}).")

        for pid in bp_users:
            hr_df, st_df, pos_df, neg_df = _bp_load_all(pid)
            all_negatives[pid] = neg_df
            try:
                tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue
            all_splits[pid] = (tr_u, val_u, te_u)

        if args.user not in all_splits:
            raise SystemExit(f"Skipping user {args.user}: no BP data or invalid splits.")
   
        for pid in bp_users:
            hr_df, st_df, pos_df, neg_df = _bp_load_all(pid)
            all_negatives[pid] = neg_df
            try:
                tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue
            all_splits[pid] = (tr_u, val_u, te_u)

        if args.user not in all_splits:
            raise SystemExit(f"Skipping user {args.user}: no BP data or invalid splits.")
    else:
        for u, pairs in ALLOWED_SCENARIOS.items():
            if (args.fruit, args.scenario) not in pairs:
                continue

            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
            orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')

            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

            all_negatives[u] = neg_df

            try:
                tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {u}: {e}")
                continue

            all_splits[u] = (tr_u, val_u, te_u)

        # ensure the target user has splits
        if args.user not in all_splits:
            print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
            sys.exit(0)

    # retrieve the one and only neg_df for test
    neg_df_u = all_negatives[args.user]
    tr_days_u, val_days_u, te_days_u = all_splits[args.user]

    # run pipelines, passing neg_df_u into each
    pipeline_results = {}
    
    if "global_supervised" in args.pipelines:


        df_gs, auc_gs_m, auc_gs_s, auc_gs_train, auc_gs_val = run_global_supervised(
            args.fruit, args.scenario, args.user,
            user_root, all_splits, shared_cnn_root,
            neg_df_u=neg_df_u,
            sample_mode=args.sample_mode,
            input_df=args.bp_input
        )
        
        pipeline_results["global_supervised"] = (df_gs, auc_gs_m, auc_gs_s, auc_gs_train, auc_gs_val)


    if "personal_ssl" in args.pipelines:
        df_ps, auc_ps_m, auc_ps_s, auc_ps_tr, auc_ps_val = run_personal_ssl(
            args.user, args.fruit, args.scenario,
            user_root, tr_days_u, val_days_u, te_days_u,
            neg_df_u=neg_df_u,
            sample_mode=args.sample_mode
        )
        pipeline_results["personal_ssl"] = (df_ps, auc_ps_m, auc_ps_s, auc_ps_tr, auc_ps_val)
    
    if "global_ssl" in args.pipelines:
        df_gl, auc_gl_m, auc_gl_s, auc_gl_tr, auc_gl_val = run_global_ssl(
            args.user, args.fruit, args.scenario,
            user_root, shared_enc_root, all_splits,
            neg_df_u=neg_df_u,
            sample_mode=args.sample_mode
        )
        pipeline_results["global_ssl"] = (df_gl, auc_gl_m, auc_gl_s, auc_gl_tr, auc_gl_val)

    # produce comparison summary
    rows = []
    for name, (df, auc_m, auc_s, auc_train, auc_val) in pipeline_results.items():

        tmp = df.copy()
        tmp["Balance"] = tmp[["Sensitivity_Mean", "Specificity_Mean"]].min(axis=1)
        best = tmp.sort_values(
            ["Balance", "Sensitivity_Mean"],
            ascending=[False, False]
        ).iloc[0]
        rows.append({
            "Pipeline":         name,
            "Best_Threshold":   best["Threshold"],
            "Accuracy_Mean":    best["Accuracy_Mean"],
            "Accuracy_STD":     best["Accuracy_STD"],
            "Sensitivity_Mean": best["Sensitivity_Mean"],
            "Sensitivity_STD":  best["Sensitivity_STD"],
            "Specificity_Mean": best["Specificity_Mean"],
            "Specificity_STD":  best["Specificity_STD"],
            "AUC_Train":       pipeline_results[name][3],
            "AUC_Val":         pipeline_results[name][4],
            "AUC_Mean":         auc_m,
            "AUC_STD":          auc_s
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(user_root / "comparison_summary.csv", index=False)

    print("\n--- Comparison Summary ---")
    print(df_summary.to_markdown(index=False))
