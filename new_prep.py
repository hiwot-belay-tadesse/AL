
from __future__ import annotations

import inspect
from pathlib import Path
import random
import numpy as np
import pandas as pd
import sys 
import shutil
sys.path.append(str(Path(__file__).resolve().parents[1]))

import src
from src.classifier_utils import (
    BASE_DATA_DIR,
    ALLOWED_SCENARIOS,
    load_signal_data,
    load_label_data,
)
from src.compare_pipelines import derive_negative_labels, collect_windows, ensure_train_val_test_days

import uq_utility

import numpy as np
import pandas as pd
import json
import os
import re
from pandas.errors import EmptyDataError


def _safe_read_csv(path, label):
    """
    Read CSV safely and raise RuntimeError with context on malformed/empty files.
    """
    try:
        df = pd.read_csv(path)
    except EmptyDataError as e:
        raise RuntimeError(f"Empty CSV for {label}: {path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed reading CSV for {label}: {path} ({e})") from e
    if df is None or df.empty or len(df.columns) == 0:
        raise RuntimeError(f"No usable rows/columns for {label}: {path}")
    return df


def _bp_load_signal_df(path, value_col="value"):
    '''
    loads raw hr, steps and bp labels
    '''
    df = _safe_read_csv(path, f"signal ({Path(path).name})")
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
    df = df.dropna(subset=["time"])
    if df.empty:
        raise RuntimeError(f"Signal file has no valid timestamps: {path}")
    df.sort_values("time", inplace=True)
    df = df.groupby("time", as_index=False)["value"].mean()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["value"] = df["value"].ffill()
    df.set_index("time", inplace=True)
    if df.empty:
        raise RuntimeError(f"Signal file has no usable rows after preprocessing: {path}")
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

    df_bp = _safe_read_csv(file_bp, f"bp labels ({file_bp.name})")

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


def _collect_processed_rows(df_processed: pd.DataFrame, days) -> pd.DataFrame:
    day_index = df_processed["datetime_local"].dt.normalize()
    days_norm = pd.to_datetime(days).normalize()
    return df_processed[day_index.isin(days_norm)].copy()


    
def prepare_data(
    args,
    top_out: Path,
    shared_enc_root: Path,
    shared_cnn_root: Path,
    batch_ssl: int,
    ssl_epochs: int,
    pool: str,
    task: str = "fruit",
    input_df: str | None = None,
):
    # random.seed(42)
    # np.random.seed(42)

    if input_df is None:
        input_df = getattr(args, "input_df", "raw")

    user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
    user_root.mkdir(parents=True, exist_ok=True)

    out_dir   = user_root / pool
    models_d  = out_dir / "models_saved"
    results_d = out_dir / "results"
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    all_splits    = {}
    all_negatives = {}
    all_positives = {}
    all_signals   = {}

    # ══════════════════════════════════════════════════════════════
    # BRANCH 1 — BP TASK
    # ══════════════════════════════════════════════════════════════
    if task == "bp":

        BASE_DATA_DIR = "DATA/Cardiomate/"
        bp_root = Path(BASE_DATA_DIR)
        candidate_dirs = (
            sorted((bp_root / "hp").glob("hp*"))
            if (bp_root / "hp").exists()
            else sorted(bp_root.glob("hp*"))
        )

        # ── Step 1: define loading helpers based on input_df ──────
        if input_df == "raw":

            def _load_signal_data(pid):
                hr_df, st_df, _, _ = _bp_load_all(pid)
                return hr_df, st_df

            def _load_label_data(pid, scenario):
                _, _, pos_df, neg_df = _bp_load_all(pid)
                if scenario == "None":
                    return neg_df.copy()
                return pos_df.copy()

            def _load_rows(pid, days):
                hr_df, st_df = _load_signal_data(pid)
                pos_df = _load_label_data(pid, "spike")
                orig_neg = _load_label_data(pid, "None")
                if len(orig_neg) < len(pos_df):
                    extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                    neg_df = pd.concat([orig_neg, extra], ignore_index=True)
                else:
                    neg_df = orig_neg
                return collect_windows(pos_df, neg_df, hr_df, st_df, days)

            # Match 10pct_eval test-time behavior: replace negatives if insufficient.
            def _load_rows_test(pid, days):
                hr_df, st_df = _load_signal_data(pid)
                pos_df = _load_label_data(pid, "spike")
                neg_df = _load_label_data(pid, "None")
                if len(neg_df) < len(pos_df):
                    neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))
                return collect_windows(pos_df, neg_df, hr_df, st_df, days)

        elif input_df == "processed":

            def _load_signal_data(pid):
                hr_df, st_df, _, _ = _bp_load_all(pid)
                return hr_df, st_df

            def _load_label_data(pid, scenario):
                _, _, pos_df, neg_df = _bp_load_all(pid)
                if scenario == "None":
                    return neg_df.copy()
                return pos_df.copy()

            def _bp_load_processed_data(pid):
                in_file = Path("processed") / f"hp{pid}" / "processed_bp_prediction_data.csv"
                if not in_file.exists():
                    raise FileNotFoundError(f"Missing processed BP file: {in_file}")

                df = _safe_read_csv(in_file, f"processed bp ({in_file.name})")
                if "datetime_local" not in df.columns:
                    raise ValueError(f"{in_file} missing 'datetime_local'")
                if "BP_spike" not in df.columns:
                    raise ValueError(f"{in_file} missing 'BP_spike'")

                df["datetime_local"] = pd.to_datetime(df["datetime_local"]).dt.tz_localize(None)
                df = df.sort_values("datetime_local").reset_index(drop=True)
                df["state_val"] = df["BP_spike"].astype(int)
                return df

            def _load_rows(pid, days):
                df = _bp_load_processed_data(pid)
                return _collect_processed_rows(df, days)

            _load_rows_test = _load_rows

        else:
            raise ValueError(f"Unknown input_df: {input_df}")

        # ── Step 2: discover participants ─────────────────────────
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
            raise SystemExit(
                f"No BP participants found under {bp_root} (or {bp_root / 'hp'})."
            )

        # ── Step 3: load signals/labels and compute splits ONCE ───
        for pid in bp_users:
            try:
                hr_df, st_df, pos_df, neg_df = _bp_load_all(pid)
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue
            all_signals[pid]   = (hr_df, st_df)
            all_positives[pid] = pos_df
            all_negatives[pid] = neg_df

            try:
                if input_df == "processed":
                    df_processed = _bp_load_processed_data(pid)

                    tr_u, val_u, te_u = ensure_train_val_test_days(
                        pos_df, neg_df, hr_df, st_df,
                        df_processed=df_processed,
                        input_df="processed",
                    )
                    
                else:
                    tr_u, val_u, te_u = ensure_train_val_test_days(
                        pos_df, neg_df, hr_df, st_df,
                        input_df="raw",
                    )
                all_splits[pid] = (tr_u, val_u, te_u)
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue

        if args.user not in all_splits:
            raise SystemExit(
                f"Skipping user {args.user}: no BP data or invalid splits."
            )

    # ══════════════════════════════════════════════════════════════
    # BRANCH 2 — FRUIT TASK
    # ══════════════════════════════════════════════════════════════
    else:
        BASE_DATA_DIR = "DATA/Banawre/"
        BP_MODE = False

        # ── Step 1: define loading helpers ────────────────────────
        def _load_signal_data(u):
            return load_signal_data(Path(BASE_DATA_DIR) / u)

        def _load_label_data(u, scenario):
            return load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, scenario)

        def _load_rows(u, days):
            hr_df, st_df = _load_signal_data(u)
            pos_df       = _load_label_data(u, args.scenario)
            orig_neg     = _load_label_data(u, "None")
            if len(orig_neg) < len(pos_df):
                extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg
            return collect_windows(pos_df, neg_df, hr_df, st_df, days)
        _load_rows_test = _load_rows

        # ── Step 2: load signals/labels and compute splits ONCE ───
        for u, pairs in ALLOWED_SCENARIOS.items():
            if (args.fruit, args.scenario) not in pairs:
                continue

            hr_df, st_df = _load_signal_data(u)
            pos_df       = _load_label_data(u, args.scenario)
            orig_neg     = _load_label_data(u, "None")

            if len(orig_neg) < len(pos_df):
                extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

            all_signals[u]   = (hr_df, st_df)
            all_positives[u] = pos_df
            all_negatives[u] = neg_df

            try:
                tr_u, val_u, te_u = ensure_train_val_test_days(
                    pos_df, neg_df, hr_df, st_df
                )
                all_splits[u] = (tr_u, val_u, te_u)
            except RuntimeError as e:
                print(f"Skipping user {u}: {e}")
                continue

        if args.user not in all_splits:
            print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
            sys.exit(0)

    # ══════════════════════════════════════════════════════════════
    # SHARED — retrieve target user splits
    # same for both bp and fruit from this point on
    # ══════════════════════════════════════════════════════════════
    uid          = args.user
    neg_df_u     = all_negatives[uid]
    if args.fruit !="BP": 
        BP_MODE =False 
        
    BP_MODE = True
    tr_days_u, val_days_u, te_days_u = all_splits[uid]

    # ── build DataFrames using _load_rows ─────────────────────────
    # _load_rows is defined per branch above — same call signature
    # splits are NEVER recomputed below this point
    train_info = {}
    val_info   = {}

    user_iter = bp_users if task == "bp" else [
        u for u, pairs in ALLOWED_SCENARIOS.items()
        if (args.fruit, args.scenario) in pairs
        and u in all_splits
    ]

    for u in user_iter:
        if u not in all_splits:
            print(f"Skipping user {u}: no valid train/val/test splits.")
            continue
        tr_days, val_days, _ = all_splits[u]
        df_tr_u  = _load_rows(u, tr_days)
        df_val_u = _load_rows(u, val_days)
        df_tr_u["user_id"]  = str(u)
        df_val_u["user_id"] = str(u)
        train_info[u] = {"days": list(tr_days), "df": df_tr_u}
        val_info[u]   = {"days": list(val_days), "df": df_val_u}

    # test — target user only, from the SAME split computed above
    if task == "bp" and input_df == "raw":
        df_te = _load_rows_test(uid, te_days_u)
    else:
        df_te = _load_rows(uid, te_days_u)
    df_te["user_id"] = str(uid)

    # ── pool branching ─────────────────────────────────────────────
    if pool == "personal":
        df_tr     = train_info[uid]["df"].copy()
        df_val    = val_info[uid]["df"].copy()
        df_all_tr = None

        if input_df == "raw":
            enc_hr = src.compare_pipelines._train_or_load_encoder(
                models_d / "hr_encoder.keras", "hr",
                all_signals[uid][0], tr_days_u, results_d
            )
            enc_st = src.compare_pipelines._train_or_load_encoder(
                models_d / "steps_encoder.keras", "steps",
                all_signals[uid][1], tr_days_u, results_d
            )
        else:
            enc_hr, enc_st = None, None

    elif pool == "global":
        df_all_tr = pd.concat(
            [info["df"] for info in train_info.values()],
            ignore_index=True
        )
        df_val = pd.concat(
            [v["df"] for v in val_info.values()],
            ignore_index=True
        )
        df_tr = train_info[uid]["df"].copy()
    
        if input_df == "raw":
            enc_hr, enc_st, enc_src = uq_utility._ensure_global_encoders(
                shared_enc_root, args.fruit, args.scenario,
                all_splits, batch_ssl, ssl_epochs,
                exclude_user_id=None, 
                BP_MODE=BP_MODE,
            )
            for fpath in Path(enc_src).glob("*"):
                if fpath.suffix == ".keras":
                    shutil.copy2(fpath, models_d / fpath.name)
                elif fpath.suffix == ".png":
                    shutil.copy2(fpath, results_d / fpath.name)
        else:
            enc_hr, enc_st = None, None

    elif pool == "global_supervised":
        df_all_tr = pd.concat(
            [info["df"] for info in train_info.values()],
            ignore_index=True
        )

        df_val = pd.concat(
            [v["df"] for v in val_info.values()],
            ignore_index=True
        )
        df_tr = train_info[uid]["df"].copy()
        enc_hr, enc_st = None, None

    else:
        raise ValueError(f"Unknown pool: {pool}")
    ()
    # ── sanity check — no date overlap between train pool and test (only for global pools)
    # only check target user's portion of df_all_tr

    date_col = (
    "datetime_local"
    if input_df == "processed"
    else "hawaii_createdat_time"
    )
    df_tr_uid = df_all_tr[df_all_tr["user_id"] == str(uid)]

    train_dates_uid = set(pd.to_datetime(df_tr_uid[date_col]).dt.date)
    test_dates_uid  = set(pd.to_datetime(df_te[date_col]).dt.date)

    overlap = train_dates_uid & test_dates_uid
    if overlap:
        raise RuntimeError(
            f"Date overlap for user {uid} between train and test: {overlap}"
        )

    # ══════════════════════════════════════════════════════════════
    # RETURN — identical signature for all branches
    # ══════════════════════════════════════════════════════════════
    return (
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
        all_negatives,
    )
