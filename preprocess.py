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


def oversample_df_round_robin(
    df_tr: pd.DataFrame,
    *,
    user_col: str = "user_id",
    label_col: str = "state_val",
    random_state: int = 42,
    # balancing knobs:
    target_ratio: float = 1.0,   # minority target = target_ratio * majority (1.0 = fully balanced)
    max_multiplier: int = 10,    # cap: new_minority <= max_multiplier * orig_minority (per user)
) -> pd.DataFrame:
    """
    Oversample the GLOBAL minority class (0 or 1) within each user on df_tr.
    Applies BEFORE splitting into labeled/unlabeled.

    target_ratio=1.0 makes minority count ~= majority count (globally, approximately).
    """
    rng = np.random.default_rng(random_state)

    # --- determine global minority class ---
    counts = df_tr[label_col].value_counts()
    if len(counts) < 2:
        print("[oversample] Only one class present; returning df_tr unchanged.")
        return df_tr.copy()

    cls0 = counts.index.min()
    cls1 = counts.index.max()
    # More robust: explicitly look for 0/1 if present
    # (but will still work if labels are floats 0.0/1.0)
    n0 = int(counts.get(0, counts.iloc[0] if counts.index[0] == 0 else 0))
    n1 = int(counts.get(1, counts.iloc[0] if counts.index[0] == 1 else 0))
    # fallback if labels aren't exactly 0/1
    if n0 == 0 and n1 == 0:
        # pick by smallest count
        minority_label = counts.idxmin()
        majority_label = counts.idxmax()
        n_min = int(counts.min())
        n_maj = int(counts.max())
    else:
        minority_label = 1 if n1 < n0 else 0
        majority_label = 0 if minority_label == 1 else 1
        n_min = min(n0, n1)
        n_maj = max(n0, n1)

    # how many minority samples we want globally
    target_min_global = int(np.ceil(target_ratio * n_maj))
    add_needed_global = max(0, target_min_global - n_min)

    print(f"[oversample] global counts before: {counts.to_dict()}")
    print(f"[oversample] minority={minority_label}, majority={majority_label}, need_add={add_needed_global}")

    if add_needed_global == 0:
        return df_tr.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # --- build per-user pools ---
    by_user = {uid: df_u.copy() for uid, df_u in df_tr.groupby(user_col)}
    uids = list(by_user.keys())

    # track per-user minority availability
    min_pools = {
        uid: by_user[uid][by_user[uid][label_col] == minority_label]
        for uid in uids
    }

    # users with at least 1 minority example
    eligible = [uid for uid in uids if len(min_pools[uid]) > 0]
    if not eligible:
        print("[oversample] No minority examples exist in any user; returning df_tr unchanged.")
        return df_tr.copy()

    # round-robin add ONE sample at a time across eligible users
    added_total = 0
    i = 0
    while add_needed_global > 0:
        uid = eligible[i % len(eligible)]
        df_u = by_user[uid]
        pool = min_pools[uid]
        orig_min_u = int((df_tr[df_tr[user_col] == uid][label_col] == minority_label).sum())
        new_min_u = int((df_u[label_col] == minority_label).sum())

        # cap: don't exceed max_multiplier * original minority count for this user
        if orig_min_u > 0 and new_min_u >= max_multiplier * orig_min_u:
            i += 1
            # if all users hit cap, stop
            if i > len(eligible) * 5:
                break
            continue

        # sample one minority row with replacement
        row = pool.sample(
            n=1, replace=True,
            random_state=int(rng.integers(0, 2**31 - 1))
        )
        by_user[uid] = pd.concat([df_u, row], axis=0, ignore_index=True)

        added_total += 1
        add_needed_global -= 1
        i += 1

    df_tr_bal = pd.concat(by_user.values(), axis=0, ignore_index=True)
    df_tr_bal = df_tr_bal.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    new_counts = df_tr_bal[label_col].value_counts().to_dict()
    print(f"[oversample] added={added_total}, global counts after: {new_counts}")

    return df_tr_bal

def _warn_small_split(df, name, min_total=20, min_per_class=5):
    if df is None or df.empty:
        print(f"Warning: {name} split is empty.")
        return
    if "state_val" not in df.columns:
        return
    pos = int((df["state_val"] == 1).sum())
    neg = int((df["state_val"] == 0).sum())
    if len(df) < min_total or pos < min_per_class or neg < min_per_class:
        print(
            f"Warning: {name} split is small "
            f"(total={len(df)}, pos={pos}, neg={neg})."
        )


def _debug_label_alignment(pos_df, neg_df, df_te, te_days, label="test"):
    if pos_df is None or neg_df is None or df_te is None:
        return
    if "hawaii_createdat_time" not in pos_df.columns:
        return
    try:
        pos_raw = pos_df[pos_df["hawaii_createdat_time"].dt.date.isin(te_days)]
        neg_raw = neg_df[neg_df["hawaii_createdat_time"].dt.date.isin(te_days)]
    except Exception:
        return
    pos_ct = len(pos_raw)
    neg_ct = len(neg_raw)
    te_pos = int((df_te["state_val"] == 1).sum())
    te_neg = int((df_te["state_val"] == 0).sum())
    print(
        f"[label_debug] {label}: raw labels pos={pos_ct}, neg={neg_ct} "
        f"-> windows pos={te_pos}, neg={te_neg}"
    )
        
import hashlib
def model_hash(model):
    weights = model.get_weights()
    flat = np.concatenate([w.ravel() for w in weights]).astype(np.float32)
    return hashlib.sha256(flat.tobytes()).hexdigest()

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

def _collect_processed_rows(df_processed: pd.DataFrame, days) -> pd.DataFrame:
    day_index = df_processed["datetime_local"].dt.normalize()
    days_norm = pd.to_datetime(days).normalize()
    return df_processed[day_index.isin(days_norm)].copy()

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
    # Seed everything for deterministic splits
    random.seed(42)
    np.random.seed(42)
    if input_df is None:
        input_df = getattr(args, "input_df", "raw")

    user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
    user_root.mkdir(parents=True, exist_ok=True)

    all_splits = {}
    all_negatives = {}

    if task == "bp":
        bp_root = Path("DATA/Cardiomate/hp")
        src.compare_pipelines.BP_MODE = True
        src.compare_pipelines.BASE_DATA_DIR = str(bp_root)

        def _bp_load_signal_data(user_dir):
            pid = src.compare_pipelines._bp_pid_from_user_dir(user_dir)
            hr_df, st_df, _, _ = src.compare_pipelines._bp_load_all(pid)
            return hr_df, st_df

        def _bp_load_label_data(user_dir, fruit, scenario):
            pid = src.compare_pipelines._bp_pid_from_user_dir(user_dir)
            _, _, pos_df, neg_df = src.compare_pipelines._bp_load_all(pid)
            if scenario == "None":
                return neg_df.copy()
            return pos_df.copy()

        global BASE_DATA_DIR, load_signal_data, load_label_data
        BASE_DATA_DIR = bp_root
        load_signal_data = _bp_load_signal_data
        load_label_data = _bp_load_label_data

        src.compare_pipelines.BASE_DATA_DIR = bp_root
        src.compare_pipelines.load_signal_data = _bp_load_signal_data
        src.compare_pipelines.load_label_data = _bp_load_label_data

        bp_users = []
        for p in sorted(Path("DATA/Cardiomate/hp").glob("hp*")):
            try:
                pid = src.compare_pipelines._bp_pid_from_user_dir(p)
            except Exception:
                continue
            base = Path("DATA/Cardiomate/hp") / f"hp{pid}"
            if not (base / f"hp{pid}_hr.csv").exists():
                continue
            if not (base / f"hp{pid}_steps.csv").exists():
                continue
            if not (base / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
                continue
            bp_users.append(pid)

        if not bp_users:
            print("No BP participants found under hp/hp*/")
            return None

        for pid in bp_users:
            hr_df, st_df, pos_df, neg_df = src.compare_pipelines._bp_load_all(pid)
            all_negatives[pid] = neg_df

            try:
                if input_df == "processed":
                    df_processed = _bp_load_processed_data(str(pid))
                    tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df,df_processed, input_df="processed")
                else:
                    tr_u, val_u, te_u = ensure_train_val_test_days(
                        pos_df, neg_df, hr_df, st_df, input_df="raw"
                    )
            except RuntimeError as e:
                print(f"Skipping user {pid}: {e}")
                continue
            all_splits[pid] = (tr_u, val_u, te_u)
    else:
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

            all_negatives[u] = neg_df

            try:
                tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {u}: {e}")
                continue

            all_splits[u] = (tr_u, val_u, te_u)

    if args.user not in all_splits:
        print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
        return None

    # target user splits + negatives
    neg_df_u = all_negatives[args.user]
    tr_days_u, val_days_u, te_days_u = all_splits[args.user]



    if pool == "personal":
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / args.user)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, "None")

        if task == "bp":
            neg_df = orig_neg
        else:
            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

        tr_days_u, val_days_u, te_days_u = ensure_train_val_test_days(
            pos_df, neg_df, hr_df, st_df
        )

        out_dir = user_root / "personal_ssl"
        models_d = out_dir / "models_saved"
        results_d = out_dir / "results"
        models_d.mkdir(parents=True, exist_ok=True)
        results_d.mkdir(parents=True, exist_ok=True)

        enc_hr = src.compare_pipelines._train_or_load_encoder(
            models_d / "hr_encoder.keras", "hr", hr_df, tr_days_u, results_d
        )
        enc_st = src.compare_pipelines._train_or_load_encoder(
            models_d / "steps_encoder.keras", "steps", st_df, tr_days_u, results_d
        )

        df_tr = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
        df_te =collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)
        df_tr["user_id"] = args.user
        df_val["user_id"] = args.user
        df_te["user_id"] = args.user
        df_all_tr = None
        
    elif pool == "global":

        print(f"\n>> Global-SSL ({args.fruit}_{args.scenario})")

        # Directories
        out_dir   = user_root / "global_ssl"
        models_d  = out_dir / "models_saved"
        results_d = out_dir / "results"
        models_d.mkdir(parents=True, exist_ok=True)
        results_d.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # 2) Train/load encoders
        # ----------------------------
        enc_hr, enc_st, enc_src = uq_utility._ensure_global_encoders(
            shared_enc_root,
            args.fruit,
            args.scenario,
            all_splits,
            batch_ssl,
            ssl_epochs,
            exclude_user_id=None,
            BP_MODE=True,
        )

        for fpath in Path(enc_src).glob("*"):
            if fpath.suffix == ".keras":
                shutil.copy2(fpath, models_d / fpath.name)
            elif fpath.suffix == ".png":
                shutil.copy2(fpath, results_d / fpath.name)

        # ----------------------------
        # 3) Build per-user train_info/val_info + pooled df_all_tr
        # ----------------------------
        train_info = {}
        val_info = {}

        for u, (tr_days, val_days, _) in all_splits.items():
            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
            # neg_df       = all_negatives[u]   # reuse (no recompute)
            orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')
            if len(orig_neg) < len(pos_df):
                extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg

            df_tr_u  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
            df_val_u = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

            df_tr_u["user_id"]  = u
            df_val_u["user_id"] = u

            train_info[u] = {"days": tr_days.tolist(), "df": df_tr_u}
            val_info[u]   = {"days": val_days.tolist(), "df": df_val_u}

        df_all_tr = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)

        # ----------------------------
        # 4) Build TARGET user's df_tr/df_val/df_te (DataFrames)
        # ----------------------------
        df_tr  = train_info[args.user]["df"]
        df_val = val_info[args.user]["df"]

        hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
        pos_df_u         = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)

        df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
        df_te["user_id"] = args.user

    elif pool == 'global_supervised':
        out_dir  = user_root / "global_supervised"
        models_d  = out_dir / "models_saved"
        results_d = out_dir / "results"
        models_d.mkdir(parents=True, exist_ok=True)
        results_d.mkdir(parents=True, exist_ok=True)

        # all_positives = {}
        # all_signals   = {}

        # if task == "bp":
        #     user_iter = list(all_splits.keys())
        # else:
        #     user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (args.fruit, args.scenario) in pairs]

        # for u in user_iter:
        #     hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        #     pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        #     orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")

        #     if task == "bp":
        #         neg_df = orig_neg
        #     else:
        #         if len(orig_neg) < len(pos_df):
        #             extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
        #             neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        #         else:
        #             neg_df = orig_neg

        #     all_signals[u]   = (hr_df, st_df)
        #     all_positives[u] = pos_df
        #     all_negatives[u] = neg_df

        #     try:
        #         tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        #     except RuntimeError as e:
        #         print(f"Skipping user {u}: {e}")
        #         continue

        #     all_splits[u] = (tr_u, val_u, te_u)

        # uid = args.user
        # if uid not in all_splits:
        #     print(f"Skipping user {uid}: no data for {args.fruit}/{args.scenario}.")
        #     return None

        # tr_days_u, val_days_u, te_days_u = all_splits[uid]

        train_info = {}
        val_info   = {}

        if input_df == "processed":
            # Match compare_pipelines processed flow: train/val/test rows are pulled
            # from processed feature tables by split days.
            for u, (tr_days, val_days, _) in all_splits.items():
                df_processed_u = _bp_load_processed_data(str(u))
                df_tr_u = _collect_processed_rows(df_processed_u, tr_days)
                df_val_u = _collect_processed_rows(df_processed_u, val_days)
                df_tr_u["user_id"] = str(u)
                df_val_u["user_id"] = str(u)
                train_info[u] = {"days": list(tr_days), "df": df_tr_u}
                val_info[u] = {"days": list(val_days), "df": df_val_u}

            df_all_tr = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)
            df_tr = train_info[args.user]["df"].copy()
            ### df_val = val_info[uid]["df"].copy()
            df_val = pd.concat([v["df"] for v in val_info.values()], ignore_index=True)
            df_processed_uid = _bp_load_processed_data(str(args.user))
            df_te = _collect_processed_rows(df_processed_uid, te_days_u)
            df_te["user_id"] = str(args.user)
            enc_hr, enc_st = None, None
        else:
            # Raw window-based path (existing behavior).
            pos_df_u = all_positives[args.user]
            neg_df_u = all_negatives[args.user]

            for u, (tr_days, val_days, _) in all_splits.items():
                hr_df, st_df = all_signals[u]
                pos_df = all_positives[u]
                neg_df = all_negatives[u]

                df_tr_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
                df_val_u = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

                df_tr_u["user_id"]  = u
                df_val_u["user_id"] = u

                train_info[u] = {"days": tr_days, "df": df_tr_u}
                val_info[u]   = {"days": val_days, "df": df_val_u}

            df_all_tr = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)
            df_tr  = train_info[uid]["df"].copy()
            # df_val = val_info[uid]["df"].copy()
            df_val = pd.concat([v["df"] for v in val_info.values()], ignore_index=True)
            hr_df_u, st_df_u = all_signals[uid]
            df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
            df_te["user_id"] = uid
            enc_hr, enc_st = None, None
    return df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives 

def compute_pool_size(args, pool: str, task: str = "fruit") -> int:
    """
    Compute training pool size for dynamic budget without training encoders.
    Returns N used in budget = ceil(unlabeled_frac * N / K).
    """
    if pool == "personal":
        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / args.user)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, "None")
        if task == "bp":
            neg_df = orig_neg
        else:
            if len(orig_neg) < len(pos_df):
                extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                neg_df = pd.concat([orig_neg, extra], ignore_index=True)
            else:
                neg_df = orig_neg
        try:
            tr_days_u, _, _ = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as e:
            print(f"Skipping user {args.user}: {e}")
            return 0
        df_tr = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
        return len(df_tr)

    if pool in ["global", "global_supervised"]:
        all_splits = {}
        all_negatives = {}
        if task == "bp":
            user_iter = list(Path("DATA/Cardiomate/hp").glob("hp*"))
        else:
            user_iter = [u for u, pairs in ALLOWED_SCENARIOS.items() if (args.fruit, args.scenario) in pairs]

        for u in user_iter:
            if task == "bp":
                pid = src.compare_pipelines._bp_pid_from_user_dir(u)
                hr_df, st_df, pos_df, neg_df = src.compare_pipelines._bp_load_all(pid)
                key = pid
            else:
                key = u
                hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
                pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
                orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")
                if len(orig_neg) < len(pos_df):
                    extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
                    neg_df = pd.concat([orig_neg, extra], ignore_index=True)
                else:
                    neg_df = orig_neg

            all_negatives[key] = neg_df
            try:
                tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
            except RuntimeError as e:
                print(f"Skipping user {key}: {e}")
                continue
            all_splits[key] = (tr_u, val_u, te_u)

        if args.user not in all_splits:
            print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
            return 0

        train_info = {}
        for u, (tr_days, _, _) in all_splits.items():
            hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
            pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
            neg_df = all_negatives[u]
            df_tr_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
            df_tr_u["user_id"] = u
            train_info[u] = {"days": tr_days.tolist(), "df": df_tr_u}

        df_all_tr = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)
        return len(df_all_tr)

    raise ValueError(f"Unknown pool type: {pool}")
        #             loss=base_model.loss, metrics=base_model.metrics)

        # ----------------------------
        # # 3) Build per-user train_info/val_info (DataFrames)
        # # ----------------------------
        # train_info = {}
        # val_info = {}
        # for u, (tr_days, val_days, _) in all_splits.items():
        #     hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
        #     pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
        #     neg_df = all_negatives[u]

        #     df_tr_u = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        #     df_tr_u["user_id"] = u

        #     df_val_u = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        #     df_val_u["user_id"] = u

        #     train_info[u] = {"days": tr_days, "df": df_tr_u}
        #     val_info[u] = {"days": val_days, "df": df_val_u}

        # df_all_tr = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)
        # # Target user's fixed val/test
        # hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
        # # pos_df_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
        # # # if neg_df_u.empty or len(neg_df_u) < len(pos_df_u):
        # # #     neg_df_u = src.compare_pipelines.derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u))
        # df_val = val_info[args.user]["df"].copy()
        # df_te =collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
        # df_te["user_id"] = args.user
        
        # df_tr = df_all_tr[df_all_tr["user_id"] == args.user].copy()


    # return df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives


   
# def prepare_data(

#     args,
#     top_out: Path,
#     # neg_df_u: pd.DataFrame,
#     shared_enc_root: Path,
#     batch_ssl: int,
#     ssl_epochs: int,
#     pool : str
# ):
#     '''
#     prepare data for training, returns df_tr, df_tr_all, df_val, df_test and enc_hr and enc_st 

#     '''
#         # Per‑user output root: under output directory
#     user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
#     user_root.mkdir(parents=True, exist_ok=True)

#     # Seed everything (same as no_leak.py)
#     random.seed(42)
#     np.random.seed(42)
#     import tensorflow as tf 
#     tf.random.set_seed(42)
#     if pool == "personal": 
#         # Directories
#         out_dir   = user_root / 'personal_ssl'
#         models_d  = out_dir / 'models_saved'
#         results_d = out_dir / 'results_personal'
#         models_d.mkdir(parents=True, exist_ok=True)
#         results_d.mkdir(parents=True, exist_ok=True)
#             # ─── build day‐level splits and negatives for all users ─────────────────
#         all_splits   = {}
#         all_negatives = {}     # ← NEW: store per-user neg_df

#         for u, pairs in ALLOWED_SCENARIOS.items():
#             if (args.fruit, args.scenario) not in pairs:
#                 continue

#             hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#             pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#             orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')

#             if len(orig_neg) < len(pos_df):
#                 extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#                 neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#             else:
#                 neg_df = orig_neg

#             all_negatives[u] = neg_df

#             try:
#                 tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
#             except RuntimeError as e:
#                 print(f"Skipping user {u}: {e}")
#                 continue

#             all_splits[u] = (tr_u, val_u, te_u)

#         # ensure the target user has splits
#         if args.user not in all_splits:
#             print(f"Skipping user {args.user}: no data for {args.fruit}/{args.scenario}.")
#             sys.exit(0)

#         # retrieve the one and only neg_df for test
#         neg_df_u = all_negatives[args.user]
#         tr_days_u, val_days_u, te_days_u = all_splits[args.user]
#         # 1) Load signals & labels
#         hr_df, st_df  = load_signal_data(Path(BASE_DATA_DIR) / args.user)
#         pos_df = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
#         neg_df = neg_df_u
#         # 2) Train or load SSL encoders
#         import inspect
#         print('inspect', inspect.signature(_train_or_load_encoder))
#         print("ENCODER FUNC FILE:", inspect.getsourcefile(_train_or_load_encoder))
#         print("ENCODER FUNC LINE:", inspect.getsourcelines(_train_or_load_encoder)[1])
#         print("ENCODER SIG:", inspect.signature(_train_or_load_encoder))
#         enc_hr = uq_utility._train_or_load_encoder(models_d / 'hr_encoder.keras',
#                                         'hr', hr_df, tr_days_u, results_d, force_retrain=True)
#         enc_st = uq_utility._train_or_load_encoder(models_d / 'steps_encoder.keras',
#                                         'steps', st_df, tr_days_u, results_d, force_retrain=True)

#         # 3) Window & label for TRAIN/VAL/TEST
#         df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days_u)
#         df_tr['user_id'] = args.user
#         df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days_u)
#         df_te  = collect_windows(pos_df, neg_df, hr_df, st_df, te_days_u)
#         _debug_label_alignment(pos_df, neg_df, df_te, te_days_u, label="test")
#         df_all_tr = None
        
#     elif pool == "global":
        
#         outdir = user_root/ 'global_ssl'
#         models_d  = outdir / 'models_saved'
#         results_d = outdir / 'results_global'
#         models_d.mkdir(parents=True, exist_ok=True)
#         results_d.mkdir(parents=True, exist_ok=True)

#         # ---------------------------------------------------------
#         # 1) Build SSL splits on OTHER users (for global encoders)
#         # ---------------------------------------------------------
#         ssl_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

#         for u, pairs in ALLOWED_SCENARIOS.items():
#             # Exclude target user
#             if u == args.user:
#                 continue
#             if (args.fruit, args.scenario) not in pairs:
#                 continue

#             hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#             pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#             orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")

#             if len(orig_neg) < len(pos_df):
#                 extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#                 neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#             else:
#                 neg_df = orig_neg

#             try:
#                 tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
#             except RuntimeError as e:
#                 print(f"Skipping user {u}: {e}")
#                 continue

#             ssl_splits[u] = (tr_u, val_u, te_u)

#         # ---------------------------------------------------------
#         # 2) Train / load global SSL encoders (OTHER users only)
#         # ---------------------------------------------------------

#         enc_hr, enc_st, enc_src = uq_utility._ensure_global_encoders(
#             shared_enc_root,
#             args.fruit,
#             args.scenario,
#             ssl_splits,
#             batch_ssl,
#             ssl_epochs,
#         )
        

#         # Optionally copy encoder artifacts under this user's root 
#         models_d = (user_root / "global_ssl" / "models_saved")
#         results_d = (user_root / "global_ssl" / "results")
#         models_d.mkdir(parents=True, exist_ok=True)
#         results_d.mkdir(parents=True, exist_ok=True)

#         for fpath in Path(enc_src).glob("*"):
#             if fpath.suffix == ".keras":
#                 (models_d / fpath.name).write_bytes(fpath.read_bytes())
#             elif fpath.suffix == ".png":
#                 (results_d / fpath.name).write_bytes(fpath.read_bytes())
#         # SSL Pretraining phase (other users only)
#         train_info = {}
#         val_info   = {}
#         for u, (tr_days, val_days, _) in ssl_splits.items():
#             hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#             pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#             orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')
#             if len(orig_neg) < len(pos_df):
#                 extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#                 neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#             else:
#                 neg_df = orig_neg

#             df_tr_others  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
#             # df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

#             train_info[u] = {"days": tr_days.tolist(),  "df": df_tr_others}
#             # val_info[u]   = {"days": val_days.tolist(), "df": df_val}

#         # 3) Apply sampling across users
#         # sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)
#         df_all_tr  = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)

#         # ---------------------------------------------------------
#         # 3) Build this user's TRAIN / VAL / TEST window DataFrames
#         # ---------------------------------------------------------
#         hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
#         pos_df_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
#         orig_neg_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, "None")

#         if len(orig_neg_u) < len(pos_df_u):
#             extra = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u) - len(orig_neg_u))
#             neg_df_u = pd.concat([orig_neg_u, extra], ignore_index=True)
#         else:
#             neg_df_u = orig_neg_u

#         tr_days_u, val_days_u, te_days_u = ensure_train_val_test_days(
#             pos_df_u, neg_df_u, hr_df_u, st_df_u
#         )

#         df_tr = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, tr_days_u)
#         df_tr['user_id'] = args.user
#         df_val = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)
#         df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
#         _debug_label_alignment(pos_df_u, neg_df_u, df_te, te_days_u, label="test")
#     elif pool == "mixed": ## here the pool is shared between group with similar scenario and drug type and target
#         outdir = user_root/ 'mixed_ssl'
#         models_d  = outdir / 'models_saved'
#         results_d = outdir / 'results_mixed'
#         models_d.mkdir(parents=True, exist_ok=True)
#         results_d.mkdir(parents=True, exist_ok=True)

#         # ---------------------------------------------------------
#         # 1) Build SSL splits 
#         # ---------------------------------------------------------

#         ssl_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
#         all_negatives = {}
#         for u, pairs in ALLOWED_SCENARIOS.items():
#             # Do not Exclude target user
#             # if u == args.user:
#             #     continue
#             if (args.fruit, args.scenario) not in pairs:
#                 continue

#             hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#             pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#             orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")

#             if len(orig_neg) < len(pos_df):
#                 extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#                 neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#             else:
#                 neg_df = orig_neg
#             all_negatives[u] = neg_df

#             try:
#                 tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
#             except RuntimeError as e:
#                 print(f"Skipping user {u}: {e}")
#                 continue

#             ssl_splits[u] = (tr_u, val_u, te_u)

#         # 
        
#         # ---------------------------------------------------------
#         # 2) Train / load global SSL encoders 
#         # ---------------------------------------------------------

#         enc_hr, enc_st, enc_src = uq_utility._ensure_global_encoders(
#             shared_enc_root,
#             args.fruit,
#             args.scenario,
#             ssl_splits,
#             batch_ssl,
#             ssl_epochs,
#             # force_retrain=True
#         )
#         # 

#         # Optionally copy encoder artifacts under this user's root 
#         models_d = (user_root / "mixed_ssl" / "models_saved")
#         results_d = (user_root / "mixed_ssl" / "results")
#         models_d.mkdir(parents=True, exist_ok=True)
#         results_d.mkdir(parents=True, exist_ok=True)

#         for fpath in Path(enc_src).glob("*"):
#             if fpath.suffix == ".keras":
#                 (models_d / fpath.name).write_bytes(fpath.read_bytes())
#             elif fpath.suffix == ".png":
#                 (results_d / fpath.name).write_bytes(fpath.read_bytes())
#         # SSL Pretraining phase (all users including target)
#         train_info = {}
#         val_info   = {}
#         for u, (tr_days, val_days, _) in ssl_splits.items():
#             hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#             pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#             orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')
#             if len(orig_neg) < len(pos_df):
#                 extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#                 neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#             else:
#                 neg_df = orig_neg

#             df_tr  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
#             df_tr['user_id'] = u    ### added user id for fairness comparision
#             # df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

#             train_info[u] = {"days": tr_days.tolist(),  "df": df_tr}
#             # val_info[u]   = {"days": val_days.tolist(), "df": df_val}

#         # 3) Apply sampling across users
#         # sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)
#         df_all_tr  = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)
#         
#         # ---------------------------------------------------------
#         # 3) Build this user's TRAIN / VAL / TEST window DataFrames
#         # ---------------------------------------------------------
#         hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
#         pos_df_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
#         orig_neg_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, "None")

#         if len(orig_neg_u) < len(pos_df_u):
#             extra = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u) - len(orig_neg_u))
#             neg_df_u = pd.concat([orig_neg_u, extra], ignore_index=True)
#         else:
#             neg_df_u = orig_neg_u

#         tr_days_u, val_days_u, te_days_u = ensure_train_val_test_days(
#             pos_df_u, neg_df_u, hr_df_u, st_df_u
#         )

#         df_tr = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, tr_days_u)
#         df_val = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)
#         df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
#         _debug_label_alignment(pos_df_u, neg_df_u, df_te, te_days_u, label="test")
#         
#     elif pool == 'global_supervised':
#         pass
#     else:
#         raise ValueError("pool must be 'personal' or 'global' or 'mixed' ")
#     _warn_small_split(df_val, "val")
#     _warn_small_split(df_te, "test")
#     return df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root


# def prepare_data(
#     args,
#     pool: str,
#     top_out: Path,
#     shared_enc_root: Path,
#     batch_ssl: int,
#     ssl_epochs: int,
# ) -> tuple:
#     """
#     High‑level preprocessing for the no_leak-style AL pipeline.

#     - Build SSL splits on OTHER users for global encoders
#     - Train/load global encoders via uq_utility._ensure_global_encoders
#     - Build this user's train/val/test window DataFrames

#     Returns:
#         enc_hr, enc_st, df_tr, df_val, df_te, user_root
#     """

#     # Per‑user output root: under output directory
#     user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
#     user_root.mkdir(parents=True, exist_ok=True)

#     # Seed everything (same as no_leak.py)
#     random.seed(42)
#     np.random.seed(42)
#     import tensorflow as tf 
#     tf.random.set_seed(42)

#     # ---------------------------------------------------------
#     # 1) Build SSL splits on OTHER users (for global encoders)
#     # ---------------------------------------------------------
#     ssl_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

#     for u, pairs in ALLOWED_SCENARIOS.items():
#         # Exclude target user
#         if u == args.user:
#             continue
#         if (args.fruit, args.scenario) not in pairs:
#             continue

#         hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#         pos_df = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#         orig_neg = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, "None")

#         if len(orig_neg) < len(pos_df):
#             extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#             neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#         else:
#             neg_df = orig_neg

#         try:
#             tr_u, val_u, te_u = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
#         except RuntimeError as e:
#             print(f"Skipping user {u}: {e}")
#             continue

#         ssl_splits[u] = (tr_u, val_u, te_u)

#     # ---------------------------------------------------------
#     # 2) Train / load global SSL encoders (OTHER users only)
#     # ---------------------------------------------------------
#     if pool != "global":
#         raise ValueError("This AL entrypoint currently supports only pool='global'.")

#     enc_hr, enc_st, enc_src = uq_utility._ensure_global_encoders(
#         shared_enc_root,
#         args.fruit,
#         args.scenario,
#         ssl_splits,
#         batch_ssl,
#         ssl_epochs,
#     )
    

#     # Optionally copy encoder artifacts under this user's root 
#     models_d = (user_root / "global_ssl" / "models_saved")
#     results_d = (user_root / "global_ssl" / "results")
#     models_d.mkdir(parents=True, exist_ok=True)
#     results_d.mkdir(parents=True, exist_ok=True)

#     for fpath in Path(enc_src).glob("*"):
#         if fpath.suffix == ".keras":
#             (models_d / fpath.name).write_bytes(fpath.read_bytes())
#         elif fpath.suffix == ".png":
#             (results_d / fpath.name).write_bytes(fpath.read_bytes())
#     # SSL Pretraining phase (other users only)
#     train_info = {}
#     val_info   = {}
#     for u, (tr_days, val_days, _) in ssl_splits.items():
#         hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / u)
#         pos_df       = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, args.scenario)
#         orig_neg     = load_label_data(Path(BASE_DATA_DIR) / u, args.fruit, 'None')
#         if len(orig_neg) < len(pos_df):
#             extra  = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
#             neg_df = pd.concat([orig_neg, extra], ignore_index=True)
#         else:
#             neg_df = orig_neg

#         df_tr_others  = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
#         # df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)

#         train_info[u] = {"days": tr_days.tolist(),  "df": df_tr_others}
#         # val_info[u]   = {"days": val_days.tolist(), "df": df_val}

#     # 3) Apply sampling across users
#     # sample_summary = sample_train_info(train_info, mode=sample_mode, random_state=42)
#     df_all_tr  = pd.concat([v["df"] for v in train_info.values()], ignore_index=True)

#     # ---------------------------------------------------------
#     # 3) Build this user's TRAIN / VAL / TEST window DataFrames
#     # ---------------------------------------------------------
#     hr_df_u, st_df_u = load_signal_data(Path(BASE_DATA_DIR) / args.user)
#     pos_df_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, args.scenario)
#     orig_neg_u = load_label_data(Path(BASE_DATA_DIR) / args.user, args.fruit, "None")

#     if len(orig_neg_u) < len(pos_df_u):
#         extra = derive_negative_labels(hr_df_u, pos_df_u, len(pos_df_u) - len(orig_neg_u))
#         neg_df_u = pd.concat([orig_neg_u, extra], ignore_index=True)
#     else:
#         neg_df_u = orig_neg_u

#     tr_days_u, val_days_u, te_days_u = ensure_train_val_test_days(
#         pos_df_u, neg_df_u, hr_df_u, st_df_u
#     )

#     df_tr = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, tr_days_u)
#     df_val = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, val_days_u)
#     df_te = collect_windows(pos_df_u, neg_df_u, hr_df_u, st_df_u, te_days_u)
    

#     return enc_hr, enc_st, df_tr, df_val, df_te, df_all_tr, user_root
