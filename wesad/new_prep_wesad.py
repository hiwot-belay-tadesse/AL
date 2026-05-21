"""
WESAD data prep — self-contained, mirrors the return signature of new_prep.prepare_data
so wesad/run_wesad.py (a copy of run.py with the import swapped) can use it without
any change to the BP pipeline.

Layout this expects (produced by wesad/convert_wesad_to_csv.py at 2-sec resample):
    DATA/WESAD/hp/hp<pid>/hp<pid>_hr.csv      (2-sec rows, columns: hawaii_createdat_time, value)
    DATA/WESAD/hp/hp<pid>/hp<pid>_steps.csv   (2-sec rows, same schema)
    DATA/WESAD/hp/hp<pid>/hp<pid>_stress.csv  (one row per protocol/PANAS segment with
                                               start_datetime, end_datetime, stress_binary)

Windowing (replaces the BP ±1h `process_label_window` which doesn't fit WESAD's
100-min sessions):
    - WINDOW_SEC = 60       : each window covers 60 seconds of signal.
    - BIN_SEC    = 2        : 30 bins of 2 seconds each (FEATURE_POINTS=30, matches encoder).
    - STRIDE_SEC = 2        : sliding stride = 1 bin (~30x overlap).
    - Strict majority label : a window is positive iff stress dominates the full 60-sec span;
                              boundary windows (containing >1 condition by >LABEL_PURITY_FRAC)
                              are dropped.

Split (interleave-by-segment within subject):
    WESAD's protocol always runs stress at ~30–60 min of a 90-min session, so a single
    time-block split (first 60% / last 20%) leaves test with zero stress windows. To keep
    a time-block design *and* guarantee class balance, we split *within each labeled
    segment*: the first TRAIN_FRAC of windows inside that segment go to train, the next
    VAL_FRAC to val, the last TEST_FRAC to test. A BUFFER_SEC gap is removed at each
    sub-block boundary so near-duplicate sliding windows don't straddle the split.
    Net result: every segment (stress, baseline, amusement, meditation) contributes
    proportionally to all three splits.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Reuse only the encoder loader; windowing is rewritten below.
from src.compare_pipelines import _train_or_load_encoder


WESAD_ROOT = Path("DATA/WESAD/hp")

# Window contract: 60-sec span × 30 bins × 2-sec stride.
WINDOW_SEC = 60
BIN_SEC = 2
STRIDE_SEC = 2
FEATURE_POINTS = WINDOW_SEC // BIN_SEC  # 30, matches src.classifier_utils.FEATURE_POINTS

# Strict majority: window is labeled iff one condition covers ≥ LABEL_PURITY_FRAC
# of its time span. Otherwise the window straddles a transition and is dropped.
LABEL_PURITY_FRAC = 1.0  # 1.0 = absolutely strict (entire window one condition).

# Time-block split fractions and buffer between blocks (in seconds).
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20
BUFFER_SEC = WINDOW_SEC  # gap between blocks ≥ one window so no overlap leaks across.

# Fabricated calendar days so we can keep the encoder's `train_days` API.
TRAIN_DAY = datetime(2020, 1, 1).date()
TEST_DAY = datetime(2020, 1, 2).date()


def _load_hr_steps(pid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load 2-sec hr/steps CSVs indexed by datetime."""
    base = WESAD_ROOT / f"hp{pid}"
    hr_path = base / f"hp{pid}_hr.csv"
    st_path = base / f"hp{pid}_steps.csv"
    if not hr_path.exists() or not st_path.exists():
        raise FileNotFoundError(f"Missing hr/steps CSVs under {base}")
    hr_df = pd.read_csv(hr_path)
    st_df = pd.read_csv(st_path)
    hr_df["hawaii_createdat_time"] = pd.to_datetime(hr_df["hawaii_createdat_time"])
    st_df["hawaii_createdat_time"] = pd.to_datetime(st_df["hawaii_createdat_time"])
    hr_df = hr_df.set_index("hawaii_createdat_time").sort_index()
    st_df = st_df.set_index("hawaii_createdat_time").sort_index().fillna(0)
    return hr_df, st_df


def _load_stress_events(pid: str) -> pd.DataFrame:
    """Returns segment table with start_datetime, end_datetime, stress_binary."""
    base = WESAD_ROOT / f"hp{pid}"
    stress_path = base / f"hp{pid}_stress.csv"
    if not stress_path.exists():
        raise FileNotFoundError(f"Missing stress CSV: {stress_path}")
    df = pd.read_csv(stress_path)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    df["end_datetime"] = pd.to_datetime(df["end_datetime"])
    df = df.dropna(subset=["start_datetime", "end_datetime", "stress_binary"])
    df["stress_binary"] = df["stress_binary"].astype(int)
    return df.sort_values("start_datetime").reset_index(drop=True)


def _label_for_window(win_start: pd.Timestamp, win_end: pd.Timestamp,
                      events: pd.DataFrame) -> int | None:
    """Strict-majority label for a sliding window.

    Returns 0/1 if one label dominates ≥ LABEL_PURITY_FRAC of [win_start, win_end].
    Returns None if the window straddles a transition (drop it).
    """
    span_sec = (win_end - win_start).total_seconds()
    if span_sec <= 0:
        return None
    overlaps = {0: 0.0, 1: 0.0}
    for _, row in events.iterrows():
        seg_start = row["start_datetime"]
        seg_end = row["end_datetime"]
        if seg_end <= win_start or seg_start >= win_end:
            continue
        ov_start = max(seg_start, win_start)
        ov_end = min(seg_end, win_end)
        overlaps[int(row["stress_binary"])] += (ov_end - ov_start).total_seconds()
    covered = overlaps[0] + overlaps[1]
    if covered <= 0:
        return None
    for lbl, secs in overlaps.items():
        if secs / span_sec >= LABEL_PURITY_FRAC:
            return lbl
    return None


def _build_windows(hr_df: pd.DataFrame, st_df: pd.DataFrame,
                   events: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp,
                   pid: str, day: object) -> pd.DataFrame:
    """Generate sliding windows over [t_start, t_end), label by strict majority.

    Per-window standard-scaling is applied (matches process_label_window's behavior).
    The window's hawaii_createdat_time is set to the window center, with date forced
    to `day` so downstream day-based filters (kept for parity) still match.
    """
    if hr_df.empty or st_df.empty:
        return pd.DataFrame()

    bin_td = pd.Timedelta(seconds=BIN_SEC)
    win_td = pd.Timedelta(seconds=WINDOW_SEC)
    stride_td = pd.Timedelta(seconds=STRIDE_SEC)

    scaler = StandardScaler()
    records = []
    cur = t_start
    while cur + win_td <= t_end:
        w_start = cur
        w_end = cur + win_td
        lbl = _label_for_window(w_start, w_end, events)
        if lbl is None:
            cur += stride_td
            continue

        # Slice raw signal then bin to FEATURE_POINTS via 2-sec resample.
        # The converter already produced 2-sec rows, so this is effectively a take.
        hr_win = hr_df.loc[w_start:w_end - pd.Timedelta(microseconds=1)]["value"]
        st_win = st_df.loc[w_start:w_end - pd.Timedelta(microseconds=1)]["value"]
        if len(hr_win) < FEATURE_POINTS or len(st_win) < FEATURE_POINTS:
            cur += stride_td
            continue

        hr_bins = hr_win.resample(bin_td, origin=w_start).mean().iloc[:FEATURE_POINTS]
        st_bins = st_win.resample(bin_td, origin=w_start).mean().iloc[:FEATURE_POINTS]
        if (len(hr_bins) != FEATURE_POINTS or hr_bins.isna().any() or
                len(st_bins) != FEATURE_POINTS or st_bins.isna().any()):
            cur += stride_td
            continue

        hr_scaled = scaler.fit_transform(hr_bins.values.reshape(-1, 1)).flatten().tolist()
        st_scaled = scaler.fit_transform(st_bins.values.reshape(-1, 1)).flatten().tolist()

        center_time = w_start + (win_td / 2)
        # Stamp window's date with `day` so downstream day-based filters match.
        stamped = pd.Timestamp.combine(day, center_time.time())
        records.append({
            "hawaii_createdat_time": stamped,
            "hr_seq": hr_scaled,
            "st_seq": st_scaled,
            "state_val": int(lbl),
            "user_id": str(pid),
        })
        cur += stride_td

    return pd.DataFrame(records)


def _split_segment(seg_start: pd.Timestamp, seg_end: pd.Timestamp) -> list[
        tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Split a single labeled segment into train/val/test sub-blocks with BUFFER_SEC gaps.

    Sub-block durations are computed from the *usable* span (total minus the two
    inter-block buffers), so train/val/test each get their fair share. Sub-blocks
    shorter than WINDOW_SEC are dropped since no full window fits.
    """
    total = (seg_end - seg_start).total_seconds()
    usable = total - 2 * BUFFER_SEC
    if usable < WINDOW_SEC:
        return []

    train_dur = usable * TRAIN_FRAC
    val_dur = usable * VAL_FRAC
    test_dur = usable * TEST_FRAC

    train_start = seg_start
    train_end = train_start + pd.Timedelta(seconds=train_dur)
    val_start = train_end + pd.Timedelta(seconds=BUFFER_SEC)
    val_end = val_start + pd.Timedelta(seconds=val_dur)
    test_start = val_end + pd.Timedelta(seconds=BUFFER_SEC)
    test_end = test_start + pd.Timedelta(seconds=test_dur)

    blocks = []
    for (s, e, name) in [
        (train_start, train_end, "train"),
        (val_start, val_end, "val"),
        (test_start, test_end, "test"),
    ]:
        if (e - s).total_seconds() >= WINDOW_SEC:
            blocks.append((s, e, name))
    return blocks


def _build_user_frame(pid: str) -> tuple[pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (hr_df, st_df, df_train, df_val, df_test) for a single subject.

    Interleaves the 60/20/20 split *within each labeled segment* (see module docstring).
    """
    hr_df, st_df = _load_hr_steps(pid)
    events = _load_stress_events(pid)

    train_frames, val_frames, test_frames = [], [], []
    for _, ev in events.iterrows():
        for (s, e, split_name) in _split_segment(ev["start_datetime"], ev["end_datetime"]):
            day = TEST_DAY if split_name == "test" else TRAIN_DAY
            df_block = _build_windows(hr_df, st_df, events, s, e, pid, day)
            if df_block.empty:
                continue
            if split_name == "train":
                train_frames.append(df_block)
            elif split_name == "val":
                val_frames.append(df_block)
            else:
                test_frames.append(df_block)

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=["hawaii_createdat_time", "hr_seq", "st_seq",
                                          "state_val", "user_id"])
        return pd.concat(frames, ignore_index=True)

    return hr_df, st_df, _concat(train_frames), _concat(val_frames), _concat(test_frames)


def _discover_users() -> list[str]:
    if not WESAD_ROOT.exists():
        return []
    pids = []
    for p in sorted(WESAD_ROOT.glob("hp*")):
        m = re.search(r"\d+", p.name)
        if not m:
            continue
        pids.append(m.group(0))
    return pids


def prepare_data(
    args,
    top_out: Path,
    shared_enc_root: Path,
    shared_cnn_root: Path,
    batch_ssl: int,
    ssl_epochs: int,
    pool: str,
    task: str = "wesad",
    input_df: str | None = None,
    seed: int | None = None,
):
    """Mirror of new_prep.prepare_data for WESAD. Returns the same tuple shape."""
    if input_df is None:
        input_df = getattr(args, "input_df", "raw")
    if input_df != "raw":
        raise NotImplementedError("WESAD branch only supports input_df='raw'.")

    user_root = top_out / args.user / f"{args.fruit}_{args.scenario}"
    user_root.mkdir(parents=True, exist_ok=True)
    out_dir = user_root / pool
    models_d = out_dir / "models_saved"
    results_d = out_dir / "results"
    models_d.mkdir(parents=True, exist_ok=True)
    results_d.mkdir(parents=True, exist_ok=True)

    excluded_users = {
        u.strip() for u in os.environ.get("BAN_AL_EXCLUDE_USERS", "").split(",") if u.strip()
    }
    if excluded_users:
        print(f"[wesad-exclude] BAN_AL_EXCLUDE_USERS active; dropped: {sorted(excluded_users)}")

    discovered = [u for u in _discover_users() if u not in excluded_users]
    if not discovered:
        raise SystemExit(f"No WESAD subjects found under {WESAD_ROOT}.")

    uid = str(args.user)
    if uid not in discovered:
        raise SystemExit(f"Target user {uid} not found in WESAD subjects: {discovered}")

    user_iter = discovered if pool == "global" else [uid]

    train_info: dict[str, dict] = {}
    val_info: dict[str, dict] = {}
    target_test_df: pd.DataFrame | None = None
    target_hr: pd.DataFrame | None = None
    target_st: pd.DataFrame | None = None

    for u in user_iter:
        try:
            hr_u, st_u, df_tr_u, df_val_u, df_te_u = _build_user_frame(u)
        except Exception as e:
            print(f"[wesad] skipping user {u}: {e}")
            continue
        if df_tr_u.empty:
            print(f"[wesad] skipping user {u}: empty train block after windowing.")
            continue
        train_info[u] = {"days": [TRAIN_DAY], "df": df_tr_u, "hr": hr_u, "st": st_u}
        val_info[u] = {"days": [TRAIN_DAY], "df": df_val_u}
        if u == uid:
            target_test_df = df_te_u
            target_hr = hr_u
            target_st = st_u

    if uid not in train_info:
        raise SystemExit(f"Failed to build train pool for target user {uid}.")

    if pool == "personal":
        df_tr = train_info[uid]["df"].copy()
        df_val = val_info[uid]["df"].copy()
        df_all_tr = None
        if input_df == "raw":
            enc_hr = _train_or_load_encoder(
                models_d / "hr_encoder.keras", "hr",
                target_hr, [TRAIN_DAY], results_d,
            )
            enc_st = _train_or_load_encoder(
                models_d / "steps_encoder.keras", "steps",
                target_st, [TRAIN_DAY], results_d,
            )
        else:
            enc_hr, enc_st = None, None
    elif pool == "global":
        df_all_tr = pd.concat([info["df"] for info in train_info.values()], ignore_index=True)
        df_val = pd.concat([v["df"] for v in val_info.values()], ignore_index=True)
        df_tr = train_info[uid]["df"].copy()

        per_user_counts = df_all_tr["user_id"].astype(str).value_counts().sort_index()
        excl_in_pool = [u for u in per_user_counts.index if u in excluded_users]
        excl_rows = int(per_user_counts.loc[excl_in_pool].sum()) if excl_in_pool else 0
        total_rows = int(per_user_counts.sum())
        share = (100.0 * excl_rows / total_rows) if total_rows else 0.0
        print(
            f"[wesad-df_all_tr] {total_rows} rows from {len(per_user_counts)} users; "
            f"excluded-user rows in pool: {excl_rows} ({share:.1f}%) "
            f"from {excl_in_pool if excl_in_pool else '[]'}"
        )

        if input_df == "raw":
            shared_enc_root.mkdir(parents=True, exist_ok=True)
            enc_hr_path = shared_enc_root / "wesad_hr_encoder.keras"
            enc_st_path = shared_enc_root / "wesad_steps_encoder.keras"
            concat_hr = pd.concat([info["hr"] for info in train_info.values()])
            concat_st = pd.concat([info["st"] for info in train_info.values()])
            enc_hr = _train_or_load_encoder(enc_hr_path, "hr", concat_hr, [TRAIN_DAY], results_d)
            enc_st = _train_or_load_encoder(enc_st_path, "steps", concat_st, [TRAIN_DAY], results_d)
            for src, dst in [
                (enc_hr_path, models_d / "hr_encoder.keras"),
                (enc_st_path, models_d / "steps_encoder.keras"),
            ]:
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        else:
            enc_hr, enc_st = None, None
    else:
        raise ValueError(f"Unknown pool: {pool!r}")

    if target_test_df is None or len(target_test_df) == 0:
        raise SystemExit(
            f"Target user {uid} has no test windows after time-block split. "
            "Either the session is too short for the 60/20/20 split or no labeled "
            "stress segments fall in the test block."
        )
    df_te = target_test_df.copy()

    all_splits = {u: ([TRAIN_DAY], [TRAIN_DAY], [TEST_DAY]) for u in train_info}
    all_negatives: dict = {}

    print(f"[wesad] target={uid} pool={pool} train={len(df_tr)} val={len(df_val)} test={len(df_te)}")
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
