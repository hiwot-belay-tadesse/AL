"""
Count per-user train/val/test window labels.

Examples:
  python scripts/count.py --task fruit
  python scripts/count.py --task bp --bp-root DATA/Cardiomate/hp
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.classifier_utils import ALLOWED_SCENARIOS, BASE_DATA_DIR, load_label_data, load_signal_data
from src.compare_pipelines import (
    _bp_load_signal_df,
    collect_windows,
    derive_negative_labels,
    ensure_train_val_test_days,
    sample_train_info,
)


def _get_bp_users(bp_root: Path) -> list[str]:
    users: list[str] = []
    for p in sorted(bp_root.glob("hp*")):
        m = re.search(r"\d+", p.name)
        if not m:
            continue
        pid = m.group(0)
        base = bp_root / f"hp{pid}"
        if not (base / f"hp{pid}_hr.csv").exists():
            continue
        if not (base / f"hp{pid}_steps.csv").exists():
            continue
        if not (base / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
            continue
        users.append(pid)
    return users


def _bp_load_all_from_root(pid: str, bp_root: Path, sys_thresh: int = 130, dia_thresh: int = 80):
    """Mirror src.compare_pipelines._bp_load_all, but with explicit bp_root."""
    base = bp_root / f"hp{pid}"
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

    pos_df = df_bp[df_bp["BP_spike"] == 1][["datetime_local"]].copy()
    neg_df = df_bp[df_bp["BP_spike"] == 0][["datetime_local"]].copy()
    pos_df = pos_df.rename(columns={"datetime_local": "hawaii_createdat_time"})
    neg_df = neg_df.rename(columns={"datetime_local": "hawaii_createdat_time"})

    return hr_df, st_df, pos_df.reset_index(drop=True), neg_df.reset_index(drop=True)


def count_fruit_per_user(fruit: str, scenario: str, sample_mode: str = "original") -> list[dict]:
    all_splits: dict[str, tuple] = {}
    cache: dict[str, tuple] = {}
    raw_counts: dict[str, tuple[int, int]] = {}

    for user, pairs in ALLOWED_SCENARIOS.items():
        if (fruit, scenario) not in pairs:
            continue

        hr_df, st_df = load_signal_data(Path(BASE_DATA_DIR) / user)
        pos_df = load_label_data(Path(BASE_DATA_DIR) / user, fruit, scenario)
        orig_neg = load_label_data(Path(BASE_DATA_DIR) / user, fruit, "None")
        raw_counts[user] = (int(len(pos_df)), int(len(orig_neg)))

        if len(orig_neg) < len(pos_df):
            extra = derive_negative_labels(hr_df, pos_df, len(pos_df) - len(orig_neg))
            neg_df = pd.concat([orig_neg, extra], ignore_index=True)
        else:
            neg_df = orig_neg

        try:
            tr_days, val_days, te_days = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as exc:
            print(f"Skipping user {user}: {exc}")
            continue

        all_splits[user] = (tr_days, val_days, te_days)
        cache[user] = (hr_df, st_df, pos_df, neg_df)

    if not all_splits:
        return []

    train_info: dict[str, dict] = {}
    val_info: dict[str, dict] = {}
    for user, (tr_days, val_days, _te_days) in all_splits.items():
        hr_df, st_df, pos_df, neg_df = cache[user]
        df_tr = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        train_info[user] = {"days": tr_days.tolist(), "df": df_tr}
        val_info[user] = {"days": val_days.tolist(), "df": df_val}

    sample_train_info(train_info, mode=sample_mode, random_state=42)

    rows: list[dict] = []
    for user in sorted(train_info.keys()):
        hr_df, st_df, pos_df, neg_df = cache[user]
        _tr_days, _val_days, te_days = all_splits[user]
        df_tr = train_info[user]["df"]
        df_val = val_info[user]["df"]
        df_te = collect_windows(pos_df, neg_df, hr_df, st_df, te_days)

        rows.append(
            {
                "task": "fruit",
                "user": user,
                "fruit_scenario": f"{fruit}_{scenario}",
                "len_df_train": int(len(df_tr)),
                "pos": int((df_tr["state_val"] == 1).sum()),
                "neg": int((df_tr["state_val"] == 0).sum()),
                "len_df_val": int(len(df_val)),
                "pos_val": int((df_val["state_val"] == 1).sum()),
                "neg_val": int((df_val["state_val"] == 0).sum()),
                "len_df_test": int(len(df_te)),
                "pos_test": int((df_te["state_val"] == 1).sum()),
                "neg_test": int((df_te["state_val"] == 0).sum()),
                "raw_pos_labels": raw_counts[user][0],
                "raw_neg_labels": raw_counts[user][1],
            }
        )
    return rows


def count_bp_per_user(sample_mode: str = "original", bp_root: Path | None = None) -> list[dict]:
    bp_root = bp_root or (PROJECT_ROOT / "DATA" / "Cardiomate" / "hp")
    bp_users = _get_bp_users(bp_root)
    if not bp_users:
        return []

    all_splits: dict[str, tuple] = {}
    cache: dict[str, tuple] = {}
    raw_counts: dict[str, tuple[int, int]] = {}

    for pid in bp_users:
        hr_df, st_df, pos_df, neg_df = _bp_load_all_from_root(pid, bp_root=bp_root)
        raw_counts[pid] = (int(len(pos_df)), int(len(neg_df)))

        try:
            tr_days, val_days, te_days = ensure_train_val_test_days(pos_df, neg_df, hr_df, st_df)
        except RuntimeError as exc:
            print(f"Skipping user {pid}: {exc}")
            continue

        all_splits[pid] = (tr_days, val_days, te_days)
        cache[pid] = (hr_df, st_df, pos_df, neg_df)

    if not all_splits:
        return []

    train_info: dict[str, dict] = {}
    val_info: dict[str, dict] = {}
    for pid, (tr_days, val_days, _te_days) in all_splits.items():
        hr_df, st_df, pos_df, neg_df = cache[pid]
        df_tr = collect_windows(pos_df, neg_df, hr_df, st_df, tr_days)
        df_val = collect_windows(pos_df, neg_df, hr_df, st_df, val_days)
        train_info[pid] = {"days": tr_days.tolist(), "df": df_tr}
        val_info[pid] = {"days": val_days.tolist(), "df": df_val}

    sample_train_info(train_info, mode=sample_mode, random_state=42)

    rows: list[dict] = []
    for pid in sorted(train_info.keys(), key=lambda x: int(x)):
        hr_df, st_df, pos_df, neg_df = cache[pid]
        _tr_days, _val_days, te_days = all_splits[pid]
        df_tr = train_info[pid]["df"]
        df_val = val_info[pid]["df"]
        df_te = collect_windows(pos_df, neg_df, hr_df, st_df, te_days)

        rows.append(
            {
                "task": "bp",
                "user": pid,
                # "fruit_scenario": "BP_spike",
                "len_df_train": int(len(df_tr)),
                "pos": int((df_tr["state_val"] == 1).sum()),
                "neg": int((df_tr["state_val"] == 0).sum()),
                "len_df_val": int(len(df_val)),
                "pos_val": int((df_val["state_val"] == 1).sum()),
                "neg_val": int((df_val["state_val"] == 0).sum()),
                "len_df_test": int(len(df_te)),
                "pos_test": int((df_te["state_val"] == 1).sum()),
                "neg_test": int((df_te["state_val"] == 0).sum()),
                # "raw_pos_labels": raw_counts[pid][0],
                # "raw_neg_labels": raw_counts[pid][1],
            }
        )
    return rows


def main() -> None:
    pa = argparse.ArgumentParser(description="Count per-user train/val/test window labels by task.")
    pa.add_argument("--task", choices=["fruit", "bp"], default="fruit")
    pa.add_argument("--sample-mode", choices=["original", "undersample", "oversample"], default="original")
    pa.add_argument(
        "--bp-root",
        default=str(PROJECT_ROOT / "DATA" / "Cardiomate" / "hp"),
        help="BP dataset root containing hp*/ folders (used for --task bp discovery and loading).",
    )
    args = pa.parse_args()

    rows: list[dict] = []
    if args.task == "fruit":
        pairs = sorted({pair for pairs in ALLOWED_SCENARIOS.values() for pair in pairs})
        for fruit, scenario in pairs:
            rows.extend(count_fruit_per_user(fruit, scenario, sample_mode=args.sample_mode))
        out_name = "df_train_counts_per_user.csv"
    else:
        rows.extend(count_bp_per_user(sample_mode=args.sample_mode, bp_root=Path(args.bp_root)))
        out_name = "df_train_counts_bp_spike_per_user.csv"

    results_df = pd.DataFrame(rows)
    print(results_df)
    (PROJECT_ROOT / "supp_plots").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(PROJECT_ROOT / "supp_plots" / out_name, index=False)


if __name__ == "__main__":
    main()
