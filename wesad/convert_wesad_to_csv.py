"""
Convert WESAD .pkl files into a CSV layout that wesad/new_prep_wesad.py expects.

Input layout:
    <wesad_raw>/S<id>/S<id>.pkl
    <wesad_raw>/S<id>/S<id>_quest.csv

Output layout (parallels DATA/Cardiomate/hp/hp<pid>/ for visual consistency, but kept
under DATA/WESAD/hp/ so there is no name collision and BP code never sees it):
    DATA/WESAD/hp/hp<id>/hp<id>_hr.csv      ← resampled mean BVP-derived HR
    DATA/WESAD/hp/hp<id>/hp<id>_steps.csv   ← resampled mean ACC magnitude
    DATA/WESAD/hp/hp<id>/hp<id>_stress.csv  ← one row per protocol condition with
                                              datetime_local, condition, panas_stressed,
                                              stress_binary

Resample rate is configurable via --resample (default '2S' = 2 seconds), to match the
2-sec bins used by the new WESAD windowing pipeline in new_prep_wesad.py.

Why this layout: keeps WESAD-derived CSVs entirely separate from any Cardiomate files.
new_prep_wesad.py will read these CSVs directly. BP discovery logic in new_prep.py is
untouched.

Two label sources are supported via --label_source:

  panas    : (default) PANAS "Stressed" item (index 20, 0-indexed) is thresholded
             at >=3 ("Somewhat" or higher) to produce stress_binary. Sparse:
             ~4 rows per subject (one per protocol condition).

  protocol : WESAD's 700 Hz `label` array from the .pkl. Condition codes:
             0 = transient/not defined, 1 = baseline, 2 = stress, 3 = amusement,
             4 = meditation. stress_binary = 1 iff label == 2, else 0. We collapse
             each contiguous protocol segment into one row, so you get many more
             labeled segments than PANAS (typically 5-10 protocol segments per
             subject, depending on the recording).

Run:
    python wesad/convert_wesad_to_csv.py --wesad_root /path/to/WESAD_raw
    python wesad/convert_wesad_to_csv.py --wesad_root /path/to/WESAD_raw \\
        --label_source protocol --out_root DATA/WESAD_protocol/hp
"""

from __future__ import annotations

import argparse
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


WRIST_BVP_HZ = 64
WRIST_ACC_HZ = 32
LABEL_HZ = 700  # WESAD chest + label stream
FAKE_EPOCH = datetime(2020, 1, 1, 0, 0, 0)

# Order of PANAS items in SX_quest.csv, per WESAD README.
PANAS_ITEMS = [
    "Active", "Distressed", "Interested", "Inspired", "Annoyed", "Strong",
    "Guilty", "Scared", "Hostile", "Excited", "Proud", "Irritable", "Enthusiastic",
    "Ashamed", "Alert", "Nervous", "Determined", "Attentive", "Jittery", "Afraid",
    "Stressed", "Frustrated", "Happy", "Sad",
]
STRESSED_INDEX = PANAS_ITEMS.index("Stressed")  # 20
STRESS_THRESHOLD = 3  # PANAS scale 1-5; ≥3 = "Somewhat" or higher → positive

# WESAD protocol-label condition codes from the .pkl's 'label' array.
PROTOCOL_CONDITIONS = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}
PROTOCOL_STRESS_CODE = 2  # protocol code that maps to stress_binary=1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert WESAD .pkl to per-subject CSVs.")
    parser.add_argument("--wesad_root", type=Path, required=True,
                        help="Root directory containing S2/, S3/, ... subject folders.")
    parser.add_argument("--out_root", type=Path, default=Path("DATA/WESAD/hp"),
                        help="Output directory. Per-subject CSVs go to <out_root>/hp<id>/.")
    parser.add_argument("--bvp_to_hr_method", default="rolling_mean", choices=["rolling_mean", "raw"],
                        help="rolling_mean: 1s rolling mean of |BVP| as a crude HR proxy. raw: keep raw BVP.")
    parser.add_argument("--label_source", default="panas", choices=["panas", "protocol"],
                        help="panas: PANAS 'Stressed' item thresholded at >=3 (sparse, ~4/subject). "
                             "protocol: WESAD 700 Hz protocol label array, one row per contiguous "
                             "condition segment (stress=condition code 2).")
    parser.add_argument("--resample", default="2S",
                        help="pandas resample rule for HR/steps CSVs. Default '2S' (2 seconds) "
                             "matches the 2-sec bins in new_prep_wesad.py. Use '1min' for the legacy "
                             "per-minute layout.")
    return parser.parse_args()


def acc_magnitude(acc_xyz: np.ndarray) -> np.ndarray:
    """L2 norm over the last axis (3 channels)."""
    return np.linalg.norm(acc_xyz, axis=-1)


def bvp_to_hr_proxy(bvp: np.ndarray, hz: int) -> np.ndarray:
    """Cheap HR proxy: 1-second rolling mean of |BVP|.

    Avoids the heavy dependencies of a real PPG → HR pipeline (heartpy, scipy peak-find).
    Downstream we resample to per-minute means anyway, so high-frequency detail is lost.
    If you want true beats-per-minute, replace this with heartpy.process(bvp, sample_rate=hz)
    and take the resulting bpm series.
    """
    window = max(1, int(hz))
    s = pd.Series(np.abs(bvp), dtype=float)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def samples_to_resampled_csv(values: np.ndarray, hz: int, start: datetime, rule: str) -> pd.DataFrame:
    """Resample a high-rate signal to `rule`-spaced means with fabricated timestamps."""
    n = len(values)
    times = pd.date_range(start=start, periods=n, freq=pd.Timedelta(seconds=1.0 / hz))
    df = pd.DataFrame({"value": values}, index=times)
    resampled = df.resample(rule).mean().reset_index()
    resampled = resampled.rename(columns={"index": "datetime"})
    resampled["hawaii_createdat_time"] = resampled["datetime"]
    return resampled[["hawaii_createdat_time", "value"]]


def parse_quest_csv(quest_path: Path) -> pd.DataFrame:
    """Pull condition order, time intervals (min.sec), and PANAS rows out of SX_quest.csv.

    Returns a DataFrame with columns: condition, start_min, end_min, panas_stressed (NaN if absent).
    """
    if not quest_path.exists():
        raise FileNotFoundError(f"Missing quest file: {quest_path}")

    # WESAD's quest CSVs are semicolon-separated, with each row tagged by its first cell.
    # Lines we need:
    #   "# ORDER" : second row, condition order (one cell per condition).
    #   "# START" : third row, start times in min.sec.
    #   "# END"   : fourth row, end times in min.sec.
    #   Five PANAS rows: each starts with "# PANAS"; order matches conditions.
    rows: dict[str, list[list[str]]] = {}
    with open(quest_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            cells = [c.strip() for c in line.split(";")]
            tag = cells[0]
            rows.setdefault(tag, []).append(cells[1:])

    order_row = rows.get("# ORDER", [[]])[0]
    start_row = rows.get("# START", [[]])[0]
    end_row = rows.get("# END", [[]])[0]
    panas_rows = rows.get("# PANAS", [])

    # Filter out the dataset-internal "bRead", "fRead", "sRead" labels per README.
    keep_idx = [i for i, c in enumerate(order_row) if c and c not in ("bRead", "fRead", "sRead")]
    conditions = [order_row[i] for i in keep_idx]
    starts = [_minsec_to_minutes(start_row[i]) if i < len(start_row) else np.nan for i in keep_idx]
    ends = [_minsec_to_minutes(end_row[i]) if i < len(end_row) else np.nan for i in keep_idx]

    # PANAS rows: one per condition, in the same order. Pull the "Stressed" item.
    panas_stressed = []
    for idx in keep_idx:
        if idx < len(panas_rows):
            cells = panas_rows[idx]
            val = cells[STRESSED_INDEX] if len(cells) > STRESSED_INDEX else ""
            try:
                panas_stressed.append(float(val))
            except (ValueError, TypeError):
                panas_stressed.append(np.nan)
        else:
            panas_stressed.append(np.nan)

    df = pd.DataFrame({
        "condition": conditions,
        "start_min": starts,
        "end_min": ends,
        "panas_stressed": panas_stressed,
    })
    df["stress_binary"] = (df["panas_stressed"] >= STRESS_THRESHOLD).astype(int)
    return df


def protocol_labels_to_df(label_array: np.ndarray, hz: int, start: datetime) -> pd.DataFrame:
    """Collapse a high-rate protocol label stream into one row per contiguous segment.

    label_array: int array sampled at `hz`. Values are WESAD condition codes
    (see PROTOCOL_CONDITIONS). Transient (0) segments are skipped.

    Returns a DataFrame with the same columns as parse_quest_csv so downstream
    code stays uniform: condition, start_min, end_min, panas_stressed (NaN here,
    since protocol mode has no PANAS readout), stress_binary, start_datetime,
    end_datetime, datetime_local. start_min/end_min are minutes from `start`.
    """
    labels = np.asarray(label_array).ravel().astype(int)
    if labels.size == 0:
        return pd.DataFrame(columns=[
            "condition", "start_min", "end_min", "panas_stressed", "stress_binary",
            "start_datetime", "end_datetime", "datetime_local",
        ])

    # Find boundaries where the label value changes.
    change_idx = np.flatnonzero(np.diff(labels) != 0) + 1
    seg_starts = np.concatenate(([0], change_idx))
    seg_ends = np.concatenate((change_idx, [labels.size]))

    rows = []
    sec_per_sample = 1.0 / hz
    for s, e in zip(seg_starts, seg_ends):
        code = int(labels[s])
        if code == 0:
            # Transient/undefined: skip so positives/negatives aren't contaminated.
            continue
        start_sec = s * sec_per_sample
        end_sec = e * sec_per_sample
        start_min = start_sec / 60.0
        end_min = end_sec / 60.0
        rows.append({
            "condition": PROTOCOL_CONDITIONS.get(code, f"code_{code}"),
            "start_min": start_min,
            "end_min": end_min,
            "panas_stressed": np.nan,
            "stress_binary": int(code == PROTOCOL_STRESS_CODE),
            "start_datetime": start + timedelta(seconds=start_sec),
            "end_datetime": start + timedelta(seconds=end_sec),
            "datetime_local": start + timedelta(seconds=(start_sec + end_sec) / 2.0),
        })
    return pd.DataFrame(rows)


def _minsec_to_minutes(value: str) -> float:
    """Convert WESAD's "min.sec" format to fractional minutes. E.g. '2.30' → 2.5."""
    if not value or value in ("nan", "NaN"):
        return float("nan")
    try:
        parts = value.split(".")
        mins = int(parts[0])
        secs = int(parts[1]) if len(parts) > 1 else 0
        return mins + secs / 60.0
    except (ValueError, IndexError):
        return float("nan")


def write_subject(subject_dir: Path, out_root: Path, bvp_method: str, label_source: str,
                  resample_rule: str) -> None:
    pid_match = re.search(r"\d+", subject_dir.name)
    if not pid_match:
        print(f"Skipping {subject_dir}: no numeric subject ID.")
        return
    pid = pid_match.group(0)

    pkl_path = subject_dir / f"S{pid}.pkl"
    quest_path = subject_dir / f"S{pid}_quest.csv"
    if not pkl_path.exists():
        print(f"Skipping S{pid}: missing {pkl_path}.")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]
    bvp = np.asarray(wrist["BVP"]).ravel().astype("float32")
    acc = np.asarray(wrist["ACC"]).astype("float32")  # (N, 3) at 32 Hz

    if bvp_method == "rolling_mean":
        hr_samples = bvp_to_hr_proxy(bvp, WRIST_BVP_HZ)
    else:
        hr_samples = bvp
    acc_mag = acc_magnitude(acc)

    # Each subject's epoch is fabricated; downstream code only needs monotonic timestamps.
    hr_df = samples_to_resampled_csv(hr_samples, WRIST_BVP_HZ, FAKE_EPOCH, resample_rule)
    st_df = samples_to_resampled_csv(acc_mag, WRIST_ACC_HZ, FAKE_EPOCH, resample_rule)

    if label_source == "panas":
        quest_df = parse_quest_csv(quest_path)
        # Translate condition midpoint (in minutes from start) → fabricated datetime.
        quest_df["datetime_local"] = [
            FAKE_EPOCH + timedelta(minutes=(s + e) / 2.0)
            for s, e in zip(quest_df["start_min"], quest_df["end_min"])
        ]
        quest_df["start_datetime"] = [FAKE_EPOCH + timedelta(minutes=s) for s in quest_df["start_min"]]
        quest_df["end_datetime"] = [FAKE_EPOCH + timedelta(minutes=e) for e in quest_df["end_min"]]
    elif label_source == "protocol":
        label_array = np.asarray(data["label"]).ravel()
        quest_df = protocol_labels_to_df(label_array, LABEL_HZ, FAKE_EPOCH)
    else:
        raise ValueError(f"Unknown label_source: {label_source}")

    out_dir = out_root / f"hp{pid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    hr_df.to_csv(out_dir / f"hp{pid}_hr.csv", index=False)
    st_df.to_csv(out_dir / f"hp{pid}_steps.csv", index=False)
    quest_df.to_csv(out_dir / f"hp{pid}_stress.csv", index=False)
    n_pos = int(quest_df["stress_binary"].sum()) if not quest_df.empty else 0
    print(f"S{pid} [{label_source}]: hr={len(hr_df)} st={len(st_df)} "
          f"segments={len(quest_df)} positive_segments={n_pos}")


def main() -> int:
    args = parse_args()
    if not args.wesad_root.exists():
        raise SystemExit(f"WESAD root does not exist: {args.wesad_root}")
    subject_dirs = sorted(p for p in args.wesad_root.glob("S*") if p.is_dir())
    if not subject_dirs:
        raise SystemExit(f"No S* subject folders under {args.wesad_root}.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Converting with label_source={args.label_source} resample={args.resample} -> {args.out_root}")
    for sd in subject_dirs:
        try:
            write_subject(sd, args.out_root, args.bvp_to_hr_method, args.label_source, args.resample)
        except Exception as e:
            print(f"Failed to convert {sd.name}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
