"""
Sanity check for the WESAD windowing/split pipeline.

For each subject under DATA/WESAD/hp/, this script:
  - Builds train/val/test sliding windows via wesad.new_prep_wesad._build_user_frame.
  - Prints per-block window counts, label balance (n_pos/n_neg), and the boundary-drop
    rate (fraction of candidate strides discarded because the window straddled a
    label transition or had insufficient signal).
  - Aggregates totals across subjects at the end.

Usage:
    python -m wesad.sanity_check_wesad
    python -m wesad.sanity_check_wesad --users 2 3 4
    python -m wesad.sanity_check_wesad --csv DATA/WESAD/hp/sanity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wesad.new_prep_wesad import (
    BUFFER_SEC,
    STRIDE_SEC,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    WINDOW_SEC,
    _build_user_frame,
    _discover_users,
    _load_stress_events,
    _split_segment,
)


def _candidate_count(t_start: pd.Timestamp, t_end: pd.Timestamp) -> int:
    """How many sliding windows of WINDOW_SEC fit in [t_start, t_end] at STRIDE_SEC stride."""
    span = (t_end - t_start).total_seconds()
    if span < WINDOW_SEC:
        return 0
    return int((span - WINDOW_SEC) // STRIDE_SEC) + 1


def _label_balance(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty:
        return 0, 0
    pos = int((df["state_val"] == 1).sum())
    neg = int((df["state_val"] == 0).sum())
    return pos, neg


def check_subject(pid: str) -> dict:
    hr_df, st_df, df_tr, df_va, df_te = _build_user_frame(pid)
    events = _load_stress_events(pid)

    # Per-split candidate counts: sum over all segments of the candidate windows
    # in that segment's train/val/test sub-block (after interleave-by-segment split).
    cand_tr = cand_va = cand_te = 0
    for _, ev in events.iterrows():
        for (s, e, name) in _split_segment(ev["start_datetime"], ev["end_datetime"]):
            c = _candidate_count(s, e)
            if name == "train":
                cand_tr += c
            elif name == "val":
                cand_va += c
            else:
                cand_te += c

    pos_tr, neg_tr = _label_balance(df_tr)
    pos_va, neg_va = _label_balance(df_va)
    pos_te, neg_te = _label_balance(df_te)

    def drop_rate(kept: int, cand: int) -> float:
        return 1.0 - (kept / cand) if cand else 0.0

    t_min = max(hr_df.index.min(), st_df.index.min())
    t_max = min(hr_df.index.max(), st_df.index.max())
    return {
        "user": pid,
        "session_sec": (t_max - t_min).total_seconds(),
        "n_segments": len(events),
        "train_kept": len(df_tr), "train_cand": cand_tr, "train_drop": drop_rate(len(df_tr), cand_tr),
        "train_pos": pos_tr, "train_neg": neg_tr,
        "val_kept": len(df_va), "val_cand": cand_va, "val_drop": drop_rate(len(df_va), cand_va),
        "val_pos": pos_va, "val_neg": neg_va,
        "test_kept": len(df_te), "test_cand": cand_te, "test_drop": drop_rate(len(df_te), cand_te),
        "test_pos": pos_te, "test_neg": neg_te,
    }


def format_row(r: dict) -> str:
    return (
        f"hp{r['user']:>2}  "
        f"sess={r['session_sec']:6.0f}s segs={r['n_segments']:>2}  "
        f"train n={r['train_kept']:>4}/{r['train_cand']:>4} drop={r['train_drop']:.2f} "
        f"pos/neg={r['train_pos']:>3}/{r['train_neg']:>3}  "
        f"val n={r['val_kept']:>4}/{r['val_cand']:>4} drop={r['val_drop']:.2f} "
        f"pos/neg={r['val_pos']:>3}/{r['val_neg']:>3}  "
        f"test n={r['test_kept']:>4}/{r['test_cand']:>4} drop={r['test_drop']:.2f} "
        f"pos/neg={r['test_pos']:>3}/{r['test_neg']:>3}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="WESAD windowing sanity check.")
    ap.add_argument("--users", nargs="+", default=None,
                    help="Specific subject IDs to check (e.g. 2 3 4). Default: all discovered.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path to dump per-subject stats as CSV.")
    args = ap.parse_args()

    print(
        f"WESAD sanity check: window={WINDOW_SEC}s, stride={STRIDE_SEC}s, "
        f"split={TRAIN_FRAC}/{VAL_FRAC}/{TEST_FRAC}, buffer={BUFFER_SEC}s"
    )

    users = args.users if args.users else _discover_users()
    if not users:
        print("No WESAD subjects discovered under DATA/WESAD/hp/. Run convert first.")
        return 1

    rows = []
    for pid in users:
        try:
            r = check_subject(pid)
        except Exception as e:
            print(f"hp{pid}: FAILED — {e}")
            continue
        rows.append(r)
        print(format_row(r))

    if not rows:
        print("No subjects processed successfully.")
        return 1

    df = pd.DataFrame(rows)
    totals = {
        "subjects": len(df),
        "train_kept": int(df["train_kept"].sum()),
        "val_kept": int(df["val_kept"].sum()),
        "test_kept": int(df["test_kept"].sum()),
        "train_pos": int(df["train_pos"].sum()),
        "train_neg": int(df["train_neg"].sum()),
        "test_pos": int(df["test_pos"].sum()),
        "test_neg": int(df["test_neg"].sum()),
        "mean_train_drop": float(df["train_drop"].mean()),
        "mean_test_drop": float(df["test_drop"].mean()),
    }
    print("\n=== totals ===")
    for k, v in totals.items():
        if isinstance(v, float):
            print(f"  {k:>16}: {v:.3f}")
        else:
            print(f"  {k:>16}: {v}")

    train_imbalance = totals["train_pos"] / max(1, totals["train_pos"] + totals["train_neg"])
    test_imbalance = totals["test_pos"] / max(1, totals["test_pos"] + totals["test_neg"])
    print(f"  train positive rate: {train_imbalance:.3f}")
    print(f"   test positive rate: {test_imbalance:.3f}")

    no_pos_test = df[df["test_pos"] == 0]["user"].tolist()
    no_neg_test = df[df["test_neg"] == 0]["user"].tolist()
    if no_pos_test:
        print(f"  ⚠ subjects with 0 positive test windows: {no_pos_test}")
    if no_neg_test:
        print(f"  ⚠ subjects with 0 negative test windows: {no_neg_test}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nPer-subject stats written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
