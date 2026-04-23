#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

PIPELINES = ["global_supervised", "global_ssl", "personal_ssl"]


def normalize_task(task_name: str) -> str:
    if task_name.lower() == "bp_spike":
        return "BP_spike"
    return task_name


def participant_sort_key(pid: str):
    try:
        return (0, int(pid))
    except Exception:
        return (1, pid)


def find_latest_summaries(root: Path):
    summaries = {}
    for path in root.rglob("comparison_summary.csv"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 3:
            continue
        participant = parts[0]
        task = normalize_task(parts[1])
        key = (participant, task)
        prev = summaries.get(key)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            summaries[key] = path
    return summaries


def build_table(root: Path):
    summaries = find_latest_summaries(root)
    table = {}
    for (participant, _task), csv_path in summaries.items():
        df = pd.read_csv(csv_path)
        if "Pipeline" not in df.columns:
            continue
        row = table.setdefault(participant, {p: "" for p in PIPELINES})
        for _, r in df.iterrows():
            pipeline = str(r.get("Pipeline", "")).strip()
            if pipeline not in PIPELINES:
                continue
            auc_mean = r.get("AUC_Mean")
            auc_std = r.get("AUC_STD")
            try:
                cell = f"{float(auc_mean):.4f} ± {float(auc_std):.4f}"
            except Exception:
                cell = ""
            row[pipeline] = cell
    rows = []
    for pid in sorted(table.keys(), key=participant_sort_key):
        row = {"participant": pid}
        row.update(table[pid])
        rows.append(row)
    return pd.DataFrame(rows, columns=["participant"] + PIPELINES)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--root", default="ALL_results", help="Results root to scan.")
    pa.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to <root>/merge_comparision.csv.",
    )
    pa.add_argument("--preview", action="store_true", help="Print a preview.")
    args = pa.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    out_path = Path(args.out) if args.out else root / "merge_comparision.csv"

    df = build_table(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if args.preview:
        print(df.head(10).to_string(index=False))
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
