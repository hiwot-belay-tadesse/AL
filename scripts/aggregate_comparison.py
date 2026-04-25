#!/usr/bin/env python3
'''
aggregate each comparison_summary.csv into a single table with MultiIndex (ID, Substance, Task, Pipeline) and columns:
syntax to run from terminal:
python src/aggregate_comparison.py --root BP_SPIKE_PRED --out user_pipeline_auc_wide.csv

'''
"""
aggregate_comparison.py
=======================

Recursively finds all `comparison_summary.csv` files under a root directory,
extracts ID, Substance, Task, and per-pipeline metrics, then outputs a
combined table with a MultiIndex (ID, Substance, Task, Pipeline) and columns:

  Threshold | Sensitivity | Specificity | Accuracy | ROC AUC | AUC_Train | AUC_Val

Usage:
  python aggregate_comparison.py --root /path/to/results --out combined_results.csv
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # two levels up

import glob
import argparse
import re

import numpy as np
import pandas as pd
from utility import bootstrap_auc


def _pipeline_aggregated_auc(root_abs, pipeline_name, substance=None, task=None):
    if substance and task:
        probs_pattern = os.path.join(
            root_abs, "**", f"{substance}_{task}", pipeline_name, "results", "test_probs_*.npy"
        )
    elif substance:
        probs_pattern = os.path.join(
            root_abs, "**", f"{substance}_*", pipeline_name, "results", "test_probs_*.npy"
        )
    else:
        probs_pattern = os.path.join(root_abs, "**", pipeline_name, "results", "test_probs_*.npy")
    probs_paths = sorted(glob.glob(probs_pattern, recursive=True))
    y_all, p_all = [], []
    for probs_path in probs_paths:
        probs_p = Path(probs_path)
        uid_part = probs_p.stem.replace("test_probs_", "", 1)
        labels_p = probs_p.with_name(f"test_labels_{uid_part}.npy")

        if not labels_p.exists():
            continue
        try:
            probs = np.load(probs_p).ravel()
            y = np.load(labels_p).ravel()
        except Exception:
            continue
        if len(probs) == 0 or len(y) == 0 or len(probs) != len(y):
            continue
        y_all.append(y)
        p_all.append(probs)

    if not y_all:
        return ""
    y_cat = np.concatenate(y_all)
    p_cat = np.concatenate(p_all)
    auc_m, auc_s, _ = bootstrap_auc(y_cat, p_cat)
    if np.isnan(auc_m) or np.isnan(auc_s):
        return ""
    return f"{auc_m:.3f} ± {auc_s:.3f}"



def parse_args():
    p = argparse.ArgumentParser(description="Aggregate comparison_summary.csv files with MultiIndex output")
    p.add_argument("--root", required=True,
                   help="Root directory containing comparison_summary.csv files")
    p.add_argument("--out", default="combined_results.csv",
                   help="Filename for aggregated output CSV")
    p.add_argument(
        "--mode",
        choices=["old", "new", "grouped"], ### note old is the original comparison from banaware, new is for the al quick comparision of train and test aucs
        default="old",
        help="Aggregation mode: 'old' = legacy MultiIndex table, 'new' = per-participant wide AUC table, 'grouped' = grouped by Substance/Task/Pipeline."
    )
    return p.parse_args()


def build_user_auc_wide_df(root_dir):
    """
    Build a wide DataFrame with one row per uid and two AUC columns per pipeline:
      - <pipeline>_AUC_Train
      - <pipeline>_AUC_Mean±STD

    The function scans all comparison_summary.csv files under root_dir.
    It appends a final row (uid=AGGREGATED) with pooled test AUC by pipeline,
    stored under each pipeline's existing <pipeline>_AUC_Train column.
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    pattern = os.path.join(root_dir, "**", "comparison_summary.csv")
    matches = sorted(glob.glob(pattern, recursive=True))

    rows_by_uid = {}
    pipeline_keys = set()
    for csv_path in matches:
        p = Path(csv_path)
        # expected: .../<uid>/<fruit_task>/comparison_summary.csv
        try:
            uid = p.parent.parent.name
        except Exception:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Pipeline" not in df.columns:
            continue

        row = rows_by_uid.setdefault(str(uid), {"uid": str(uid)})

        for _, r in df.iterrows():
            pipe = str(r.get("Pipeline", "")).strip()
            if not pipe:
                continue
            key = pipe.replace(" ", "_")
            pipeline_keys.add(key)

            train_auc = r.get("AUC_Train", "")
            if pd.notna(train_auc) and str(train_auc) != "":
                row[f"{key}_AUC_Train"] = train_auc

            # Prefer already formatted current column, then fallback to legacy numeric columns.
            if "AUC_Mean ± STD" in df.columns and pd.notna(r.get("AUC_Mean ± STD")):
                test_auc = str(r.get("AUC_Mean ± STD"))
            elif "AUC_Mean" in df.columns and "AUC_STD" in df.columns:
                try:
                    test_auc = f"{float(r.get('AUC_Mean')):.3f} ± {float(r.get('AUC_STD')):.3f}"
                except Exception:
                    test_auc = ""
            elif "ROC AUC" in df.columns and pd.notna(r.get("ROC AUC")):
                test_auc = str(r.get("ROC AUC"))
            else:
                test_auc = ""

            if test_auc:
                row[f"{key}_AUC_Mean±STD"] = test_auc

    out = pd.DataFrame(rows_by_uid.values())
    if out.empty:
        return out

    def _uid_sort_key(v):
        s = str(v)
        num = "".join(ch for ch in s if ch.isdigit())
        return int(num) if num else 10**9

    out = out.sort_values("uid", key=lambda c: c.map(_uid_sort_key)).reset_index(drop=True)

    # Append one final row with pooled test AUC (across users) per pipeline.
    # Store it under the existing <pipeline>_AUC_Train columns.
    agg_row = {"uid": "AGGREGATED"}
    for key in sorted(pipeline_keys):
        agg_row[f"{key}_AUC_Mean±STD"] = _pipeline_aggregated_auc(root_dir, key)
    out = pd.concat([out, pd.DataFrame([agg_row])], ignore_index=True)
    return out


def _legacy_aggregate(root_dir):
    pattern = os.path.join(root_dir, "**", "comparison_summary.csv")
    matches = glob.glob(pattern, recursive=True)
    print(f"Found {len(matches)} comparison_summary.csv files")

    records = []
    for csv_path in matches:
        parts = csv_path.split(os.sep)
        if len(parts) < 3:
            continue
        id_ = parts[-3]
        fruit_task = parts[-2]
        if "_" in fruit_task:
            substance, task = fruit_task.split("_", 1)
        else:
            substance, task = fruit_task, ""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping unreadable {csv_path}: {e}")
            continue
        for _, row in df.iterrows():
            records.append({
                "ID": id_,
                "Substance": substance,
                "Task": task,
                "Pipeline": row.get("Pipeline", ""),
                "Threshold": row.get("Best_Threshold", ""),
                "Sensitivity": row.get("Sensitivity_Mean ± STD", ""),
                "Specificity": row.get("Specificity_Mean ± STD", ""),
                "Accuracy": row.get("Accuracy (Mean ± STD)", ""),
                "ROC AUC": row.get("AUC_Mean ± STD", row.get("ROC AUC", "")),
                "AUC_Train": row.get("AUC_Train", ""),
                "AUC_Val": row.get("AUC_Val", ""),
            })

    if not records:
        return pd.DataFrame()

    combined = pd.DataFrame.from_records(records)
    combined = combined[["ID", "Substance", "Task", "Pipeline",
                         "Threshold", "Sensitivity", "Specificity",
                         "Accuracy", "ROC AUC", "AUC_Train", "AUC_Val"]]
    combined.set_index(["ID", "Substance", "Task", "Pipeline"], inplace=True)
    combined.sort_index(level=["ID", "Pipeline"], inplace=True)
    return combined


def _build_substance_user_tables(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    pattern = os.path.join(root_dir, "**", "comparison_summary.csv")
    matches = sorted(glob.glob(pattern, recursive=True))

    rows_by_group = {}
    pipes_by_group = {}

    for csv_path in matches:
        p = Path(csv_path)
        try:
            uid = str(p.parent.parent.name)
            fruit_task = str(p.parent.name)
        except Exception:
            continue

        if "_" in fruit_task:
            substance, task = fruit_task.split("_", 1)
        else:
            substance = fruit_task
            task = ""
        group_name = f"{substance}_{task}" if task else substance

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "Pipeline" not in df.columns:
            continue

        sub_rows = rows_by_group.setdefault(group_name, {})
        sub_pipes = pipes_by_group.setdefault(group_name, set())
        row = sub_rows.setdefault(uid, {"uid": uid})

        for _, r in df.iterrows():
            pipe = str(r.get("Pipeline", "")).strip()
            if not pipe:
                continue
            key = pipe.replace(" ", "_")
            sub_pipes.add(key)

            train_auc = r.get("AUC_Train", "")
            mean_std = r.get("AUC_Mean ± STD", r.get("ROC AUC", ""))

            if pd.notna(train_auc) and str(train_auc) != "":
                cur = row.get(f"{key}_AUC_Train", "")
                nxt = str(train_auc)
                if cur and cur != nxt:
                    row[f"{key}_AUC_Train"] = cur
                else:
                    row[f"{key}_AUC_Train"] = nxt

            if pd.notna(mean_std) and str(mean_std) != "":
                cur = row.get(f"{key}_AUC_Mean±STD", "")
                nxt = str(mean_std)
                if cur and cur != nxt:
                    row[f"{key}_AUC_Mean±STD"] = cur
                else:
                    row[f"{key}_AUC_Mean±STD"] = nxt

    tables = {}
    for group_name, uid_map in rows_by_group.items():
        sub_df = pd.DataFrame(uid_map.values())
        if sub_df.empty:
            continue

        def _uid_sort_key(v):
            s = str(v)
            num = "".join(ch for ch in s if ch.isdigit())
            return int(num) if num else 10**9

        sub_df = sub_df.sort_values("uid", key=lambda c: c.map(_uid_sort_key)).reset_index(drop=True)
        keys = sorted(pipes_by_group.get(group_name, []))

        if "_" in group_name:
            substance, task = group_name.split("_", 1)
        else:
            substance, task = group_name, ""

        agg_row = {"uid": "AGGREGATED"}
        for key in keys:
            agg_row[f"{key}_AUC_Mean±STD"] = _pipeline_aggregated_auc(
                root_dir, key, substance=substance, task=task
            )
        sub_df = pd.concat([sub_df, pd.DataFrame([agg_row])], ignore_index=True)

        ordered_cols = ["uid"]
        for key in keys:
            ordered_cols.append(f"{key}_AUC_Train")
            ordered_cols.append(f"{key}_AUC_Mean±STD")
        for col in ordered_cols:
            if col not in sub_df.columns:
                sub_df[col] = ""
        sub_df = sub_df[ordered_cols]
        tables[group_name] = sub_df

    return tables


def main(root_dir, out_file, mode="old"):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    print(f"Searching for comparison_summary.csv under '{root_dir}'...")
    if mode == "new":
        wide = build_user_auc_wide_df(root_dir)
        if wide.empty:
            print(f"No comparison_summary.csv files found under '{root_dir}'")
            return
        wide.to_csv(out_file, index=False)
        print(f"Aggregated {len(wide)} participants to {out_file}")
        print("\nPer-participant AUC summary:")
        print(wide.to_string(index=False))
        return

    if mode == "grouped":
        tables = _build_substance_user_tables(root_dir)
        if not tables:
            print(f"No comparison_summary.csv files found under '{root_dir}'")
            return

        out_path = Path(out_file)
        stem = out_path.stem
        suffix = out_path.suffix if out_path.suffix else ".csv"
        parent = out_path.parent if str(out_path.parent) != "" else Path(".")
        parent.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        for substance, sub_df in sorted(tables.items()):
            safe_substance = re.sub(r"[^A-Za-z0-9_-]+", "_", substance).strip("_") or "Unknown"
            sub_out = parent / f"{stem}_{safe_substance}{suffix}"
            sub_df.to_csv(sub_out, index=False)
            total_rows += len(sub_df)
            print(f"\n{substance} group -> {sub_out}")
            print(sub_df.to_markdown(index=False))

        print(f"\nWrote {len(tables)} grouped tables ({total_rows} rows total).")
        return

    combined = _legacy_aggregate(root_dir)
    if combined.empty:
        print(f"No comparison_summary.csv files found under '{root_dir}'")
        return
    combined.to_csv(out_file)
    print(f"Aggregated {len(combined)} rows to {out_file}")
    print("\nCombined results (MultiIndex):")
    print(combined.to_markdown())


if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.out, args.mode)
