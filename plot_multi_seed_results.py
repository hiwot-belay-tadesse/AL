"""
Plot multi-seed AL progress from existing al_progress.csv files.

This plots mean AUC per round across seeds with +/- 1 std shaded bands.
python plot_multi_seed_results.py \
  --outdir multiseeds \
  --user 15 \
  --participant_id 20 \
  --seeds 41,42,43,44,45,46,47,48,49,50

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="cluster_results/multiseeds")
    parser.add_argument("--seeds", default="41,42,43", help="Comma-separated seeds")
    parser.add_argument("--user", default="15")
    parser.add_argument("--participant_id", default=None)
    parser.add_argument("--pool", default="global")
    parser.add_argument("--fruit", default="BP")
    parser.add_argument("--scenario", default="spike")
    parser.add_argument("--task", default="bp")
    parser.add_argument("--methods", default="random,coreset")
    parser.add_argument("--outfile", default="auc_mean_std_across_seeds.png")
    args = parser.parse_args()

    if args.task == "bp":
        args.participant_id = args.participant_id or args.user
        args.user = str(args.participant_id)

    repo_root = Path(__file__).resolve().parent
    base_dir = Path(args.outdir)
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir

    seeds_sorted = sorted(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    aq_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    scenario_dir = base_dir / args.pool / args.user / f"{args.fruit}_{args.scenario}"
    breakpoint()
    method_colors = {
        "uncertainty": "tab:blue",
        "random": "darkorange",
        "coreset": "seagreen",
    }

    rows = []
    for aq in aq_methods:
        for seed in seeds_sorted:
            seed_dir = scenario_dir / aq / f"seed_{seed}"
            al_paths = sorted(seed_dir.rglob("al_progress.csv"))
            if not al_paths:
                print(f"Missing al_progress.csv for {seed_dir}")
                continue

            df = pd.read_csv(al_paths[0])
            x = df["round"] if "round" in df.columns else df.index
            y = df["AUC_Mean"] if "AUC_Mean" in df.columns else df.get("auc_mean")
            if y is None:
                print(f"Missing AUC_Mean column for {seed_dir}")
                continue

            seed_rows = pd.DataFrame(
                {
                    "method": aq,
                    "seed": seed,
                    "round": pd.to_numeric(x, errors="coerce"),
                    "auc": pd.to_numeric(y, errors="coerce"),
                }
            )
            rows.append(seed_rows.dropna(subset=["round", "auc"]))

    if not rows:
        raise SystemExit(f"No al_progress.csv files found under {scenario_dir}")

    all_progress = pd.concat(rows, ignore_index=True)
    summary = (
        all_progress.groupby(["method", "round"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"), n_seeds=("seed", "nunique"))
        .sort_values(["method", "round"])
    )
    summary["auc_std"] = summary["auc_std"].fillna(0.0)

    plt.figure(figsize=(10, 6))
    for aq in aq_methods:
        method_df = summary[summary["method"] == aq]
        if method_df.empty:
            continue

        x_vals = method_df["round"].to_numpy(dtype=float)
        mean_vals = method_df["auc_mean"].to_numpy(dtype=float)
        std_vals = method_df["auc_std"].to_numpy(dtype=float)
        color = method_colors.get(aq)

        plt.plot(
            x_vals,
            mean_vals,
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=f"{aq} mean",
        )
        plt.fill_between(
            x_vals,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.18,
            linewidth=0,
            label=f"{aq} +/- 1 std",
        )

    plt.xlabel("Round")
    plt.ylabel("AUC Mean")
    plt.title("Mean AUC per Round Across Seeds")
    plt.legend(ncol=1, fontsize=8)
    plt.tight_layout()
    out_path = scenario_dir / args.outfile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved mean/std plot to {out_path}")


if __name__ == "__main__":
    main()
