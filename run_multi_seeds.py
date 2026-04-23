'''
Run multiple seeds for all methods (uncertainty, random, coreset) 
'''

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import run as run_module
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import preprocess
from sklearn.model_selection import train_test_split

run_module.OUTPUT_DIR = "/Users/hiwotbelaytadesse/Desktop/Banaware_AL/multi_seed_results"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--seeds", default="41,42,43", help="Comma-separated seeds")
    pa.add_argument("--results_subdir", default="results")
    pa.add_argument("--user", default="35")
    pa.add_argument("--pool", default="global")
    pa.add_argument("--fruit", default="BP") 
    pa.add_argument("--scenario", default="spike")
    pa.add_argument("--unlabeled_frac", default="0.00018")
    pa.add_argument("--dropout_rate", default="0.5")
    pa.add_argument("--warm_start", default="0")
    pa.add_argument("--task", default="bp")
    pa.add_argument("--input_df", default="raw")
    pa.add_argument("--analyze_round0", action="store_true")
    args = pa.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No seeds provided.")

    aq_methods = [
        # "uncertainty",
        "random",
        "coreset",
    ]
    
    colors = {"uncertainty": "tab:blue", "random": "darkorange", "coreset": "seagreen"}
    base_dir = Path(run_module.OUTPUT_DIR)

    scenario_dir = base_dir / args.pool / args.user / f"{args.fruit}_{args.scenario}"
    seed_dirs_by_aq = {aq: [] for aq in aq_methods}
    seeds_sorted = sorted(seeds)
    shade_vals = np.linspace(0.35, 0.95, max(len(seeds_sorted), 1))
    seed_colors = {
        "uncertainty": {s: plt.cm.Blues(v) for s, v in zip(seeds_sorted, shade_vals)},
        "random": {s: plt.cm.Reds(v) for s, v in zip(seeds_sorted, shade_vals)},
        "coreset": {s: plt.cm.Greens(v) for s, v in zip(seeds_sorted, shade_vals)},
    }

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        if hasattr(run_module, "reset_seeds"):
            run_module.reset_seeds(seed)
        for exp_name in aq_methods:
            exp_kwargs = {
                "user": args.user,
                "pool": args.pool,
                "fruit": args.fruit,
                "scenario": args.scenario,
                "unlabeled_frac": float(args.unlabeled_frac),
                "dropout_rate": float(args.dropout_rate),
                "warm_start": int(args.warm_start),
                "seed": seed,
                "aq": exp_name,
                "K": run_module.K[0],
                "Budget": run_module.Budget[0],
                "T": run_module.T[0],
                "input_df": args.input_df,
                "task": args.task,
            }

            exp_dir = scenario_dir / exp_name
            seed_dir = exp_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            run_module.run(str(seed_dir), exp_name, exp_kwargs)
            seed_dirs_by_aq[exp_name].append(seed_dir)

    # Single plot: one line per seed for both methods (no averaging)
    plt.figure(figsize=(10, 6))
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
            prefix_map = {"uncertainty": "U", "random": "R", "coreset": "C"}
            prefix = prefix_map.get(aq, aq[:1].upper())
            plt.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=1.5,
                alpha=0.9,
                color=seed_colors[aq][seed],
                label=f"{prefix}-{seed}",
            )

    plt.xlabel("Round")
    plt.ylabel("AUC Mean")
    plt.title("AUC per Round by Seed (Uncertainty vs Random vs Coreset)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = scenario_dir / "auc_one_line_per_seed_both.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined per-seed plot to {out_path}")

    if args.analyze_round0:
        args_ns = SimpleNamespace(
            user=args.user,
            pool=args.pool,
            fruit=args.fruit,
            scenario=args.scenario,
            unlabeled_frac=float(args.unlabeled_frac),
            dropout_rate=float(args.dropout_rate),
            warm_start=int(args.warm_start),
            results_subdir=args.results_subdir,
        )
        top_out = Path(run_module.OUTPUT_DIR)
        prep = preprocess.prepare_data(
            args=args_ns,
            top_out=top_out,
            shared_enc_root=top_out / "_global_encoders",
            shared_cnn_root=top_out / "global_cnns",
            batch_ssl=32,
            ssl_epochs=100,
            pool=args.pool,
        )
        if prep is not None:
            df_tr, df_all_tr, *_ = prep
            split_source = df_tr.reset_index(drop=True) if args.pool == "personal" else df_all_tr.reset_index(drop=True)
            rows = []
            for seed in seeds_sorted:
                lab, _ = train_test_split(
                    split_source,
                    test_size=float(1 -(args.unlabeled_frac)),
                    stratify=split_source["state_val"],
                    random_state=seed,
                )
                al_paths = sorted((scenario_dir / "uncertainty" / f"seed_{seed}").rglob("al_progress.csv"))
                round0_auc = np.nan
                if al_paths:
                    al = pd.read_csv(al_paths[0])
                    r0 = al[al["round"] == 0]
                    if not r0.empty and "AUC_Mean" in r0.columns:
                        round0_auc = float(r0["AUC_Mean"].iloc[0])
                rows.append(
                    {
                        "seed": seed,
                        "round0_auc_uncertainty": round0_auc,
                        "n_labeled": int(len(lab)),
                        "pos_labeled": int((lab["state_val"] == 1).sum()),
                        "neg_labeled": int((lab["state_val"] == 0).sum()),
                        "pos_rate": float((lab["state_val"] == 1).mean()),
                    }
                )
            if rows:
                out_df = pd.DataFrame(rows).sort_values("round0_auc_uncertainty", ascending=False)
                out_csv = scenario_dir / "seed_round0_split_analysis.csv"
                out_df.to_csv(out_csv, index=False)
                print(f"Saved round0 split analysis to {out_csv}")


if __name__ == "__main__":
    main()
