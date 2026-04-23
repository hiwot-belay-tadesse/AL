import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utility
import new_helper
from preprocess import prepare_data


def build_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", default="fruit")
    pa.add_argument("--participant_id", default=None)
    pa.add_argument("--user", default="ID5")
    pa.add_argument("--pool", default="personal")
    pa.add_argument("--fruit", default="Melon")
    pa.add_argument("--scenario", default="Crave")
    pa.add_argument("--unlabeled_frac", type=float, default=0.7)
    pa.add_argument("--dropout_rate", type=float, default=0.2)
    pa.add_argument("--warm_start", type=int, default=0)
    pa.add_argument("--K", type=int, default=50)
    pa.add_argument("--T", type=int, default=30)
    pa.add_argument("--clf_epochs", type=int, default=500)
    pa.add_argument("--clf_patience", type=int, default=20)
    pa.add_argument("--n_runs", type=int, default=5)
    pa.add_argument("--seed_start", type=int, default=42)
    pa.add_argument("--out_csv", default="")
    pa.add_argument("--plot_path", default="")
    return pa.parse_args()


def make_namespace(args):
    # prepare_data/run helpers expect an argparse-like object
    return args


def run_one_seed(args, seed):
    new_helper.reset_seeds(seed)

    args_ns = make_namespace(args)
    top_out = Path(new_helper.set_output_dir(args.pool))
    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"

    prep = prepare_data(
        args=args_ns,
        top_out=top_out,
        shared_enc_root=shared_enc_root,
        shared_cnn_root=shared_cnn_root,
        batch_ssl=32,
        ssl_epochs=100,
        pool=args.pool,
        task=args.task,
    )
    if prep is None:
        return []

    df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives = prep

    budget = new_helper.compute_budget(args.pool, df_tr, df_all_tr, args.unlabeled_frac, args.K)

    fit_kwargs = dict(epochs=args.clf_epochs, batch_size=32, verbose=0)
    if args.pool in ["personal", "global"]:
        Z_val_hr, Z_val_st = utility.encode(df_val, enc_hr, enc_st)
        Z_val = np.concatenate([Z_val_hr, Z_val_st], axis=1).astype("float32")
        y_val = df_val["state_val"].values.astype("float32")
    elif args.pool == "global_supervised":
        Z_val, y_val = new_helper.build_XY_from_windows(df_val)
    else:
        Z_val, y_val = None, None

    if Z_val is not None and len(Z_val) > 0:
        fit_kwargs["validation_data"] = (Z_val, y_val)

    if args.pool == "personal":
        split_source = df_tr.reset_index(drop=True)
    else:
        split_source = df_all_tr.reset_index(drop=True)

    # Same initial split for both methods in this seed.
    new_helper.reset_seeds(seed)
    df_init_labeled, df_init_unlabeled = utility.make_labeled_unlabeled_with_target_quota(
        split_source,
        target_uid=args.user,
        unlabeled_frac=args.unlabeled_frac,
        seed=seed,
    )

    rows = []
    for method in ["random", "uncertainty"]:
        new_helper.reset_seeds(seed)
        al_progress, active_model, df_lab_final, df_unlab_final, queried_all, count_df = new_helper.run_al_refactored(
            Aq=method,
            df_tr_labeled=df_init_labeled.copy(),
            df_tr_unlabeled=df_init_unlabeled.copy(),
            df_val=df_val,
            df_te=df_te,
            enc_hr=enc_hr,
            enc_st=enc_st,
            build_classifier=new_helper.build_classifier,
            mc_predict=new_helper.mc_predict,
            K=args.K,
            budget=budget,
            T=args.T if method != "random" else None,
            CLF_PATIENCE=args.clf_patience,
            dropout_rate=float(args.dropout_rate),
            fit_kwargs=fit_kwargs,
            pool=args.pool,
            models_d=models_d,
            results_d=results_d,
            active_model=None,
            init_weights_trained=None,
            warm_start=bool(int(args.warm_start)),
            seed=seed,
        )

        if al_progress.empty:
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "round": np.nan,
                    "auc": np.nan,
                    "final_labeled": int(len(df_lab_final)),
                    "final_unlabeled": int(len(df_unlab_final)),
                    "budget": int(budget),
                }
            )
        else:
            for _, r in al_progress[["round", "AUC_Mean"]].iterrows():
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "round": int(r["round"]),
                        "auc": float(r["AUC_Mean"]),
                        "final_labeled": int(len(df_lab_final)),
                        "final_unlabeled": int(len(df_unlab_final)),
                        "budget": int(budget),
                    }
                )

    return rows


def main():
    args = build_args()

    if args.task == "bp":
        if args.participant_id is None:
            raise SystemExit("For --task bp, provide --participant_id")
        args.user = str(args.participant_id)
        args.fruit = "BP"
        args.scenario = "spike"

    all_rows = []
    for i in range(args.n_runs):
        seed = args.seed_start + i
        rows = run_one_seed(args, seed)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No results produced.")
        return

    print("\nPer-run results:")
    print(df)

    final_df = df.sort_values(["seed", "method", "round"]).groupby(["seed", "method"], as_index=False).tail(1)
    summary = (
        final_df.groupby("method", as_index=False)
        .agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"), n=("auc", "count"))
    )
    print("\nSummary (mean/std across independent runs):")
    print(summary)

    round_summary = (
        df.dropna(subset=["round", "auc"])
        .groupby(["method", "round"], as_index=False)
        .agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"))
    )

    plt.figure(figsize=(8, 5))
    for method in ["uncertainty", "random"]:
        sub = round_summary[round_summary["method"] == method].sort_values("round")
        if sub.empty:
            continue
        plt.errorbar(
            sub["round"].to_numpy(),
            sub["mean_auc"].to_numpy(),
            yerr=sub["std_auc"].fillna(0.0).to_numpy(),
            marker="o",
            capsize=4,
            label=method,
        )
    plt.xlabel("Active Learning Round")
    plt.ylabel("AUC (Bootstrap Mean)")
    plt.title("Mean ± Std AUC Across Independent Runs")
    plt.legend()
    plt.tight_layout()

    if args.plot_path:
        plot_out = Path(args.plot_path)
        plt.savefig(plot_out, dpi=200)
        print(f"Saved plot to {plot_out}")
    else:
        plt.show()

    if args.out_csv:
        out = Path(args.out_csv)
        df.to_csv(out, index=False)
        print(f"Saved per-run results to {out}")


if __name__ == "__main__":
    main()
