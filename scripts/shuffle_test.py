import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import preprocess
import utility
import new_helper
from preprocess import prepare_data


def reset_all(seed: int):
    # Clear TF graph/session state between trials, then reseed all RNGs.
    tf.keras.backend.clear_session()
    tf.random.set_global_generator(tf.random.Generator.from_seed(seed))

    new_helper.reset_seeds(seed)


def main():
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
    pa.add_argument("--n_runs", type=int, default=1)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--vary_seed", action="store_true")
    args = pa.parse_args()

    if args.task == "bp":
        if args.participant_id is None:
            raise SystemExit("For --task bp, please provide --participant_id.")
        args.user = str(args.participant_id)
        args.fruit = "BP"
        args.scenario = "spike"

    output_dir = new_helper.set_output_dir(args.pool)
    top_out = Path(output_dir)
    shared_enc_root = top_out / "_global_encoders"
    shared_cnn_root = top_out / "global_cnns"

    reset_all(args.seed)
    prep = prepare_data(
        args=args,
        top_out=top_out,
        shared_enc_root=shared_enc_root,
        shared_cnn_root=shared_cnn_root,
        batch_ssl=32,
        ssl_epochs=100,
        pool=args.pool,
        task=args.task,
    )

    if prep is None:
        print("prepare_data returned None.")
        return

    df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, *_ = prep

    # Compute budget
    budget = new_helper.compute_budget(args.pool, df_tr, df_all_tr, args.unlabeled_frac, args.K)

    fit_kwargs = dict(epochs=500, batch_size=32, verbose=0)
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

    rows = []
    methods = ["random", "uncertainty"]

    for method in methods:
        for i in range(args.n_runs):
            seed = args.seed + i if args.vary_seed else args.seed
            reset_all(seed)

            if args.pool == "personal":
                split_source = df_tr.reset_index(drop=True)
            else:
                split_source = df_all_tr.reset_index(drop=True)

            # reset_all(seed)
            df_tr_labeled, df_tr_unlabeled = utility.make_labeled_unlabeled_with_target_quota(
                split_source, target_uid=args.user, unlabeled_frac=args.unlabeled_frac, seed=seed
            )

            # reset_all(seed)
            al_progress, active_model, *_ = new_helper.run_al_refactored(
                method,
                df_tr_labeled.copy(),
                df_tr_unlabeled.copy(),
                df_val,
                df_te,
                enc_hr,
                enc_st,
                new_helper.build_classifier,
                new_helper.mc_predict,
                args.K,
                budget,
                T=args.T if method != "random" else None,
                CLF_PATIENCE=20,
                dropout_rate=float(args.dropout_rate),
                fit_kwargs=fit_kwargs,
                pool=args.pool,
                models_d={},
                results_d={},
                active_model=None,
                init_weights_trained=None,
                warm_start=bool(int(args.warm_start)),
                seed=seed,
            )

            if al_progress.empty:
                print(f"{method} run {i+1}: no progress")
                auc = float("nan")
            else:
                auc = float(al_progress.iloc[-1]["AUC_Mean"])
                print(f"{method} run {i+1} (seed={seed}): AUC_Mean={auc:.6f}")

            rows.append({"method": method, "run": i + 1, "auc": auc, "seed": seed})

    df_multi_runs = pd.DataFrame(rows)
    df_wide = df_multi_runs.pivot(index="run", columns="method", values="auc").reset_index()

    print(df_multi_runs)
    for method in methods:

        vals = df_wide[method].to_numpy()
        if np.isfinite(vals).all():
            print(
                f"{method} mean={vals.mean():.6f}, std={vals.std(ddof=1) if len(vals) > 1 else 0.0:.6f}"
            )
    out_path = "multi_run_auc.csv"
    df_wide.to_csv(out_path, index=False)
    print(f"Saved {out_path}")



if __name__ == "__main__":
    main()
