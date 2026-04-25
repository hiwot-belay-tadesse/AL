import os
import re
from pathlib import Path
from types import SimpleNamespace

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from new_helper import (
    bootstrap_auc,
    build_XY_from_processed,
    parse_args,
    reset_seeds,
    set_output_dir,
)
from new_prep import prepare_data
from src import compare_pipelines as cp


def mlp_model_builder(input_shape, dropout_rate: float):
    return Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ])


def discover_bp_users() -> list[str]:
    bp_root = Path("DATA/Cardiomate")
    candidate_dirs = (
        sorted((bp_root / "hp").glob("hp*"))
        if (bp_root / "hp").exists()
        else sorted(bp_root.glob("hp*"))
    )

    users = []
    for folder in candidate_dirs:
        match = re.search(r"\d+", folder.name)
        if not match:
            continue
        pid = match.group(0)
        if not (folder / f"hp{pid}_hr.csv").exists():
            continue
        if not (folder / f"hp{pid}_steps.csv").exists():
            continue
        if not (folder / f"blood_pressure_readings_ID{pid}_cleaned.csv").exists():
            continue
        users.append(pid)
    return users


def parse_auc_mean_std(auc_str: str) -> tuple[float, float]:
    parts = str(auc_str).split("±")
    mean = float(parts[0].strip())
    std = float(parts[1].strip()) if len(parts) > 1 else np.nan
    return mean, std


def plot_seed_comparison(df_seed_results: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt

    tmp = df_seed_results.copy()
    tmp["auc_mean_num"] = pd.to_numeric(
        tmp["test_auc_mean ± std"].astype(str).str.split("±").str[0].str.strip(),
        errors="coerce",
    )

    seed_df = tmp[tmp["seed"] != "full_data"].copy()
    if seed_df.empty:
        return

    seed_df["seed"] = seed_df["seed"].astype(str)
    auc_100_vals = tmp.loc[tmp["seed"] == "full_data", "auc_mean_num"].dropna()
    auc_100 = float(auc_100_vals.iloc[0]) if not auc_100_vals.empty else np.nan

    x = np.arange(len(seed_df))
    plt.figure(figsize=(7, 4))
    plt.scatter(x, seed_df["auc_mean_num"], s=70, color="tab:blue")

    for i, row in seed_df.reset_index(drop=True).iterrows():
        plt.text(i, row["auc_mean_num"] + 0.002, f"seed {row['seed']}", ha="center", fontsize=9)

    if not np.isnan(auc_100):
        plt.axhline(auc_100, color="red", linestyle="--", label=f"100% AUC = {auc_100:.3f}")
        plt.legend()

    plt.xticks(x, [f"seed {s}" for s in seed_df["seed"]])
    plt.ylabel("Test AUC mean")
    plt.title("Per-seed 10% AUC vs 100% AUC")
    plt.tight_layout()
    plt.savefig(out_dir / "seed_comparison.png")
    plt.close()


def run_for_user(
    user_id: str,
    args,
    top_out: Path,
    shared_enc_root: Path,
    shared_cnn_root: Path,
    results_subdir: str,
    input_df: str,
    seeds: list[int],
) -> tuple[pd.DataFrame | None, dict | None]:
    args_ns = SimpleNamespace(
        user=str(user_id),
        pool=args.pool,
        fruit=args.fruit,
        scenario=args.scenario,
        task=args.task,
        participant_id=getattr(args, "participant_id", None),
        unlabeled_frac=args.unlabeled_frac,
        dropout_rate=args.dropout_rate,
        warm_start=args.warm_start,
        results_subdir=results_subdir,
        input_df=input_df,
    )

    try:
        prep = prepare_data(
            args=args_ns,
            top_out=top_out,
            shared_enc_root=shared_enc_root,
            shared_cnn_root=shared_cnn_root,
            batch_ssl=32,
            ssl_epochs=100,
            pool=args_ns.pool,
            task=args_ns.task,
            input_df=args_ns.input_df,
        )
    except (SystemExit, FileNotFoundError, ValueError) as exc:
        print(f"Skipping user {user_id}: {exc}")
        return None, None

    (
        _df_tr,
        df_all_tr,
        df_val,
        df_te,
        _enc_hr,
        _enc_st,
        _user_root,
        _all_splits,
        _models_d,
        _results_d,
        _all_negatives,
    ) = prep

    out_dir = top_out / results_subdir / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in seeds:
        try:
            strat_key = df_all_tr["user_id"].astype(str) + "_" + df_all_tr["state_val"].astype(str)
            df_labeled, _ = train_test_split(
                df_all_tr,
                test_size=0.9,
                stratify=strat_key,
                random_state=seed,
            )
        except ValueError as exc:
            print(f"User {user_id} seed {seed} skipped: {exc}")
            continue

        X_seed, y_seed, feature_cols, train_median, scaler = build_XY_from_processed(df_labeled, fit=True)
        X_val_s, y_val_s, *_ = build_XY_from_processed(
            df_val,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,
        )
        X_te_s, y_te_s, *_ = build_XY_from_processed(
            df_te,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,
        )

        reset_seeds(seed)
        model = mlp_model_builder(input_shape=(X_seed.shape[1],), dropout_rate=args_ns.dropout_rate)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        es = EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=0,
        )
        lr_cb = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        )

        classes = np.unique(y_seed)
        if len(classes) == 2:
            cw_vals = compute_class_weight("balanced", classes=classes, y=y_seed)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
        else:
            class_weight = {0: 1.0, 1: 1.0}

        model.fit(
            X_seed,
            y_seed,
            validation_data=(X_val_s, y_val_s),
            epochs=200,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[es, lr_cb],
            verbose=0,
            shuffle=True,
        )

        probs_te = model.predict(X_te_s, verbose=0).ravel()
        test_auc_mean_seed, test_auc_std_seed, _ = bootstrap_auc(y_te_s, probs_te)
        run_tag = f"frac10pct_seed{seed}"
        np.save(out_dir / f"test_probs_{user_id}_{run_tag}.npy", probs_te)
        np.save(out_dir / f"test_labels_{user_id}_{run_tag}.npy", y_te_s)

        seed_results.append(
            {
                "user": str(user_id),
                "run": "10%",
                "seed": seed,
                "n_train": len(df_labeled),
                "test_auc_mean ± std": f"{test_auc_mean_seed:.4f} ± {test_auc_std_seed:.4f}",
            }
        )

    if not seed_results:
        print(f"Skipping user {user_id}: no valid seed runs")
        return None, None

    df_seed_results = pd.DataFrame(seed_results)

    X_tr, y_tr, feature_cols, train_median, scaler = build_XY_from_processed(df_all_tr, fit=True)
    X_val, y_val, *_ = build_XY_from_processed(
        df_val,
        feature_cols=feature_cols,
        train_median=train_median,
        scaler=scaler,
        fit=False,
    )
    X_te, y_te, *_ = build_XY_from_processed(
        df_te,
        feature_cols=feature_cols,
        train_median=train_median,
        scaler=scaler,
        fit=False,
    )

    full_data_model = mlp_model_builder(input_shape=(X_tr.shape[1],), dropout_rate=args_ns.dropout_rate)
    full_data_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=1,
    )
    lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    classes = np.unique(y_tr)
    if len(classes) == 2:
        cw_vals = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
    else:
        class_weight = {0: 1.0, 1: 1.0}

    hist = full_data_model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[es, lr_cb],
        class_weight=class_weight,
        verbose=1,
        shuffle=True,
    )

    cp.plot_clf_losses(hist.history["loss"], hist.history["val_loss"], out_dir, "clf_loss_full_data")

    probs = full_data_model.predict(X_te, verbose=0).ravel()
    auc = roc_auc_score(y_te, probs)
    val_auc = roc_auc_score(y_val, full_data_model.predict(X_val, verbose=0).ravel())
    print(f"User {user_id} Test AUC: {auc:.4f}")
    print(f"User {user_id} Validation AUC on 100% Data: {val_auc:.4f}")

    auc_mean, auc_std, _ = bootstrap_auc(y_te, probs)
    full_row = pd.DataFrame(
        [
            {
                "user": str(user_id),
                "run": "100%",
                "seed": "full_data",
                "n_train": len(df_all_tr),
                "test_auc_mean ± std": f"{auc_mean:.4f} ± {auc_std:.4f}",
            }
        ]
    )
    df_seed_results = pd.concat([df_seed_results, full_row], ignore_index=True)

    out_file = out_dir / "LR_results.csv"
    df_seed_results.to_csv(out_file, index=False)
    print(f"Saved results for user {user_id} to: {out_file}")

    plot_seed_comparison(df_seed_results, out_dir)

    y_true_path = out_dir / f"test_labels_{user_id}_full_data.npy"
    y_pred_path = out_dir / f"test_probs_{user_id}_full_data.npy"
    np.save(y_true_path, y_te)
    np.save(y_pred_path, probs)

    full_payload = {
        "user": str(user_id),
        "full_auc": float(auc_mean),
        "full_auc_std": float(auc_std),
        "y_true_path": str(y_true_path),
        "y_pred_path": str(y_pred_path),
    }
    return df_seed_results, full_payload


def plot_aggregate_auc(df_full_results: pd.DataFrame, top_out: Path, results_subdir: str):
    import matplotlib.pyplot as plt

    if df_full_results.empty:
        return

    df = df_full_results.copy()
    df = df.dropna(subset=["full_auc"]).sort_values("user")
    if df.empty:
        return

    out_root = top_out / results_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    # OLD aggregate plotting (mean of per-user means) is intentionally disabled.
    # Kept requested behavior: pooled AUC from concatenated predictions.
    all_true = np.concatenate(df["y_true"].values)
    all_pred = np.concatenate(df["y_pred"].values)
    pooled_auc, _, _ = bootstrap_auc(all_true, all_pred)

    x = np.arange(len(df))
    plt.figure(figsize=(8, 4))
    plt.scatter(x, df["full_auc"].values, s=70, color="tab:purple")
    for i, row in df.reset_index(drop=True).iterrows():
        plt.text(i, row["full_auc"] + 0.002, f"user {row['user']}", ha="center", fontsize=8)

    plt.axhline(pooled_auc, color="black", linestyle="--", label=f"Pooled AUC = {pooled_auc:.3f}")
    plt.xticks(x, [str(u) for u in df["user"]], rotation=45, ha="right")
    plt.ylabel("Full-data test AUC")
    plt.xlabel("User")
    plt.title("Aggregate Full-data AUC Across Users")
    plt.legend()
    plt.tight_layout()
    out_plot = out_root / "aggregate_full_data_auc.png"
    plt.savefig(out_plot)
    plt.close()

    print(f"Saved: {out_plot} | Pooled AUC = {pooled_auc:.3f}")


def plot_aggregate_10pct(
    df_seed_results: pd.DataFrame,
    seed_to_plot: int,
    run_tag: str,
    out_path: Path,
):
    """
    Plot across users 10% AUC for a single fixed seed.
    """
    import matplotlib.pyplot as plt

    df = df_seed_results.copy()
    df = df[df["seed"] == seed_to_plot].copy()
    if df.empty:
        print(f"No rows found for seed={seed_to_plot}; skipping aggregate 10% plot.")
        return

    df = df.sort_values("user").reset_index(drop=True)
    x = np.arange(len(df))

    plt.figure(figsize=(8, 4))
    plt.scatter(x, df["agg_auc_mean"], s=70)
    for i, row in df.iterrows():
        plt.text(i, row["agg_auc_mean"] + 0.002, f"user {row['user']}", ha="center", fontsize=8)

    all_true_parts = []
    all_pred_parts = []
    for _, row in df.iterrows():
        uid = str(row["user"])
        user_root_i = Path(row["user_root"])
        labels_path = user_root_i / f"test_labels_{uid}_{run_tag}.npy"
        probs_path = user_root_i / f"test_probs_{uid}_{run_tag}.npy"
        if labels_path.exists() and probs_path.exists():
            all_true_parts.append(np.load(labels_path))
            all_pred_parts.append(np.load(probs_path))

    if all_true_parts and all_pred_parts:
        all_true = np.concatenate(all_true_parts)
        all_pred = np.concatenate(all_pred_parts)
        pooled_auc, _, _ = bootstrap_auc(all_true, all_pred)
        plt.axhline(
            pooled_auc,
            color="black",
            linestyle="--",
            label=f"Pooled AUC = {pooled_auc:.3f}",
        )

    plt.xticks(x, [str(u) for u in df["user"]], rotation=45, ha="right")
    plt.ylabel("Test AUC mean")
    plt.xlabel("User")
    plt.title(f"Aggregate 10% AUC Across Users (seed={seed_to_plot})")
    if all_true_parts and all_pred_parts:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args, _ = parse_args()

    args.user = "all"
    args.pool = "global_supervised"
    args.fruit = "BP"
    args.task = "bp"
    args.scenario = "spike"
    args.unlabeled_frac = float(0.1)
    args.dropout_rate = float(0.3)
    args.warm_start = bool(int(0))

    results_subdir = "LR_check"
    input_df = "processed"
    seeds = [42, 123, 456, 789, 1011]

    output_dir = set_output_dir(args.pool)
    top_out = Path(output_dir)
    shared_cnn_root = top_out / "global_cnns"
    shared_enc_root = top_out / "_global_encoders"

    requested_user = str(getattr(args, "user", "all") or "all")
    if requested_user.lower() == "all":
        users = discover_bp_users() if args.task == "bp" else []
    else:
        users = [requested_user]

    if not users:
        raise SystemExit("No users found to process.")

    print(f"Running LR_check for {len(users)} user(s): {users}")

    aggregate_full_rows = []
    aggregate_10pct_rows = []
    for uid in users:
        reset_seeds(42)
        df_res, full_payload = run_for_user(
            user_id=str(uid),
            args=args,
            top_out=top_out,
            shared_enc_root=shared_enc_root,
            shared_cnn_root=shared_cnn_root,
            results_subdir=results_subdir,
            input_df=input_df,
            seeds=seeds,
        )
        if df_res is not None:
            user_out_dir = top_out / results_subdir / str(uid)
            seed_only = df_res[df_res["seed"] != "full_data"].copy()
            for _, row in seed_only.iterrows():
                mean_auc, std_auc = parse_auc_mean_std(row["test_auc_mean ± std"])
                aggregate_10pct_rows.append(
                    {
                        "user": str(uid),
                        "user_root": str(user_out_dir),
                        "seed": int(row["seed"]),
                        "agg_auc_mean": float(mean_auc),
                        "agg_auc_std": float(std_auc),
                        "num_train": int(row["n_train"]),
                    }
                )
        if full_payload is not None:
            y_true = np.load(full_payload["y_true_path"])
            y_pred = np.load(full_payload["y_pred_path"])
            aggregate_full_rows.append(
                {
                    "user": full_payload["user"],
                    "full_auc": full_payload["full_auc"],
                    "full_auc_std": full_payload["full_auc_std"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    aggregate_full_df = pd.DataFrame(aggregate_full_rows)
    out_root = top_out / results_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    aggregate_full_df.drop(columns=["y_true", "y_pred"], errors="ignore").to_csv(
        out_root / "df_full_results_summary.csv",
        index=False,
    )
    print(f"Constructed aggregate_full_df with {len(aggregate_full_df)} users")

    plot_aggregate_auc(aggregate_full_df, top_out, results_subdir)
    aggregate_10pct_df = pd.DataFrame(aggregate_10pct_rows)
    for seed in seeds:
        run_tag = f"frac10pct_seed{seed}"
        plot_aggregate_10pct(
            aggregate_10pct_df,
            seed_to_plot=seed,
            run_tag=run_tag,
            out_path=out_root / f"aggregate_10pct_seed_{seed}_{args.fruit}_{args.scenario}.png",
        )


if __name__ == "__main__":
    main()
