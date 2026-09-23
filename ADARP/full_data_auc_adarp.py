"""The 100%-data AUC, per participant, from the learned representations.

This is the ceiling the AL curves are measured against: no acquisition, no
rounds -- the classifier simply sees every labeled training window at once,
encoded by the frozen SimCLR encoders, and is scored on the participant's
held-out test sessions.

It is the same computation `new_helper.run_experiment` does once per run and
saves as `upper_bound_auc.npy`, lifted out so it can be swept over every usable
target and plotted in one figure without running active learning at all.

    python ADARP/full_data_auc_adarp.py --pool global --classifier lr
    python ADARP/full_data_auc_adarp.py --participants 112,101 --classifier mlp
    python ADARP/full_data_auc_adarp.py --compare --output_dir adarp_global_results

Encoders are loaded from disk when they exist and trained once when they do
not, exactly as a run would -- `prepare_data` decides, not this script.
"""

import argparse
import json
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_ADARP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ADARP_DIR.parent
for _path in (str(_REPO_ROOT), str(_ADARP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sklearn.preprocessing import StandardScaler

import utility
from new_helper import (
    set_output_dir,
    set_classifier,
    build_classifier,
    build_fit_kwargs,
    predict_positive_scores,
    reset_seeds,
)
from preprocess_adarp_data import (
    PROCESSED_DIR,
    FEATURE_POINTS,
    bin_mean,
    load_windows,
    windows_to_frame,
    split_sessions,
    usable_targets,
    prepare_data,
)

# Mirrors run_adarp.py: the SSL knobs are read from the environment so a sweep
# here trains the same encoders a run would, not a second set beside them.
BATCH_SSL_ADARP = int(os.environ.get("ADARP_BATCH_SSL", "32"))
SSL_EPOCHS_ADARP = int(os.environ.get("ADARP_SSL_EPOCHS", "100"))


def parse_args():
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--pool", default="global", choices=["personal", "global"],
                    help="global trains on every contributor's train sessions; "
                         "personal on the target's own.")
    pa.add_argument("--classifier", default="lr", choices=["mlp", "lr"])
    pa.add_argument("--participants", default=None,
                    help="Comma-separated targets. Default: every usable target.")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--dropout_rate", type=float, default=0.5)
    pa.add_argument("--epochs", type=int, default=200, help="MLP only.")
    pa.add_argument("--patience", type=int, default=15, help="MLP only.")
    pa.add_argument("--output_dir", default=None,
                    help="Run tree the encoders live under. Defaults to "
                         "BAN_AL_OUTPUT_DIR, else new_helper.set_output_dir.")
    pa.add_argument("--results_dir", default=str(_ADARP_DIR / "results" / "full_data_auc"),
                    help="Where the CSV, JSON and figure are written.")
    pa.add_argument("--exclude_users", default="",
                    help="Comma-separated users dropped from the global pool.")
    pa.add_argument("--level", action="store_true",
                    help="Append the per-window mean and SD of each channel to Z "
                         "-- exactly what zscore_rows discards -- and standardise "
                         "before the fit.")
    pa.add_argument("--level_ablation", action="store_true",
                    help="Run both arms (encoder alone, encoder+level) over the "
                         "same targets and add a figure pairing them.")
    pa.add_argument("--scale_features", action="store_true",
                    help="StandardScaler on Z before the fit. Implied by --level.")
    pa.add_argument("--compare", action="store_true",
                    help="Sweep global AND personal, and add a figure pairing "
                         "the two per participant. Overrides --pool.")
    pa.add_argument("--processed_dir", default=str(PROCESSED_DIR))
    pa.add_argument("--channels", default="hr_eda", choices=["hr_eda", "eda"],
                    help="Which channels to encode: hr_eda uses both, eda uses EDA only.")
    pa.add_argument("--n_splits", type=int, default=1,
                    help="Number of random train/test splits to run. Results aggregated as mean ± std.")
    return pa.parse_args()


def all_usable_targets(processed_dir):
    """Every participant the session split can actually score."""
    eda, hr, meta = load_windows(processed_dir)
    frame = windows_to_frame(eda, hr, meta)
    return usable_targets(frame)



LEVEL_COLS = ["hr_level_mean", "hr_level_sd", "eda_level_mean", "eda_level_sd"]


def load_level_table(processed_dir):
    """The per-window statistics `zscore_rows` throws away, keyed for joining.

    `windows_to_frame` bins each window to FEATURE_POINTS and then standardises
    the row, which discards exactly two numbers per channel: the row's mean and
    its SD. Those are recomputed here from the same binned arrays, so appending
    them to `Z` restores precisely the information the pipeline drops -- nothing
    more, and no rebuild or retraining required.

    Note what the EDA pair carries. It is measured after `prepare_eda`'s
    per-segment min-max, so it holds the tonic level AND the label-dependent
    scaling artifact together. That is the point: if AUC jumps when these are
    appended, the raw-feature baseline's 0.73 was reachable from inside the
    existing pipeline, and the next question is which of the two did it.
    """
    eda, hr, meta = load_windows(processed_dir)
    hr_b = bin_mean(hr, FEATURE_POINTS)
    eda_b = bin_mean(eda, FEATURE_POINTS)

    table = pd.DataFrame({
        "hr_level_mean": hr_b.mean(axis=1),
        "hr_level_sd": hr_b.std(axis=1),
        "eda_level_mean": eda_b.mean(axis=1),
        "eda_level_sd": eda_b.std(axis=1),
    })
    table["_key"] = _window_key(
        meta["participant"], meta["session"], meta["segment_id"],
        pd.to_datetime(meta["window_start"], utc=True))
    return table.drop_duplicates("_key").set_index("_key")[LEVEL_COLS]


def _window_key(user, session, segment, start):
    """Participant, session, segment and start instant identify a window uniquely.

    The start is carried as integer nanoseconds: windows are joined across a
    round-trip through CSV, and float timestamps do not survive that reliably.
    """
    return (user.astype(str) + "|" + session.astype(str) + "|"
            + segment.astype(str) + "|"
            + pd.to_datetime(start, utc=True).astype("int64").astype(str))


def attach_level(Z, df, level_table):
    """Append the level block to an encoded matrix, in `df` row order."""
    keys = _window_key(df["user_id"], df["session"], df["segment_id"], df["window_start"])
    block = level_table.reindex(keys.to_numpy())
    missing = int(block.isna().any(axis=1).sum())
    if missing:
        raise SystemExit(
            f"{missing} of {len(df)} windows found no level row. The frame and "
            f"{WINDOWS_STEM_HINT} have drifted apart; rebuild the dataset."
        )
    return np.hstack([Z, block.to_numpy(dtype="float32")]).astype("float32")


WINDOWS_STEM_HINT = "adarp_windows.npz"


def full_data_auc_for_user(user, args, top_out, pool, level_table=None, channels="hr_eda"):
    """Train on 100% of the training pool, score the target's test sessions.

    `pool` is passed rather than read off `args` so one invocation can sweep
    both the global and the personal model over the same targets.

    Returns the row that goes into the table, or None when `prepare_data`
    declines the user.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_class_weight

    args_ns = SimpleNamespace(
        user=str(user),
        pool=pool,
        fruit="ADARP",
        scenario="stress",
        task="adarp",
        participant_id=str(user),
        unlabeled_frac=0.0,
        dropout_rate=float(args.dropout_rate),
        warm_start=1,
        results_subdir="results",
        input_df="raw",
        classifier=args.classifier,
    )

    reset_seeds(args.seed)
    prep = prepare_data(
        args=args_ns,
        top_out=top_out,
        shared_enc_root=top_out / "_global_encoders",
        shared_cnn_root=top_out / "global_cnns",
        batch_ssl=BATCH_SSL_ADARP,
        ssl_epochs=SSL_EPOCHS_ADARP,
        pool=pool,
        task="adarp",
        input_df="raw",
        seed=args.seed,
        processed_dir=Path(args.processed_dir),
    )
    if prep is None:
        return None

    df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, *_ = prep

    # The 100% pool is what AL would eventually have labeled everything of:
    # the pooled train frame under global, the target's own under personal.
    df_full = df_all_tr if df_all_tr is not None else df_tr

    # LR has no early stopping, so val is training data it would otherwise
    # waste -- the same fold-in run_adarp.py does under --classifier lr.
    if args.classifier == "lr" and df_val is not None and len(df_val) > 0:
        df_full = pd.concat([df_full, df_val], ignore_index=True)
        df_val = df_val.iloc[0:0].copy()

    set_classifier(args.classifier)
    reset_seeds(args.seed)

    if channels == "eda":
        # EDA-only: encode with stress encoder, skip HR
        Z_full = utility.encode_single_df(df_full, None, enc_st, pool)
        Z_te = utility.encode_single_df(df_te, None, enc_st, pool)
    else:
        # HR + EDA: both encoders
        Z_full = utility.encode_single_df(df_full, enc_hr, enc_st, pool)
        Z_te = utility.encode_single_df(df_te, enc_hr, enc_st, pool)
    y_full = df_full["state_val"].values.astype("float32")
    y_te = df_te["state_val"].values.astype("float32")

    if level_table is not None:
        Z_full = attach_level(Z_full, df_full, level_table)
        Z_te = attach_level(Z_te, df_te, level_table)

    # Standardising is not optional once the level block is attached: HR sits in
    # bpm, EDA in [0,1] and the encoder's units in neither, and `_LRAdapter` caps
    # lbfgs at 1000 iterations. Unscaled, the fit stops wherever it ran out of
    # budget and the comparison would measure the solver, not the features.
    scaler = None
    if args.scale_features or level_table is not None:
        scaler = StandardScaler().fit(Z_full)
        Z_full = scaler.transform(Z_full).astype("float32")
        Z_te = scaler.transform(Z_te).astype("float32")

    model, callbacks = build_classifier(
        Z_full.shape[1],
        args.patience,
        float(args.dropout_rate),
        args.seed,
    )

    class_weight = None
    classes = np.unique(y_full)
    if len(classes) == 2:
        cw = compute_class_weight("balanced", classes=classes, y=y_full)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    if args.classifier == "lr":
        # _LRAdapter turns class_weight into per-sample weights; passing it is
        # what keeps this the same estimator new_helper.run_experiment scores.
        model.fit(Z_full, y_full, class_weight=class_weight)
    else:
        fit_kwargs = dict(epochs=args.epochs, batch_size=16, verbose=0)
        if df_val is not None and len(df_val) > 0:
            if channels == "eda":
                Z_val = utility.encode_single_df(df_val, None, enc_st, pool)
            else:
                Z_val = utility.encode_single_df(df_val, enc_hr, enc_st, pool)
            if level_table is not None:
                Z_val = attach_level(Z_val, df_val, level_table)
            if scaler is not None:
                Z_val = scaler.transform(Z_val).astype("float32")
            y_val = df_val["state_val"].values.astype("float32")
            fit_kwargs["validation_data"] = (Z_val, y_val)
        fit_kwargs = build_fit_kwargs(fit_kwargs, callbacks, use_early_stopping=True)
        model.fit(Z_full, y_full, class_weight=class_weight, **fit_kwargs)

    scores = predict_positive_scores(model, Z_te)
    auc = float(roc_auc_score(y_te, scores))

    # In-sample AUC over the entire fitted pool -- the objective the fit actually
    # optimised. It counts cross-participant pairs, which the per-participant
    # figure below does not, so the two differ by however much of the model's
    # ranking rides on between-subject offsets rather than within-subject change.
    pooled_train_auc = float("nan")
    if len(np.unique(y_full)) == 2:
        pooled_train_auc = float(roc_auc_score(
            y_full, predict_positive_scores(model, Z_full)))

    # In-sample AUC on the target's OWN rows inside the pool. The gap against
    # `auc` is what separates an overfit model from one that never fit at all:
    # overfitting shows train near 1 and test near chance, a flat representation
    # shows both near chance.
    df_tr_user = df_full[df_full["user_id"].astype(str) == str(user)]
    y_tr_user = df_tr_user["state_val"].values.astype("float32")
    if len(np.unique(y_tr_user)) < 2:
        train_auc = float("nan")
    else:
        if channels == "eda":
            Z_tr_user = utility.encode_single_df(df_tr_user, None, enc_st, pool)
        else:
            Z_tr_user = utility.encode_single_df(df_tr_user, enc_hr, enc_st, pool)
        if level_table is not None:
            Z_tr_user = attach_level(Z_tr_user, df_tr_user, level_table)
        if scaler is not None:
            Z_tr_user = scaler.transform(Z_tr_user).astype("float32")
        train_auc = float(roc_auc_score(
            y_tr_user, predict_positive_scores(model, Z_tr_user)))

    return {
        "user_id": str(user),
        "auc": auc,
        "train_auc": train_auc,
        "pooled_train_auc": pooled_train_auc,
        "gap": train_auc - auc,
        "train_n_user": int(len(df_tr_user)),
        "train_pos_user": int(y_tr_user.sum()),
        # Events, not windows, are the real sample size: one press yields ~79
        # near-duplicate windows, so `test_pos` overstates the evidence badly.
        "train_events": int(df_tr_user.loc[df_tr_user["state_val"] == 1, "event_id"].nunique()),
        "test_events": int(df_te.loc[df_te["state_val"] == 1, "event_id"].nunique()),
        "n_train": int(len(df_full)),
        "n_test": int(len(df_te)),
        "test_pos": int(y_te.sum()),
        "test_neg": int((y_te == 0).sum()),
        "test_pos_rate": float(y_te.mean()),
        "pool": pool,
        "level": level_table is not None,
        "n_features": int(Z_full.shape[1]),
        "classifier": args.classifier,
        "seed": args.seed,
        "_test_scores": scores,  # Raw scores for pooling
        "_test_labels": y_te,    # Raw labels for pooling
    }


def plot_full_data_auc(table, out_path, pool, classifier):
    """Test AUC per participant against chance, the mean, and the training fit.

    The training fit is drawn as one line, not a dot per participant: under the
    global pool every target is scored by the same model fitted on the same
    pooled windows, so there is exactly one in-sample number. How far the dots
    sit below that line is the generalisation gap; how far the line itself sits
    above 0.5 is whether the representation separates the classes at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = table.sort_values("auc").reset_index(drop=True)
    x = np.arange(len(table))
    mean_auc = float(table["auc"].mean())

    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * len(table) + 4), 5.4))

    # Bar plot with error bars
    colors = ["#c44e52" if a < 0.5 else "#4c72b0" for a in table["auc"]]
    yerr = table["auc_std"].fillna(0).values if "auc_std" in table.columns else None

    ax.bar(x, table["auc"], color=colors, alpha=0.7, edgecolor="black", linewidth=1,
           yerr=yerr, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "gray"},
           label="test AUC ± std" if yerr is not None else "test AUC")

    # Show aggregated test AUC line
    # Only show ±std if it came from multiple splits (auc_std column exists)
    if "auc_std" in table.columns and table["auc_std"].max() > 1e-10:
        std_auc = float(table["auc_std"].mean())  # Std from cross-split variation
        ax.axhline(mean_auc, color="#55a868", linestyle="-", linewidth=2.0,
                   label=f"aggregated test AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    else:
        # Single split: no cross-split variation
        ax.axhline(mean_auc, color="#55a868", linestyle="-", linewidth=2.0,
                   label=f"aggregated test AUC = {mean_auc:.3f}")

    # Under `global` every target shares one fitted model, so the in-sample AUC
    # is a single number and belongs on a line. Under `personal` each target has
    # its own model and its own in-sample AUC, so it has to be drawn per dot.
    if "pooled_train_auc" in table and table["pooled_train_auc"].notna().any():
        vals = table["pooled_train_auc"]
        if vals.dropna().nunique() == 1:
            pooled = float(vals.dropna().iloc[0])
            # Add std band if available
            if "pooled_train_auc_std" in table.columns:
                pooled_std = float(table["pooled_train_auc_std"].dropna().mean())
                ax.fill_between([-0.5, len(table)-0.5], pooled - pooled_std, pooled + pooled_std,
                               alpha=0.15, color="#e5ae38")
                ax.axhline(pooled, color="#e5ae38", linestyle="-", linewidth=2.0,
                           label=f"train AUC = {pooled:.3f} ± {pooled_std:.3f}")
            else:
                ax.axhline(pooled, color="#e5ae38", linestyle="-", linewidth=2.0,
                           label=f"train AUC (in-sample) = {pooled:.3f}")
        else:
            ax.scatter(x, vals, s=45, marker="_", linewidths=2.0,
                       color="#e5ae38", zorder=2,
                       label=f"train AUC (in-sample), mean {vals.mean():.3f}")

    for xi, row in zip(x, table.itertuples()):
        ax.annotate(f"{row.auc:.2f}", (xi, row.auc), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    # Event counts under the tick: the sample size the AUC above actually rests on.
    if "test_events" in table:
        labels = [f"{u}\n({e} ev)" for u, e in zip(table["user_id"], table["test_events"])]
    else:
        labels = list(table["user_id"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Participant (test-set button presses)")
    ax.set_ylabel("AUC")
    ax.set_title(f"ADARP full-data ceiling -- pool={pool}, classifier={classifier}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path



def plot_pool_comparison(tables, out_path, classifier):
    """Global against personal, one pair of dots per participant.

    Sorted by the personal model, because that is the axis the question is
    about: does a model trained on this person alone beat one trained on
    everyone. An upward arrow means personalisation paid; a downward one means
    the person had too little of their own data to learn from.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = tables["global"][["user_id", "auc", "test_events"]].rename(columns={"auc": "global_auc"})
    pr = tables["personal"][["user_id", "auc", "train_events"]].rename(columns={"auc": "personal_auc"})
    both = g.merge(pr, on="user_id", how="inner").sort_values("personal_auc").reset_index(drop=True)
    if both.empty:
        print("[full-data] no participant ran under both pools; skipping comparison plot")
        return None

    x = np.arange(len(both))
    fig, ax = plt.subplots(figsize=(max(7.5, 0.8 * len(both) + 4), 5.4))

    ax.vlines(x, both["global_auc"], both["personal_auc"],
              color="#b0b0b0", linewidth=1.5, zorder=1)
    ax.scatter(x, both["global_auc"], s=70, marker="s", zorder=3,
               color="#4c72b0", label=f"global, mean {both['global_auc'].mean():.3f}")
    ax.scatter(x, both["personal_auc"], s=70, marker="o", zorder=3,
               color="#dd8452", label=f"personal, mean {both['personal_auc'].mean():.3f}")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="chance (0.5)")

    # Train events, not test: personalisation is limited by what the person's
    # own training sessions hold, and that is the number to read a win against.
    labels = [f"{u}\n({e} tr ev)" for u, e in zip(both["user_id"], both["train_events"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Participant (button presses in their own training sessions)")
    ax.set_ylabel("Test AUC")
    ax.set_title(f"ADARP full-data ceiling: global vs personal -- classifier={classifier}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path



def plot_level_ablation(tables, out_path, pool, classifier):
    """Encoder alone against encoder plus the level the pipeline discards.

    A jump means the four appended numbers carry what SimCLR could not reach,
    and the raw-feature baseline's 0.73 is available without a rebuild. A flat
    pair means level is not what is missing, and the ceiling is where it is for
    some other reason.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = tables["encoder"][["user_id", "auc", "test_events"]].rename(columns={"auc": "base_auc"})
    lvl = tables["level"][["user_id", "auc"]].rename(columns={"auc": "level_auc"})
    both = base.merge(lvl, on="user_id").sort_values("level_auc").reset_index(drop=True)
    if both.empty:
        print("[full-data] no participant ran under both arms; skipping ablation plot")
        return None

    x = np.arange(len(both))
    fig_, ax = plt.subplots(figsize=(max(7.5, 0.8 * len(both) + 4), 5.4))

    ax.vlines(x, both["base_auc"], both["level_auc"],
              color="#b0b0b0", linewidth=1.5, zorder=1)
    ax.scatter(x, both["base_auc"], s=70, marker="s", zorder=3, color="#8172b3",
               label=f"encoder only, mean {both['base_auc'].mean():.3f}")
    ax.scatter(x, both["level_auc"], s=70, zorder=3, color="#4c72b0",
               label=f"encoder + level, mean {both['level_auc'].mean():.3f}")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="chance (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{u}\n({e} ev)" for u, e in zip(both["user_id"], both["test_events"])],
                       fontsize=9)
    ax.set_xlabel("Participant (test-set button presses)")
    ax.set_ylabel("Test AUC")
    ax.set_title(f"Does the discarded level matter? -- pool={pool}, classifier={classifier}")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    fig_.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig_)
    return out_path


def sweep_pool(args, pool, top_out, users, results_dir, level_table=None, channels="hr_eda"):
    """Every target under one pool: the table, the JSON summary and the figure."""
    rows, skipped = [], []

    # If multiple splits, run each user across n_splits with different seeds
    if args.n_splits > 1:
        # Collect all rows per user across splits
        per_user_splits = {str(u): [] for u in users}

        for split_id in range(args.n_splits):
            split_seed = args.seed + split_id
            args_split = SimpleNamespace(**vars(args))
            args_split.seed = split_seed

            print(f"\n[full-data] === Split {split_id + 1}/{args.n_splits} (seed={split_seed}) ===", flush=True)

            for user in users:
                arm_name = "encoder+level" if level_table is not None else "encoder"
                try:
                    row = full_data_auc_for_user(user, args_split, top_out, pool, level_table, channels)
                except SystemExit as exc:
                    if split_id == 0:
                        print(f"[full-data] skipping {user}: {exc}")
                        skipped.append({"user_id": str(user), "reason": str(exc)})
                    continue
                if row is None:
                    if split_id == 0:
                        skipped.append({"user_id": str(user), "reason": "prepare_data returned None"})
                    continue
                per_user_splits[str(user)].append(row)

        # Aggregate: mean and std per participant across splits
        for user in users:
            user_str = str(user)
            split_rows = per_user_splits[user_str]
            if not split_rows:
                continue

            # Compute mean and std across splits
            agg_row = split_rows[0].copy()  # Start with first row as template
            for key in ['auc', 'train_auc', 'pooled_train_auc', 'gap']:
                vals = [r[key] for r in split_rows if not np.isnan(r[key])]
                if vals:
                    agg_row[f'{key}_mean'] = float(np.mean(vals))
                    agg_row[f'{key}_std'] = float(np.std(vals, ddof=0))
                    agg_row[key] = agg_row[f'{key}_mean']  # Use mean as main value

            agg_row['n_splits'] = len(split_rows)
            rows.append(agg_row)

            print(f"[full-data] {user}: test AUC={agg_row['auc']:.4f}±{agg_row['auc_std']:.4f} "
                  f"train AUC={agg_row['train_auc']:.4f}±{agg_row['train_auc_std']:.4f} "
                  f"({len(split_rows)} splits)")

    else:
        # Single split: original logic
        for user in users:
            arm_name = "encoder+level" if level_table is not None else "encoder"
            print(f"\n[full-data] === {pool} / {arm_name} / participant {user} ===",
                  flush=True)
            try:
                row = full_data_auc_for_user(user, args, top_out, pool, level_table, channels)
            except SystemExit as exc:      # prepare_data's own refusals
                print(f"[full-data] skipping {user}: {exc}")
                skipped.append({"user_id": str(user), "reason": str(exc)})
                continue
            if row is None:
                skipped.append({"user_id": str(user), "reason": "prepare_data returned None"})
                continue
            print(f"[full-data] {user}: test AUC={row['auc']:.4f} "
                  f"train AUC={row['train_auc']:.4f} gap={row['gap']:+.4f} "
                  f"(pool={row['n_train']}, test={row['n_test']} windows / "
                  f"{row['test_events']} events)")
            rows.append(row)

    # Save individual split results if multiple splits
    if args.n_splits > 1:
        split_rows_all = []
        for split_id in range(args.n_splits):
            for user in per_user_splits:
                if split_id < len(per_user_splits[user]):
                    row = per_user_splits[user][split_id].copy()
                    row['split_id'] = split_id
                    split_rows_all.append(row)

        if split_rows_all:
            # Remove internal fields
            for row in split_rows_all:
                row.pop("_test_scores", None)
                row.pop("_test_labels", None)

            split_table = pd.DataFrame(split_rows_all)
            split_stem = f"full_data_auc_{pool}{arm}_{args.classifier}_seed{args.seed}_{args.n_splits}splits_splits{channel_suffix}"
            split_csv_path = results_dir / f"{split_stem}.csv"
            split_table.to_csv(split_csv_path, index=False)
            print(f"[full-data] saved per-split results to {split_csv_path}", flush=True)

    if not rows:
        print(f"[full-data] no participant produced a full-data AUC under {pool}.")
        return None

    # Compute true aggregated test AUC from pooled predictions and labels
    from sklearn.metrics import roc_auc_score
    all_scores = []
    all_labels = []
    for row in rows:
        if "_test_scores" in row and "_test_labels" in row:
            all_scores.extend(row["_test_scores"].flatten())
            all_labels.extend(row["_test_labels"].flatten())

    pooled_test_auc = None
    if all_scores and len(np.unique(all_labels)) == 2:
        pooled_test_auc = float(roc_auc_score(all_labels, all_scores))
        print(f"[full-data] pooled test AUC (all data): {pooled_test_auc:.4f}", flush=True)

    # Remove internal fields before creating DataFrame
    for row in rows:
        row.pop("_test_scores", None)
        row.pop("_test_labels", None)

    arm = "_level" if level_table is not None else ""
    split_suffix = f"_{args.n_splits}splits" if args.n_splits > 1 else ""
    channel_suffix = f"_{channels}" if channels != "hr_eda" else "_hr_eda"
    stem = f"full_data_auc_{pool}{arm}_{args.classifier}_seed{args.seed}{split_suffix}{channel_suffix}"
    table = pd.DataFrame(rows)
    csv_path = results_dir / f"{stem}.csv"
    table.to_csv(csv_path, index=False)
    print(f"[full-data] saved results to {csv_path}", flush=True)

    summary = {
        "pool": pool,
        "classifier": args.classifier,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "n_participants": len(table),
        "per_participant_mean_auc": float(table["auc"].mean()),
        "pooled_test_auc": pooled_test_auc,
        "mean_train_auc": float(table["train_auc"].mean()),
        "pooled_train_auc": float(table["pooled_train_auc"].dropna().iloc[0])
        if table["pooled_train_auc"].dropna().nunique() == 1 else None,
        "mean_pooled_train_auc": float(table["pooled_train_auc"].mean()),
        "mean_gap": float(table["gap"].mean()),
        "median_auc": float(table["auc"].median()),
        "std_auc": float(table["auc"].std(ddof=0)),
        "min_auc": float(table["auc"].min()),
        "max_auc": float(table["auc"].max()),
        "skipped": skipped,
    }
    if args.n_splits > 1 and "auc_std" in table.columns:
        summary["mean_auc_std"] = float(table["auc_std"].mean())

    json_path = results_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[full-data] saved summary to {json_path}", flush=True)

    png_path = plot_full_data_auc(table, results_dir / f"{stem}.png",
                                  pool, args.classifier)

    print(f"\n[full-data] {pool}: {len(table)} participants, "
          f"per-participant mean test AUC {summary['per_participant_mean_auc']:.4f} "
          f"(min {summary['min_auc']:.4f}, max {summary['max_auc']:.4f}); "
          f"pooled train AUC {summary.get('pooled_train_auc', 'N/A')}")
    print(f"[full-data] wrote {csv_path}")
    print(f"[full-data] wrote {png_path}")
    return table


def main():
    args = parse_args()

    if args.exclude_users:
        os.environ["BAN_AL_EXCLUDE_USERS"] = args.exclude_users

    top_out = Path(args.output_dir or os.environ.get("BAN_AL_OUTPUT_DIR")
                   or set_output_dir(args.pool, True))

    if args.participants:
        users = [u.strip() for u in args.participants.split(",") if u.strip()]
    else:
        users = all_usable_targets(Path(args.processed_dir))
    print(f"[full-data] targets: {users}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.level_ablation:
        level_table = load_level_table(Path(args.processed_dir))
        print(f"[full-data] level ablation: encoder alone, then encoder + "
              f"{len(LEVEL_COLS)} level features {LEVEL_COLS}")
        arms = {}
        for arm, tbl in (("encoder", None), ("level", level_table)):
            table = sweep_pool(args, args.pool, top_out, users, results_dir, tbl, args.channels)
            if table is not None:
                arms[arm] = table
        if len(arms) == 2:
            png = plot_level_ablation(
                arms, results_dir / f"level_ablation_{args.pool}_"
                                    f"{args.classifier}_seed{args.seed}.png",
                args.pool, args.classifier)
            d = arms["level"]["auc"].mean() - arms["encoder"]["auc"].mean()
            print(f"\n[full-data] level is worth {d:+.4f} mean test AUC "
                  f"({arms['encoder']['auc'].mean():.4f} -> "
                  f"{arms['level']['auc'].mean():.4f})")
            if png is not None:
                print(f"[full-data] wrote {png}")
        return 0

    level_table = load_level_table(Path(args.processed_dir)) if args.level else None
    pools = ["global", "personal"] if args.compare else [args.pool]
    if args.compare:
        print("[full-data] --compare: sweeping both pools. The personal pool "
              "trains a SimCLR encoder pair per participant the first time it "
              "runs; that is the slow part, and it is cached afterwards.")

    tables = {}
    for pool in pools:
        table = sweep_pool(args, pool, top_out, users, results_dir, level_table, args.channels)
        if table is not None:
            tables[pool] = table

    if not tables:
        raise SystemExit("No participant produced a full-data AUC.")

    if len(tables) == 2:
        cmp_path = plot_pool_comparison(
            tables, results_dir / f"full_data_auc_global_vs_personal_"
                                  f"{args.classifier}_seed{args.seed}.png",
            args.classifier)
        if cmp_path is not None:
            print(f"[full-data] wrote {cmp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
