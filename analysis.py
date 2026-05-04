"""
analysis.py
-----------
Phase 1: BP spike dataset analysis.

Produces:
  1. Dataset overview   — per-participant stats table
  2. Label distribution — class balance, spike rate
  3. SSL representation — t-SNE colored by label / participant
  4. Upper bound AUC    — global-train ceiling per participant
  5. Summary statement  — why dataset is hard for AL

Run:
    python analysis.py \
        --pool global \
        --input_df processed \
        --output_dir ./analysis_results

All heavy lifting (data loading, encoding, windowing)
reuses the existing new_prep.py / new_helper.py stack.
No new data-loading code is written here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping

# ── project imports ────────────────────────────────────────
from new_helper import (
    reset_seeds,
    bootstrap_auc,
    build_XY_from_processed,
    build_classifier,
)
from new_prep import prepare_data, _bp_load_all
from src.compare_pipelines import collect_windows, derive_negative_labels


# ─────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BP spike dataset analysis")
    p.add_argument("--pool",       default="global",
                   choices=["personal", "global", "global_supervised"])
    p.add_argument("--input_df",   default="processed",
                   choices=["raw", "processed"])
    p.add_argument("--output_dir", default="./analysis_results")
    p.add_argument("--n_bootstrap",type=int, default=500)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────
# 1.  DATASET OVERVIEW
# ─────────────────────────────────────────────────────────

def compute_dataset_overview(all_splits: dict,
                              all_positives: dict,
                              all_negatives: dict) -> pd.DataFrame:
    """
    Per-participant summary:
        n_pos, n_neg, total, spike_rate,
        n_train_days, n_val_days, n_test_days
    """
    rows = []
    for pid, (tr_days, val_days, te_days) in all_splits.items():
        pos = all_positives.get(pid)
        neg = all_negatives.get(pid)
        n_pos = len(pos) if pos is not None else 0
        n_neg = len(neg) if neg is not None else 0
        rows.append({
            "participant":   pid,
            "n_pos":         n_pos,
            "n_neg":         n_neg,
            "total_labels":  n_pos + n_neg,
            "spike_rate":    round(n_pos / max(n_pos + n_neg, 1), 3),
            "n_train_days":  len(tr_days),
            "n_val_days":    len(val_days),
            "n_test_days":   len(te_days),
        })
    return pd.DataFrame(rows).sort_values("participant")


def plot_spike_rates(overview: pd.DataFrame,
                     output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Bar chart — spike rate per participant
    ax = axes[0]
    colors = ["#E24B4A" if r > 0.3
              else "#185FA5" if r > 0.1
              else "#B4B2A9"
              for r in overview["spike_rate"]]
    ax.bar(overview["participant"].astype(str),
           overview["spike_rate"], color=colors)
    ax.axhline(overview["spike_rate"].mean(),
               color="black", linestyle="--", linewidth=1,
               label=f"mean={overview['spike_rate'].mean():.2f}")
    ax.set_xlabel("Participant")
    ax.set_ylabel("BP spike rate")
    ax.set_title("Spike rate per participant")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # Histogram — distribution of spike rates
    ax2 = axes[1]
    ax2.hist(overview["spike_rate"], bins=10,
             color="#185FA5", edgecolor="white")
    ax2.set_xlabel("Spike rate")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of spike rates")

    plt.tight_layout()
    path = output_dir / "spike_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────
# 2.  COLLECT REPRESENTATIONS FROM ALL PARTICIPANTS
# ─────────────────────────────────────────────────────────


def _bp_raw_rows_like_new_prep(pid, days, all_signals, *, test: bool = False):
    hr_df, st_df, pos_df, orig_neg = all_signals[pid]
    pos_df = pos_df.copy()
    orig_neg = orig_neg.copy()

    if test:
        if len(orig_neg) < len(pos_df):
            neg_df = derive_negative_labels(hr_df, pos_df, len(pos_df))
        else:
            neg_df = orig_neg
    elif len(orig_neg) < len(pos_df):
        extra = derive_negative_labels(
            hr_df, pos_df, len(pos_df) - len(orig_neg)
        )
        neg_df = pd.concat([orig_neg, extra], ignore_index=True)
    else:
        neg_df = orig_neg

    return collect_windows(pos_df, neg_df, hr_df, st_df, days)


def collect_all_representations(
    bp_users:    list,
    all_splits:  dict,
    input_df:    str | None,
    # only needed for raw branch
    all_signals: dict | None = None,
    enc_hr=None,
    enc_st=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each participant load their data, extract
    (Z, y, pid) arrays and concatenate across all days.
 
    input_df='processed':
        Uses precomputed feature CSV via build_XY_from_processed.
 
    input_df='raw':
        Uses global SSL encoders (enc_hr, enc_st) to encode
        HR + step windows via utility.encode.
        Requires all_signals, enc_hr, enc_st.
 
    Returns:
        Z_all   : (N, d)  float32
        y_all   : (N,)    int
        pid_all : (N,)    str
    """
    from new_prep import _bp_load_processed_data, _collect_processed_rows

    input_df = input_df or "processed"
    if input_df not in {"raw", "processed"}:
        raise ValueError(f"Unknown input_df: {input_df}")

    if input_df == "raw":
        if any(v is None for v in [all_signals, enc_hr, enc_st]):
            raise ValueError(
                "raw branch requires all_signals, enc_hr, enc_st."
            )
        import utility
 
    Z_list, y_list, pid_list = [], [], []
 
    for pid in bp_users:
        if pid not in all_splits:
            continue
        tr_days, val_days, te_days = all_splits[pid]
        all_days = list(tr_days) + list(val_days) + list(te_days)
 
        try:
            # ── processed branch ──────────────────────────
            if input_df == "processed":
                df     = _bp_load_processed_data(pid)
                df_pid = _collect_processed_rows(df, all_days)
 
                if df_pid.empty:
                    print(f"  {pid}: empty processed data — skipping")
                    continue
 
                Z, y, *_ = build_XY_from_processed(df_pid, fit=True)
 
            # ── raw branch ────────────────────────────────
            elif input_df == "raw":
                df_pid = _bp_raw_rows_like_new_prep(
                    pid, all_days, all_signals
                )
 
                if df_pid is None or df_pid.empty:
                    print(f"  {pid}: empty raw windows — skipping")
                    continue
 
                Z_hr, Z_st = utility.encode(df_pid, enc_hr, enc_st)
                Z = np.concatenate([Z_hr, Z_st], axis=1)
                y = df_pid["state_val"].values.astype(int)
 
            Z_list.append(Z.astype(np.float32))
            y_list.append(y.astype(int))
            pid_list.append(np.array([str(pid)] * len(y)))
            print(f"  {pid}: {len(y)} windows collected")
 
        except Exception as e:
            print(f"  Skipping {pid}: {e}")
            continue
 
    if not Z_list:
        raise RuntimeError("No representations collected.")
 
    return (
        np.concatenate(Z_list,   axis=0),
        np.concatenate(y_list,   axis=0),
        np.concatenate(pid_list, axis=0),
    )
 

# ─────────────────────────────────────────────────────────
# 3.  T-SNE VISUALIZATION
# ─────────────────────────────────────────────────────────

PARTICIPANT_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9A6324","#fffac8","#800000",
    "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
]


def plot_tsne(Z: np.ndarray,
              y: np.ndarray,
              pid: np.ndarray,
              output_dir: Path,
              seed: int = 42):
    print("  Running t-SNE...")
    tsne  = TSNE(n_components=2, random_state=seed,
                 perplexity=min(30, len(Z) - 1))
    Z2    = tsne.fit_transform(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: color by label ──────────────────────────────
    ax = axes[0]
    colors_y = np.where(y == 1, "#E24B4A", "#185FA5")
    ax.scatter(Z2[y == 0, 0], Z2[y == 0, 1],
               c="#185FA5", alpha=0.4, s=8, label="no spike")
    ax.scatter(Z2[y == 1, 0], Z2[y == 1, 1],
               c="#E24B4A", alpha=0.6, s=10, label="spike")
    ax.set_title("t-SNE colored by label")
    ax.legend(markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Right: color by participant ───────────────────────
    ax2    = axes[1]
    pids_u = sorted(np.unique(pid))
    cmap   = {p: PARTICIPANT_COLORS[i % len(PARTICIPANT_COLORS)]
               for i, p in enumerate(pids_u)}
    patches = []
    for p in pids_u:
        mask = pid == p
        ax2.scatter(Z2[mask, 0], Z2[mask, 1],
                    c=cmap[p], alpha=0.5, s=8)
        patches.append(mpatches.Patch(color=cmap[p], label=str(p)))
    ax2.set_title("t-SNE colored by participant")
    ax2.legend(handles=patches, fontsize=6,
               ncol=2, loc="upper right")
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.suptitle("SSL Representation Space — BP Spike Dataset",
                 fontsize=13)
    plt.tight_layout()
    path = output_dir / "tsne.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return Z2


# ─────────────────────────────────────────────────────────
# 4.  WITHIN / BETWEEN CLASS SIMILARITY
# ─────────────────────────────────────────────────────────

def class_separation_analysis(Z: np.ndarray,
                               y: np.ndarray,
                               output_dir: Path) -> dict:
    """
    Measure how well-separated the SSL representations are.
    If within-class sim ≈ 1 and between-class sim ≈ 0
    → representations are trivially separable
    → AUC = 1 from round 0 even with few labels
    """
    Z_norm = normalize(Z)
    sim    = Z_norm @ Z_norm.T

    stress_idx    = np.where(y == 1)[0]
    nonstress_idx = np.where(y == 0)[0]

    within_s  = sim[np.ix_(stress_idx, stress_idx)].mean()
    within_ns = sim[np.ix_(nonstress_idx, nonstress_idx)].mean()
    between   = sim[np.ix_(stress_idx, nonstress_idx)].mean()

    stats = {
        "within_spike_sim":    round(float(within_s),  4),
        "within_nospike_sim":  round(float(within_ns), 4),
        "between_class_sim":   round(float(between),   4),
        "separability_ratio":  round(float(
            (within_s + within_ns) / 2 / max(between, 1e-8)
        ), 4),
    }

    print("\n  Class separation:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    with open(output_dir / "class_separation.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ─────────────────────────────────────────────────────────
# 5.  UPPER BOUND AUC — GLOBAL TRAINING CEILING
# ─────────────────────────────────────────────────────────

def compute_upper_bounds(
    bp_users:   list,
    all_splits: dict,
    input_df:   str,
    n_bootstrap: int,
    seed:       int,
    all_signals:  dict | None = None,
    shared_enc_root: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Train one upper-bound classifier on all users' train days,
    validate on all users' val days, and evaluate each participant
    on their own held-out test days.

    Protocol:
        Train  → train days aggregated across users
        Val    → val days aggregated across users
        Test   → test days (final evaluation, never touched)
    """
    from new_prep import _bp_load_processed_data, _collect_processed_rows

    # ── raw branch: load global encoders once ─────────────
    enc_hr, enc_st = None, None
    if input_df == "raw":
        if all_signals is None or shared_enc_root is None:
            raise ValueError(
                "raw branch requires all_signals and "
                "shared_enc_root to be provided."
            )
        try:
            import uq_utility
            enc_hr, enc_st, _ = uq_utility._ensure_global_encoders(
                shared_enc_root,
                fruit="BP",
                scenario="spike",
                all_splits=all_splits,
                BATCH_SSL=32,
                SSL_EPOCHS=100,
                exclude_user_id=None,
                BP_MODE=True,
            )
            print("  Global encoders loaded ✓")
        except Exception as e:
            raise RuntimeError(
                f"Could not load global encoders for raw branch: {e}"
            )

    input_df = input_df or "processed"
    if input_df not in {"raw", "processed"}:
        raise ValueError(f"Unknown input_df: {input_df}")

    train_parts = []
    val_parts = []
    test_by_pid = {}

    for pid in bp_users:
        if pid not in all_splits:
            continue
        tr_days, val_days, te_days = all_splits[pid]

        try:
            if input_df == "processed":
                df = _bp_load_processed_data(pid)

                df_tr = _collect_processed_rows(df, list(tr_days))
                df_val = _collect_processed_rows(df, list(val_days))
                df_te = _collect_processed_rows(df, list(te_days))

                if df_tr.empty or df_val.empty or df_te.empty:
                    print(f"  {pid}: empty split — skipping")
                    continue

            elif input_df == "raw":
                df_tr = _bp_raw_rows_like_new_prep(
                    pid, list(tr_days), all_signals
                )
                df_val = _bp_raw_rows_like_new_prep(
                    pid, list(val_days), all_signals
                )
                df_te = _bp_raw_rows_like_new_prep(
                    pid, list(te_days), all_signals, test=True
                )

                if df_tr is None or df_val is None or df_te is None:
                    print(f"  {pid}: collect_windows returned None — skipping")
                    continue
                if df_tr.empty or df_val.empty or df_te.empty:
                    print(f"  {pid}: empty windows — skipping")
                    continue

            else:
                raise ValueError(f"Unknown input_df: {input_df}")

            df_tr = df_tr.copy()
            df_val = df_val.copy()
            df_te = df_te.copy()
            for df_split in (df_tr, df_val, df_te):
                df_split["u_id"] = str(pid)
                df_split["user_id"] = str(pid)

            train_parts.append(df_tr)
            val_parts.append(df_val)
            test_by_pid[pid] = df_te

        except Exception as e:
            print(f"  {pid}: error — {e}")
            continue

    if not train_parts or not val_parts:
        raise RuntimeError("No aggregated train/val data collected.")

    df_tr_all = pd.concat(train_parts, ignore_index=True)
    df_val_all = pd.concat(val_parts, ignore_index=True)
   

    if input_df == "processed":
        def _model_processed_df(df_model):
            return df_model.drop(columns=["u_id"], errors="ignore")

        Z_tr, y_tr, feature_cols, train_median, scaler = (
            build_XY_from_processed(_model_processed_df(df_tr_all), fit=True)
        )

        def _build_processed_eval(df_eval):
            df_eval = _model_processed_df(df_eval.copy())
            for col in feature_cols:
                if col not in df_eval.columns:
                    df_eval[col] = np.nan
            return build_XY_from_processed(
                df_eval,
                feature_cols=feature_cols,
                train_median=train_median,
                scaler=scaler,
                fit=False,
            )

        Z_val, y_val, *_ = _build_processed_eval(df_val_all)
    else:
        import utility

        Z_tr_hr, Z_tr_st = utility.encode(df_tr_all, enc_hr, enc_st)
        Z_val_hr, Z_val_st = utility.encode(df_val_all, enc_hr, enc_st)
        Z_tr = np.concatenate([Z_tr_hr, Z_tr_st], axis=1).astype("float32")
        Z_val = np.concatenate([Z_val_hr, Z_val_st], axis=1).astype("float32")
        y_tr = df_tr_all["state_val"].values.astype(int)
        y_val = df_val_all["state_val"].values.astype(int)
    if len(np.unique(y_tr)) < 2:
        raise RuntimeError("Aggregated training split has a single class.")

    clf, _ = build_classifier(
        input_dim=Z_tr.shape[1],
        CLF_PATIENCE=15,
        dropout_rate=0.2,
        seed=seed,
        n_labeled=len(y_tr),
    )

    es = EarlyStopping(
        monitor="val_auc",
        patience=15,
        restore_best_weights=True,
        mode="max",
        verbose=0,
    )
    classes = np.unique(y_tr)      
    from sklearn.utils.class_weight import compute_class_weight
    cw      = compute_class_weight(
        "balanced",
        classes=classes,
        y=y_tr
    )
    cw_d = {int(c): float(w)
            for c, w in zip(classes, cw)}
    import tensorflow as tf


    history = clf.fit(
        Z_tr,
        y_tr,
        validation_data=(Z_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=[es],
        verbose=0,
        class_weight=cw_d,
    )
    
    print("Train spike rate:", y_tr.mean())
    print("Val spike rate:  ", y_val.mean())

    if output_dir is not None:
        plot_classifier_loss(history, output_dir)


    train_proba = clf.predict(Z_tr, verbose=0).ravel()

    train_auc_mean, train_auc_std, _ = bootstrap_auc(
        y_tr, train_proba,
        n_iters=n_bootstrap, rng_seed=seed,
    )


    rows = []
    all_y_true = []
    all_y_prob = []
    for pid, df_te in test_by_pid.items():
        try:
            if input_df == "processed":
                Z_te, y_te, *_ = _build_processed_eval(df_te)
            else:
                Z_te_hr, Z_te_st = utility.encode(df_te, enc_hr, enc_st)
                Z_te = np.concatenate([Z_te_hr, Z_te_st], axis=1).astype("float32")
                y_te = df_te["state_val"].values.astype(int)

            if len(np.unique(y_te)) < 2:
                print(f"  {pid}: single class in test — skipping")
                continue

            proba = clf.predict(Z_te, verbose=0).ravel()


            ub_mean, ub_std, valid_frac = bootstrap_auc(
                y_te, proba,
                n_iters=n_bootstrap,
                rng_seed=seed,
            )

            train_test_gap = train_auc_mean - ub_mean
            ap = average_precision_score(y_te, proba)

            all_y_true.append(y_te)
            all_y_prob.append(proba)

            rows.append({
                "participant":     pid,
                "ub_auc_mean":     round(ub_mean,     4),
                "ub_auc_std":      round(ub_std,      4),
                "ub_valid_frac":   round(valid_frac,  4),
                "train_auc_mean":  round(train_auc_mean, 4),
                "train_auc_std":   round(train_auc_std,  4),
                "train_test_gap":  round(train_test_gap, 4),
                "ub_ap":           round(ap,           4),

                "n_train":         len(y_tr),
                "n_val":           len(y_val),
                "n_test":          len(y_te),
                "test_spike_rate": round(float(y_te.mean()), 3),
            })
            print(f"  {pid}: UB AUC={ub_mean:.3f}±{ub_std:.3f} "

                  f"valid_frac={valid_frac:.2f}  AP={ap:.3f}  "
                  f"(global_n_tr={len(y_tr)}, global_n_val={len(y_val)}, "
                  f"n_te={len(y_te)})")

        except Exception as e:
            print(f"  {pid}: error — {e}")
            continue

    df_out = (
        pd.DataFrame(rows).sort_values("participant")
        if rows
        else pd.DataFrame()
    )

    # ── pooled grand AUC ──────────────────────────────────
    pooled_auc = float("nan")
    pooled_ap  = float("nan")
    if all_y_true:
        y_pool = np.concatenate(all_y_true)
        p_pool = np.concatenate(all_y_prob)
        if len(np.unique(y_pool)) > 1:
            pooled_auc, _, _  = bootstrap_auc(y_pool, p_pool)
            pooled_ap  = average_precision_score(y_pool, p_pool)
            print(f"\n  Pooled grand AUC = {pooled_auc:.4f}  "
                  f"AP = {pooled_ap:.4f}  "
                  f"(n={len(y_pool)} windows)")

    return df_out, pooled_auc, pooled_ap


def plot_classifier_loss(history, output_dir: Path):
    hist = getattr(history, "history", {})
    if "loss" not in hist or "val_loss" not in hist:
        print("  Skipping classifier loss plot: loss history unavailable.")
        return

    epochs = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, hist["loss"], label="train loss", linewidth=1.8)
    ax.plot(epochs, hist["val_loss"], label="val loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary crossentropy loss")
    ax.set_title("Classifier train vs validation loss")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = output_dir / "classifier_train_val_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_upper_bounds(ub_df: pd.DataFrame,
                      pooled_auc: float,
                      output_dir: Path):
    if ub_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4))

    x     = np.arange(len(ub_df))
    means = ub_df["ub_auc_mean"].values
    stds  = ub_df["ub_auc_std"].values

    colors = ["#E24B4A" if m < 0.65 else
              "#f58231" if m < 0.75 else
              "#3cb44b" for m in means]

    ax.bar(x, means, color=colors, alpha=0.8,
           yerr=stds, capsize=3,
           error_kw={"color": "black", "linewidth": 1})

    # ax.axhline(0.65, color="red", linestyle="--",
    #            linewidth=1, label="0.65 (noise floor)")

    # pooled grand AUC — computed from all y_true and y_prob
    if not np.isnan(pooled_auc):
        ax.axhline(pooled_auc, color="gray", linestyle=":",
                   linewidth=1.5,
                   label=f"pooled AUC={pooled_auc:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(ub_df["participant"].astype(str),
                       rotation=45, ha="right")
    ax.set_ylabel("Upper bound AUC")
    ax.set_title(
        "Global-train upper bound AUC per participant\n"
        "(trained on all users' train days, validated on all users' val days)\n"
        "error bars = bootstrap std | dashed = pooled grand AUC"
    )
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    path = output_dir / "upper_bounds.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────
# 6.  ISOLATION SCORE — WHO NEEDS OWN LABELS
# ─────────────────────────────────────────────────────────

def compute_isolation_scores(Z: np.ndarray,
                              pid: np.ndarray) -> pd.DataFrame:
    """
    For each participant measure how far their centroid
    is from the global centroid in SSL space.

    High isolation → collective labeling may not transfer well
    """
    Z_norm   = normalize(Z)
    global_c = Z_norm.mean(axis=0, keepdims=True)
    global_c = normalize(global_c)[0]

    rows = []
    for p in np.unique(pid):
        mask = pid == p
        Z_p  = Z_norm[mask]
        c_p  = Z_p.mean(axis=0)
        c_p  = c_p / (np.linalg.norm(c_p) + 1e-8)
        cos_dist = 1 - float(c_p @ global_c)
        rows.append({
            "participant":      p,
            "isolation_score":  round(cos_dist, 4),
            "n_windows":        int(mask.sum()),
        })

    return pd.DataFrame(rows).sort_values("isolation_score",
                                          ascending=False)


def plot_isolation(iso_df: pd.DataFrame,
                   output_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 4))
    colors  = ["#E24B4A" if s > 0.3 else
               "#f58231" if s > 0.15 else
               "#185FA5"
               for s in iso_df["isolation_score"]]
    ax.bar(iso_df["participant"].astype(str),
           iso_df["isolation_score"], color=colors)
    ax.axhline(0.3, color="red", linestyle="--",
               linewidth=1, label="high isolation threshold")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Isolation score (cosine distance to pool centroid)")
    ax.set_title("Participant isolation in SSL space\n"
                 "(high = collective labeling less likely to help)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    path = output_dir / "isolation_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────
# 7.  SUMMARY STATEMENT
# ─────────────────────────────────────────────────────────

def write_summary_report(overview: pd.DataFrame,
                          ub_df:    pd.DataFrame,
                          pooled_auc: float,
                          pooled_ap:  float,
                          sep:      dict,
                          iso_df:   pd.DataFrame,
                          output_dir: Path):
    n_participants = len(overview)
    mean_spike     = overview["spike_rate"].mean()
    low_spike      = (overview["spike_rate"] < 0.10).sum()
    low_ub         = (ub_df["ub_auc_mean"] < 0.65).sum() \
                     if not ub_df.empty else "N/A"
    mean_ub        = ub_df["ub_auc_mean"].mean() \
                     if not ub_df.empty else float("nan")
    mean_ub_std    = ub_df["ub_auc_std"].mean() \
                     if not ub_df.empty else float("nan")
    high_iso       = (iso_df["isolation_score"] > 0.3).sum()

    lines = [
        "=" * 60,
        "BP SPIKE DATASET ANALYSIS — SUMMARY",
        "=" * 60,
        "",
        f"Participants:         {n_participants}",
        f"Mean spike rate:      {mean_spike:.1%}",
        f"Low spike rate (<10%): {low_spike}/{n_participants} participants",
        "",
        "SSL Representation Analysis:",
        f"  Within-spike sim:    {sep.get('within_spike_sim', 'N/A')}",
        f"  Within-nospike sim:  {sep.get('within_nospike_sim', 'N/A')}",
        f"  Between-class sim:   {sep.get('between_class_sim', 'N/A')}",
        f"  Separability ratio:  {sep.get('separability_ratio', 'N/A')}",
        "",
        "Upper Bound AUC (global-train ceiling):",
        f"  Mean UB AUC:              {mean_ub:.3f}",
        f"  Participants with UB<0.65: {low_ub}/{len(ub_df)}",
        "",
        f"Isolation (high = needs own labels): {high_iso}/{n_participants}",
        "",
        "=" * 60,
        "WHY THIS DATASET IS HARD FOR ACTIVE LEARNING",
        "=" * 60,
        "",
        "1. Label noise:",
        "   BP spike labels are self-reported via EMA.",
        "   Participants may misremember or misattribute spikes.",
        "   This creates an irreducible noise floor.",
        "",
        "2. Class imbalance:",
        f"  Mean spike rate = {mean_spike:.1%}.",
        "   Rare positive events make selection unstable.",
        "",
        "3. Low global-train upper bound:",
        f"  Pooled grand AUC = {pooled_auc:.3f}  AP = {pooled_ap:.3f}.",
        f"  Mean per-participant test UB = {mean_ub:.3f} ± {mean_ub_std:.3f}.",
        "   Even with aggregated training labels, performance is limited.",
        "   No selection strategy can exceed this ceiling.",
        "",
        "4. Participant heterogeneity:",
        f"  {high_iso}/{n_participants} participants are isolated in SSL space.",
        "   Collective labeling transfers poorly to isolated participants.",
        "",
        "Conclusion:",
        "   These properties make it impossible to demonstrate",
        "   active learning benefit in this dataset.",
        "   We validate the framework on WESAD where labels",
        "   are protocol-controlled and reliable.",
        "",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    (output_dir / "summary.txt").write_text(text)
    print(f"  Saved: {output_dir / 'summary.txt'}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    reset_seeds(args.seed)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BP SPIKE DATASET ANALYSIS")
    print(f"  pool:      {args.pool}")
    print(f"  input_df:  {args.input_df}")
    print(f"  output:    {out}")
    print("=" * 60)

    # ── Load all participant data once ────────────────────
    print("\n[1/6] Loading participant data...")
    from new_prep import _bp_load_all, _collect_processed_rows
    import re

    base = Path("DATA/Cardiomate/hp")
    shared_enc_root = out / "_global_encoders"
    candidate_dirs = sorted(base.glob("hp*")) if base.exists() else []

    all_splits, all_positives, all_negatives = {}, {}, {}
    all_signals = {}   # pid → (hr_df, st_df, pos_df, neg_df)
    bp_users = []

    for p in candidate_dirs:
        m = re.search(r"\d+", p.name)
        if not m:
            continue
        pid = m.group(0)
        try:
            hr_df, st_df, pos_df, neg_df = _bp_load_all(pid)
        except Exception as e:
            print(f"  Skipping {pid}: {e}")
            continue

        all_positives[pid]  = pos_df
        all_negatives[pid]  = neg_df
        all_signals[pid]    = (hr_df, st_df, pos_df, neg_df)

        from src.compare_pipelines import ensure_train_val_test_days
        try:
            tr_u, val_u, te_u = ensure_train_val_test_days(
                pos_df, neg_df, hr_df, st_df,
                input_df=args.input_df,
            )
            all_splits[pid] = (tr_u, val_u, te_u)
            bp_users.append(pid)
        except RuntimeError as e:
            print(f"  Skipping {pid}: {e}")

    print(f"  Found {len(bp_users)} valid participants: {bp_users}")

    # ── 1. Dataset overview ───────────────────────────────
    print("\n[2/6] Computing dataset overview...")
    overview = compute_dataset_overview(
        all_splits, all_positives, all_negatives
    )
    overview.to_csv(out / "dataset_overview.csv", index=False)
    print(overview.to_string(index=False))
    plot_spike_rates(overview, out)

    # ── 2. Collect representations ────────────────────────
    print("\n[3/6] Collecting SSL representations...")
    try:
        if args.input_df == "raw":
            print("  Loading global encoders for raw branch...")
            import uq_utility
            enc_hr, enc_st, _ = uq_utility._ensure_global_encoders(
                shared_enc_root,
                fruit="BP",
                scenario="spike",
                all_splits=all_splits,
                BATCH_SSL=32,
                SSL_EPOCHS=100,
                exclude_user_id=None,
                BP_MODE=True,
            )
            Z_all, y_all, pid_all = collect_all_representations(
                    bp_users, all_splits, args.input_df,
                    all_signals=all_signals, enc_hr=enc_hr, enc_st=enc_st,
                )
        else:
            Z_all, y_all, pid_all = collect_all_representations(
                    bp_users, all_splits, args.input_df,
                )
        print(f"  Z_all shape: {Z_all.shape}")
        print(f"  Spike rate:  {y_all.mean():.2%}")
        np.save(out / "Z_all.npy",   Z_all)
        np.save(out / "y_all.npy",   y_all)
        np.save(out / "pid_all.npy", pid_all)
    except RuntimeError as e:
        print(f"  Could not collect representations: {e}")
        Z_all = y_all = pid_all = None

    # ── 3. t-SNE ─────────────────────────────────────────
    if Z_all is not None:
        print("\n[4/6] Running t-SNE visualization...")
        plot_tsne(Z_all, y_all, pid_all, out, seed=args.seed)
        sep = class_separation_analysis(Z_all, y_all, out)
        iso_df = compute_isolation_scores(Z_all, pid_all)
        iso_df.to_csv(out / "isolation_scores.csv", index=False)
        plot_isolation(iso_df, out)
    else:
        sep    = {}
        iso_df = pd.DataFrame()

    # ── 4. Upper bound AUC ────────────────────────────────
    print("\n[5/6] Computing global-train upper bounds...")
    shared_enc_root = out / "_global_encoders"
    ub_df, pooled_auc, pooled_ap = compute_upper_bounds(
        bp_users, all_splits, args.input_df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        all_signals=all_signals if args.input_df == "raw" else None,
        shared_enc_root=shared_enc_root if args.input_df == "raw" else None,
        output_dir=out,
    )
    if not ub_df.empty:
        ub_df.to_csv(out / "upper_bounds.csv", index=False)
        with open(out / "pooled_grand_auc.json", "w") as f:
            json.dump({"pooled_auc": pooled_auc,
                       "pooled_ap":  pooled_ap}, f, indent=2)
        plot_upper_bounds(ub_df, pooled_auc, out)

    # ── 5. Summary ────────────────────────────────────────
    print("\n[6/6] Writing summary report...")
    write_summary_report(overview, ub_df, pooled_auc, pooled_ap,
                         sep, iso_df, out)

    print("\n" + "=" * 60)
    print(f"Analysis complete. All outputs in: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
