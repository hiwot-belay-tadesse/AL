"""
Helpers for refactor_run.py.
"""
import json
import os


os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf


import random
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

try:
    import seaborn as sns
except Exception:
    sns = None

from sklearn.utils import compute_class_weight
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from typing import Callable

import utility
from src import compare_pipelines as cp
from utility import (
    compute_representations,
    compute_user_stats,
    augment_labeled_windows,
    encode_single_df,
)
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
tf.random.set_global_generator(tf.random.Generator.from_seed(42)) 
# Independent RNG streams for sampling (MC dropout) and training.
sampling_rng = tf.random.Generator.from_seed(123)
training_rng = tf.random.Generator.from_seed(456)

# 
####Hyperparameters for global SSL model

WINDOW_SIZE = 30 

def reset_seeds(seed=42):
    import random, numpy as np, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def parse_args():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", default="fruit")
    pa.add_argument("--participant_id", default=None)
    pa.add_argument("--user", default="ID12")
    pa.add_argument("--pool", default="global_supervised", choices=["personal", "global", "global_supervised"])
    pa.add_argument("--fruit", default="Nectarine")
    pa.add_argument("--scenario", default="Crave")
    pa.add_argument("--sample_mode", default="original")
    pa.add_argument("--unlabeled_frac", default=0.0018)
    # pa.add_argument("--unlabeled_frac", default=0.1)

    pa.add_argument("--dropout_rate", default=0.5)
    pa.add_argument("--warm_start", default=0) ## 0 is retrain from scratch each round, 1 is finetune each round
    pa.add_argument("--results_subdir", default="results")
    pa.add_argument("--input_df", default="raw", choices=["raw", "processed"])
    return pa.parse_known_args()



def split_labeled_unlabeled_kmeans(
    df_tr_all,
    B,
    n_clusters=10,
    random_state=42,
    feature_cols=None,
):
    """
    Split full training dataframe into labeled and unlabeled sets using k-means.

    Args:
        df_tr_all: Full training dataframe.
        B: Label budget (number of samples to place in labeled set).
        n_clusters: Number of k-means clusters.
        random_state: Random seed for reproducibility.
        feature_cols: Optional list of numeric columns to use for clustering.
                      If None, numeric columns are auto-selected.

    Returns:
        df_tr_labeled, df_tr_unlabeled
    """
    if df_tr_all is None or len(df_tr_all) == 0:
        return df_tr_all.copy(), df_tr_all.copy()

    df_all = df_tr_all.copy()
    N = len(df_all)
    B = min(max(1, int(B)), N)

    if feature_cols is not None:
        X = df_all[feature_cols].to_numpy(dtype=float)
    else:
        excluded = {"state_val", "label", "target"}
        candidate_cols = [c for c in df_all.columns if c not in excluded]
        feat_df = pd.DataFrame(index=df_all.index)

        # 1) Use native numeric columns directly.
        numeric_cols = df_all[candidate_cols].select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if numeric_cols:
            feat_df = pd.concat([feat_df, df_all[numeric_cols].astype(float)], axis=1)

        # 2) Convert datetime columns to numeric timestamps.
        datetime_cols = df_all[candidate_cols].select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
        for col in datetime_cols:
            feat_df[f"{col}_ts"] = pd.to_datetime(df_all[col], errors="coerce").astype("int64") / 1e9

        # 3) For sequence/object cols (e.g., hr_seq, st_seq), derive compact stats.
        obj_cols = df_all[candidate_cols].select_dtypes(include=["object"]).columns.tolist()
        for col in obj_cols:
            series = df_all[col]
            first_valid = next((v for v in series if v is not None and not (isinstance(v, float) and np.isnan(v))), None)
            if isinstance(first_valid, (list, tuple, np.ndarray)):
                arrs = series.apply(lambda v: np.asarray(v, dtype=float) if isinstance(v, (list, tuple, np.ndarray)) else np.asarray([], dtype=float))
                feat_df[f"{col}_mean"] = arrs.apply(lambda a: float(np.nanmean(a)) if a.size else np.nan)
                feat_df[f"{col}_std"] = arrs.apply(lambda a: float(np.nanstd(a)) if a.size else np.nan)
                feat_df[f"{col}_min"] = arrs.apply(lambda a: float(np.nanmin(a)) if a.size else np.nan)
                feat_df[f"{col}_max"] = arrs.apply(lambda a: float(np.nanmax(a)) if a.size else np.nan)

        # Keep only finite/numeric columns and fill missing values.
        feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

        if feat_df.shape[1] == 0:
            raise ValueError(
                "No usable k-means features found in df_tr_all. "
                "Pass feature_cols explicitly."
            )

        # Median-impute column-wise; any all-NaN columns fall back to 0.
        med = feat_df.median(numeric_only=True)
        feat_df = feat_df.fillna(med).fillna(0.0)
        X = feat_df.to_numpy(dtype=float)

    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=min(int(n_clusters), N), random_state=random_state, n_init=10)
    cluster_id = km.fit_predict(Xs)
    centers = km.cluster_centers_

    counts = np.bincount(cluster_id, minlength=km.n_clusters)
    alloc = np.floor(B * counts / counts.sum()).astype(int)

    non_empty = np.where(counts > 0)[0]
    for c in non_empty:
        if alloc[c] == 0 and alloc.sum() < B:
            alloc[c] = 1

    while alloc.sum() < B:
        c = int(np.argmax(counts - alloc))
        alloc[c] += 1
    while alloc.sum() > B:
        c = int(np.argmax(alloc))
        alloc[c] -= 1

    labeled_pos = []
    for c in range(km.n_clusters):
        members = np.where(cluster_id == c)[0]
        if len(members) == 0 or alloc[c] == 0:
            continue
        d = np.linalg.norm(Xs[members] - centers[c], axis=1)
        chosen = members[np.argsort(d)[: alloc[c]]]
        labeled_pos.extend(chosen.tolist())

    labeled_pos = np.array(sorted(set(labeled_pos)), dtype=int)
    unlabeled_mask = np.ones(N, dtype=bool)
    unlabeled_mask[labeled_pos] = False

    df_tr_labeled = df_all.iloc[labeled_pos].copy()
    df_tr_unlabeled = df_all.iloc[np.where(unlabeled_mask)[0]].copy()
    return df_tr_labeled, df_tr_unlabeled



def k_means_clustering(Z_labeled, Z_unlabeled, K, max_iterations=100):
    
    centeroids = Z_labeled.copy()
    for i in range(max_iterations):
        distances = np.linalg.norm(Z_unlabeled[:, None, :] - centeroids[None, :, :], axis=2)
        closest_centroids = np.argmin(distances, axis=1)
        new_centroids = np.array([Z_unlabeled[closest_centroids == k].mean(axis=0)
                                  if np.any(closest_centroids == k) else centeroids[k] for k in range(K)])
    return new_centroids

def set_output_dir(pool, BP_MODE=None):
    if BP_MODE:
        if pool == "personal":
            return "Cardiomate_AL/P_SSL"
        if pool == "global":
        # return "Cardiomate_AL/G_SSL_all_val"
        # return "Cardiomate_AL/G_SSL_with_target_quota"
            return "Cardiomate_AL/G_SSL_augmented"
        if pool == "global_supervised":
           return "Cardiomate_AL/GS"
        if pool == "p_global_ssl":
           return "Cardiomate_AL/P_Global_SSL"
        raise ValueError(f"Unknown pool type: {pool}")
    else:
        if pool == "personal":
            return "Ban_AL/Personal"
        if pool == "global":
            return "Ban_AL/Global"
        if pool == "global_supervised":
            return "Ban_AL/Global_Supervised"
        raise ValueError(f"Unknown pool type: {pool}")


def pick_random(K, df_tr_unlabeled, seed=42):
    """Return K random indices from the unlabeled pool."""
    if len(df_tr_unlabeled) < K:
        K = len(df_tr_unlabeled)
    rng = random.Random(seed)
    queried_indices = rng.sample(list(df_tr_unlabeled.index), K)
    df_queried = df_tr_unlabeled.loc[queried_indices]
    return queried_indices, df_queried


def pick_most_uncertain(
    clf, df_tr_unlabeled, Z_tr_unlabeled, K, T, mc_predict, density_k: int = 10
):
    """MC-Dropout acquisition weighted by local density."""
    # Freeze BatchNorm for MC dropout
    for layer in clf.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    # Step 1: MC-Dropout uncertainty
    _, std_p, BALD = mc_predict(clf, Z_tr_unlabeled, T=T)

    if len(std_p) != len(df_tr_unlabeled):
        raise ValueError(
            f"std_p has {len(std_p)} entries but df_tr_unlabeled has {len(df_tr_unlabeled)} rows."
        )
    top_pos = np.argsort(std_p)[::-1][:K]
    queried_indices = df_tr_unlabeled.iloc[top_pos].index.tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices].copy()
    ## Revisit this 
    # Compute density weight from unlabeled feature geometry. ##
    # Higher weight for points in denser regions (smaller avg kNN distance).
    # z = np.asarray(Z_tr_unlabeled)
    # if z.ndim == 1:
    #     z = z.reshape(-1, 1)
    # if z.shape[0] < 3:
    #     density_weight = np.ones(z.shape[0], dtype=float)
    # else:
    #     k_eff = max(1, min(int(density_k), z.shape[0] - 1))
    #     nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    #     nn.fit(z)
    #     dists, _ = nn.kneighbors(z)
    #     # exclude self-distance at column 0
    #     mean_knn_dist = dists[:, 1:].mean(axis=1)
    #     density_raw = 1.0 / (mean_knn_dist + 1e-8)
    #     d_min, d_max = density_raw.min(), density_raw.max()
    #     if d_max > d_min:
    #         density_weight = (density_raw - d_min) / (d_max - d_min)
    #     else:
    #         density_weight = np.ones_like(density_raw)

    # Attach uncertainty and density to rows (keeping original indices).
    #Rank by uncertainty without adding a new column

    # df_unlb = df_tr_unlabeled.copy()
    # df_unlb["uncertainty"] = std_p
    # df_unlb["density_weight"] = density_weight
    # df_unlb["acquisition_score"] = df_unlb["uncertainty"] * df_unlb["density_weight"]

    # Sort high → low acquisition score
    # df_unlb = df_unlb.sort_values("acquisition_score", ascending=False)
    # Step 2: Select top‑K uncertain windows
    # queried_indices = df_unlb.head(K).index.tolist()
    # df_queried = df_unlb.loc[queried_indices]
    return queried_indices, df_queried

def coreset_greedy(
    clf, df_tr_labeled, Z_tr_labeled, df_tr_unlabeled, Z_tr_unlabeled, K
):
    """Core-Set greedy acquisition based on farthest point sampling."""
    # Z_tr_labeled:   (n_labeled, d)   
    # Z_tr_unlabeled: (n_unlabeled, d) 
    # K: number of points to query

    centers = Z_tr_labeled.copy()  # start with labeled points as centers
    n_unlabeled = len(Z_tr_unlabeled)
    if n_unlabeled == 0:
        return [], df_tr_unlabeled.head(0).copy()

    k_actual = min(int(K), n_unlabeled)
    selected_positions = []  # positional indices into Z_tr_unlabeled

    for _ in range(k_actual):
        # for each unlabeled point, find distance to nearest center
        dists = np.min(
            np.linalg.norm(
                Z_tr_unlabeled[:, None, :] - centers[None, :, :],
                axis=2
            ),
            axis=1
        )  # shape: (n_unlabeled,)

        # Prevent re-selecting the same unlabeled position within this round.
        if selected_positions:
            dists[np.array(selected_positions, dtype=int)] = -np.inf

        # pick the unlabeled point furthest from any center
        pos = np.argmax(dists)
        if not np.isfinite(dists[pos]):
            break
        selected_positions.append(pos)

        # add it to centers for next iteration
        centers = np.vstack([centers, Z_tr_unlabeled[pos]])

    # convert positional indices to dataframe index labels
    queried_indices = df_tr_unlabeled.iloc[selected_positions].index.tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices].copy()
    return queried_indices, df_queried


def kmeans_query_with_labeled_centroid(
    df_tr_labeled,
    Z_tr_labeled,
    df_tr_unlabeled,
    Z_tr_unlabeled,
    K,
    max_iter=50,
    tol=1e-6,
):
    """
    Query K unlabeled samples by k-means clustering on unlabeled embeddings.
    Initialization is anchored by the labeled-set centroid.

    Args:
        df_tr_labeled: DataFrame of labeled samples (used for shape/consistency only).
        Z_tr_labeled:  Labeled embeddings, shape (n_labeled, d).
        df_tr_unlabeled: DataFrame of unlabeled samples.
        Z_tr_unlabeled:  Unlabeled embeddings, shape (n_unlabeled, d).
        K: Number of unlabeled samples to query.
        max_iter: Max Lloyd iterations.
        tol: Convergence threshold on center movement.

    Returns:
        queried_indices: List of dataframe indices selected from df_tr_unlabeled.
        df_queried:      DataFrame slice of queried samples.
    """

    z_u = np.asarray(Z_tr_unlabeled, dtype=float)
    if z_u.ndim != 2:
        raise ValueError("Z_tr_unlabeled must be a 2D array (n_unlabeled, d).")
    n_unlabeled, dim = z_u.shape
    if n_unlabeled == 0:
        return [], df_tr_unlabeled.head(0).copy()

    k_actual = min(int(K), n_unlabeled)
    if k_actual <= 0:
        return [], df_tr_unlabeled.head(0).copy()

    # Anchor initialization with labeled centroid when labeled points exist.
    z_l = np.asarray(Z_tr_labeled, dtype=float) if Z_tr_labeled is not None else None
    if z_l is not None and z_l.size > 0:
        if z_l.ndim != 2 or z_l.shape[1] != dim:
            raise ValueError("Z_tr_labeled must be shape (n_labeled, d) with matching d.")
        labeled_centroid = z_l.mean(axis=0)
    else:
        labeled_centroid = z_u.mean(axis=0)

    # Deterministic seeded initialization:
    # first center = labeled centroid, others = farthest-first from existing centers.
    centers = [labeled_centroid]
    while len(centers) < k_actual:
        c = np.vstack(centers)
        dists = np.min(np.linalg.norm(z_u[:, None, :] - c[None, :, :], axis=2), axis=1)
        next_idx = int(np.argmax(dists))
        centers.append(z_u[next_idx])
    centers = np.vstack(centers)

    # Lloyd's algorithm on unlabeled pool.
    for _ in range(max_iter):
        dmat = np.linalg.norm(z_u[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dmat, axis=1)
        new_centers = centers.copy()
        for j in range(k_actual):
            members = z_u[labels == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift <= tol:
            break

    # Pick one representative unlabeled point per cluster center.
    chosen_pos = []
    used = set()
    for j in range(k_actual):
        order = np.argsort(np.linalg.norm(z_u - centers[j], axis=1))
        for pos in order:
            p = int(pos)
            if p not in used:
                used.add(p)
                chosen_pos.append(p)
                break

    queried_indices = df_tr_unlabeled.iloc[chosen_pos].index.tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices].copy()
    return queried_indices, df_queried




def plot_queried_windows(df_queried, round_num, results_d, max_n: int = 24):
    if df_queried is None or len(df_queried) == 0:
        return
    if "hr_seq" not in df_queried.columns or "st_seq" not in df_queried.columns:
        return

    try:
        from matplotlib import pyplot as plt
    except Exception:
        return

    out_dir = Path(results_d) / "queried_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    q = df_queried
    n = len(q)
    ncols = 4
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)

    for i, (_, row) in enumerate(q.iterrows()):
        ax = axes[i // ncols][i % ncols]
        hr = np.asarray(row["hr_seq"], dtype=float)
        st = np.asarray(row["st_seq"], dtype=float)
        ax.plot(hr, lw=1.6, label="HR")
        ax.plot(st, lw=1.6, label="Steps")
        uid = row.get("user_id", "?")
        y = row.get("state_val", "?")
        ax.set_title(f"user={uid} y={y}")
        ax.grid(alpha=0.25)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"Queried Windows - Round {int(round_num)}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"queried_windows_round_{int(round_num):02d}.png", dpi=160)
    plt.close(fig)


def plot_feature_space_tsne(
    Z_tr_labeled,
    Z_tr_unlabeled,
    df_tr_labeled,
    df_tr_unlabeled,
    round_num,
    results_d,
):
    if (
        Z_tr_labeled is None
        or Z_tr_unlabeled is None
        or df_tr_labeled is None
        or df_tr_unlabeled is None
    ):
        return
    n_labeled = int(len(Z_tr_labeled))
    n_unlabeled = int(len(Z_tr_unlabeled))
    n_total = n_labeled + n_unlabeled
    if n_total < 3:
        return

    try:
        from matplotlib import pyplot as plt
    except Exception:
        return

    Z_lab = np.asarray(Z_tr_labeled, dtype=np.float32)
    Z_unlab = np.asarray(Z_tr_unlabeled, dtype=np.float32)
    Z_all = np.vstack([Z_lab, Z_unlab])

    perplexity = min(30.0, max(2.0, float((n_total - 1) // 3)))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    emb = tsne.fit_transform(Z_all)

    out_dir = Path(results_d) / "tsne_feature_space"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        emb[:n_labeled, 0],
        emb[:n_labeled, 1],
        c="red",
        s=18,
        alpha=0.75,
        label=f"Labeled (n={n_labeled})",
    )
    ax.scatter(
        emb[n_labeled:, 0],
        emb[n_labeled:, 1],
        c="blue",
        s=18,
        alpha=0.55,
        label=f"Unlabeled (n={n_unlabeled})",
    )
    ax.set_title(f"t-SNE Feature Space - Round {int(round_num)}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    labeled_user_ids = (
        df_tr_labeled["user_id"].astype(str).tolist()
        if "user_id" in df_tr_labeled.columns
        else ["?"] * n_labeled
    )
    unlabeled_user_ids = (
        df_tr_unlabeled["user_id"].astype(str).tolist()
        if "user_id" in df_tr_unlabeled.columns
        else ["?"] * n_unlabeled
    )

    for i, uid in enumerate(labeled_user_ids):
        ax.annotate(uid, (emb[i, 0], emb[i, 1]), fontsize=6, alpha=0.8)
    for j, uid in enumerate(unlabeled_user_ids):
        k = n_labeled + j
        ax.annotate(uid, (emb[k, 0], emb[k, 1]), fontsize=6, alpha=0.65)

    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_round_{int(round_num):02d}.png", dpi=180)

    plt.close(fig)


def compute_budget(pool, df_tr, df_all_tr, uf_val, k_val):
    if pool == "personal":
        N = len(df_tr)
    elif pool in ["global", "global_supervised"]:
        N = len(df_all_tr)
    else:
        raise ValueError(f"Unknown pool type: {pool}")
    budget = int(np.ceil(uf_val * N / k_val))
    return max(1, budget)


def pick_most_uncertain_ensemble(
    df_tr_labeled,
    X_tr_labeled,
    y_tr_labeled,
    df_tr_unlabeled,
    X_tr_unlabeled,
    K,
    X_val=None,
    y_val=None,
    n_models: int = 5,
    seeds=None,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    class_weight=None,
    callbacks=None,
    verbose: int = 0,
):
    """
    Ensemble-based acquisition equivalent to pick_most_uncertain.

    Trains an MLP ensemble on current labeled set and uses predictive
    variance on unlabeled samples as uncertainty.

    Returns:
      queried_indices, df_queried
    """
    if len(df_tr_unlabeled) == 0:
        return [], df_tr_unlabeled.copy()

    ens = train_mlp_ensemble(
        X_train=X_tr_labeled,
        y_train=y_tr_labeled,
        X_val=X_val,
        y_val=y_val,
        X_eval=X_tr_unlabeled,
        n_models=n_models,
        seeds=seeds,
        lr=1e-4, ##reduced lr for ensemble training
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )

    uncertainty = ens["pred_var"]
    if len(uncertainty) != len(df_tr_unlabeled):
        raise ValueError(
            f"uncertainty has {len(uncertainty)} entries but df_tr_unlabeled has {len(df_tr_unlabeled)} rows."
        )

    df_unlb = df_tr_unlabeled.copy()
    df_unlb["uncertainty"] = uncertainty
    df_unlb = df_unlb.sort_values("uncertainty", ascending=False)

    k_actual = min(int(K), len(df_unlb))
    queried_indices = df_unlb.head(k_actual).index.tolist()
    df_queried = df_unlb.loc[queried_indices]
    return queried_indices, df_queried


def build_hp_folder(uf_val, k_val, budget, t_val, dr_val):
    hp = [
        f"UF{float(uf_val * 100)}",
        f"K{k_val}",
        f"B{budget}",
    ]
    if t_val is not None:
        hp.append(f"T{int(t_val)}")
    hp.append(f"DR{int(dr_val * 100)}")
    return "_".join(hp)


def write_summary(exp_dir, user, pool, fruit, scenario, uf_val, dr_val, k_val, budget, t_val):
    summary = {
        "user": user,
        "pool": pool,
        "fruit": fruit,
        "scenario": scenario,
        "dropout_rate": dr_val,
        "unlabeled_frac": uf_val,
        "T": t_val,
        "K": k_val,
        "Budget": budget,
    }
    summary_path = os.path.join(exp_dir, "exp_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved exp_summary.json to:", summary_path)


def mc_predict(model, x, T, mc_seed=42):
    """
    Runs MC dropout on the final dropout layer of the classifier.
    """
    tf.random.set_global_generator(tf.random.Generator.from_seed(mc_seed))

    ## Snapshot/restore state so MC dropout does not mutate active model state (e.g. BN stats).
    orig_weights = model.get_weights()
    # reset_seeds(mc_seed)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    
    preds = []
    
    for t in range(T):
        # Run the ENTIRE model with training=True to enable dropout
        logits = model(x, training=True)
        probs = tf.math.sigmoid(logits)
        preds.append(tf.reshape(probs, [-1]).numpy())
    
    preds = np.stack(preds, axis=0)
    mean_probs = preds.mean(axis=0)
    std_probs = preds.std(axis=0)
    entropy = - (mean_probs * np.log(mean_probs + 1e-8)
             + (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))
    # Entropy per MC sample
    H_per_t = - (preds * np.log(preds + 1e-8)
            + (1 - preds) * np.log(1 - preds + 1e-8))  # shape (T, N)

    H_exp = H_per_t.mean(axis=0)  # expected entropy

    BALD = entropy - H_exp         # epistemic uncertainty estimate
    
    # Restore original model state after MC sampling.
    model.set_weights(orig_weights)

    return mean_probs, std_probs, BALD


def build_classifier(input_dim, CLF_PATIENCE, dropout_rate, seed):
    
    # 7) Train classifier
    reset_seeds(42)  # Ensure deterministic model initialization
    clf = Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(), layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(dropout_rate),
        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.01)), layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    

    clf.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss='binary_crossentropy', metrics=['accuracy'])

    es  = EarlyStopping(monitor='val_loss', patience=CLF_PATIENCE,
                                restore_best_weights=True, verbose=1)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    # cw_vals     = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    # class_weight = {i: cw_vals[i] for i in range(len(cw_vals))}

    return clf, [es, lr_cb] 
# def build_classifier(input_dim, CLF_PATIENCE, dropout_rate, seed):
#     # init = tf.keras.initializers.GlorotUniform(seed=seed)
#     inputs = tf.keras.Input(shape=(input_dim,), name="input")
#     # x = layers.Dense(128, activation="relu",
#     #                  kernel_regularizer=l2(1e-3))(inputs)
#     x = layers.Dense(64, activation="relu",
#                      #kernel_regularizer=l2(1e-2),
#                      kernel_initializer=0.01)(inputs)
#     ##New:
#     x = layers.BatchNormalization()(x) 
#     x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    
#     # x = layers.Dense(64, activation="relu",
#     #                  kernel_regularizer=l2(1e-3))(x)
#     x = layers.Dense(32, activation="relu",
#                      #kernel_regularizer=l2(1e-2), 
#                      kernel_initializer=0.01)(x)

#     x = layers.Dropout(dropout_rate, name="dropout_2")(x)

#     # ##New layer
#     # x = layers.Dense(32, activation="relu",
#     #                  kernel_regularizer=l2(1e-3))(x)
#     x = layers.Dense(16, activation="relu",
#                      #kernel_regularizer=l2(1e-2)
#                      )(x)
#     outputs = layers.Dense(1,  activation=None,
#                         #    activation="sigmoid"
#                            )(x)


#     model = tf.keras.Model(inputs, outputs)
#     model.compile(
#         # optimizer=Adam(1e-3),
#         optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-3),
#         # loss="binary_crossentropy",
#         loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
#         metrics=["accuracy"]
#         #  metrics=[tf.keras.metrics.AUC(name="auc")]
#     )

#     es = EarlyStopping(
#         monitor="val_loss",
#         # monitor="val_auc",
#         patience=CLF_PATIENCE,
#         restore_best_weights=True,
#         verbose=0
#     )
#     return model, es


def run_experiment(exp_dir, exp_name, exp_kwargs, args, prep, clf_epochs, clf_patience, df_tr_labeled, df_tr_unlabeled):
    df_tr, df_all_tr, df_val, df_te, enc_hr, enc_st, user_root, all_splits, models_d, results_d, all_negatives = prep
    ## sorting for determinisim 
    # df_tr = df_tr.sort_index()

    if df_tr is None or df_val is None or df_te is None:
        print(f"Skipping user {args.user}: missing train/val/test splits.")
        return None
    if len(df_tr) == 0 or len(df_val) == 0 or len(df_te) == 0:
        print(f"Skipping user {args.user}: empty train/val/test split(s).")
        return None
    if args.pool in ["global", "global_supervised"] and (df_all_tr is None or len(df_all_tr) == 0):
        print(f"Skipping user {args.user}: empty pooled training set.")
        return None

    warm_start = bool(int(args.warm_start))
    
    K = exp_kwargs.get("K")
    T = exp_kwargs.get("T")
    budget = exp_kwargs.get("Budget")
    method = exp_kwargs.get("aq", exp_name)
    split_seed = int(exp_kwargs.get("seed", 42))
    # Labeled/unlabeled split is provided by caller.

    fit_kwargs = dict(epochs=clf_epochs, batch_size=16, verbose=0)
    #Commenting this out for now to handle the processed branching
    if args.pool in ["personal", "global"]:
        Z_val_hr, Z_val_st = utility.encode(df_val, enc_hr, enc_st)
        Z_val = np.concatenate([Z_val_hr, Z_val_st], axis=1).astype("float32")
        y_val = df_val["state_val"].values.astype("float32")
    elif args.pool == "global_supervised":
        if args.input_df != "processed":
            Z_val, y_val = build_XY_from_windows(df_val)
        else:
            Z_val, y_val, _, _, _ = build_XY_from_processed(df_val, fit=True)


    if Z_val is not None and len(Z_val) > 0:
        fit_kwargs["validation_data"] = (Z_val, y_val)


    df_tr_labeled_m = df_tr_labeled.copy()
    df_tr_unlabeled_m = df_tr_unlabeled.copy()
    

    reset_seeds(split_seed)

    (
        al_progress,
        active_model,
        df_tr_labeled_final,
        df_tr_unlabeled_final,
        queried_indices,
        count_df,
        round_labeled_history,
        round_eval_payloads,
    ) = run_al_refactored(
        exp_name,
        df_tr_labeled_m,
        df_tr_unlabeled_m,
        df_val,
        df_te,
        enc_hr,
        enc_st,
        build_classifier,
        mc_predict,
        K,
        budget,
        T if method != "random" else None,
        CLF_PATIENCE=clf_patience,
        dropout_rate=float(args.dropout_rate),
        fit_kwargs=fit_kwargs,
        pool=args.pool,
        models_d=models_d,
        results_d=results_d,
        active_model=None,
        init_weights_trained=None,
        warm_start=warm_start,
        seed=split_seed,
        input_df=args.input_df,
    )

    print('len of final labeled', len(df_tr_labeled_final))
    print('len of final unlabeled', len(df_tr_unlabeled_final))
    method_dir = Path(exp_dir) / method if exp_name == "combined" else Path(exp_dir)
    method_dir.mkdir(parents=True, exist_ok=True)
    auc_cols = [
        "round",
        "Num_Labeled",
        "Total_Data",
        "Pct_Total_Labeled",
        "AUC_Mean_Train",
        "AUC_STD_Train",
        "AUC_Mean_Val",
        "AUC_STD_Val",
        "AUC_Mean",
        "AUC_STD",
    ]
    keep_cols = [c for c in auc_cols if c in al_progress.columns]
    al_progress_min = al_progress[keep_cols].copy() if keep_cols else al_progress.copy()
    al_progress_min.to_csv(os.path.join(method_dir, "al_progress.csv"), index=False)

    upper_bound_path = method_dir / "upper_bound_auc.npy"
    full_data_eval_payload = None
    # if not upper_bound_path.exists():
    if warm_start:
        df_full = (
            pd.concat([df_tr_labeled_final, df_tr_unlabeled_final], axis=0)
            if df_tr_unlabeled_final is not None
            else df_tr_labeled_final
        )

        if args.pool in ["personal", "global"]:
            Z_full = encode_single_df(df_full, enc_hr, enc_st, args.pool)
            y_full = df_full["state_val"].values.astype("float32")
            Z_val_full = encode_single_df(df_val, enc_hr, enc_st, args.pool)
            y_val_full = df_val["state_val"].values.astype("float32")
            Z_te = encode_single_df(df_te, enc_hr, enc_st, args.pool)
            y_te = df_te["state_val"].values.astype("float32")
            auc_m, _, _, _, _, _, _, full_model = utility.fit_and_eval(
                fit_kwargs,
                build_classifier,
                Z_full.shape[1],
                Z_full,
                y_full,
                Z_val_full,
                y_val_full,
                Z_te,
                y_te,
                clf_patience,
                float(args.dropout_rate),
                clf=None,
            )
            upper_bound_auc = auc_m
            full_probs_te = full_model.predict(Z_te, verbose=0).ravel()
            full_data_eval_payload = {
                "y_true": np.asarray(y_te).ravel(),
                "y_score": np.asarray(full_probs_te).ravel(),
            }
        else:
            X_full, y_full = build_XY_from_windows(df_full)
            Z_val_full, y_val_full = build_XY_from_windows(df_val)
            Z_te, y_te = build_XY_from_windows(df_te)

            class_weight = None
            classes = np.unique(y_full)
            if len(classes) == 2:
                cw_vals = compute_class_weight("balanced", classes=classes, y=y_full)
                class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

            ceiling_model = build_global_cnn_lstm(dropout_rate=float(args.dropout_rate))
            ceiling_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            reset_seeds(42)
            ceiling_model.fit(
                X_full,
                y_full,
                validation_data=(Z_val_full, y_val_full),
                epochs=50,
                batch_size=32,
                class_weight=class_weight,
                verbose=0,
                shuffle=True,
            )
            full_probs_te = ceiling_model.predict(Z_te, verbose=0).ravel()
            upper_bound_auc, _, _ = bootstrap_auc(y_te, full_probs_te)
            full_data_eval_payload = {
                "y_true": np.asarray(y_te).ravel(),
                "y_score": np.asarray(full_probs_te).ravel(),
            }
    else:
        Z_final = encode_single_df(df_all_tr, enc_hr, enc_st, args.pool)
        y_final = df_all_tr["state_val"].values.astype("float32")
        Z_val_final, y_val_final = encode_single_df(df_val, enc_hr, enc_st, args.pool), df_val["state_val"].values.astype("float32")

    
        es = EarlyStopping(
            monitor="val_loss",
            patience=clf_patience,
            restore_best_weights=True,
            verbose=1,
        )
        lr_cb = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        )
        final_model = build_classifier(
            input_dim=Z_final.shape[1],
            CLF_PATIENCE=clf_patience,
            dropout_rate=float(args.dropout_rate),
            seed=split_seed,
        )[0]
        final_model.fit(
        Z_final,
        y_final,
        validation_data=(Z_val_final, y_val_final),
        callbacks=[es, lr_cb],
        epochs=200,
        batch_size=16,
        verbose=0,
        shuffle=True,
        

    )
        Z_te_final, y_te_final = encode_single_df(df_te, enc_hr, enc_st, args.pool), df_te["state_val"].values.astype("float32")

        full_probs_te = final_model.predict(Z_te_final, verbose=0).ravel()
        upper_bound_auc, _, _ = bootstrap_auc(y_te_final, full_probs_te)
        full_data_eval_payload = {
            "y_true": np.asarray(y_te_final).ravel(),
            "y_score": np.asarray(full_probs_te).ravel(),
        }
        # upper_bound_auc = (
        #     float(al_progress.iloc[-1]["AUC_Mean"])
        #     if not al_progress.empty
        #     else float("nan")
        # )
    np.save(upper_bound_path, upper_bound_auc)

    with open(os.path.join(method_dir, f"queried_participant_counts_{args.user}_{args.pool}.pkl"), "wb") as f:
        pickle.dump(count_df, f)
    return {
        "labeled_len": len(df_tr_labeled_final),
        "unlabeled_len": len(df_tr_unlabeled_final) if df_tr_unlabeled_final is not None else 0,
        "al_progress": al_progress_min.copy(),
        "round_labeled_history": round_labeled_history,
        "round_eval_payloads": round_eval_payloads,
        "full_data_eval_payload": full_data_eval_payload,
        "df_val": df_val.copy(),
        "df_test": df_te.copy(),
    }

def build_mlp_classifier(input_dim, lr=1e-3):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu", kernel_regularizer=l2(1e-3)),
        # layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        # layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model




def train_mlp_ensemble(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_eval=None,
    n_models: int = 10,
    seeds=None,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 64,
    class_weight=None,
    callbacks=None,
    verbose: int = 0,
):
    """
    Train an ensemble of MLPs and return prediction-variance statistics.

    Returns a dict with per-sample predictive mean/variance and summary stats.
    """
    if seeds is None:
        seeds = list(range(42, 42 + n_models))
    if len(seeds) < n_models:
        raise ValueError("len(seeds) must be >= n_models")

    if X_eval is None:
        X_eval = X_val if X_val is not None else X_train

    all_eval_probs = []
    train_losses = []
    val_losses = []
    trained_models = []
    
    tf.keras.backend.clear_session()
    for i in range(n_models):
        seed = int(seeds[i])
        reset_seeds(seed)
        

        model = build_mlp_classifier(input_dim=X_train.shape[1], lr=lr)
        
        model_callbacks = None
        if callbacks is not None:
            model_callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,  
                    verbose=0
                ), 
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-5,
                    verbose=0
                )
            ]
        
        fit_kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": verbose,
            "shuffle": True,
            "callbacks":  model_callbacks, 
        }
        if class_weight is not None:
            fit_kwargs["class_weight"] = class_weight
        # if callbacks is not None:
        #     fit_kwargs["callbacks"] = model_callbacks
        #     # fit_kwargs["callbacks"] = callbacks
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_kwargs["validation_data"] = (X_val, y_val)

        hist = model.fit(X_train, y_train, **fit_kwargs)
        if "loss" in hist.history and len(hist.history["loss"]) > 0:
            train_losses.append(float(hist.history["loss"][-1]))
        if "val_loss" in hist.history and len(hist.history["val_loss"]) > 0:
            val_losses.append(float(hist.history["val_loss"][-1]))

        eval_probs = model.predict(X_eval, verbose=0).ravel()
        all_eval_probs.append(eval_probs)
        trained_models.append(model)

    probs = np.stack(all_eval_probs, axis=0)
    pred_mean = probs.mean(axis=0)
    pred_var = probs.var(axis=0)
    pred_std = np.sqrt(pred_var)

    return {
        "models": trained_models,
        "pred_mean": pred_mean,
        "pred_var": pred_var,
        "pred_std": pred_std,
        "ensemble_variance_mean": float(pred_var.mean()),
        "ensemble_variance_median": float(np.median(pred_var)),
        "train_loss_mean": float(np.mean(train_losses)) if train_losses else np.nan,
        "train_loss_std": float(np.std(train_losses)) if train_losses else np.nan,
        "val_loss_mean": float(np.mean(val_losses)) if val_losses else np.nan,
        "val_loss_std": float(np.std(val_losses)) if val_losses else np.nan,
        "n_models": int(n_models),
    }


def predict_mlp_ensemble(models, X, verbose: int = 0):
    """
    Average probabilities from a list of trained MLP models.
    """
    if not models:
        raise ValueError("models list is empty")

    probs = np.stack([m.predict(X, verbose=verbose).ravel() for m in models], axis=0)
    return probs.mean(axis=0)


def build_global_cnn_lstm(dropout_rate):
    inp = layers.Input(shape=(WINDOW_SIZE, 2))

    x = layers.Conv1D(64, 8, padding='same', activation='relu',
                      kernel_regularizer=l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(4, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 64))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(128, 5, padding='same', activation='relu',
                      kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(8, activation='relu')(se)
    se = layers.Dense(128, activation='sigmoid')(se)
    se = layers.Reshape((1, 128))(se)
    x = layers.Multiply()([x, se])

    x = layers.Conv1D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    cnn_out = layers.GlobalAveragePooling1D()(x)

    lstm_out = layers.LSTM(128)(inp)
    lstm_out = layers.Dropout(dropout_rate)(lstm_out)

    combined = layers.concatenate([cnn_out, lstm_out])
    out = layers.Dense(1, activation='sigmoid')(combined)

    model = Model(inp, out)
    return model

def build_XY_from_windows(
    df: pd.DataFrame,
):
    """
    Build X (N, window_size, 2) and y (N,) from df with columns hr_seq, st_seq, state_val.
    """
    
    X = np.stack([
    np.vstack([h, s]).T
    for h, s in zip(df["hr_seq"], df["st_seq"])
])

    y = df["state_val"].values.astype(int)

    return X, y



def _processed_feature_cols(df: pd.DataFrame):
    # Match compare_pipelines processed-feature extraction baseline.
    base_drop = {"id", "user_id", "reading_id", "device_type", "data_type_hr", "data_type_steps"}
    cols = [c for c in df.columns if c not in base_drop]
    # Never allow target columns in model input.
    cols = [c for c in cols if c not in {"BP_spike", "state_val"}]
    return cols


def build_XY_from_processed(
    df: pd.DataFrame,
    feature_cols=None,
    train_median: pd.Series | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = False,
):
    df = df.copy()
    if "datetime_local" in df.columns:
        df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
        df = df.sort_values("datetime_local").reset_index(drop=True)

    if "BP_spike" in df.columns:
        y = df["BP_spike"].values.astype("float32")
    elif "state_val" in df.columns:
        y = df["state_val"].values.astype("float32")
    else:
        raise ValueError("processed df needs 'BP_spike' or 'state_val'")

    if feature_cols is None:
        feature_cols = _processed_feature_cols(df)
        # force model to learn physiological signal
        # harder task, more room for AL to demonstrate value
        drop_lag = [c for c in feature_cols
                    if 'lag' in c.lower()
                    or 'spike' in c.lower()
                    or 'time_since' in c.lower()
                    or 'systolic' in c.lower()
                    or 'diastolic' in c.lower()
                    or 'datetime' in c.lower()
                    or 'datetime_local' in c.lower()
                    or 'time' in c.lower()
                    or 'created_at' in c.lower()]

        feature_cols = [c for c in feature_cols if c not in drop_lag]
        print(f"Features without lags: {feature_cols}")

    X_df = df[feature_cols].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    if fit:
        train_median = X_df.median(numeric_only=True)
    if train_median is None:
        train_median = X_df.median(numeric_only=True)

    X_df = X_df.fillna(train_median).fillna(0.0)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df.values).astype("float32")
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X = scaler.transform(X_df.values).astype("float32")

    return X, y, feature_cols, train_median, scaler

def check_model_initialization(model, method_name, round_num):
    # Sum up all weights in the model
    total_weight_sum = sum([np.sum(w) for w in model.get_weights()])
    # Get the value of the very first weight in the first layer
    first_weight_sample = model.get_weights()[0].flatten()[0]
    
    print(f"\n>>> [INIT CHECK] Method: {method_name} | Round: {round_num}")
    print(f">>> Total Weight Sum: {total_weight_sum:.10f}")
    print(f">>> First Weight Sample: {first_weight_sample:.10f}")
    return total_weight_sum

def initialize_active_model(
    pool,
    Z_tr_labeled,
    y_tr_labeled,
    df_tr_labeled,
    Z_val,
    y_val,
    build_classifier,
    CLF_PATIENCE,
    dropout_rate,
    fit_kwargs,
    active_model,
    warm_start,
    round_num,
    seed,
    input_df,
):
    if pool in ["personal", "global"]:
        
        
        # tf.keras.backend.clear_session()
        # tf.random.set_global_generator(tf.random.Generator.from_seed(42))
        # reset_seeds(42)
        input_dim = Z_tr_labeled.shape[1]
        clf, [es, lr_cb ] = build_classifier(input_dim, CLF_PATIENCE, dropout_rate, seed)
        fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}
        # if "shuffle" not in fit_kwargs_with_callbacks:
        #     fit_kwargs_with_callbacks["shuffle"] = True
        # if "callbacks" in fit_kwargs_with_callbacks:
        #     callbacks = fit_kwargs_with_callbacks["callbacks"]
        #     if not isinstance(callbacks, list):
        #         callbacks = [callbacks]
        #     callbacks = callbacks + [es, lr_cb]
        #     fit_kwargs_with_callbacks["callbacks"] = callbacks
        # else:
        #     fit_kwargs_with_callbacks["callbacks"] = [es, lr_cb]

        unique_classes = np.unique(y_tr_labeled)
        cw_vals = compute_class_weight("balanced", classes=unique_classes, y=y_tr_labeled)
        class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
        # total_weights = check_model_initialization(clf, pool, round_num)
        # reset_seeds(42)
        # tf.random.set_global_generator(tf.random.Generator.from_seed(42))
        # reset_seeds(42)
        clf.fit(
            Z_tr_labeled,
            y_tr_labeled,
            # class_weight=class_weight,
            **fit_kwargs_with_callbacks,
        )
        return clf

    elif pool == "global_supervised":
        if warm_start:
            if active_model is None:
                if round_num > 0:
                    raise RuntimeError("warm_start=True requires an initial active_model.")
            else:
                return active_model
            
        if input_df != "processed":
             X_init, y_init = build_XY_from_windows(df_tr_labeled)
        else:
            X_init, y_init,  _, _, _ = build_XY_from_processed(df_tr_labeled, fit=True)

             
        class_weight = None
        classes = np.unique(y_init)
        if len(classes) == 2:
            cw_vals = compute_class_weight("balanced", classes=classes, y=y_init)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

        model = build_mlp_classifier(X_init.shape[1], lr=1e-4)  
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
        lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #     loss="binary_crossentropy",
        #     metrics=["accuracy"],
        # )
        


        reset_seeds(42)

        model.fit(
            X_init,
            y_init,
            validation_data=(Z_val, y_val),
            epochs=200,
            batch_size=32,
            class_weight=class_weight,
            verbose=0,
            shuffle=True,
            callbacks=[lr_cb,es]
        )

        return model
    else: 
        raise ValueError(f"Unknown pool type: {pool}")


def pre_al_metrics(model, Z_tr_labeled, y_tr_labeled, Z_val, y_val, Z_te, y_te):
    
    probs_train = model.predict(Z_tr_labeled, verbose=0).ravel()
    probs_val = model.predict(Z_val, verbose=0).ravel()
    probs_te = model.predict(Z_te, verbose=0).ravel()
    
    auc_m_pre, auc_s_pre, _ = bootstrap_auc(y_te, probs_te)
    auc_m_train_pre, auc_s_train_pre, _ = bootstrap_auc(y_tr_labeled, probs_train)
    auc_m_val_pre, auc_s_val_pre, _ = bootstrap_auc(y_val, probs_val)
    
    m = pd.DataFrame(index=[0])
    m["round"] = 0
    m["AUC_Mean_Train"] = auc_m_train_pre
    m["AUC_STD_Train"] = auc_s_train_pre
    m["AUC_Mean_Val"] = auc_m_val_pre
    m["AUC_STD_Val"] = auc_s_val_pre
    m["AUC_Mean"] = auc_m_pre
    m["AUC_STD"] = auc_s_pre
    eval_payload = {
        "train": {"y_true": np.asarray(y_tr_labeled).ravel(), "y_score": np.asarray(probs_train).ravel()},
        "val": {"y_true": np.asarray(y_val).ravel(), "y_score": np.asarray(probs_val).ravel()},
        "test": {"y_true": np.asarray(y_te).ravel(), "y_score": np.asarray(probs_te).ravel()},
    }
    return m, eval_payload


def _coerce_eval_split_payload(split_payload):
    if not isinstance(split_payload, dict):
        return None, None
    y_true = split_payload.get("y_true")
    y_score = split_payload.get("y_score")
    if y_true is None or y_score is None:
        return None, None
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        return None, None
    return y_true, y_score


def _build_eval_payload(y_tr_labeled, probs_train, y_val, probs_val, y_te, probs_te):
    return {
        "train": {"y_true": np.asarray(y_tr_labeled).ravel(), "y_score": np.asarray(probs_train).ravel()},
        "val": {"y_true": np.asarray(y_val).ravel(), "y_score": np.asarray(probs_val).ravel()},
        "test": {"y_true": np.asarray(y_te).ravel(), "y_score": np.asarray(probs_te).ravel()},
    }


def run_al_refactored(
    Aq,
    df_tr_labeled,
    df_tr_unlabeled,
    df_val,
    df_te,
    enc_hr,
    enc_st,
    build_classifier,
    mc_predict,
    K,
    budget,
    T,
    CLF_PATIENCE,
    dropout_rate,
    fit_kwargs,
    pool,
    models_d,
    results_d,
    active_model=None,
    init_weights_trained=None,
    warm_start: bool = True,
    seed: int = 42,
    input_df: str = "processed",
):
    """
    Refactored Active Learning Loop - Main Entry Point.

    """
    # acquisition utilities use index labels for select/drop.
    # Non-unique labels can duplicate selections and inflate labeled counts.
    if not df_tr_labeled.index.is_unique:
        print("Warning: non-unique labeled index detected. Resetting index.")
        df_tr_labeled = df_tr_labeled.reset_index(drop=True)
    if not df_tr_unlabeled.index.is_unique:
        print("Warning: non-unique unlabeled index detected. Resetting index.")
        df_tr_unlabeled = df_tr_unlabeled.reset_index(drop=True)

    if input_df == "processed":
        Z_tr_labeled, y_tr_labeled, feature_cols, train_median, scaler = build_XY_from_processed(
            df_tr_labeled, fit=True
        )
        Z_tr_unlabeled, y_tr_unlabeled, _, _, _ = build_XY_from_processed(
            df_tr_unlabeled,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,
        )
        Z_val, y_val, _, _, _ = build_XY_from_processed(
            df_val,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,
        )
        Z_te, y_te, _, _, _ = build_XY_from_processed(
            df_te,
            feature_cols=feature_cols,
            train_median=train_median,
            scaler=scaler,
            fit=False,
        )

        # corr = pd.DataFrame(Z_tr_labeled, columns=feature_cols).corr().abs()
        # upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
        # drop_idx = [feature_cols.index(f) for f in to_drop if f in feature_cols]
        # feature_cols_low_corr = [f for f in feature_cols if f not in to_drop]

        # if drop_idx:
        #     Z_tr_labeled = np.delete(Z_tr_labeled, drop_idx, axis=1)
        #     Z_tr_unlabeled = np.delete(Z_tr_unlabeled, drop_idx, axis=1)
        #     Z_val = np.delete(Z_val, drop_idx, axis=1)
        #     Z_te = np.delete(Z_te, drop_idx, axis=1)

        # print(f"Dropping {len(to_drop)} features due to high correlation: {to_drop}")
        # print(feature_cols_low_corr)

        # try:
        #     from matplotlib import pyplot as plt
        #     plt.figure(figsize=(12, 10))
        #     if sns is not None:
        #         sns.heatmap(corr, cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
        #     else:
        #         plt.imshow(corr.values, cmap='coolwarm', aspect='auto')
        #         plt.colorbar()
        #     plt.title('Feature correlation matrix')
        #     plt.tight_layout()
        #     Path(results_d).mkdir(parents=True, exist_ok=True)
        #     plt.savefig(Path(results_d) / 'feature_correlation_matrix.png')
        #     plt.close()
        # except Exception as e:
        #     print(f"Skipping correlation heatmap save: {e}")
    else:
        Z_tr_labeled, y_tr_labeled, Z_tr_unlabeled, y_tr_unlabeled, Z_val, y_val, Z_te, y_te = compute_representations(
            df_tr_labeled, df_tr_unlabeled, df_val, df_te, enc_hr, enc_st, pool
        )
        y_tr_labeled = df_tr_labeled["state_val"].values.astype("float32")

    results = []
    queried_all = []
    queried_participants = {}
    participants_count_per_round = {}
    round_labeled_history = {0: df_tr_labeled.copy()}
    total_data = int(len(df_tr_labeled) + len(df_tr_unlabeled))
    df_counts_wide = pd.DataFrame()
    

    active_model = initialize_active_model(
        pool,
        Z_tr_labeled,
        y_tr_labeled,
        df_tr_labeled,
        Z_val,
        y_val,
        build_classifier,
        CLF_PATIENCE,
        dropout_rate,
        fit_kwargs,
        active_model,
        warm_start,
        0,
        seed=42,
        input_df=input_df,
    )
    m0, eval_payload_0 = pre_al_metrics(active_model, Z_tr_labeled, y_tr_labeled, Z_val, y_val, Z_te, y_te)
    m0["Num_Labeled"] = int(len(df_tr_labeled))
    m0["Total_Data"] = total_data
    m0["Pct_Total_Labeled"] = (
        100.0 * float(m0["Num_Labeled"].iloc[0]) / float(total_data)
        if total_data > 0
        else np.nan
    )
    results.append(m0)
    round_eval_payloads = {0: eval_payload_0}

    initial_unlabeled = len(df_tr_unlabeled)
    # planned_rounds = int(np.ceil(initial_unlabeled / K)) if K and initial_unlabeled > 0 else 0
    planned_rounds = 20
    round_num = 0

    while len(df_tr_unlabeled) > 0 and round_num < planned_rounds:
        round_num += 1
        print(f"\n{'='*70}")
        print(f"AL Round {round_num}/{planned_rounds if planned_rounds > 0 else '?'}")
        print(f"  Labeled: {len(df_tr_labeled)}")
        print(f"  Unlabeled: {len(df_tr_unlabeled)}")
        print(f"{'='*70}")

        k_actual = min(K, len(df_tr_unlabeled))
        if input_df == "processed":
            Z_tr_unlabeled, y_tr_unlabeled, _, _, _ = build_XY_from_processed(
                df_tr_unlabeled,
                feature_cols=feature_cols,
                train_median=train_median,
                scaler=scaler,
                fit=False,
            )
            # if 'drop_idx' in locals() and drop_idx:
            #     Z_tr_unlabeled = np.delete(Z_tr_unlabeled, drop_idx, axis=1)
        else:
            Z_tr_unlabeled = encode_single_df(df_tr_unlabeled, enc_hr, enc_st, pool)

        # try:
        #     plot_feature_space_tsne(
        #         Z_tr_labeled,
        #         Z_tr_unlabeled,
        #         df_tr_labeled,
        #         df_tr_unlabeled,
        #         round_num,
        #         results_d,
        #     )
        # except Exception as e:
        #     print(f"Skipping t-SNE feature-space plot for round {round_num}: {e}")

        if Aq == "random":
            queried_indices, df_queried = pick_random(k_actual, df_tr_unlabeled, seed=42+round_num)
            # pos_query = len(df_tr_labeled[df_tr_labeled["state_val"] == 1])
            # neg_query = len(df_tr_labeled[df_tr_labeled["state_val"] == 0])
            # target_ratio = pos_query/(pos_query + neg_query) if (pos_query + neg_query) > 0 else 0
            # df_queried = augment_labeled_windows(df_queried_original, target_ratio=target_ratio)

        elif Aq == "uncertainty":
            queried_indices, df_queried = pick_most_uncertain(
                active_model, df_tr_unlabeled, Z_tr_unlabeled, k_actual, T, mc_predict
            )
        
            df_queried_original = df_queried.copy()
            # queried_indices, df_queried = pick_most_uncertain_ensemble(
            #     df_tr_labeled=df_tr_labeled,
            #     X_tr_labeled=Z_tr_labeled,
            #     y_tr_labeled=y_tr_labeled,
            #     df_tr_unlabeled=df_tr_unlabeled,
            #     X_tr_unlabeled=Z_tr_unlabeled,
            #     K=k_actual,
            #     X_val=Z_val,
            #     y_val=y_val
            # )
        elif Aq == "coreset":
            # queried_indices, df_queried = coreset_greedy(
            #     active_model, df_tr_labeled, Z_tr_labeled, df_tr_unlabeled, Z_tr_unlabeled, k_actual
            # )
            # pos_query = len(df_tr_labeled[df_tr_labeled["state_val"] == 1])
            # neg_query = len(df_tr_labeled[df_tr_labeled["state_val"] == 0])
            # target_ratio = pos_query/(pos_query + neg_query) if (pos_query + neg_query) > 0 else 0

            queried_indices, df_queried = coreset_greedy(
                active_model, df_tr_labeled, Z_tr_labeled, df_tr_unlabeled, Z_tr_unlabeled, k_actual
            )
            
            # user_stats = compute_user_stats(pd.concat([df_tr_labeled, df_tr_unlabeled], axis=0)) 
            # df_queried = augment_labeled_windows(df_queried_original, target_ratio=target_ratio)
                                            
            
            # breakpoint()
        elif Aq == "kmeans":
            queried_indices, df_queried = kmeans_query_with_labeled_centroid(
                df_tr_labeled, Z_tr_labeled, df_tr_unlabeled, Z_tr_unlabeled, k_actual
            )
            df_queried_original = df_queried.copy()
        else:
            raise ValueError(
                f"Unknown acquisition function: {Aq}. Must be 'random', 'uncertainty', 'coreset', or 'mixed'."
            )

        if len(queried_indices) == 0:
            print("No samples were queried in this round. Stopping to avoid infinite loop.")
            break

        queried_all.extend(queried_indices)
        try:
            plot_queried_windows(df_queried_original, round_num, results_d)
        except Exception as e:
            print(f"Skipping queried-window plot for round {round_num}: {e}")
        queried_participants_per_round = df_queried["user_id"].tolist()
        queried_participants[round_num] = queried_participants_per_round
        participants_count_per_round[round_num] = Counter(queried_participants_per_round)

        df_tr_labeled = pd.concat([df_tr_labeled, df_queried], axis=0)
        df_tr_unlabeled = df_tr_unlabeled.drop(index=queried_indices)
        round_labeled_history[int(round_num)] = df_tr_labeled.copy()

        
        ## sorting so last round match for random and uncertainty
        #df_tr_labeled = df_tr_labeled.sort_index()

        ##Sort only on true final round (before querying next round)
        if len(df_tr_unlabeled) <= K:
            df_tr_labeled = df_tr_labeled.sort_index()

        if input_df == "processed":
            Z_tr_labeled, y_tr_labeled, _, _, _ = build_XY_from_processed(
                df_tr_labeled,
                feature_cols=feature_cols,
                train_median=train_median,
                scaler=scaler,
                fit=False,
            )
            # if 'drop_idx' in locals() and drop_idx:
            #     Z_tr_labeled = np.delete(Z_tr_labeled, drop_idx, axis=1)
        else:
            Z_tr_labeled = encode_single_df(df_tr_labeled, enc_hr, enc_st, pool)
            y_tr_labeled = df_tr_labeled["state_val"].values.astype("float32")
        # Restore RNG state before training so both methods are equivalent


        use_class_weight = True
        (
            auc_m_post,
            auc_s_post,
            auc_m_val_post,
            auc_s_val_post,
            auc_m_train_post,
            auc_s_train_post,
            active_model,
            eval_payload,
        ) = train_and_evaluate_by_pool(
            pool=pool,
            df_tr_labeled=df_tr_labeled,
            Z_tr_labeled=Z_tr_labeled,
            y_tr_labeled=y_tr_labeled,
            Z_val=Z_val,
            y_val=y_val,
            Z_te=Z_te,
            y_te=y_te,
            build_classifier=build_classifier,
            CLF_PATIENCE=CLF_PATIENCE,
            dropout_rate=dropout_rate,
            fit_kwargs=fit_kwargs,
            warm_start=warm_start,
            active_model=active_model,
            seed=seed,
            method=Aq,
            round_num=round_num,
            last_round_only=True,
            total_rounds=planned_rounds,
            shuffle_seed=99,
            use_early_stopping=True,
            use_class_weight=use_class_weight,
            input_df=input_df,
        )
        round_eval_payloads[int(round_num)] = eval_payload
        metrics = pd.DataFrame(index=[0])
        metrics["round"] = round_num
        metrics["length_labeled"] = len(df_tr_labeled)
        metrics["AUC_Mean_Train"] = auc_m_train_post
        metrics["AUC_STD_Train"] = auc_s_train_post
        metrics["AUC_Mean_Val"] = auc_m_val_post
        metrics["AUC_STD_Val"] = auc_s_val_post
        metrics["AUC_Mean"] = auc_m_post
        metrics["AUC_STD"] = auc_s_post
        metrics["Num_Labeled"] = int(len(df_tr_labeled))
        metrics["Total_Data"] = total_data
        metrics["Pct_Total_Labeled"] = (
            100.0 * float(len(df_tr_labeled)) / float(total_data)
            if total_data > 0
            else np.nan
        )
        results.append(metrics)

        df_counts_wide = (
            pd.DataFrame.from_dict(participants_count_per_round, orient="index")
            .fillna(0)
            .astype(int)
        )
        df_counts_wide.index.name = "round"
        df_counts_wide.columns.name = "user_id"

        print(f"  → Round {round_num}: AUC={auc_m_post:.3f} ± {auc_s_post:.3f}")

    al_progress = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    return (
        al_progress,
        active_model,
        df_tr_labeled,
        df_tr_unlabeled,
        queried_all,
        df_counts_wide,
        round_labeled_history,
        round_eval_payloads,
    )


def build_fit_kwargs(fit_kwargs, es, use_early_stopping: bool = True):
    fit_kwargs_with_callbacks = fit_kwargs.copy() if fit_kwargs else {}
    # if "shuffle" not in fit_kwargs_with_callbacks:
        # Keep batch order fixed for reproducible round-to-round retraining.
        # fit_kwargs_with_callbacks["shuffle"] = True
    if use_early_stopping:
        if "callbacks" in fit_kwargs_with_callbacks:
            callbacks = fit_kwargs_with_callbacks["callbacks"]
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            callbacks = callbacks + [es]
            fit_kwargs_with_callbacks["callbacks"] = callbacks
        else:
            fit_kwargs_with_callbacks["callbacks"] = [es]
    else:
        fit_kwargs_with_callbacks.pop("callbacks", None)
    return fit_kwargs_with_callbacks


def compute_class_weight_map(y_lab):
    unique_classes = np.unique(y_lab)
    cw_vals = compute_class_weight("balanced", classes=unique_classes, y=y_lab)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    if len(class_weight) < 2:
        print(f"Warning: Only one class present in y_lab. Class weights: {class_weight}")
    return class_weight

def bootstrap_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sample_frac: float = 0.7,
    n_iters: int = 1000,
    rng_seed: int = 42,
    return_samples: bool = False,
) -> tuple:
    """
    Compute AUC with bootstrap confidence intervals.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred_proba : np.ndarray
        Predicted probabilities
    sample_frac : float, default=0.7
        Fraction of samples to use in each bootstrap iteration
    n_iters : int, default=1000
        Number of bootstrap iterations
    rng_seed : int, default=42
        Random seed for reproducibility
    return_samples : bool, default=False
        If True, return all bootstrap AUC samples
    
    Returns
    -------
    auc_mean : float
        Mean AUC across bootstrap samples
    auc_std : float
        Standard deviation of AUC across bootstrap samples
    auc_samples : np.ndarray (optional)
        All bootstrap AUC scores if return_samples=True
    
    """
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    k = int(np.round(sample_frac * n))
    idx_all = np.arange(n)
    
    aucs = []
    
    for _ in range(n_iters):
        # Sample with replacement
        sidx = rng.choice(idx_all, size=k, replace=True)
        y_samp = y_true[sidx]
        p_samp = y_pred_proba[sidx]
        
        # AUC only valid with both classes present
        if len(np.unique(y_samp)) > 1:
            aucs.append(roc_auc_score(y_samp, p_samp))
    
    auc_samples = np.array(aucs)
    auc_mean = np.nanmean(auc_samples)
    auc_std = np.nanstd(auc_samples, ddof=1)
    
    if return_samples:
        return auc_mean, auc_std, auc_samples
    
    valid_frac = len(auc_samples) / n_iters
    return auc_mean, auc_std, valid_frac



def _safe_auc_from_scores(y_true, y_score):
    """
    Compute ROC-AUC safely; returns NaN when AUC is undefined.
    """
    if y_true is None or y_score is None:
        return np.nan
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _bootstrap_auc_safe(y_true, y_score, rng_seed=42):
    """
    Safe wrapper around bootstrap_auc: returns (nan, nan) when inputs are invalid.
    """
    if y_true is None or y_score is None:
        return np.nan, np.nan
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        return np.nan, np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    auc_m, auc_s, _ = bootstrap_auc(y_true, y_score, rng_seed=rng_seed)
    return float(auc_m), float(auc_s)



def _normalize_user_df_map(user_df_map):
    """
    Normalize either dict[user_id -> df] or list[dict(user_id, df)] into dict form.
    """
    if user_df_map is None:
        return {}
    if isinstance(user_df_map, dict):
        return user_df_map
    if isinstance(user_df_map, list):
        out = {}
        for item in user_df_map:
            if isinstance(item, dict) and "user_id" in item and "df" in item:
                out[item["user_id"]] = item["df"]
        return out
    raise TypeError("Expected dict[user_id -> df] or list of {'user_id','df'} records.")


def aggregate_per_round_labeled_and_compute_auc(
    per_user_al_progress,
    per_user_round_eval=None,
    per_user_full_data_eval=None,
):
    """
    Aggregate per-user AL progress by round.

    AUC columns are computed
    only from pooled raw predictions/labels .
    """
    if not isinstance(per_user_al_progress, dict):
        raise TypeError("per_user_al_progress must be dict[user_id -> DataFrame].")

    metric_cols = [
        "AUC_Mean_Train",
        "AUC_STD_Train",
        "AUC_Mean_Val",
        "AUC_STD_Val",
        "AUC_Mean",
        "AUC_STD",
        "Num_Labeled",
        "Total_Data",
        "Pct_Total_Labeled",
    ]

    dfs = []
    for uid, df in per_user_al_progress.items():
        if df is None or len(df) == 0 or "round" not in df.columns:
            continue
        dfx = df.copy()
        dfx["user_id"] = uid
        keep_cols = ["user_id", "round"] + [c for c in metric_cols if c in dfx.columns]
        dfs.append(dfx[keep_cols])

    if not dfs:
        return {"auc_per_round": pd.DataFrame(columns=["round"] + metric_cols)}

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    auc_metric_cols = [
        "AUC_Mean_Train",
        "AUC_STD_Train",
        "AUC_Mean_Val",
        "AUC_STD_Val",
        "AUC_Mean",
        "AUC_STD",
    ]
    non_auc_metric_cols = [c for c in metric_cols if c not in auc_metric_cols]
    numeric_cols = [c for c in non_auc_metric_cols if c in combined.columns]
    combined["round"] = pd.to_numeric(combined["round"], errors="coerce")
    combined = combined.dropna(subset=["round"])
    combined["round"] = combined["round"].astype(int)

    if numeric_cols:
        auc_per_round = (
            combined.groupby("round", as_index=False)[numeric_cols]
            .mean(numeric_only=True)
            .sort_values("round")
            .reset_index(drop=True)
        )
    else:
        auc_per_round = (
            combined[["round"]]
            .drop_duplicates()
            .sort_values("round")
            .reset_index(drop=True)
        )

    per_user_round_eval = per_user_round_eval if isinstance(per_user_round_eval, dict) else {}

    pooled_metric_map = {
        "train": ("AUC_Mean_Train", "AUC_STD_Train"),
        "val": ("AUC_Mean_Val", "AUC_STD_Val"),
        "test": ("AUC_Mean", "AUC_STD"),
    }
    pooled_by_round = {}

    for round_payloads in per_user_round_eval.values():
        if not isinstance(round_payloads, dict):
            continue
        for round_key, eval_payload in round_payloads.items():
            try:
                round_num = int(round_key)
            except (TypeError, ValueError):
                continue
            pooled_by_round.setdefault(round_num, {split: {"y_true": [], "y_score": []} for split in pooled_metric_map})
            if not isinstance(eval_payload, dict):
                continue
            for split_name in pooled_metric_map:
                y_true, y_score = _coerce_eval_split_payload(eval_payload.get(split_name))
                if y_true is None:
                    continue
                pooled_by_round[round_num][split_name]["y_true"].append(y_true)
                pooled_by_round[round_num][split_name]["y_score"].append(y_score)

    pooled_rows = []
    for round_num in sorted(auc_per_round["round"].tolist()):
        row = {"round": round_num}
        for split_name, (mean_col, std_col) in pooled_metric_map.items():
            split_payload = pooled_by_round.get(round_num, {}).get(split_name, {"y_true": [], "y_score": []})
            if split_payload["y_true"] and split_payload["y_score"]:
                y_true = np.concatenate(split_payload["y_true"])
                y_score = np.concatenate(split_payload["y_score"])
                auc_m, auc_s = _bootstrap_auc_safe(y_true, y_score)
            else:
                auc_m, auc_s = np.nan, np.nan
            row[mean_col] = auc_m
            row[std_col] = auc_s
        pooled_rows.append(row)

    pooled_auc_df = pd.DataFrame(pooled_rows)
    auc_per_round = auc_per_round.merge(pooled_auc_df, on="round", how="left")

    per_user_full_data_eval = per_user_full_data_eval if isinstance(per_user_full_data_eval, dict) else {}
    full_y_true = []
    full_y_score = []
    total_data_vals = []

    for uid, payload in per_user_full_data_eval.items():
        y_true, y_score = _coerce_eval_split_payload(payload)
        if y_true is not None:
            full_y_true.append(y_true)
            full_y_score.append(y_score)
        user_df = per_user_al_progress.get(uid)
        if isinstance(user_df, pd.DataFrame) and not user_df.empty and "Total_Data" in user_df.columns:
            td = pd.to_numeric(user_df["Total_Data"], errors="coerce").dropna()
            if not td.empty:
                total_data_vals.append(float(td.iloc[0]))

    full_data_auc = None
    if full_y_true and full_y_score:
        pooled_y_true = np.concatenate(full_y_true)
        pooled_y_score = np.concatenate(full_y_score)
        full_auc_mean, full_auc_std = _bootstrap_auc_safe(pooled_y_true, pooled_y_score)
        total_data = float(np.nansum(total_data_vals)) if total_data_vals else float(len(pooled_y_true))
        full_data_auc = {
            "round": None,
            "Num_Labeled": total_data,
            "Total_Data": total_data,
            "Pct_Total_Labeled": 100.0,
            "AUC_Mean_Train": np.nan,
            "AUC_STD_Train": np.nan,
            "AUC_Mean_Val": np.nan,
            "AUC_STD_Val": np.nan,
            "AUC_Mean": full_auc_mean,
            "AUC_STD": full_auc_std,
        }

    return {"auc_per_round": auc_per_round, "full_data_auc": full_data_auc}


def train_and_evaluate_by_pool(
    pool,
    df_tr_labeled,
    Z_tr_labeled,
    y_tr_labeled,
    Z_val,
    y_val,
    Z_te,
    y_te,
    build_classifier,
    CLF_PATIENCE,
    dropout_rate,
    fit_kwargs,
    warm_start,
    active_model=None,
    seed: int = 42,
    method: str = None,
    round_num: int = 0,
    last_round_only: bool = False,
    total_rounds: int = 0,
    shuffle_seed:int=99,
    use_early_stopping: bool = True,
    use_class_weight: bool = True,
    input_df: str = "processed",
):
    """
    Unified train+eval for both MLP and global-supervised CNN.
    Returns: auc_mean, auc_std, auc_m_val, auc_s_val, auc_m_train, auc_s_train, model
    """
    if pool in ["personal", "global"]:

        if Z_tr_labeled is None or len(Z_tr_labeled) == 0:
            raise ValueError("Empty training set in train_and_evaluate_by_pool.")

        assert Z_tr_labeled.shape[0] > 0, f"Empty Z_tr_labeled: {Z_tr_labeled.shape}"
        assert y_tr_labeled.shape[0] > 0, f"Empty y_tr_labeled: {y_tr_labeled.shape}"
        assert Z_tr_labeled.shape[0] == y_tr_labeled.shape[0], (
            f"Mismatch Z_tr_labeled {Z_tr_labeled.shape} vs y_tr_labeled {y_tr_labeled.shape}"
        )

        # rs = shuffle_seed + int(round_num)
        # rng = np.random.default_rng(rs)
        # perm = rng.permutation(Z_tr_labeled.shape[0])
        # Z_tr_labeled = Z_tr_labeled[perm]
        # y_tr_labeled = y_tr_labeled[perm]

        tf.keras.backend.clear_session()
        tf.random.set_global_generator(tf.random.Generator.from_seed(42))
        reset_seeds(42)
        
        clf, es = build_classifier(Z_tr_labeled.shape[1], CLF_PATIENCE, dropout_rate, seed)
        # class_weight = compute_class_weight_map(y_tr_labeled) if use_class_weight else {0: 1.0, 1: 1.0}
        fit_kwargs_with_callbacks = build_fit_kwargs(fit_kwargs, es, use_early_stopping=use_early_stopping)
       
        print(f"First weight after build: {clf.get_weights()[0].flatten()[0]:.10f}")

        print("\n=== Class Weights Check ===")
        print(
            f"Training class distribution: Positive (1): {(y_tr_labeled == 1).sum()}, "
            f"Negative (0): {(y_tr_labeled == 0).sum()}"
        )
        
        tf.random.set_global_generator(tf.random.Generator.from_seed(42))
        reset_seeds(42)
        # clf.fit(Z_tr_labeled, y_tr_labeled, class_weight=class_weight, **fit_kwargs_with_callbacks)
        # clf.fit(Z_tr_labeled, y_tr_labeled, class_weight=class_weight, shuffle=True, **fit_kwargs_with_callbacks)
        
        # rng = np.random.default_rng(42)
        # perm = rng.permutation(len(Z_tr_labeled))
        # Z_tr_labeled = Z_tr_labeled[perm]
        # y_tr_labeled = y_tr_labeled[perm]
        # clf.fit(Z_tr_labeled, y_tr_labeled, class_weight=class_weight, shuffle=True, **fit_kwargs_with_callbacks) ## to match 10_pct_eval
        clf.fit(Z_tr_labeled, y_tr_labeled, **fit_kwargs_with_callbacks)

        print(f"First weight after fit: {clf.get_weights()[0].flatten()[0]:.10f}")
        # 
        all_weights = np.concatenate([w.flatten() for w in clf.get_weights()])
        print(f"Method: {method}, Weight sum: {all_weights.sum():.10f}, norm: {np.linalg.norm(all_weights):.10f}")
        # clf.summary()

        
        # 
        # Save weights per method to compare
        # np.save(f"weights_{method}.npy", all_weights)
        
        probs_train = clf.predict(Z_tr_labeled, verbose=0).ravel()

        probs_te = clf.predict(Z_te, verbose=0).ravel()
        probs_val = clf.predict(Z_val, verbose=0).ravel()

        # val_auc = roc_auc_score(y_val, probs_val)
        # print(f"Validation AUC = {val_auc:.3f}")
        # print(f"Test size: {len(y_te)}")
        print(
            f"Test probs: min={probs_te.min():.3f}, "
            f"max={probs_te.max():.3f}, mean={probs_te.mean():.3f}"
        )
        if len(np.unique(y_te)) == 2:
            direct_auc = roc_auc_score(y_te, probs_te)
            flipped_auc = roc_auc_score(y_te, 1 - probs_te)
            print(f"Direct Test AUC: {direct_auc:.3f}")
            print(f"Flipped Test AUC: {flipped_auc:.3f}")


        auc_m_post, auc_s_post, _ = bootstrap_auc(y_te, probs_te)
        auc_m_train_post, auc_s_train_post, _ = bootstrap_auc(y_tr_labeled, probs_train)
        auc_m_val_post, auc_s_val_post, _ = bootstrap_auc(y_val, probs_val)
        eval_payload = _build_eval_payload(
            y_tr_labeled=y_tr_labeled,
            probs_train=probs_train,
            y_val=y_val,
            probs_val=probs_val,
            y_te=y_te,
            probs_te=probs_te,
        )
        # 
        return (
            auc_m_post,
            auc_s_post,
            auc_m_val_post,
            auc_s_val_post,
            auc_m_train_post,
            auc_s_train_post,
            clf,
            eval_payload,
        )

    elif pool == "global_supervised":

        ##trains from scratch
        if not warm_start:
            if input_df == "processed":
                # Keep processed tensors aligned with pruning done in run_al_refactored.
                X_init, y_init = Z_tr_labeled, y_tr_labeled
            else:
                X_init, y_init = build_XY_from_windows(df_tr_labeled)

            class_weight = None
            classes = np.unique(y_init)
            if len(classes) == 2:
                cw_vals = compute_class_weight("balanced", classes=classes, y=y_init)
                class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
            active_model = build_mlp_classifier(X_init.shape[1], lr=1e-3)
            es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
            lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1)
            # active_model = build_global_cnn_lstm(dropout_rate=dropout_rate)
            # active_model.compile(
            #     # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            #     loss="binary_crossentropy",
            #     metrics=["accuracy"],
            # )
            
            reset_seeds(42)
            active_model.fit(
                X_init,
                y_init,
                validation_data=(Z_val, y_val),
                epochs=200,
                batch_size=32,
                class_weight=class_weight,
                verbose=0,
                shuffle=True,
                callbacks=[lr_cb, es]
            )

        #     ##Note the below is the changes for the ensemble
        #     ens = train_mlp_ensemble(
        #         X_train=X_init,
        #         y_train=y_init,
        #         X_val=Z_val,
        #         y_val=y_val,
        #         X_eval=Z_te,          # evaluate on test directly
        #         n_models=10,
        #         seeds=list(range(      # round-dependent seeds — no fixed seed
        #             round_num * 10,
        #             round_num * 10 + 10
        #         )),
        #         class_weight=class_weight,
        #         callbacks=[es, lr_cb],
        #         epochs=200,
        #         batch_size=32,
        #     )

        #     ## variance of the ensemble 
        #     var_per_sample = ens["pred_var"]
        #     std_per_sample = ens["pred_std"]
            
        #     ##Top uncertain samples 
        #     top_idx = np.argsort(-var_per_sample)[:20]  # top 5 most uncertain samples
        #     print(top_idx, var_per_sample[top_idx])
        #     # 
        #     # ── AUC from ensemble mean ─────────────────────────
        #     probs_te  = np.stack([
        #         m.predict(Z_te, verbose=0).ravel()
        #         for m in ens["models"]
        #     ]).mean(axis=0)

        #     probs_val = np.stack([
        #         m.predict(Z_val, verbose=0).ravel()
        #         for m in ens["models"]
        #     ]).mean(axis=0)

        #     probs_tr  = np.stack([
        #         m.predict(Z_tr_labeled,  verbose=0).ravel()
        #         for m in ens["models"]
        #     ]).mean(axis=0)

        #     # keep active_model as the best single model from ensemble
        #     # (lowest val loss) — used for warm_start continuity if needed
        #     best_idx    = np.argmin([
        #         m.evaluate(Z_val, y_val, verbose=0)[0]
        #         for m in ens["models"]
        #     ])
        #     active_model = ens["models"][best_idx]


        # ##warm start
        # else:
        #     if active_model is None:
        #         raise RuntimeError("warm_start=True requires an initial active_model.")
        #     if input_df == "processed":
        #         X_lab, y_lab = Z_tr_labeled, y_tr_labeled
        #     else:
        #         X_lab, y_lab = build_XY_from_windows(df_tr_labeled)
        #     class_weight = None
        #     classes = np.unique(y_lab)
        #     if len(classes) == 2:
        #         cw_vals = compute_class_weight("balanced", classes=classes, y=y_lab)
        #         class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}
        #     active_model.compile(
        #         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        #         loss="binary_crossentropy",
        #         metrics=["accuracy"],
        #     )
        #     reset_seeds(42)
        #     active_model.fit(
        #         X_lab,
        #         y_lab,
        #         validation_data=(Z_val, y_val),
        #         epochs=50,
        #         batch_size=32,
        #         class_weight=class_weight,
        #         verbose=0,
        #         shuffle=True,
        #     )
        # auc_m_post,     auc_s_post,     _ = bootstrap_auc(y_te,           probs_te)
        # auc_m_val_post, auc_s_val_post, _ = bootstrap_auc(y_val,          probs_val)
        # auc_m_train_post,  auc_s_train_post,  _ = bootstrap_auc(y_tr_labeled,   probs_tr)
        probs_te = active_model.predict(Z_te, verbose=0).ravel()
        probs_train = active_model.predict(Z_tr_labeled, verbose=0).ravel()
        probs_val = active_model.predict(Z_val, verbose=0).ravel()
        auc_m_post, auc_s_post, _ = bootstrap_auc(y_te, probs_te)
        auc_m_train_post, auc_s_train_post, _ = bootstrap_auc(y_tr_labeled, probs_train)
        auc_m_val_post, auc_s_val_post, _ = bootstrap_auc(y_val, probs_val)
        eval_payload = _build_eval_payload(
            y_tr_labeled=y_tr_labeled,
            probs_train=probs_train,
            y_val=y_val,
            probs_val=probs_val,
            y_te=y_te,
            probs_te=probs_te,
        )
        return (
            auc_m_post,
            auc_s_post,
            auc_m_val_post,
            auc_s_val_post,
            auc_m_train_post,
            auc_s_train_post,
            active_model,
            eval_payload,
        )

    else:
       raise ValueError(f"Unknown pool type: {pool}")
