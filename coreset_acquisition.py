import numpy as np
from sklearn.preprocessing import normalize


def _cosine_min_dist_to_centers(Z_query, centers):
    Zn = normalize(Z_query)
    Cn = normalize(centers)
    sims = Zn @ Cn.T
    dists = 1.0 - sims
    return np.min(dists, axis=1)


def _knn_density(Z, k_neighbors=10):
    """Mean cosine similarity to k nearest neighbors. Higher = denser region."""
    Zn = normalize(Z)
    sims = Zn @ Zn.T
    np.fill_diagonal(sims, -1.0)
    k = min(k_neighbors, sims.shape[1] - 1)
    if k <= 0:
        return np.ones(len(Z))
    topk = np.partition(sims, -k, axis=1)[:, -k:]
    density = topk.mean(axis=1)
    rng = density.max() - density.min()
    if rng < 1e-12:
        return np.ones_like(density)
    return (density - density.min()) / rng


def coreset_kmeanspp(
    clf,
    df_tr_labeled,
    Z_tr_labeled,
    df_tr_unlabeled,
    Z_tr_unlabeled,
    K,
    rng_seed=None,
):
    """
    K-means++ flavored coreset acquisition.

    Vanilla k-center always picks the deterministic farthest point, which at
    larger K drifts toward outliers and exhausts the "extreme" points after a
    few picks (per-pick informational gain collapses).

    K-means++ instead samples each pick with probability proportional to
    squared distance from the labeled set. This:
      - still favors far points (preserves diversity),
      - does NOT lock onto a single deterministic outlier,
      - covers multiple high-distance regions instead of stacking near one,
      - degrades more gracefully as K grows.

    Same signature as `coreset_greedy` so it can be slotted in directly.
    """
    del clf, df_tr_labeled

    n_unlabeled = len(Z_tr_unlabeled)
    if n_unlabeled == 0:
        return [], df_tr_unlabeled.head(0).copy()

    rng = np.random.default_rng(rng_seed)
    k_actual = min(int(K), n_unlabeled)

    centers = Z_tr_labeled.copy()
    selected_positions = []
    available = np.ones(n_unlabeled, dtype=bool)

    for _ in range(k_actual):
        dists = _cosine_min_dist_to_centers(Z_tr_unlabeled, centers)
        dists = np.clip(dists, 0.0, None)
        dists[~available] = 0.0

        sq = dists ** 2
        total = sq.sum()
        if total <= 1e-12:
            remaining_idx = np.where(available)[0]
            if len(remaining_idx) == 0:
                break
            pos = int(rng.choice(remaining_idx))
        else:
            probs = sq / total
            pos = int(rng.choice(n_unlabeled, p=probs))

        selected_positions.append(pos)
        available[pos] = False
        centers = np.vstack([centers, Z_tr_unlabeled[pos]])

    queried_indices = df_tr_unlabeled.iloc[selected_positions].index.tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices].copy()
    return queried_indices, df_queried


def coreset_margin(clf, df_tr_labeled, Z_tr_labeled,
                   df_tr_unlabeled, Z_tr_unlabeled, K,
                   alpha=0.5, **kwargs):
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Z_tr_unlabeled)
        p = proba[:, 1]
    else:
        p = clf.predict(Z_tr_unlabeled, verbose=0).ravel()

    # if proba.ndim == 2 and proba.shape[1] == 2:
    #     p = proba[:, 1]
    # else:
    #     p = np.asarray(proba).ravel()
    uncertainty = 1.0 - 2.0 * np.abs(p - 0.5)

    centers = Z_tr_labeled.copy()
    selected = []
    masked = np.zeros(len(Z_tr_unlabeled), dtype=bool)

    for _ in range(min(int(K), len(Z_tr_unlabeled))):
        D = _cosine_min_dist_to_centers(Z_tr_unlabeled, centers)   # ← here
        score = alpha * uncertainty + (1.0 - alpha) * D
        score[masked] = -np.inf

        pos = int(np.argmax(score))
        selected.append(pos)
        masked[pos] = True
        centers = np.vstack([centers, Z_tr_unlabeled[pos]])

    queried_indices = df_tr_unlabeled.iloc[selected].index.tolist()
    return queried_indices, df_tr_unlabeled.loc[queried_indices].copy()

    
    

def coreset_adaptive(
    clf,
    df_tr_labeled,
    Z_tr_labeled,
    df_tr_unlabeled,
    Z_tr_unlabeled,
    K,
    M_modes=5,
    k_neighbors=10,
    rng_seed=None,
):
    """
    K-adaptive coreset acquisition.

    Combines three ingredients whose strengths automatically scale with K:
      - Distance D(x): updated after each pick (k-means++ style)
      - Density ρ(x): kNN density, suppresses outliers (precomputed once)
      - Temperature τ and density power p: scheduled by K

    Schedules:
        τ(K) = 1 + 8/K        # high τ → deterministic; low τ → spread
        p(K) = min(K/M, 1.5)  # high p → strong outlier penalty

    Resulting behavior:
        K=1..3   → near-deterministic, mild density penalty (vanilla coreset)
        K=4..5   → balanced (matches your K=4 winning regime)
        K=10+    → spread sampling + strong outlier penalty (fixes K=10)

    Same signature as coreset_greedy.
    """
    del clf, df_tr_labeled

    n_unlabeled = len(Z_tr_unlabeled)
    if n_unlabeled == 0:
        return [], df_tr_unlabeled.head(0).copy()

    rng = np.random.default_rng(rng_seed)
    k_actual = min(int(K), n_unlabeled)

    # K-dependent schedules.
    tau = 1.0 + 8.0 / max(int(K), 1)
    p = min(float(K) / float(max(M_modes, 1)), 1.5)

    # Precompute density once (depends only on unlabeled pool geometry).
    density = _knn_density(Z_tr_unlabeled, k_neighbors=k_neighbors)
    density_w = density ** p

    centers = Z_tr_labeled.copy()
    selected_positions = []
    available = np.ones(n_unlabeled, dtype=bool)

    for _ in range(k_actual):
        dists = _cosine_min_dist_to_centers(Z_tr_unlabeled, centers)
        dists = np.clip(dists, 0.0, None)

        score = dists * density_w
        score[~available] = 0.0

        weights = score ** tau
        total = weights.sum()
        if total <= 1e-12:
            remaining_idx = np.where(available)[0]
            if len(remaining_idx) == 0:
                break
            pos = int(rng.choice(remaining_idx))
        else:
            probs = weights / total
            pos = int(rng.choice(n_unlabeled, p=probs))

        selected_positions.append(pos)
        available[pos] = False
        centers = np.vstack([centers, Z_tr_unlabeled[pos]])

    queried_indices = df_tr_unlabeled.iloc[selected_positions].index.tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices].copy()
    return queried_indices, df_queried


def coreset_density_weighted(
    clf,
    df_tr_labeled,
    Z_tr_labeled,
    df_tr_unlabeled,
    Z_tr_unlabeled,
    K,
    density_power=1.0,
    k_neighbors=10,
):
    """
    Density-weighted greedy coreset.

    Standard k-center: pick argmax of min-distance-to-labeled.
    This variant: pick argmax of (min-distance-to-labeled) * (density)^p

    Outliers have low density (few neighbors), so they get downweighted even
    though they're geometrically "far." The selected points are far AND in
    a populated region — i.e., representative diverse picks rather than
    edge-of-space noise.

    `density_power` controls the trade-off:
      - 0.0 → vanilla k-center (no density weighting)
      - 1.0 → balanced
      - >1  → strongly avoid outliers, picks become more representative
    """
    del clf, df_tr_labeled

    n_unlabeled = len(Z_tr_unlabeled)
    if n_unlabeled == 0:
        return [], df_tr_unlabeled.head(0).copy()

    k_actual = min(int(K), n_unlabeled)

    density = _knn_density(Z_tr_unlabeled, k_neighbors=k_neighbors)
    density_w = density ** density_power

    centers = Z_tr_labeled.copy()
    selected_positions = []
    masked = np.zeros(n_unlabeled, dtype=bool)

    for _ in range(k_actual):
        dists = _cosine_min_dist_to_centers(Z_tr_unlabeled, centers)
        score = dists * density_w
        score[masked] = -np.inf

        pos = int(np.argmax(score))
        if not np.isfinite(score[pos]):
            break

        selected_positions.append(pos)
        masked[pos] = True
        centers = np.vstack([centers, Z_tr_unlabeled[pos]])

    queried_indices = df_tr_unlabeled.iloc[selected_positions].index.tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices].copy()
    return queried_indices, df_queried
