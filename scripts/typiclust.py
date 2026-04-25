import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def pick_typiclust(
    df_tr_labeled: pd.DataFrame,
    df_tr_unlabeled: pd.DataFrame,
    Z_L: np.ndarray,
    Z_U: np.ndarray,
    K: int,
    n_clusters: int = None,
    k_nn: int = 10,
    random_state: int = 42,
):
    """
    TypiClust-style batch acquisition.
    Returns:
      queried_indices: list of df_tr_unlabeled indices
      df_queried: subset df_tr_unlabeled with those indices
    """

    if len(df_tr_unlabeled) == 0:
        return [], df_tr_unlabeled.iloc[[]]

    K = min(K, len(df_tr_unlabeled))

    # Heuristic: number of clusters
    # Paper uses a clustering step; in practice choose something like O(sqrt(N))
    if n_clusters is None:
        n_clusters = int(np.clip(np.sqrt(len(df_tr_unlabeled)), 10, 200))

    # --- 1) Cluster the unlabeled pool embeddings ---
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init="auto",
    )
    clu_U = km.fit_predict(Z_U)

    # Assign labeled points to clusters using nearest centroid
    clu_L = km.predict(Z_L) if len(Z_L) else np.array([], dtype=int)

    # Coverage counts per cluster
    cov_L = np.bincount(clu_L, minlength=n_clusters) if len(clu_L) else np.zeros(n_clusters, dtype=int)

    # For each cluster, keep indices of unlabeled points belonging to it
    U_indices = df_tr_unlabeled.index.to_numpy()
    cluster_to_u_positions = {}
    for pos, c in enumerate(clu_U):
        cluster_to_u_positions.setdefault(c, []).append(pos)

    # We'll build the batch iteratively (without replacement)
    chosen_u_positions = []
    chosen_clusters = []

    # Precompute typicality per cluster (kNN inside cluster)
    # typicality score for point i in cluster c = -mean distance to kNN within that cluster
    typicality_scores = {}  # c -> array of scores aligned with positions list
    for c, positions in cluster_to_u_positions.items():
        if len(positions) < 2:
            continue
        Zc = Z_U[positions]
        k_eff = min(k_nn, len(Zc) - 1)
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(Zc)
        dists, _ = nbrs.kneighbors(Zc)  # first neighbor is itself (distance 0)
        mean_knn_dist = dists[:, 1:].mean(axis=1)
        typicality_scores[c] = -mean_knn_dist  # higher is better

    # --- 2) Select K points: prefer uncovered clusters, then most typical within cluster ---
    for _ in range(K):
        # among clusters that still have available points, pick the cluster with lowest labeled coverage
        available_clusters = [
            c for c, positions in cluster_to_u_positions.items()
            if any(pos not in chosen_u_positions for pos in positions)
        ]
        if not available_clusters:
            break

        # sort clusters by labeled coverage (ascending), tie-break by cluster size (descending)
        available_clusters.sort(key=lambda c: (cov_L[c], -len(cluster_to_u_positions[c])))
        c_star = available_clusters[0]

        # pick most typical remaining point in c_star
        positions = cluster_to_u_positions[c_star]
        remaining = [pos for pos in positions if pos not in chosen_u_positions]

        if len(remaining) == 0:
            continue

        if c_star in typicality_scores:
            # map remaining positions to local indices within the cluster list
            local_scores = typicality_scores[c_star]
            # positions list aligns with local_scores
            # choose argmax among remaining
            best_pos = max(
                remaining,
                key=lambda pos: local_scores[positions.index(pos)]
            )
        else:
            # fallback: if no typicality computed (tiny cluster), just take first remaining
            best_pos = remaining[0]

        chosen_u_positions.append(best_pos)
        chosen_clusters.append(c_star)

        # update coverage to reflect that we will label one point from this cluster
        cov_L[c_star] += 1

    queried_indices = U_indices[chosen_u_positions].tolist()
    df_queried = df_tr_unlabeled.loc[queried_indices]
    return queried_indices, df_queried


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def pick_hybrid_uncertainty_typiclust(
    clf,
    df_tr_unlabeled: pd.DataFrame,
    Z_tr_unlabeled: np.ndarray,
    K: int,
    T: int,
    mc_predict,
    *,
    unc_frac: float = 0.3,
    n_clusters: int | None = None,
    use_pca: bool = True,
    pca_dim: int = 64,
    l2_normalize: bool = True,
    random_state: int = 42,
):
    """
    Hybrid acquisition: filter to top-uncertainty points, then apply TypiClust
    (pick typical points in diverse clusters).

    Returns:
      queried_indices: list of DataFrame indices (original df_tr_unlabeled indices)
      df_queried: DataFrame of queried rows (includes an 'uncertainty' column)
    """
    if len(df_tr_unlabeled) == 0:
        return [], df_tr_unlabeled.copy()

    K = min(K, len(df_tr_unlabeled))

    # ---------- 1) Uncertainty (MC dropout) ----------
    _, std_p, _ = mc_predict(clf, Z_tr_unlabeled, T=T)

    if len(std_p) != len(df_tr_unlabeled):
        raise ValueError(
            f"std_p has {len(std_p)} entries but df_tr_unlabeled has {len(df_tr_unlabeled)} rows."
        )

    # Attach uncertainty (preserve original indices)
    df_unlb = df_tr_unlabeled.copy()
    df_unlb["uncertainty"] = std_p

    # ---------- 2) Keep top-uncertainty subset ----------
    m = max(K, int(np.ceil(unc_frac * len(df_unlb))))
    df_unc = df_unlb.nlargest(m, "uncertainty")  # highest uncertainty first

    # Subset embedding matrix in the same order as df_unc
    # (df_unc.index are original indices; need positional mapping)
    pos_map = {idx: i for i, idx in enumerate(df_tr_unlabeled.index)}
    sel_pos = np.array([pos_map[idx] for idx in df_unc.index], dtype=int)
    Z_unc = Z_tr_unlabeled[sel_pos]

    # ---------- 3) Prep embeddings for clustering ----------
    Zc = Z_unc.astype(np.float32)

    if l2_normalize:
        norms = np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8
        Zc = Zc / norms

    if use_pca and Zc.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        Zc = pca.fit_transform(Zc)

    # ---------- 4) TypiClust inside uncertain subset ----------
    if n_clusters is None:
        # a common heuristic: 10*K clusters (cap by subset size)
        n_clusters = min(len(df_unc), max(2, 10 * K))
    else:
        n_clusters = min(n_clusters, len(df_unc))

    if n_clusters < 2:
        # Not enough points to cluster — fall back to pure uncertainty
        queried_indices = df_unc.head(K).index.tolist()
        df_queried = df_unlb.loc[queried_indices]
        return queried_indices, df_queried

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Zc)
    centers = km.cluster_centers_

    # Typicality = distance to cluster centroid (smaller is more typical)
    dists = np.linalg.norm(Zc - centers[labels], axis=1)

    # For each cluster, rank points by typicality (closest first)
    cluster_to_rows = {}
    df_unc_idx = df_unc.index.to_numpy()
    for j in range(len(df_unc)):
        c = int(labels[j])
        cluster_to_rows.setdefault(c, []).append((dists[j], df_unc_idx[j]))

    for c in cluster_to_rows:
        cluster_to_rows[c].sort(key=lambda x: x[0])  # closest to centroid first

    # ---------- 5) Select K points: one per cluster, then cycle ----------
    clusters_sorted = sorted(cluster_to_rows.keys(), key=lambda c: cluster_to_rows[c][0][0])

    picked = []
    ptr = {c: 0 for c in clusters_sorted}

    while len(picked) < K:
        any_added = False
        for c in clusters_sorted:
            i = ptr[c]
            if i < len(cluster_to_rows[c]):
                _, idx = cluster_to_rows[c][i]
                ptr[c] += 1
                if idx not in picked:
                    picked.append(idx)
                    any_added = True
                    if len(picked) >= K:
                        break
        if not any_added:
            break  # safety

    queried_indices = picked
    df_queried = df_unlb.loc[queried_indices]  # includes uncertainty column
    return queried_indices, df_queried