import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize


def _mean_pairwise_cosine_distance(Z_group):
    """Mean pairwise cosine distance among rows of Z_group. Returns NaN if <2 rows."""
    n = len(Z_group)
    if n < 2:
        return float("nan")
    Zn = normalize(np.asarray(Z_group, dtype=np.float64))
    sims = Zn @ Zn.T
    # Sum over upper triangle (excluding diagonal), divided by pair count.
    iu = np.triu_indices(n, k=1)
    mean_sim = float(sims[iu].mean())
    return 1.0 - mean_sim


def compute_user_class_spread(
    Z,
    df,
    user_id=None,
    label_col="state_val",
    user_col="user_id",
    out_path=None,
):
    """
    Within-user, within-class spread vs. between-class spread.

    For each (user, class), compute the mean pairwise cosine distance between
    embeddings in that group. Average across users to get a per-class
    within-spread number. Compare to the between-class centroid distance.

    Higher within-spread = noisier embeddings for that class.
    The ratio between/within is what matters: large ratio => classes are
    well separated relative to intra-class noise.

    Parameters
    ----------
    Z : (n, d) array — encoded rows of `df`, aligned by row order.
    df : DataFrame with `user_col` and `label_col`.
    user_id : optional tag for the JSON summary (the participant being run).
    label_col : column holding the class label.
    user_col : column holding the user identifier.
    out_path : optional path to write a JSON summary.

    Returns
    -------
    dict with within/between spread and the ratio.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if len(Z) != len(df):
        raise ValueError(
            f"Z has {len(Z)} rows but df has {len(df)} — must be aligned."
        )

    if label_col not in df.columns:
        raise ValueError(f"df missing label column '{label_col}'")
    if user_col not in df.columns:
        raise ValueError(f"df missing user column '{user_col}'")

    df = df.reset_index(drop=True)
    labels = df[label_col].to_numpy()
    users = df[user_col].to_numpy()
    classes = sorted(np.unique(labels).tolist())

    # Within-user, within-class: mean pairwise cosine distance per (user,class),
    # then average across users for each class.
    per_class_within = {}
    per_user_class_detail = {}
    for cls in classes:
        per_user_vals = []
        for uid in np.unique(users):
            mask = (users == uid) & (labels == cls)
            if mask.sum() < 2:
                continue
            d = _mean_pairwise_cosine_distance(Z[mask])
            if not np.isnan(d):
                per_user_vals.append(d)
                per_user_class_detail.setdefault(str(uid), {})[int(cls)] = d
        per_class_within[int(cls)] = (
            float(np.mean(per_user_vals)) if per_user_vals else float("nan")
        )

    overall_within = (
        float(np.nanmean(list(per_class_within.values())))
        if per_class_within
        else float("nan")
    )

    # Between-class: cosine distance between class centroids (computed on all
    # users pooled together, since the question is class separability).
    Zn = normalize(Z)
    centroids = {}
    for cls in classes:
        mask = labels == cls
        if mask.sum() == 0:
            continue
        c = Zn[mask].mean(axis=0)
        # Re-normalize centroid so the inner product is cosine similarity.
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        centroids[int(cls)] = c

    between_pairs = {}
    if len(centroids) >= 2:
        cls_list = list(centroids.keys())
        for i in range(len(cls_list)):
            for j in range(i + 1, len(cls_list)):
                a, b = cls_list[i], cls_list[j]
                sim = float(centroids[a] @ centroids[b])
                between_pairs[f"{a}_vs_{b}"] = 1.0 - sim
        overall_between = float(np.mean(list(between_pairs.values())))
    else:
        overall_between = float("nan")

    ratio = (
        overall_between / overall_within
        if overall_within and not np.isnan(overall_within) and overall_within > 0
        else float("nan")
    )

    summary = {
        "user_id": str(user_id) if user_id is not None else None,
        "n_rows": int(len(Z)),
        "n_users": int(len(np.unique(users))),
        "classes": [int(c) for c in classes],
        "within_class_spread_per_class": per_class_within,
        "within_class_spread_overall": overall_within,
        "between_class_distance_per_pair": between_pairs,
        "between_class_distance_overall": overall_between,
        "between_to_within_ratio": ratio,
        "per_user_per_class_spread": per_user_class_detail,
    }

    print(
        f"[embedding-spread] user={user_id} "
        f"within={overall_within:.4f} "
        f"between={overall_between:.4f} "
        f"ratio={ratio:.3f}"
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary
