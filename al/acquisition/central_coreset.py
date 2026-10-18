"""k-center greedy restricted to the dense core of the pool.

Across 27 screened ExtraSensory labels, Spearman(outlyingness, source quality) is negative almost
everywhere (FIX_walking -0.49): a participant far from the pooled centroid is far because their
data is unusual, so a model fit on them extrapolates badly to everyone else. k-center maximizes
distance from what is already selected, so its per-participant Delta tracks dispersion
(rho = +0.70) -- precisely the quantity that predicts POOR transfer.

This keeps k-center but removes the part that is anti-correlated: candidates beyond the
`quantile` of distance-to-pooled-centroid are blocked, and the usual farthest-point greedy runs
on the rest. quantile=1.0 recovers plain coreset exactly.

Blocking is done the way class_conditional_coreset does it -- flip the excluded rows in
labeled_mask, call coreset_greedy, flip them back -- so the incremental min_dist_to_centers state
stays consistent with the caller's.
"""
import numpy as np

_CENTROID_CACHE = {}


def centroid_distance(Z, seed=(0,)):
    """||z - mean(Z)|| per row, cached per pool (the pool is shared by all targets in a fold)."""
    key = (Z.shape, float(Z[0, 0]), float(Z[-1, -1]), float(Z.sum()), tuple(seed))
    if key not in _CENTROID_CACHE:
        if len(_CENTROID_CACHE) > 8:
            _CENTROID_CACHE.pop(next(iter(_CENTROID_CACHE)))
        _CENTROID_CACHE[key] = np.sqrt(((Z - Z.mean(0)) ** 2).sum(1))
    return _CENTROID_CACHE[key]


def central_coreset(Z, labeled_mask, K, min_dist_to_centers, coreset_greedy, metric="euclidean",
                    quantile=0.8, seed=(0,)):
    """K pool indices by k-center greedy among the `quantile` least outlying unlabeled rows."""
    d = centroid_distance(Z, seed)
    unlabeled = np.where(~labeled_mask)[0]
    if len(unlabeled) == 0:
        return np.array([], dtype=int)
    cut = np.quantile(d[unlabeled], quantile)
    blocked = unlabeled[d[unlabeled] > cut]
    if len(unlabeled) - len(blocked) < K:          # too few candidates left: fall back to all
        blocked = np.array([], dtype=int)
    labeled_mask[blocked] = True
    try:
        queried = coreset_greedy(Z, labeled_mask, K, min_dist_to_centers, metric)
    finally:
        labeled_mask[blocked] = False
    return queried
