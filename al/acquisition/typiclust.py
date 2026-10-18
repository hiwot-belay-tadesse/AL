"""TypiClust acquisition: query the most TYPICAL point of each uncovered cluster.

Hacohen, Dekel & Weinshall, "Active Learning on a Budget: Opposite Strategies Suit High and Low
Budgets" (ICML 2022) / "Active Learning Through a Covering Lens" (NeurIPS 2022). Their claim is a
phase transition: in the LOW-budget regime typical (dense-region) points are the right query and
atypical/uncertain ones only pay off once the budget is large. This repo's walking runs are
squarely in the low-budget regime -- 210 labels out of a 41849-row pool, 0.5% -- which is exactly
where k-center is predicted to be the wrong strategy.

It is also the direct test of the mechanism measured in walking/results/al_global_rolling_cnn64/:
CoreSet's Delta tracks participant dispersion (rho = +0.70) while outlyingness tracks transfer
NEGATIVELY (rho = -0.54). Graph cut already removed the outlier bias but ranked by centrality and
still found nothing (rho(Delta, quality) = -0.05). TypiClust differs from both by being
cluster-local: it takes the densest point of each REGION rather than scoring the pool globally,
so it spreads over the space without ever leaving a dense neighbourhood.

Per round, with |L| labels already acquired and a budget of K:
  1. cluster the whole pool into |L| + K clusters (MiniBatchKMeans)
  2. keep the clusters containing no labeled row -- the "uncovered" ones -- largest first
  3. from each of the K largest, query the unlabeled row with the highest typicality,
         typicality(x) = 1 / mean distance to its n_nn nearest neighbours INSIDE its cluster
  4. if fewer than K clusters are uncovered, continue down the covered ones by size

The clustering depends only on the pool and on |L| + K, and |L| follows the same trajectory for
every target (n_init, then +K per round), so one clustering per (pool, round) is cached and
shared by all 22 targets instead of being recomputed 22 times.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

_CLUSTER_CACHE = {}
N_NN = 20


def _fingerprint(Z, n_clusters, seed):
    return (Z.shape, float(Z[0, 0]), float(Z[-1, -1]), float(Z.sum()), n_clusters, tuple(seed))


def cluster_and_typicality(Z, n_clusters, seed=(0,), n_nn=N_NN):
    """(cluster label per row, typicality per row), cached per (pool, n_clusters)."""
    key = _fingerprint(Z, n_clusters, seed)
    if key in _CLUSTER_CACHE:
        return _CLUSTER_CACHE[key]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=int(abs(hash(key)) % (2 ** 31)),
                         n_init=3, batch_size=2048, max_iter=100)
    labels = km.fit_predict(Z)
    typ = np.zeros(len(Z), dtype=np.float32)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            typ[idx] = 0.0                      # a singleton cluster is an outlier, not a typical point
            continue
        k = int(min(n_nn, len(idx) - 1))
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[idx])
        d, _ = nn.kneighbors(Z[idx])
        typ[idx] = 1.0 / (d[:, 1:].mean(1) + 1e-12)
    if len(_CLUSTER_CACHE) > 64:
        _CLUSTER_CACHE.pop(next(iter(_CLUSTER_CACHE)))
    _CLUSTER_CACHE[key] = (labels, typ)
    return labels, typ


def typiclust_select(Z, labeled_mask, K, seed=(0,), n_nn=N_NN):
    """K pool indices, one per uncovered cluster, largest clusters first.
    Marks them in labeled_mask, as coreset_greedy and badge_select do."""
    unlabeled = np.where(~labeled_mask)[0]
    if len(unlabeled) == 0:
        return np.array([], dtype=int)
    K = int(min(K, len(unlabeled)))
    n_clusters = int(min(labeled_mask.sum() + K, len(Z)))
    labels, typ = cluster_and_typicality(Z, n_clusters, seed, n_nn)

    covered = set(np.unique(labels[labeled_mask]))
    sizes = np.bincount(labels, minlength=n_clusters)
    has_free = np.bincount(labels[unlabeled], minlength=n_clusters) > 0
    # largest first, uncovered before covered, and SINGLETON clusters last: a cluster of one is an
    # outlier that k-means has isolated, which is precisely what TypiClust is meant to avoid
    # querying (it is also the failure mode measured for CoreSet in the CNN embedding).
    big = [c for c in np.argsort(sizes)[::-1] if has_free[c] and sizes[c] > 1]
    lone = [c for c in np.argsort(sizes)[::-1] if has_free[c] and sizes[c] <= 1]
    order = ([c for c in big if c not in covered] + [c for c in big if c in covered] + lone)

    queried = []
    for c in order:
        if len(queried) == K:
            break
        cand = np.where((labels == c) & ~labeled_mask)[0]
        if len(cand):
            queried.append(int(cand[np.argmax(typ[cand])]))
    if len(queried) < K:                        # pool nearly exhausted: fall back to global typicality
        rest = np.setdiff1d(unlabeled, queried)
        queried += list(rest[np.argsort(typ[rest])[::-1][: K - len(queried)]])
    queried = np.array(queried, dtype=int)
    labeled_mask[queried] = True
    return queried
