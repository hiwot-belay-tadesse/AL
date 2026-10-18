"""Graph-cut acquisition: representation minus redundancy, instead of k-center's minimax.

Why this exists (walking/results/al_global_rolling_cnn64/): CoreSet's Delta_p turned out to be a
dispersion score -- rho(Delta, participant dispersion) = +0.70 -- and dispersion is not what makes
a source transfer. Outlyingness actively hurts (rho(outlyingness, transfer) = -0.54), and
k-center greedy is a MINIMAX objective, so the farthest point always wins the argmax and
outliers are selected first. P45 is the case in point: most dispersed and most outlying of the
22 participants, 4th most queried, and the worst source of all (0.481).

Graph cut replaces the minimax criterion with

    f(S) = sum_{i in V} sum_{j in S} s(i, j)  -  lambda * sum_{i, j in S} s(i, j)

whose marginal gain for a candidate e is

    gain(e) = sum_{i in V} s(i, e)  -  lambda * (2 * sum_{j in S} s(j, e) + s(e, e)) .

The first term rewards candidates that are similar to the pool AS A WHOLE (an outlier scores
near zero there, which is the fix), the second penalises similarity to what is already labeled
(which is what keeps it from collapsing onto one dense blob). DeepCore (arXiv 2204.08499)
reports graph cut as the strongest method at 0.1-5% selection fractions, which is the regime
here (232-rows-of-41849 budgets).

Two implementation choices, both deliberate:

  * V is a fixed random ANCHOR SAMPLE of the pool (n_anchors rows), not all 41849 rows: the
    representation term is a sum over the ground set and would otherwise cost O(n^2) per round.
    The anchors are drawn once per pool and cached, so the objective is identical across rounds
    and across methods within a (seed, fold).
  * s is an RBF kernel with sigma = the median pairwise anchor distance, so s is in (0, 1] and
    positive, as graph cut requires, and the scale adapts to the representation (46 handcrafted
    features and a 64-d CNN embedding have very different distance scales).

lambda is in units of the anchor sum: |V| = n_anchors and |S| <= n_init + budget * K, so
lambda ~ 1 balances the two terms for the budgets used here.
"""
import numpy as np

_ANCHOR_CACHE = {}


def _fingerprint(Z, n_anchors, seed):
    """Cheap, collision-safe-enough key for one pool matrix (id() alone is reused after GC)."""
    return (Z.shape, float(Z[0, 0]), float(Z[-1, -1]), float(Z.sum()), n_anchors, tuple(seed))


def anchors_and_sigma(Z, n_anchors, seed):
    """The ground-set sample V, the RBF bandwidth, and the representation term of EVERY pool row.

    rep(e) = sum_{i in V} s(i, e) does not depend on the labeled set, so it is computed once per
    pool and reused by every round and every target that shares this pool -- without that the
    round loop would recompute a 41849 x n_anchors kernel 1320 times."""
    key = _fingerprint(Z, n_anchors, seed)
    if key not in _ANCHOR_CACHE:
        rng = np.random.default_rng(list(seed))
        idx = np.sort(rng.choice(len(Z), min(n_anchors, len(Z)), replace=False))
        A = Z[idx]
        sub = A[rng.choice(len(A), min(500, len(A)), replace=False)]
        d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        sigma = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        sigma = max(sigma, 1e-6)
        _ANCHOR_CACHE[key] = (A, sigma, rbf(Z, A, sigma).sum(1))
        if len(_ANCHOR_CACHE) > 8:                      # one entry per (fold, pool); keep it bounded
            _ANCHOR_CACHE.pop(next(iter(_ANCHOR_CACHE)))
    return _ANCHOR_CACHE[key]


def rbf(Z, C, sigma, block=4096):
    """(len(Z), len(C)) RBF similarities, blocked over rows to bound memory."""
    out = np.empty((len(Z), len(C)), dtype=np.float32)
    for b in range(0, len(Z), block):
        d2 = ((Z[b:b + block, None, :] - C[None, :, :]) ** 2).sum(-1)
        out[b:b + block] = np.exp(-d2 / (2 * sigma ** 2))
    return out


def graph_cut_select(Z, labeled_mask, K, lam=1.0, n_anchors=1000, seed=(0,)):
    """K pool indices by greedy graph-cut gain. Same interface as coreset_greedy, minus the
    incremental min-distance state (graph cut keeps its own redundancy vector instead).
    Marks the queried rows in labeled_mask, as coreset_greedy and badge_select do."""
    unlabeled = np.where(~labeled_mask)[0]
    if len(unlabeled) == 0:
        return np.array([], dtype=int)
    K = int(min(K, len(unlabeled)))
    _, sigma, rep_all = anchors_and_sigma(Z, n_anchors, seed)

    rep = rep_all[unlabeled]                                       # representation: similarity to V
    lab = np.where(labeled_mask)[0]
    red = (rbf(Z[unlabeled], Z[lab], sigma).sum(1) if len(lab) else
           np.zeros(len(unlabeled), dtype=np.float32))             # redundancy: similarity to S
    picked = []
    for _ in range(K):
        gain = rep - lam * (2 * red + 1.0)
        gain[picked] = -np.inf
        j = int(np.argmax(gain))
        picked.append(j)
        red += rbf(Z[unlabeled], Z[unlabeled[j]][None, :], sigma)[:, 0]
    queried = unlabeled[np.array(picked, dtype=int)]
    labeled_mask[queried] = True
    return queried
