"""BADGE (Ash et al., ICLR 2020) for the binary logistic-regression pipeline.

Batch Active learning by Diverse Gradient Embeddings: score each unlabelled row by the gradient
the model WOULD receive if its own most-likely label were the truth, then pick a batch that is
both high-magnitude (uncertain) and mutually diverse (k-means++ seeding on those gradients).

For binary logistic regression with linear predictor w.x + b, the loss gradient w.r.t. (w, b) at
a row x with pseudo-label y_hat is

    g(x) = (p - y_hat) * [x, 1],      p = P(y=1|x),  y_hat = 1[p > 0.5]

so ||g(x)|| = min(p, 1-p) * ||[x, 1]||: magnitude is exactly the model's uncertainty, scaled by
the row's norm. k-means++ seeding on g then spreads the batch over gradient directions, which is
what stops BADGE from returning K near-duplicates of the single most uncertain row.

The features are the same preprocessed Z the classifier sees (walking/models.make_prep output:
median imputation + standardisation, fit on the training pool), so no new preprocessing exists.

Memory note: materialising G = c[:, None] * Xb costs another (n x d+1) array, ~140 MB for the
IN_A_MEETING pool. badge_select therefore never forms G; it uses

    ||g_i - g_j||^2 = c_i^2 ||xb_i||^2 + c_j^2 ||xb_j||^2 - 2 c_i c_j (xb_i . xb_j)

which is the same number from one matrix-vector product per pick. gradient_embedding() builds G
explicitly and exists for the unit test and for small pools.
"""
import numpy as np


def gradient_embedding(Z, p):
    """G[i] = (p_i - y_hat_i) * [z_i, 1], the BADGE gradient embedding. Shape (n, d+1).

    Z: (n, d) preprocessed features, p: (n,) P(y=1|z) from the current model."""
    p = np.asarray(p, dtype=float)
    y_hat = (p > 0.5).astype(float)
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    return (p - y_hat)[:, None] * Zb


def kmeanspp_seed(G, K, rng):
    """k-means++ seeding on rows of G: K indices, first = largest ||g||, then P ∝ D^2.

    This is BADGE's selection step. The deterministic first pick is Ash et al.'s convention and
    makes the batch reproducible up to `rng`; subsequent picks are sampled, not argmax'd, which
    is what keeps the batch diverse rather than collapsing onto one direction."""
    n = len(G)
    K = min(K, n)
    picks = [int(np.argmax(np.linalg.norm(G, axis=1)))]
    d2 = ((G - G[picks[0]]) ** 2).sum(axis=1)
    while len(picks) < K:
        tot = d2.sum()
        if not np.isfinite(tot) or tot <= 0:                 # all remaining rows coincide
            left = np.setdiff1d(np.arange(n), picks)
            picks += list(rng.choice(left, K - len(picks), replace=False))
            break
        j = int(rng.choice(n, p=d2 / tot))
        picks.append(j)
        np.minimum(d2, ((G - G[j]) ** 2).sum(axis=1), out=d2)
    return np.array(picks, dtype=int)


def badge_select(Z, labeled_mask, K, p, rng):
    """K pool indices chosen by BADGE from the unlabelled rows of Z.

    Z: (n, d) the whole pool in the classifier's feature space;
    labeled_mask: (n,) bool, True = already labelled (mutated in place, like coreset_greedy);
    p: (n,) current model's P(y=1) for every pool row -- candidate LABELS are never touched;
    rng: np.random.Generator (run_al_rounds keys it by (seed, fold, round) via rng_key).

    Returns the queried pool indices in acquisition order."""
    unlabeled = np.where(~labeled_mask)[0]
    if len(unlabeled) == 0:
        return np.array([], dtype=int)
    K = int(min(K, len(unlabeled)))

    Zu = Z[unlabeled]
    c = p[unlabeled] - (p[unlabeled] > 0.5)                  # (p - y_hat), in [-0.5, 0.5]
    sq = (Zu ** 2).sum(axis=1) + 1.0                         # ||[z, 1]||^2
    gnorm2 = c ** 2 * sq                                     # ||g||^2 without forming g

    def d2_to(j):
        """||g_i - g_j||^2 for every unlabelled i, from one mat-vec."""
        dot = Zu @ Zu[j] + 1.0                               # [z_i,1].[z_j,1]
        return np.maximum(gnorm2 + gnorm2[j] - 2.0 * c * c[j] * dot, 0.0)

    local = [int(np.argmax(gnorm2))]                         # largest ||g||
    d2 = d2_to(local[0])
    while len(local) < K:
        tot = d2.sum()
        if not np.isfinite(tot) or tot <= 0:
            left = np.setdiff1d(np.arange(len(unlabeled)), local)
            local += list(rng.choice(left, K - len(local), replace=False))
            break
        j = int(rng.choice(len(unlabeled), p=d2 / tot))
        local.append(j)
        np.minimum(d2, d2_to(j), out=d2)

    queried = unlabeled[np.array(local, dtype=int)]
    labeled_mask[queried] = True
    return queried
