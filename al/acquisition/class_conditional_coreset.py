"""Coreset run separately inside each predicted class.

Plain k-center greedy covers the pool geometrically and is blind to the label, which on a
skewed target means the batch inherits the pool's class ratio. This variant splits the
unlabelled rows by the CURRENT MODEL's predicted label (never by the candidates' true labels),
gives the predicted-positive side a floor of ceil(K/2) picks when it can supply them, splits
the rest in proportion to group size, and runs the existing coreset_greedy inside each group --
same metric, same incremental min-distance bookkeeping, same centers (the labelled set).

Restricting a group is done by temporarily marking the other group's unlabelled rows as
labelled, so coreset_greedy itself is used unmodified and the shared min_dist_to_centers is
still updated by every pick.
"""
import math

import numpy as np


def class_conditional_coreset(Z, labeled_mask, K, min_dist_to_centers, p_pos,
                              coreset_greedy, metric="euclidean"):
    """K pool indices, coreset-selected within each predicted class.

    Z, labeled_mask, min_dist_to_centers, metric: exactly as walking/active_learning.coreset_greedy.
    p_pos: (n,) current model's P(y=1) for every pool row.
    coreset_greedy: the function itself, injected so this module does not import walking/.

    Returns the queried pool indices (positive group first)."""
    unlabeled = np.where(~labeled_mask)[0]
    if len(unlabeled) == 0:
        return np.array([], dtype=int)
    K = int(min(K, len(unlabeled)))

    pred_pos = unlabeled[p_pos[unlabeled] > 0.5]
    pred_neg = unlabeled[p_pos[unlabeled] <= 0.5]
    n_pos, n_neg = len(pred_pos), len(pred_neg)

    k_pos = int(round(K * n_pos / len(unlabeled)))            # proportional allocation ...
    k_pos = max(k_pos, min(math.ceil(K / 2), n_pos))          # ... with the positive-class floor
    k_pos = min(k_pos, n_pos, K)
    k_neg = min(K - k_pos, n_neg)
    k_pos = min(K - k_neg, n_pos)                             # give unusable negative picks back

    queried = []
    for group, k in ((pred_pos, k_pos), (pred_neg, k_neg)):
        if k <= 0 or len(group) == 0:
            continue
        other = np.setdiff1d(np.where(~labeled_mask)[0], group)
        labeled_mask[other] = True                            # block the other class ...
        q = coreset_greedy(Z, labeled_mask, k, min_dist_to_centers, metric)
        labeled_mask[other] = False                           # ... and unblock it
        queried.append(q)
    return np.concatenate(queried) if queried else np.array([], dtype=int)
