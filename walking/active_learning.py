"""Active learning loop (random vs. CoreSet) on handcrafted watch accelerometer features.
Terminology follows run.py / new_helper.py: a labeled set and an unlabeled set drawn
from the training pool, K queries per round, and a budget = number of query rounds.

Per (seed, fold, user):
  * round 0: initial labeled/unlabeled split: N_INIT labeled rows, stratified,
    with >= MIN_PER_CLASS of each class; everything else is unlabeled. The SAME
    initial labeled set is used for every acquisition function (random and coreset),
    so the methods are compared from an identical starting point.
  * each round: the acquisition function picks K queried_indices from the unlabeled
    set, they move to the labeled set, a fresh classifier is trained on the labeled
    set and scored on the test set. Stops when the budget (rounds) is spent or the
    unlabeled set is empty (100% labeled).
  * random:  uniform sample of the unlabeled set
  * coreset: k-center greedy (farthest point) on Z, labeled rows as centers;
    distance = Euclidean (default) or cosine (--metric cosine)
  * class_balanced_coreset: same coverage criterion, re-weighted by the current
    classifier's predicted P(walking) so that candidates expected to move the labeled
    set toward a 50/50 class mix are favored (ground-truth labels of candidates are
    never used):  score(x) = min_dist(x) * (1 + lambda * d * (2 p(x) - 1)),
    d = 0.5 - expected walking fraction of the labeled set (updated within the batch
    with the picked candidates' predicted probabilities).
  * 100% reference: the same model trained on the whole pool.

pool="personal": Z = user's own train rows, standardized on them.
pool="global":   Z = all users' train rows pooled and standardized together; the
                 target's test set is still its own. Query provenance (which user
                 each queried row came from) is recorded.

Aggregate (per seed, fold, method, round): the labeled sets of ALL users are pooled
into one training set, one model is fit and scored on the concatenation of all
users' test sets; its 100% reference is the model trained on all pools.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import folds_for
from models import evaluate, fit_evaluate, make_prep

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from al.acquisition import (badge_select, central_coreset, class_conditional_coreset,  # noqa: E402
                            graph_cut_select, typiclust_select)

N_INIT = 10            # default size of the initial labeled set (override with main.py --n_init)
MIN_PER_CLASS = 2      # initial labeled set must contain >= this many rows of each class
K = 10                 # default queries per round (override with main.py --K)
N_REPORT_ROUND = 20    # round reported in the per-seed progress line
# The two newcomers are APPENDED: 1 + ALL_METHODS.index(m) is the per-method RNG key, so
# inserting them anywhere else would re-key (and silently change) every existing run.
ALL_METHODS = ["random", "coreset", "class_balanced_coreset", "badge", "class_conditional_coreset", "graph_cut", "typiclust", "central_coreset"]
DEFAULT_METHODS = ALL_METHODS[:3]   # what main.py runs when --methods is not given (unchanged)
METHODS = list(DEFAULT_METHODS)   # active set; main.py --methods overrides
NEEDS_MIN_DIST = ("coreset", "class_balanced_coreset", "class_conditional_coreset", "central_coreset")
CB_LAMBDA = 1.0        # strength of the class-balance term in class_balanced_coreset
GC_LAMBDA = 1.0        # redundancy weight in graph_cut (al/acquisition/graph_cut.py)
CENTRAL_QUANTILE = 0.8  # central_coreset: keep the 80% least outlying candidates


# ----------------------------------------------------------------------------- labeled / unlabeled split
def initial_labeled_unlabeled_split(y_pool, rng, n_init=N_INIT):
    """Indices of the initial labeled set: n_init rows, stratified, floored to
    MIN_PER_CLASS of each class (cf. the MIN_PER_CLASS floor in run.py)."""
    idx = np.arange(len(y_pool))
    if len(idx) <= n_init:
        return idx
    labeled_idx, _ = train_test_split(idx, train_size=n_init, stratify=y_pool, random_state=int(rng.integers(1 << 31)))
    labeled_idx = list(labeled_idx)
    for c in (0, 1):
        have = [i for i in labeled_idx if y_pool[i] == c]
        need = min(MIN_PER_CLASS, int((y_pool == c).sum())) - len(have)
        if need > 0:
            extra = rng.choice(np.setdiff1d(idx[y_pool == c], labeled_idx), need, replace=False)
            labeled_idx += list(extra)
    return np.array(labeled_idx)


# ----------------------------------------------------------------------------- acquisition functions
def pick_random(K, unlabeled_idx, rng):
    """Return K random indices from the unlabeled set (cf. new_helper.pick_random)."""
    return rng.choice(unlabeled_idx, min(K, len(unlabeled_idx)), replace=False)


def pairwise_dist_to(Z, j, metric="euclidean"):
    """Distance from every row of Z to row j. euclidean: L2 norm. cosine: 1 - cos(z, z_j),
    computed on the same (standardized) features (the cosine branch that is commented out
    in new_helper.coreset_greedy)."""
    if metric == "cosine":
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        return 1.0 - Zn @ Zn[j]
    return np.linalg.norm(Z - Z[j], axis=1)


def coreset_greedy(Z, labeled_mask, K, min_dist_to_centers, metric="euclidean"):
    """Core-Set greedy acquisition by farthest-point sampling (cf. new_helper.coreset_greedy).
    Centers = labeled rows; each step queries the unlabeled row farthest from its nearest
    center and adds it to the centers. Incremental form: min_dist_to_centers[i] holds the
    distance of pool row i to its nearest center and is updated in place.
    Marks the queried rows in labeled_mask and returns queried_indices."""
    queried_indices = []
    for _ in range(K):
        unlabeled_idx = np.where(~labeled_mask)[0]
        if len(unlabeled_idx) == 0:
            break
        j = unlabeled_idx[np.argmax(min_dist_to_centers[unlabeled_idx])]
        labeled_mask[j] = True
        queried_indices.append(j)
        np.minimum(min_dist_to_centers, pairwise_dist_to(Z, j, metric), out=min_dist_to_centers)
    return np.array(queried_indices, dtype=int)


def class_balanced_coreset(Z, labeled_mask, K, min_dist_to_centers, p_walk, n_labeled, n_labeled_pos,
                           metric="euclidean", lam=CB_LAMBDA):
    """CoreSet coverage score re-weighted toward the class the labeled set is short of.
    p_walk: current classifier's P(walking) for every pool row (unlabeled rows' true labels
    are never used). Within the batch the expected walking count of the labeled set is
    updated with the picked rows' predicted probabilities."""
    exp_pos, n = float(n_labeled_pos), float(n_labeled)
    queried_indices = []
    for _ in range(K):
        unlabeled_idx = np.where(~labeled_mask)[0]
        if len(unlabeled_idx) == 0:
            break
        deficit = 0.5 - exp_pos / n                                   # > 0: walking under-represented
        weight = 1.0 + lam * deficit * (2.0 * p_walk[unlabeled_idx] - 1.0)   # in [1 - lam/2, 1 + lam/2]
        j = unlabeled_idx[np.argmax(min_dist_to_centers[unlabeled_idx] * weight)]
        labeled_mask[j] = True
        queried_indices.append(j)
        exp_pos += p_walk[j]; n += 1
        np.minimum(min_dist_to_centers, pairwise_dist_to(Z, j, metric), out=min_dist_to_centers)
    return np.array(queried_indices, dtype=int)


# ----------------------------------------------------------------------------- AL rounds
def run_al_rounds(Z_pool, y_pool, Z_te, y_te, Aq, initial_labeled_idx, rng, budget=None, K=K, metric="euclidean", cb_lambda=CB_LAMBDA, rng_key=None):
    """Active learning for one (user, fold, acquisition function Aq), starting from the
    given initial labeled set (shared by all acquisition functions so they are compared
    from the same starting point). rng is used only by the random acquisition.
    Returns (al_progress, queried_all): al_progress has one dict per round;
    queried_all lists pool indices in acquisition order, so the labeled set at
    round r is queried_all[:al_progress[r]['n_labeled']].
    budget = number of query rounds; None runs until the unlabeled set is empty.
    rng_key: optional [seed, fold] -- BADGE then draws from default_rng(rng_key + [round]),
    the (seed, fold, round) keying its batch sampling needs; without it BADGE uses `rng`."""
    n = len(y_pool)
    labeled_mask = np.zeros(n, dtype=bool)
    queried_all = list(initial_labeled_idx)
    labeled_mask[queried_all] = True
    if Aq in NEEDS_MIN_DIST:
        min_dist_to_centers = np.full(n, np.inf)
        for j in queried_all:
            np.minimum(min_dist_to_centers, pairwise_dist_to(Z_pool, j, metric), out=min_dist_to_centers)

    al_progress, round_num = [], 0
    while True:
        Z_tr_labeled, y_tr_labeled = Z_pool[labeled_mask], y_pool[labeled_mask]
        metrics, clf = fit_evaluate(Z_tr_labeled, y_tr_labeled, Z_te, y_te)
        al_progress.append(dict(round=round_num, n_labeled=len(y_tr_labeled), n_labeled_pos=int(y_tr_labeled.sum()),
                                pct_labeled=100 * len(y_tr_labeled) / n, **metrics))
        if labeled_mask.all() or (budget is not None and round_num >= budget):
            break
        if Aq == "random":
            queried_indices = pick_random(K, np.where(~labeled_mask)[0], rng)
            labeled_mask[queried_indices] = True
        elif Aq == "coreset":
            queried_indices = coreset_greedy(Z_pool, labeled_mask, K, min_dist_to_centers, metric)
        elif Aq == "badge":
            p_pos = clf.predict_proba(Z_pool)[:, 1]
            r = np.random.default_rng(list(rng_key) + [round_num]) if rng_key is not None else rng
            queried_indices = badge_select(Z_pool, labeled_mask, K, p_pos, r)
        elif Aq == "graph_cut":
            # representation - redundancy (al/acquisition/graph_cut.py); no min-distance state,
            # and no labels or model scores, so it is directly comparable with coreset
            queried_indices = graph_cut_select(Z_pool, labeled_mask, K, lam=GC_LAMBDA,
                                               seed=tuple(rng_key) if rng_key is not None else (0,))
        elif Aq == "central_coreset":
            # k-center inside the dense core (al/acquisition/central_coreset.py): outlying rows
            # are blocked because outlyingness predicts POOR transfer (rho = -0.54)
            queried_indices = central_coreset(Z_pool, labeled_mask, K, min_dist_to_centers,
                                              coreset_greedy, metric, quantile=CENTRAL_QUANTILE,
                                              seed=tuple(rng_key) if rng_key is not None else (0,))
        elif Aq == "typiclust":
            # most typical point of each uncovered cluster (al/acquisition/typiclust.py)
            queried_indices = typiclust_select(Z_pool, labeled_mask, K,
                                               seed=tuple(rng_key) if rng_key is not None else (0,))
        elif Aq == "class_conditional_coreset":
            p_pos = clf.predict_proba(Z_pool)[:, 1]
            queried_indices = class_conditional_coreset(Z_pool, labeled_mask, K, min_dist_to_centers,
                                                        p_pos, coreset_greedy, metric)
        else:
            p_walk = clf.predict_proba(Z_pool)[:, 1]        # current model's belief, no candidate labels used
            queried_indices = class_balanced_coreset(Z_pool, labeled_mask, K, min_dist_to_centers, p_walk,
                                                     len(y_tr_labeled), int(y_tr_labeled.sum()), metric, cb_lambda)
        queried_all.extend(queried_indices.tolist())
        round_num += 1
    return al_progress, np.array(queried_all, dtype=int)


# ----------------------------------------------------------------------------- pool subsampling
def subsample_pool(splits, users, rng, neg_ratio=None):
    """Subsample each user's TRAINING rows (the pool) before pooling; test rows untouched.
    neg_ratio: keep at most neg_ratio non-walking rows per walking row (all walking kept)."""
    if neg_ratio is None:
        return splits
    out = {}
    for u in users:
        Xtr, Xte, ytr, yte = splits[u]
        idx = np.arange(len(ytr))
        pos, neg = idx[ytr == 1], idx[ytr == 0]
        n_neg = min(len(neg), int(round(neg_ratio * len(pos))))
        idx = np.sort(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
        out[u] = (Xtr[idx], Xte, ytr[idx], yte)
    return out


# ----------------------------------------------------------------------------- one seed: all folds x users x Aq
def run_seed(seed, data, users, pool, split, budget=None, train_days=3, test_days=1, n_folds=3, K=K, n_init=N_INIT,
             metric="euclidean", neg_ratio=None, cb_lambda=CB_LAMBDA, methods=None, folds=None):
    """All folds x users x acquisition functions for one seed (one process).
    Returns (per-user al_progress rows, aggregate rows, 100%-reference rows, query-source rows).

    `folds` overrides the split: pass a ready [(fold_id, {user: (Xtr, Xte, ytr, yte)})] list to use
    a split data.folds_for does not provide (meeting_al.py passes day-group folds, because
    meetings cluster on a few workdays and the walk-forward split drops every participant).
    Everything downstream -- acquisition, provenance, aggregates -- is unchanged."""
    METHODS = list(methods) if methods else ALL_METHODS   # local override (workers import the module fresh)
    progress_rows, agg_rows, full_rows, src_rows = [], [], [], []
    for fold, splits in (folds if folds is not None else folds_for(data, users, split, seed, train_days, test_days, n_folds)):
        splits = subsample_pool(splits, users, np.random.default_rng([seed, fold, 999]), neg_ratio)
        owner = np.concatenate([[u] * len(splits[u][2]) for u in users])  # source user of each pooled train row
        X_all_tr = np.vstack([splits[u][0] for u in users]); y_all_tr = np.concatenate([splits[u][2] for u in users])
        X_all_te = np.vstack([splits[u][1] for u in users]); y_all_te = np.concatenate([splits[u][3] for u in users])
        offsets = dict(zip(users, np.cumsum([0] + [len(splits[u][2]) for u in users[:-1]])))
        if pool == "global":
            prep_g = make_prep().fit(X_all_tr)
            Z_pool_g = prep_g.transform(X_all_tr)

        acquired = {Aq: {} for Aq in METHODS}   # acquired[Aq][user] = (queried_all in pooled indices, n_labeled per round)
        for u in users:
            Xtr, Xte, ytr, yte = splits[u]
            ui = users.index(u)
            if pool == "personal":
                prep = make_prep().fit(Xtr)
                Z_pool, y_pool = prep.transform(Xtr), ytr
            else:
                prep, Z_pool, y_pool = prep_g, Z_pool_g, y_all_tr
            Z_te = prep.transform(Xte)
            full_rows.append(dict(seed=seed, fold=fold, participant=u, n_pool=len(y_pool), n_pool_user=len(ytr),
                                  n_test=len(yte), walking_test=int(yte.sum()), **evaluate(Z_pool, y_pool, Z_te, yte)))
            # same initial labeled set for every acquisition function (keyed on seed, fold, user only)
            initial_labeled_idx = initial_labeled_unlabeled_split(y_pool, np.random.default_rng([seed, fold, ui]), n_init)
            for Aq in METHODS:
                rng = np.random.default_rng([seed, fold, ui, 1 + ALL_METHODS.index(Aq)])   # query stream (random Aq only)
                al_progress, queried_all = run_al_rounds(Z_pool, y_pool, Z_te, yte, Aq, initial_labeled_idx, rng, budget, K, metric, cb_lambda, rng_key=[seed, fold])
                progress_rows += [dict(seed=seed, fold=fold, participant=u, method=Aq, n_pool=len(y_pool), **x) for x in al_progress]
                queried_pooled = queried_all + (offsets[u] if pool == "personal" else 0)
                n_labeled_per_round = [x["n_labeled"] for x in al_progress]
                acquired[Aq][u] = (queried_pooled, n_labeled_per_round)
                lo = 0   # provenance: which user each queried row came from, per round
                for round_num, hi in enumerate(n_labeled_per_round):
                    srcs, n = np.unique(owner[queried_pooled[lo:hi]], return_counts=True)
                    src_rows += [dict(seed=seed, fold=fold, target=u, method=Aq, round=round_num, source=sv, n=int(nv))
                                 for sv, nv in zip(srcs, n)]
                    lo = hi

        # aggregate: union of all users' labeled sets -> one model -> concatenated test sets
        n_total = len(y_all_tr)
        for Aq in METHODS:
            n_rounds = max(len(c) for _, c in acquired[Aq].values())
            for round_num in range(n_rounds):
                labeled_idx = np.unique(np.concatenate([q[:c[min(round_num, len(c) - 1)]] for q, c in acquired[Aq].values()]))
                prep = make_prep().fit(X_all_tr[labeled_idx])
                agg_rows.append(dict(seed=seed, fold=fold, method=Aq, round=round_num, n_labeled=len(labeled_idx),
                                     pct_labeled=100 * len(labeled_idx) / n_total, n_users=len(users),
                                     **evaluate(prep.transform(X_all_tr[labeled_idx]), y_all_tr[labeled_idx],
                                                prep.transform(X_all_te), y_all_te)))
        prep = make_prep().fit(X_all_tr)
        full = evaluate(prep.transform(X_all_tr), y_all_tr, prep.transform(X_all_te), y_all_te)
        for Aq in METHODS:
            agg_rows.append(dict(seed=seed, fold=fold, method=Aq, round=-1, n_labeled=n_total, pct_labeled=100.0,
                                 n_users=len(users), **full))
        print(f"seed {seed} fold {fold}: {len(users)} users, aggregate 100%={full['test_auroc']:.3f}", flush=True)

    d, a = pd.DataFrame(progress_rows), pd.DataFrame(agg_rows)
    r20 = d[d["round"] == N_REPORT_ROUND].groupby("method").test_auroc.mean()
    a20 = a[a["round"] == N_REPORT_ROUND].groupby("method").test_auroc.mean()
    print(f"seed {seed}: round-{N_REPORT_ROUND} per-user AUROC " + " ".join(f"{m}={r20.get(m, np.nan):.3f}" for m in METHODS)
          + "   aggregate AUROC " + " ".join(f"{m}={a20.get(m, np.nan):.3f}" for m in METHODS), flush=True)
    return progress_rows, agg_rows, full_rows, src_rows
