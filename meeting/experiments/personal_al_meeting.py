"""Personalized active learning on IN_A_MEETING with a mixed-effects model (prompt_active_mixed.txt).

For each target p: seed with 10 of p's OWN labels, then query 10 windows per round for 20 rounds
from the pool of all 21 participants' unlabeled training windows, refitting a
BinomialBayesMixedGLM (fit_map) every round and scoring p's held-out test days.

Two acquisition functions only:
  var_reduction  targeted reduction of the summed logit variance over p's unlabeled windows,
                 in the augmented space [1, x, person intercepts, person slopes], picked greedily
                 with Sherman-Morrison updates of the Laplace covariance
  random         K windows uniformly from the whole pool

Everything about the data is reused from experiments/mixed_vs_global.py: the 175-feature set
(raw_acc, proc_gyro, watch_acceleration, location, location_quick_features, audio_naive, discrete),
day_group_folds(3 folds, seed 0), and the median-imputer + standard-scaler preprocessing fit on
each fold's pooled training rows.

    python experiments/personal_al_meeting.py --unit_test      # the closed-form score check
    python experiments/personal_al_meeting.py --time_one       # one target x fold x repeat, projected total
    python experiments/personal_al_meeting.py                  # the full run
"""
import argparse
import os
import sys
import time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")      # the acquisition is BLAS-bound; the AL loop is serial

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "walking"), HERE, os.path.join(HERE, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mixed_vs_global import (DROP_GROUPS, fit_global, fit_mixed, frame, logit_fixed,  # noqa: E402
                             parse_vc, random_part)
from models import make_prep  # noqa: E402

OUT = os.path.join(HERE, "results", "personal_al_meeting")
N_SEED, N_SEED_POS, K, ROUNDS, N_SLOPES = 10, 2, 10, 20, 10
CHECKPOINTS = (50, 100, 200)


# ----------------------------------------------------------------------------- data
def load_folds(seed=0, n_folds=3):
    """Folds plus, for every training row, which participant and which DAY it came from.

    day_group_folds returns arrays, not indices, so the day of a training row cannot be recovered
    from its output. Feeding it the row index as the feature matrix returns the indices the split
    actually used (the trick walking/representations/experiments/semisynth_walking.py uses), so the
    real split is reused rather than re-implemented."""
    import meeting_data
    meeting_data.LABEL = "label:IN_A_MEETING"
    groups = [g for g in meeting_data.GROUPS if g not in DROP_GROUPS]
    data, cols = meeting_data.load_meeting(groups)
    assert len(cols) == 175, f"expected the 175-feature set, got {len(cols)}"
    el = meeting_data.eligibility(data, 20).set_index("uuid")
    data = {u: v for u, v in data.items() if el.loc[u, "eligible"]}
    users_all = sorted(data)
    folds, users = meeting_data.day_group_folds(data, users_all, n_folds, 0.3, seed=seed, min_test=5)
    proxy = {u: (np.arange(len(data[u][1]))[:, None], data[u][1], data[u][2]) for u in users_all}
    pfolds, pusers = meeting_data.day_group_folds(proxy, users_all, n_folds, 0.3, seed=seed, min_test=5)
    assert pusers == users, "proxy split diverged from the real split"
    days = {f: {u: data[u][2][sp[u][0].ravel()] for u in users} for f, sp in pfolds}
    # M-ids run over the participants day_group_folds KEEPS (21 of the 38 eligible), which is the
    # convention run_al.py / mixed_vs_global.py use -- numbering over users_all shifts every label
    return folds, users, cols, days, {u: f"M{i + 1}" for i, u in enumerate(users)}


def seed_set(y_p, day_p, rng):
    """10 of p's own labels: N_SEED_POS positives + the rest negatives, spread over >= 2 days."""
    pos, neg = np.where(y_p == 1)[0], np.where(y_p == 0)[0]
    flag = len(pos) < N_SEED_POS
    take = []
    for idx, n in ((pos, min(N_SEED_POS, len(pos))), (neg, N_SEED - min(N_SEED_POS, len(pos)))):
        if n <= 0 or len(idx) == 0:
            continue
        dd = day_p[idx]
        chosen = []
        for d in rng.permutation(np.unique(dd)):            # one per day first, then fill
            cand = idx[dd == d]
            chosen.append(rng.choice(cand))
            if len(chosen) >= min(n, len(np.unique(dd))):
                break
        rest = np.setdiff1d(idx, chosen)
        if n > len(chosen) and len(rest):
            chosen += list(rng.choice(rest, min(n - len(chosen), len(rest)), replace=False))
        take += list(chosen)
    return np.array(sorted(set(take)), dtype=int), flag


# ----------------------------------------------------------------------------- augmented space
class Augment:
    """x~ = [1, x (175), person intercepts (P), person slopes (P x n_slopes)] in a fixed order."""

    def __init__(self, n_feat, users, slope_idx):
        self.n_feat, self.users, self.slope_idx = n_feat, list(users), list(slope_idx)
        self.P, self.S = len(users), len(slope_idx)
        self.pos = {u: i for i, u in enumerate(self.users)}
        self.d = 1 + n_feat + self.P + self.P * self.S

    def build(self, Z, owners):
        """Rows of Z augmented with each row's OWNER slots."""
        n = len(Z)
        A = np.zeros((n, self.d))
        A[:, 0] = 1.0
        A[:, 1:1 + self.n_feat] = Z
        oi = np.array([self.pos[o] for o in owners])
        A[np.arange(n), 1 + self.n_feat + oi] = 1.0
        if self.S:
            base = 1 + self.n_feat + self.P
            cols = base + oi[:, None] * self.S + np.arange(self.S)[None, :]
            np.put_along_axis(A, cols, Z[:, self.slope_idx], axis=1)
        return A

    def prior_precision(self, fe_p, sd_int, sd_slope):
        lam = np.empty(self.d)
        lam[:1 + self.n_feat] = 1.0 / fe_p ** 2
        lam[1 + self.n_feat:1 + self.n_feat + self.P] = 1.0 / max(sd_int, 1e-3) ** 2
        if self.S:
            base = 1 + self.n_feat + self.P
            lam[base:] = np.repeat(1.0 / np.maximum(np.asarray(sd_slope, float), 1e-3) ** 2, 1)[None, :]\
                .repeat(self.P, 0).ravel()
        return lam


def laplace_cov(Xl, pi, lam):
    """Sigma = (X~_L' W X~_L + Lambda)^-1 at the MAP fit."""
    w = np.clip(pi * (1 - pi), 1e-8, None)
    A = (Xl * w[:, None]).T @ Xl
    A[np.diag_indices_from(A)] += lam
    return np.linalg.inv(A)


def var_reduction_batch(Sigma, M, Xc, pi_c, K):
    """Greedy K by score(x) = w v'M v / (1 + w x'v), v = Sigma x, with Sherman-Morrison updates.

    V = Xc Sigma and VM = V M are maintained by rank-1 updates, so only the first round of each
    batch costs a dense product."""
    w = np.clip(pi_c * (1 - pi_c), 1e-8, None)
    V = Xc @ Sigma
    VM = V @ M
    picked, mask = [], np.ones(len(Xc), bool)
    for _ in range(K):
        num = w * np.einsum("ij,ij->i", VM, V)
        den = 1.0 + w * np.einsum("ij,ij->i", Xc, V)
        score = np.where(mask, num / den, -np.inf)
        j = int(np.argmax(score))
        picked.append(j); mask[j] = False
        v = V[j].copy()                             # = Sigma x_j (copy: V is updated in place below)
        c = w[j] / (1.0 + w[j] * (Xc[j] @ v))
        Xv = V @ (Xc[j])                            # (n,) = Xc Sigma x_j
        V -= c * np.outer(Xv, v)                    # V = Xc (Sigma - c v v')
        VM -= c * np.outer(Xv, M @ v)
        Sigma = Sigma - c * np.outer(v, v)
    return np.array(picked, dtype=int), Sigma


# ----------------------------------------------------------------------------- unit test (spec 5)
def unit_test(seed=0):
    rng = np.random.default_rng(seed)
    n, d, m = 40, 6, 7
    A = rng.normal(size=(d, d))
    Sigma = np.linalg.inv(A @ A.T + 3 * np.eye(d))      # symmetric PSD: the identity needs it
    Zp = rng.normal(size=(m, d)); M = Zp.T @ Zp
    Xc = rng.normal(size=(n, d)); pi = rng.uniform(0.2, 0.8, n)
    w = pi * (1 - pi)
    before = np.einsum("ij,jk,ik->", Zp, Sigma, Zp)
    worst = 0.0
    for i in range(n):
        v = Sigma @ Xc[i]
        closed = w[i] * (v @ M @ v) / (1 + w[i] * (Xc[i] @ v))
        S2 = np.linalg.inv(np.linalg.inv(Sigma) + w[i] * np.outer(Xc[i], Xc[i]))
        brute = before - np.einsum("ij,jk,ik->", Zp, S2, Zp)
        worst = max(worst, abs(closed - brute))
    print(f"unit test: max |closed-form - brute force| over {n} candidates = {worst:.3e}")
    assert worst < 1e-8, "closed-form score does not match the brute-force variance reduction"

    # the greedy batch path: Sherman-Morrison updates must track a full recompute
    Sg, M2 = Sigma.copy(), M
    pick, Sig_end = var_reduction_batch(Sigma, M, Xc, pi, 4)
    for j in pick:
        v = Sg @ Xc[j]; c = w[j] / (1 + w[j] * (Xc[j] @ v))
        Sg = Sg - c * np.outer(v, v)
    print(f"greedy batch: max |Sigma_incremental - Sigma_recomputed| = {np.abs(Sg - Sig_end).max():.3e}")
    assert np.abs(Sg - Sig_end).max() < 1e-10, "Sherman-Morrison updates drifted"

    chosen, Sb = [], Sigma.copy()                    # brute-force greedy, rescoring every candidate
    for _ in range(4):
        best, bj = -np.inf, None
        for i in range(n):
            if i in chosen:
                continue
            vv = Sb @ Xc[i]
            sc = w[i] * (vv @ M @ vv) / (1 + w[i] * (Xc[i] @ vv))
            if sc > best:
                best, bj = sc, i
        chosen.append(bj)
        vv = Sb @ Xc[bj]; cc = w[bj] / (1 + w[bj] * (Xc[bj] @ vv))
        Sb = Sb - cc * np.outer(vv, vv)
    print(f"greedy picks: fast {list(pick)} vs brute force {chosen}")
    assert list(pick) == chosen, "the vectorised greedy picked a different set"
    print("unit test PASSED")


# ----------------------------------------------------------------------------- one (fold, target, repeat)
def run_one(fold, sp, users, ids, Zpool, ytr, owner, prep, aug, slope_idx, t, rep, args, feats):
    """Both methods from the same seed set. Returns rows and the per-method fit-failure count."""
    rows, fails = [], 0
    ytr_t = sp[t][2]
    i_t = np.where(owner == t)[0]
    rng_seed = np.random.default_rng([args.seed, fold, users.index(t), rep])
    loc, flagged = seed_set(ytr_t, args.days[fold][t], rng_seed)
    seed_rows = i_t[loc]
    Zte, yte = prep.transform(sp[t][1]), sp[t][3]
    if not (0 < yte.sum() < len(yte)):
        return rows, fails

    for method in ("var_reduction", "random"):
        labeled = list(seed_rows)
        rng = np.random.default_rng([args.seed, fold, users.index(t), rep, 99])
        for rnd in range(ROUNDS + 1):
            idx = np.array(labeled, dtype=int)
            df = frame(Zpool[idx], ytr[idx], owner[idx], ids)
            status, auc = "ok", np.nan
            try:
                res = fit_mixed(df, feats, args.vc, "map")
                fe = np.asarray(res.fe_mean, float)
                if np.abs(fe).max() == 0:
                    status = "all_zero"; fails += 1
                else:
                    vc = parse_vc(res)[0]
                    auc = roc_auc_score(yte, logit_fixed(res, Zte) + random_part(vc, ids[t], Zte, slope_idx))
            except Exception as e:                                   # logged, never substituted
                status = f"fail:{type(e).__name__}"; fails += 1
                res = None
            rows.append(dict(fold=fold, target=ids[t], repeat=rep, method=method, round=rnd,
                             n_labeled=len(idx), auroc=auc, status=status,
                             self_share=float(np.mean(owner[idx] == t)),
                             n_pos=int(ytr[idx].sum()), seed_flagged=bool(flagged)))
            if rnd == ROUNDS or status != "ok":
                if status != "ok":
                    print(f"      [{method}] round {rnd}: {status}", flush=True)
                break
            unl = np.setdiff1d(np.arange(len(ytr)), idx)
            if method == "random":
                new = rng.choice(unl, min(K, len(unl)), replace=False)
            else:
                pi_l = expit(logit_fixed(res, Zpool[idx]) +
                             np.array([random_part(parse_vc(res)[0], ids[o], Zpool[[i]], slope_idx)[0]
                                       for i, o in zip(idx, owner[idx])]))
                Xl = aug.build(Zpool[idx], [ids[o] for o in owner[idx]])
                sd_i, sd_s = args.sd_of(res)
                Sigma = laplace_cov(Xl, pi_l, aug.prior_precision(args.fe_p, sd_i, sd_s))
                unl_t = np.intersect1d(unl, i_t)                     # p's own unlabeled windows
                Zp_aug = aug.build(Zpool[unl_t], [ids[t]] * len(unl_t))
                M = Zp_aug.T @ Zp_aug
                Xc = aug.build(Zpool[unl], [ids[o] for o in owner[unl]])
                pi_c = expit(logit_fixed(res, Zpool[unl]))
                sel, _ = var_reduction_batch(Sigma, M, Xc, pi_c, min(K, len(unl)))
                new = unl[sel]
            labeled += list(new)
    return rows, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--targets", nargs="*", default=None, help="M-ids; default every participant")
    ap.add_argument("--unit_test", action="store_true"); ap.add_argument("--time_one", action="store_true")
    ap.add_argument("--fold", type=int, default=None, help="run this fold only (one process per fold)")
    ap.add_argument("--max_repeat", type=int, default=None, help="stop after this repeat index (inclusive)")
    ap.add_argument("--repeat", type=int, default=None,
                    help="run exactly this repeat (the cluster shard selector, with --fold)")
    ap.add_argument("--tag", default="", help="suffix for this shard's part file, e.g. _r0")
    ap.add_argument("--summarize", action="store_true", help="build summary.md + figures from what exists")
    ap.add_argument("--fe_p", type=float, default=2.0, help="BinomialBayesMixedGLM fixed-effect prior SD")
    args = ap.parse_args()
    if args.unit_test:
        unit_test(args.seed); return
    os.makedirs(OUT, exist_ok=True)

    folds, users, cols, days, ids = load_folds(args.seed, args.folds)
    args.days = days
    feats = [f"f{i}" for i in range(len(cols))]
    print(f"IN_A_MEETING: {len(users)} participants, {len(cols)} features, {len(folds)} folds", flush=True)

    def sd_of(res):
        vcp = np.asarray(res.vcp_mean, float)
        return float(np.exp(vcp[0])), np.exp(vcp[1:]) if len(vcp) > 1 else np.array([])
    args.sd_of = sd_of

    if args.summarize:
        summarize(args); return

    t0_all = time.time()
    for fold, sp in folds[:args.folds]:
        if args.fold is not None and fold != args.fold:
            continue
        # one part file per SHARD: concurrent jobs on the same fold must not append to one file.
        # --summarize globs per_round_fold*.csv, so any sharding scheme is picked up unchanged.
        tag = args.tag or (f"_r{args.repeat}" if args.repeat is not None else "")
        part = os.path.join(OUT, f"per_round_fold{fold}{tag}.csv")
        done = set()
        if os.path.exists(part):
            d0 = pd.read_csv(part)
            done = {(r.target, int(r.repeat)) for r in d0.itertuples()}
            print(f"  fold {fold}: {len(done)} units already finished, skipping them", flush=True)
        Xtr = np.vstack([sp[u][0] for u in users]); ytr = np.concatenate([sp[u][2] for u in users])
        owner = np.concatenate([[u] * len(sp[u][2]) for u in users])
        prep = make_prep().fit(Xtr); Zpool = prep.transform(Xtr)
        g = fit_global(frame(Zpool, ytr, owner, ids), feats)          # slope features: once per fold
        fe_g = np.asarray(g.params)[1:]
        order = [int(i) for i in np.argsort(-np.abs(fe_g))]
        slope_idx = [i for i in order if not cols[i].startswith("discrete:")][:N_SLOPES]
        print(f"  fold {fold}: slopes = {', '.join(cols[i] for i in slope_idx)}", flush=True)
        args.vc = dict({"person": "0 + C(person)"},
                       **{f"slope_f{k}": f"0 + C(person):f{k}" for k in slope_idx})
        aug = Augment(len(cols), [ids[u] for u in users], slope_idx)

        tgts = [u for u in users if args.targets is None or ids[u] in set(args.targets)]
        if args.repeat is not None:
            reps = [args.repeat]
        else:
            last = args.repeats - 1 if args.max_repeat is None else min(args.max_repeat, args.repeats - 1)
            reps = list(range(last + 1))
        for rep in reps:                                # repeat-first: repeat 0 for every target first
            for t in tgts:
                if (ids[t], rep) in done:
                    continue
                t0 = time.time()
                rows, fails = run_one(fold, sp, users, ids, Zpool, ytr, owner, prep, aug,
                                      slope_idx, t, rep, args, feats)
                if rows:                                 # append per unit: a crash keeps the rest
                    pd.DataFrame(rows).to_csv(part, mode="a", header=not os.path.exists(part),
                                              index=False)
                print(f"    rep {rep} {ids[t]}: {time.time() - t0:.1f}s ({fails} failed fits)", flush=True)
                if args.time_one:
                    dt = time.time() - t0
                    n_units = args.folds * len(users) * args.repeats
                    print(f"\nprojection: {dt:.1f}s per (fold,target,repeat) x {n_units} units "
                          f"= {dt * n_units / 3600:.1f} h for both methods", flush=True)
                    return
            print(f"  fold {fold}: repeat {rep} complete", flush=True)
    print(f"\ntotal {(time.time() - t0_all) / 3600:.2f} h; saved {OUT}", flush=True)


# ----------------------------------------------------------------------------- reporting
def summarize(args):
    """summary.md + the three figures from whatever per_round_fold*.csv exist (interim-safe)."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = sorted(glob.glob(os.path.join(OUT, "per_round_fold*.csv")))
    if not parts:
        print("nothing to summarize"); return
    D = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    D.to_csv(os.path.join(OUT, "per_round.csv"), index=False)
    D["acquired"] = D.n_labeled - N_SEED
    reps = sorted(D.repeat.unique())
    print(f"summarizing {len(D)} rows: folds {sorted(D.fold.unique())}, repeats {reps}, "
          f"{D.target.nunique()} targets", flush=True)

    # mean over repeats, then over folds
    per_t = (D.groupby(["method", "target", "fold", "acquired"]).auroc.mean()
             .groupby(["method", "target", "acquired"]).mean().reset_index())
    curve = per_t.groupby(["method", "acquired"]).auroc.mean().unstack(0)
    share = (D.groupby(["method", "fold", "target", "acquired"]).self_share.mean()
             .groupby(["method", "acquired"]).mean().unstack(0))
    pos = (D.groupby(["method", "fold", "target", "acquired"]).n_pos.mean()
           .groupby(["method", "acquired"]).mean().unstack(0))
    seed_only = per_t[per_t.acquired == 0].groupby("target").auroc.mean()

    ref = os.path.join(HERE, "results", "mixed_vs_global", "table_meeting.csv")
    full = pd.read_csv(ref, index_col=0)["mixed_map"] if os.path.exists(ref) else None

    rows = []
    for m in curve.columns:
        r = dict(method=m)
        for c in CHECKPOINTS:
            r[f"auroc@{c}"] = curve[m].get(c, np.nan)
        r["final_self_share"] = share[m].dropna().iloc[-1]
        rows.append(r)
    S = pd.DataFrame(rows)
    for c in CHECKPOINTS:                       # paired difference per target
        w = per_t[per_t.acquired == c].pivot(index="target", columns="method", values="auroc")
        if {"var_reduction", "random"}.issubset(w.columns):
            d = (w.var_reduction - w.random).dropna()
            S.loc[S.method == "var_reduction", f"delta@{c}"] = d.mean()
            S.loc[S.method == "var_reduction", f"se@{c}"] = d.std(ddof=1) / np.sqrt(len(d))
            S.loc[S.method == "var_reduction", f"won@{c}"] = f"{int((d > 0).sum())}/{len(d)}"
    S.to_csv(os.path.join(OUT, "summary_table.csv"), index=False)

    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# Personalized AL on IN_A_MEETING (mixed effects, fit_map)\n\n")
        fh.write(f"folds {sorted(D.fold.unique())}, repeats {reps}, {D.target.nunique()} targets; "
                 f"{int((D.status != 'ok').sum())} failed fits of {len(D)} rounds\n\n")
        fh.write(S.round(4).to_markdown(index=False) + "\n\n")
        w = per_t[per_t.acquired == CHECKPOINTS[-1]].pivot(index="target", columns="method", values="auroc")
        if {"var_reduction", "random"}.issubset(w.columns):
            d = (w.var_reduction - w.random).dropna().sort_values(ascending=False)
            fh.write("## Conclusions\n\n")
            fh.write(f"- At {CHECKPOINTS[-1]} acquired labels variance reduction is "
                     f"{d.mean():+.4f} vs random (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f}), "
                     f"winning on {int((d > 0).sum())}/{len(d)} targets.\n")
            fh.write(f"- It places {share['var_reduction'].dropna().iloc[-1]:.1%} of the labels on the "
                     f"target itself, against {share['random'].dropna().iloc[-1]:.1%} for random "
                     f"(the pool share of one participant).\n")
            fh.write(f"- Positives acquired by the end: {pos['var_reduction'].dropna().iloc[-1]:.1f} "
                     f"vs {pos['random'].dropna().iloc[-1]:.1f}.\n")
            fh.write(f"- Benefits most: {', '.join(f'{i} {v:+.3f}' for i, v in d.head(4).items())}; "
                     f"hurts most: {', '.join(f'{i} {v:+.3f}' for i, v in d.tail(3).items())}.\n")
            if full is not None:
                fh.write(f"- References: seed-only {seed_only.mean():.4f}; mixed model on 100% of the "
                         f"training labels {full.reindex(seed_only.index).mean():.4f}.\n")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m, c in (("var_reduction", "seagreen"), ("random", "darkorange")):
        if m in curve:
            ax.plot(curve.index, curve[m], "-o", ms=3, color=c, label=m)
    ax.axhline(seed_only.mean(), color="grey", ls=":", lw=1, label="seed only")
    if full is not None:
        ax.axhline(full.reindex(seed_only.index).mean(), color="crimson", ls="--", lw=1,
                   label="mixed, 100% of labels")
    ax.set_xlabel("labels acquired"); ax.set_ylabel("test AUROC"); ax.legend(fontsize=8)
    ax.set_title("IN_A_MEETING personalized AL: mean over targets", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "auroc_curve.png"), dpi=150, facecolor="white")

    fig, ax = plt.subplots(figsize=(7, 4.0))
    for m, c in (("var_reduction", "seagreen"), ("random", "darkorange")):
        if m in share:
            ax.plot(share.index, share[m], "-o", ms=3, color=c, label=m)
    ax.set_xlabel("labels acquired"); ax.set_ylabel("share of labels from the target")
    ax.legend(fontsize=8); ax.set_title("self-share", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "self_share.png"), dpi=150, facecolor="white")

    tg = sorted(per_t.target.unique(), key=lambda x: int(x[1:]))
    nc = 5; nr = int(np.ceil(len(tg) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.4 * nr), sharex=True, sharey=True)
    for ax, t in zip(np.asarray(axes).ravel(), tg):
        g = per_t[per_t.target == t]
        for m, c in (("var_reduction", "seagreen"), ("random", "darkorange")):
            h = g[g.method == m]
            ax.plot(h.acquired, h.auroc, color=c, lw=1, label=m)
        if full is not None and t in full.index:
            ax.axhline(full[t], color="crimson", ls="--", lw=0.8)
        ax.set_title(t, fontsize=8); ax.set_ylim(0.3, 1.02)
    for ax in np.asarray(axes).ravel()[len(tg):]:
        ax.axis("off")
    np.asarray(axes).ravel()[0].legend(fontsize=6)
    fig.suptitle("per-target AUROC vs labels acquired (red = mixed model on 100% of labels)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_target_grid.png"), dpi=150, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
