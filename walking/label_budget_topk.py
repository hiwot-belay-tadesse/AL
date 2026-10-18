"""Are CoreSet's over-queried participants more label-efficient than under-queried or random
participants, at a matched label budget AND a matched pool size?  (pool.txt)

Subset selection from an existing CoreSet run (<run>/al_query_sources.csv, al_full_reference.csv):
  Delta_p(fold) = CoreSet cumulative query share of source p through round 20 (summed over
  targets, rounds >= 1) - pool share of p in that fold      [same arithmetic as the Fig. 6 heatmap]
  TOP-k    : Delta_p > 0 in all 3 folds, ranked by mean Delta (largest first), first k
  BOTTOM-k : Delta_p < 0 in all 3 folds, ranked by mean Delta (most negative first), first k
  RANDOM-k : --n_random draws of k participants from the remaining ones (TOP-k, BOTTOM-k excluded)

Size matching (per fold): n_match = smallest train-pool row count among TOP-k, BOTTOM-k and every
RANDOM-k draw; every pool is subsampled to n_match rows, stratified by participant (each
participant keeps its proportion within the pool), with --n_sub subsampling seeds.

Label-budget experiment (per fold x subsample seed x pool): initial labeled set of n_init rows
(same size and same stratified draw as the main AL runs), RANDOM acquisition with K per round for
`budget` rounds; after every round the classifier is retrained on the labeled set and scored on
EVERY target's test day (22 targets). Reference: the same procedure on the full 22-participant
pool (no size matching). 100% = the model trained on the whole (subsampled) pool.

Outputs (results/<run>/label_budget/):
  label_budget_topk.png/.pdf   one panel per k, mean AUROC over targets vs acquired labels,
                               band = min-max over folds
  label_budget_top<k>_grid.png/.pdf  one panel per target (22), same four curves + dashed
                               100%-of-TOP-k-pool and dotted 100%-of-FULL-pool references
  label_budget_diff_summary.png/.pdf  one panel per k: paired TOP-BOTTOM and TOP-RANDOM AUROC
                               difference (mean over targets) vs acquired labels, 95% bootstrap CI
                               over targets; dashed = excluding targets whose rows are in TOP/BOTTOM
  label_budget_raw.csv         per (k, pool, draw, fold, sub_seed, round, target) AUROC
  label_budget_summary.csv/.tex  at 50/100/200 acquired labels and 100%: mean AUROC per pool,
                               paired TOP-BOTTOM and TOP-RANDOM over targets with 95% bootstrap
                               CI, #targets with TOP > ..., two-sided sign-test p, and the same
                               Delta excluding targets whose own rows are in TOP-k or BOTTOM-k

    python label_budget_topk.py --run al_global_rolling_K10_baseline --ks 3 5
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from active_learning import N_INIT, initial_labeled_unlabeled_split, pick_random
from data import RESULTS_DIR, folds_for, load_sorted, select_users, short_ids
from models import make_clf, make_prep

BUDGETS = [50, 100, 200]          # acquired labels (excluding the initial labeled set)
N_BOOT = 1000


# ----------------------------------------------------------------------------- subset selection
def coreset_delta_per_fold(run_dir, final_round):
    """Delta_p per fold: CoreSet cumulative query share through final_round (rounds >= 1, summed over
    targets) minus pool share (rows of p / rows in the fold's pool). Returns DataFrame [participant x fold]."""
    src = pd.read_csv(os.path.join(run_dir, "al_query_sources.csv"))
    src = src[(src.method == "coreset") & (src["round"] >= 1) & (src["round"] <= final_round)]
    q = src.groupby(["fold", "source"]).n.sum().unstack("source").fillna(0)
    share = q.div(q.sum(axis=1), axis=0)
    ref = pd.read_csv(os.path.join(run_dir, "al_full_reference.csv")).drop_duplicates(["fold", "participant"])
    pool = ref.pivot(index="fold", columns="participant", values="n_pool_user").reindex(columns=share.columns)
    pool_share = pool.div(pool.sum(axis=1), axis=0)
    return (share - pool_share).T          # participant x fold


def select_subsets(delta, k, n_random, rng, mapping):
    """TOP-k / BOTTOM-k by sign-consistent Delta, RANDOM-k draws from the rest."""
    mean_d = delta.mean(axis=1)
    top = mean_d[(delta > 0).all(axis=1)].sort_values(ascending=False).index.tolist()[:k]
    bottom = mean_d[(delta < 0).all(axis=1)].sort_values().index.tolist()[:k]
    rest = [u for u in delta.index if u not in top and u not in bottom]
    draws = [sorted(rng.choice(rest, k, replace=False).tolist()) for _ in range(n_random)]
    ids = lambda us: ", ".join(mapping[u] for u in us)
    print(f"k={k}: TOP-{k} = [{ids(top)}]  (mean Delta {', '.join(f'{mean_d[u]:+.3f}' for u in top)})")
    print(f"k={k}: BOTTOM-{k} = [{ids(bottom)}]  (mean Delta {', '.join(f'{mean_d[u]:+.3f}' for u in bottom)})")
    for i, d in enumerate(draws):
        print(f"k={k}: RANDOM-{k} #{i + 1} = [{ids(d)}]")
    if len(top) < k or len(bottom) < k:
        print(f"WARNING: only {len(top)} sign-consistent positive / {len(bottom)} negative participants available")
    return top, bottom, draws


# ----------------------------------------------------------------------------- pools
def pool_rows(splits, members):
    X = np.vstack([splits[u][0] for u in members]); y = np.concatenate([splits[u][2] for u in members])
    owner = np.concatenate([[u] * len(splits[u][2]) for u in members])
    return X, y, owner


def stratified_subsample(owner, n_match, rng):
    """Row indices: n_match rows, each participant keeps its proportion (largest-remainder rounding)."""
    members, counts = np.unique(owner, return_counts=True)
    exact = counts * n_match / counts.sum()
    take = np.floor(exact).astype(int)
    for i in np.argsort(exact - take)[::-1][: n_match - take.sum()]:
        take[i] += 1
    idx = [rng.choice(np.where(owner == u)[0], t, replace=False) for u, t in zip(members, take)]
    return np.sort(np.concatenate(idx))


# ----------------------------------------------------------------------------- label-budget run
def random_budget_curve(X_pool, y_pool, tests, budget, K, n_init, rng_init, rng_q):
    """Random acquisition on one pool; after every round fit on the labeled set and score all targets.
    tests: {target: (X_te, y_te)}. Returns rows (round, n_labeled, acquired, target, test_auroc)
    plus 100%-of-pool rows (round=-1)."""
    prep = make_prep().fit(X_pool); Z = prep.transform(X_pool)
    Zte = {u: prep.transform(Xte) for u, (Xte, _) in tests.items()}
    n = len(y_pool); labeled = np.zeros(n, dtype=bool)
    labeled[initial_labeled_unlabeled_split(y_pool, rng_init, n_init)] = True
    n0 = int(labeled.sum()); rows = []

    def score(mask, round_num):
        clf = make_clf().fit(Z[mask], y_pool[mask])
        for u, (_, yte) in tests.items():
            rows.append(dict(round=round_num, n_labeled=int(mask.sum()), acquired=int(mask.sum()) - n0 if round_num >= 0 else n,
                             target=u, test_auroc=roc_auc_score(yte, clf.predict_proba(Zte[u])[:, 1])))

    score(labeled, 0)
    for r in range(1, budget + 1):
        unl = np.where(~labeled)[0]
        if len(unl) == 0:
            break
        labeled[pick_random(K, unl, rng_q)] = True
        score(labeled, r)
    score(np.ones(n, dtype=bool), -1)
    return rows


# ----------------------------------------------------------------------------- statistics
def paired_stats(a, b, rng):
    """a, b: per-target arrays (aligned). Mean difference, 95% bootstrap CI over targets, #a>b, sign-test p."""
    d = a - b
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses > 0 else np.nan
    return dict(delta=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), wins=wins, n=len(d), p_sign=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="al_global_rolling_K10_baseline")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--n_random", type=int, default=10); ap.add_argument("--n_sub", type=int, default=3)
    ap.add_argument("--budget", type=int, default=20); ap.add_argument("--K", type=int, default=10); ap.add_argument("--n_init", type=int, default=N_INIT)
    ap.add_argument("--final_round", type=int, default=20, help="round at which Delta is read")
    ap.add_argument("--plots_only", action="store_true")
    args = ap.parse_args()
    run_dir = os.path.join(RESULTS_DIR, args.run); out = os.path.join(run_dir, "label_budget"); os.makedirs(out, exist_ok=True)

    data = load_sorted(); users, _, _ = select_users(data, "rolling"); mapping = short_ids(users, RESULTS_DIR)
    delta = coreset_delta_per_fold(run_dir, args.final_round).reindex(users)
    delta.rename(index=mapping).round(4).to_csv(os.path.join(out, "coreset_delta_per_fold.csv"))
    subsets = {k: select_subsets(delta, k, args.n_random, np.random.default_rng(k), mapping) for k in args.ks}

    if not args.plots_only:
        folds = folds_for(data, users, "rolling")
        raw, sizes = [], []
        for fold, splits in folds:
            tests = {u: (splits[u][1], splits[u][3]) for u in users}
            # reference: full pool, no size matching (one run per subsample seed so the band is comparable)
            Xf, yf, _ = pool_rows(splits, users)
            for s in range(args.n_sub):
                rows = random_budget_curve(Xf, yf, tests, args.budget, args.K, args.n_init,
                                           np.random.default_rng([fold, s, 0]), np.random.default_rng([fold, s, 1]))
                raw += [dict(k=0, pool="FULL", draw=0, fold=fold, sub_seed=s, n_pool=len(yf), **r) for r in rows]
            for k, (top, bottom, draws) in subsets.items():
                pools = {("TOP", 0): top, ("BOTTOM", 0): bottom, **{("RANDOM", i + 1): d for i, d in enumerate(draws)}}
                raw_rows = {key: pool_rows(splits, m) for key, m in pools.items()}
                n_match = min(len(y) for _, y, _ in raw_rows.values())
                for (name, i), (X, y, owner) in raw_rows.items():
                    sizes.append(dict(k=k, pool=name, draw=i, fold=fold, n_rows=len(y), n_match=n_match))
                print(f"fold {fold} k={k}: n_match={n_match}; rows before subsampling: "
                      + ", ".join(f"{n}{'#' + str(i) if i else ''}={len(y)}" for (n, i), (_, y, _) in raw_rows.items()), flush=True)
                for (name, i), (X, y, owner) in raw_rows.items():
                    for s in range(args.n_sub):
                        idx = stratified_subsample(owner, n_match, np.random.default_rng([fold, s, k, i, {"TOP": 1, "BOTTOM": 2, "RANDOM": 3}[name]]))
                        rows = random_budget_curve(X[idx], y[idx], tests, args.budget, args.K, args.n_init,
                                                   np.random.default_rng([fold, s, 0]), np.random.default_rng([fold, s, 1]))
                        raw += [dict(k=k, pool=name, draw=i, fold=fold, sub_seed=s, n_pool=n_match, **r) for r in rows]
            print(f"fold {fold} done", flush=True)
        raw = pd.DataFrame(raw); raw["target"] = raw.target.map(mapping)
        raw.to_csv(os.path.join(out, "label_budget_raw.csv"), index=False)
        pd.DataFrame(sizes).assign(pool=lambda d: d.pool).to_csv(os.path.join(out, "pool_sizes.csv"), index=False)
    else:
        raw = pd.read_csv(os.path.join(out, "label_budget_raw.csv"))

    # ------------------------------------------------------------------ aggregate: per (k, pool, round): mean over targets, seeds, draws -> per fold
    full = raw[raw.pool == "FULL"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"TOP": "#c0392b", "BOTTOM": "#2e6fa7", "RANDOM": "gray", "FULL": "black"}
    fig, axes = plt.subplots(1, len(args.ks), figsize=(6.5 * len(args.ks), 5), sharey=True, squeeze=False); axes = axes[0]
    for ax, k in zip(axes, args.ks):
        d = pd.concat([raw[raw.k == k], full])
        d = d[d["round"] >= 0]
        per_fold = d.groupby(["pool", "fold", "acquired"]).test_auroc.mean().reset_index()      # mean over targets, seeds, draws
        curve = per_fold.groupby(["pool", "acquired"]).test_auroc.agg(["mean", "min", "max"]).reset_index()
        for name in ["TOP", "BOTTOM", "RANDOM", "FULL"]:
            c = curve[curve.pool == name]
            lab = {"TOP": f"TOP-{k}", "BOTTOM": f"BOTTOM-{k}", "RANDOM": f"RANDOM-{k} (mean of {args.n_random} draws)", "FULL": "FULL pool (22 participants)"}[name]
            ax.plot(c.acquired, c["mean"], color=colors[name], lw=2, label=lab)
            ax.fill_between(c.acquired, c["min"], c["max"], color=colors[name], alpha=0.12)
        ax.set_title(f"k = {k} (pools size-matched per fold)"); ax.set_xlabel(f"acquired labels (random acquisition, K={args.K}; excl. {args.n_init} initial)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("mean test AUROC over 22 targets (band = min-max over folds)")
    fig.suptitle("Label efficiency of CoreSet's over-queried (TOP) vs under-queried (BOTTOM) vs random participants")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"label_budget_topk.{ext}"), dpi=150, facecolor="white")

    # ------------------------------------------------------------------ per-target grid, one file per k
    order = sorted(mapping.values(), key=lambda p_: int(p_[1:]))
    ncols = 5; nrows = int(np.ceil(len(order) / ncols))
    for k in args.ks:
        top, bottom, _ = subsets[k]; in_top = {mapping[u] for u in top}; in_bot = {mapping[u] for u in bottom}
        d = pd.concat([raw[raw.k == k], full]); d = d[d["round"] >= 0]
        per_fold = d.groupby(["pool", "target", "fold", "acquired"]).test_auroc.mean().reset_index()   # mean over seeds, draws
        curve = per_fold.groupby(["pool", "target", "acquired"]).test_auroc.agg(["mean", "min", "max"]).reset_index()
        full100 = raw[(raw["round"] == -1)].groupby(["k", "pool", "target"]).test_auroc.mean()
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), sharey=True); axes = np.asarray(axes).reshape(-1)
        for ax, pid in zip(axes, order):
            for name in ["TOP", "BOTTOM", "RANDOM", "FULL"]:
                c = curve[(curve.pool == name) & (curve.target == pid)]
                ax.plot(c.acquired, c["mean"], color=colors[name], lw=1.8, label=f"{name}-{k}" if name != "FULL" else "FULL")
                ax.fill_between(c.acquired, c["min"], c["max"], color=colors[name], alpha=0.12)
            for name, kk, ls in [("TOP", k, "--"), ("FULL", 0, ":")]:
                v = full100.get((kk, name, pid), np.nan)
                ax.axhline(v, color=colors[name], ls=ls, lw=1, label=f"100% {name}{'-' + str(k) if kk else ''} pool = {v:.3f}")
            tag = " (in TOP)" if pid in in_top else " (in BOTTOM)" if pid in in_bot else ""
            ax.set_title(pid + tag, fontsize=10, fontweight="bold" if tag else "normal")
            ax.grid(alpha=0.3); ax.legend(fontsize=5.5, loc="lower right"); ax.tick_params(labelsize=7)
            ax.set_xlabel("acquired labels", fontsize=8)
        for ax in axes[len(order):]:
            ax.axis("off")
        for i in range(0, len(order), ncols):
            axes[i].set_ylabel("test AUROC")
        axes[0].set_ylim(0.3, 1.0)
        fig.suptitle(f"Per-target label-budget curves, k={k}: TOP = {', '.join(sorted(in_top, key=lambda p_: int(p_[1:])))}; "
                     f"BOTTOM = {', '.join(sorted(in_bot, key=lambda p_: int(p_[1:])))}; random acquisition on size-matched pools; "
                     f"mean over 3 folds (band = min-max over folds)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"label_budget_top{k}_grid.{ext}"), dpi=150, facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------ paired-difference summary: TOP - BOTTOM, TOP - RANDOM vs acquired labels
    def boot_ci(d, rng):
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
        return np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    fig, axes = plt.subplots(1, len(args.ks), figsize=(6.5 * len(args.ks), 5), sharey=True, squeeze=False); axes = axes[0]
    dcol = {"BOTTOM": "#2e6fa7", "RANDOM": "gray"}
    for ax, k in zip(axes, args.ks):
        top, bottom, _ = subsets[k]; own = {mapping[u] for u in top + bottom}
        d = raw[(raw.k == k) & (raw["round"] >= 0)]
        pt = d.groupby(["pool", "target", "acquired"]).test_auroc.mean().unstack("pool")     # per target x acquired: mean over folds, seeds, draws
        rng_b = np.random.default_rng([k, 7])
        for other in ["BOTTOM", "RANDOM"]:
            for excl, ls, alpha in [(False, "-", 0.15), (True, "--", 0.0)]:
                q = pt[~pt.index.get_level_values("target").isin(own)] if excl else pt
                diff = (q["TOP"] - q[other]).unstack("target")           # acquired x target
                mean = diff.mean(axis=1); ci = np.array([boot_ci(diff.loc[a].values, rng_b) for a in diff.index])
                lab = f"TOP-{k} $-$ {other}-{k}" + (f" (excl. {len(own)} own targets, n={diff.shape[1]})" if excl else f" (n={diff.shape[1]} targets)")
                ax.plot(diff.index, mean, color=dcol[other], ls=ls, lw=2, label=lab)
                if excl:
                    ax.plot(diff.index, ci[:, 0], color=dcol[other], ls=ls, lw=0.6, alpha=0.6); ax.plot(diff.index, ci[:, 1], color=dcol[other], ls=ls, lw=0.6, alpha=0.6)
                else:
                    ax.fill_between(diff.index, ci[:, 0], ci[:, 1], color=dcol[other], alpha=alpha)
        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"k = {k}: TOP = {', '.join(mapping[u] for u in top)}; BOTTOM = {', '.join(mapping[u] for u in bottom)}", fontsize=10)
        ax.set_xlabel(f"acquired labels (random acquisition, K={args.K}; excl. {args.n_init} initial)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("paired AUROC difference, mean over targets\n(band / thin lines = 95% bootstrap CI over targets)")
    fig.suptitle("Label efficiency: TOP-k minus BOTTOM-k and minus RANDOM-k, size-matched pools, mean over folds and subsample seeds")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"label_budget_diff_summary.{ext}"), dpi=150, facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------ summary table
    rng = np.random.default_rng(0); summ = []
    for k in args.ks:
        top, bottom, _ = subsets[k]; own = {mapping[u] for u in top + bottom}
        d = pd.concat([raw[raw.k == k], full])
        for budget in BUDGETS + ["100%"]:
            sel = d[d["round"] == -1] if budget == "100%" else d[(d["round"] >= 0) & (d.acquired == budget)]
            pt = sel.groupby(["pool", "target"]).test_auroc.mean().unstack("pool")      # per target: mean over folds, seeds, draws
            row = dict(k=k, budget=budget, **{f"auroc_{p}": pt[p].mean() for p in ["TOP", "BOTTOM", "RANDOM", "FULL"]})
            for other in ["BOTTOM", "RANDOM"]:
                st = paired_stats(pt["TOP"].values, pt[other].values, rng)
                row.update({f"top_minus_{other.lower()}": st["delta"], f"top_minus_{other.lower()}_ci": f"[{st['ci_lo']:+.3f}, {st['ci_hi']:+.3f}]",
                            f"top_gt_{other.lower()}": f"{st['wins']}/{st['n']}", f"p_sign_{other.lower()}": st["p_sign"]})
                ex = pt[~pt.index.isin(own)]
                st2 = paired_stats(ex["TOP"].values, ex[other].values, rng)
                row.update({f"top_minus_{other.lower()}_excl_own": st2["delta"], f"top_minus_{other.lower()}_excl_own_ci": f"[{st2['ci_lo']:+.3f}, {st2['ci_hi']:+.3f}]",
                            f"top_gt_{other.lower()}_excl_own": f"{st2['wins']}/{st2['n']}"})
            row["targets_in_top_or_bottom"] = ", ".join(sorted(own, key=lambda p: int(p[1:])))
            summ.append(row)
    summ = pd.DataFrame(summ); summ.to_csv(os.path.join(out, "label_budget_summary.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(summ.round(3).to_string(index=False))

    lines = []
    for k in args.ks:
        top, bottom, _ = subsets[k]
        lines.append(r"\midrule" + "\n" + r"\multicolumn{11}{l}{$k=%d$: TOP = %s; BOTTOM = %s} \\" % (
            k, ", ".join(mapping[u] for u in top), ", ".join(mapping[u] for u in bottom)))
        for _, r in summ[summ.k == k].iterrows():
            b = r"100\%" if r.budget == "100%" else str(r.budget)
            lines.append(f"{b} & {r.auroc_TOP:.3f} & {r.auroc_BOTTOM:.3f} & {r.auroc_RANDOM:.3f} & {r.auroc_FULL:.3f} & "
                         f"{r.top_minus_bottom:+.3f} {r.top_minus_bottom_ci} & {r.top_gt_bottom} ($p$={r.p_sign_bottom:.3f}) & {r.top_minus_bottom_excl_own:+.3f} {r.top_minus_bottom_excl_own_ci} & "
                         f"{r.top_minus_random:+.3f} {r.top_minus_random_ci} & {r.top_gt_random} ($p$={r.p_sign_random:.3f}) & {r.top_minus_random_excl_own:+.3f} {r.top_minus_random_excl_own_ci} \\\\")
    tex = r"""\begin{table*}[t]
\centering
\caption{Label efficiency of CoreSet's over-queried participants (TOP-$k$: $\Delta_p>0$ in all 3 folds, ranked by mean $\Delta_p$) vs.\ under-queried (BOTTOM-$k$: $\Delta_p<0$ in all folds) vs.\ RANDOM-$k$ draws from the remaining participants (%d draws), with every pool subsampled per fold to the smallest pool's row count (stratified by participant, %d subsampling seeds). Random acquisition, $K=%d$, %d rounds; after each round the model is scored on all 22 targets' test days. Values at 50/100/200 acquired labels (excluding the %d initial) and at 100\%% of the pool; FULL = the unmatched 22-participant pool. $\Delta$ = paired difference over targets with a 95\%% bootstrap CI (targets resampled %d times); wins = targets where TOP is higher, with a two-sided sign-test $p$; ``excl.\ own'' drops the targets whose own rows are in TOP-$k$ or BOTTOM-$k$. Mean over 3 walk-forward folds.}
\label{tab:label_budget_topk}
\scriptsize
\begin{tabular}{lcccccccccc}
\toprule
Labels & TOP & BOTTOM & RANDOM & FULL & TOP$-$BOTTOM [CI] & wins ($p$) & excl.\ own & TOP$-$RANDOM [CI] & wins ($p$) & excl.\ own \\
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (args.n_random, args.n_sub, args.K, args.budget, args.n_init, N_BOOT, "\n".join(lines))
    open(os.path.join(out, "label_budget_summary.tex"), "w").write(tex)
    # standalone compile -> label_budget_summary.pdf
    import subprocess
    sa = os.path.join(out, "label_budget_summary_standalone.tex")
    open(sa, "w").write("\\documentclass[landscape]{article}\\usepackage[margin=0.3in]{geometry}\\usepackage{booktabs,amsmath}"
                        "\\pagestyle{empty}\\begin{document}\n" + tex.replace("table*", "table") + "\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory", out, sa], capture_output=True)
    pdf = sa.replace(".tex", ".pdf")
    if r.returncode == 0 and os.path.exists(pdf):
        os.replace(pdf, os.path.join(out, "label_budget_summary.pdf")); print("compiled label_budget_summary.pdf")
    else:
        print("pdflatex failed; see", sa.replace(".tex", ".log"))
    for ext in (".aux", ".log", ".tex"):
        q = sa.replace(".tex", ext); os.path.exists(q) and r.returncode == 0 and os.remove(q)
    print("saved", out)


if __name__ == "__main__":
    main()
