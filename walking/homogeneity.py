"""Cross-participant homogeneity of a dataset (homogenity.txt).

Same features, logistic-regression pipeline and 3 walk-forward folds (train 3 days -> test 1 day)
as the main runs; P participants (the ones valid for the rolling split).

1. Transfer matrix A[s, t]: model trained on 100% of source s's training rows, AUROC on target t's
   test day (incl. t = s); per fold, then mean over folds. Cells where t's test day has one class
   are NaN (counted). FULL[t]: AUROC on t of the model trained on the pooled training rows.
2. Homogeneity metrics, each with a 95% bootstrap CI over participants (N_BOOT resamples):
   - self-vs-cross gap      mean diag(A) - mean offdiag(A)
   - source variance share  off-diagonal A = mu + a_s + b_t + e;  var(a_s) / var(offdiag A)
   - source sufficiency     suff_s = mean_{t != s} (A[s,t] - FULL[t]); median over s and the
                            fraction of sources with suff_s >= -0.02
   - feature-space separability  mean over participants of (mean distance to other participants'
                            rows / mean distance within own rows), per class, averaged over classes;
                            and a participant-ID logistic regression (5-fold stratified CV) accuracy
                            vs chance 1/P
3. Link to CoreSet: Spearman correlation of each source's CoreSet Delta_p (round 20, mean over
   folds; from <run>/al_query_sources.csv) with (a) its mean off-diagonal transfer AUROC and
   (b) its training-row count.

Outputs in homogeneity_results/<dataset>/:
   transfer_matrix_<dataset>.png/.pdf   heatmap of A (rows = source, cols = target, Fig. 6 order,
                                        diagonal outlined, colour scale 0.5-1.0)
   source_transfer_<dataset>.png/.pdf   per-source mean off-diagonal AUROC sorted by Delta_p,
                                        dashed = mean FULL
   homogeneity_<dataset>.csv/.tex (+pdf) all metrics with CIs, correlations, NaN-cell count
   transfer_matrix_<dataset>_values.csv, per_source_<dataset>.csv

    python homogeneity.py --dataset extrasensory --run al_global_rolling_K10_baseline
"""
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "homogeneity_results")
N_BOOT = 1000
FINAL_ROUND = 20
SUFF_TOL = 0.02
MAX_ROWS_SEP = 200      # rows per participant per class for the distance ratio
MAX_ROWS_ID = 400       # rows per participant for the participant-ID classifier
VMIN, VMAX = 0.5, 1.0   # shared colour scale


# ----------------------------------------------------------------------------- datasets
def load_extrasensory(run):
    """Returns folds [(fold_id, {uuid: (Xtr, Xte, ytr, yte)})], users, mapping uuid->pid,
    delta (DataFrame participant(pid) x fold of CoreSet Delta_p at FINAL_ROUND), make_clf, make_prep."""
    from data import RESULTS_DIR, folds_for, load_sorted, select_users, short_ids
    from label_budget_topk import coreset_delta_per_fold
    from models import make_clf, make_prep
    data = load_sorted(); users, _, _ = select_users(data, "rolling"); mapping = short_ids(users, RESULTS_DIR)
    folds = folds_for(data, users, "rolling")
    delta = coreset_delta_per_fold(os.path.join(RESULTS_DIR, run), FINAL_ROUND).reindex(users).rename(index=mapping)
    return folds, users, mapping, delta, make_clf, make_prep


def load_bp(run):
    raise NotImplementedError("BP spike loader: return (folds, users, mapping, delta, make_clf, make_prep) like load_extrasensory")


DATASETS = {"extrasensory": load_extrasensory, "bp": load_bp}


# ----------------------------------------------------------------------------- 1. transfer matrix
def transfer_matrix(folds, users, make_clf, make_prep):
    """A[fold][s, t], FULL[fold][t], n_train[fold][s]. NaN where target test day has one class."""
    P = len(users); A = []; FULL = []; NTR = []; n_nan = 0
    for fold, splits in folds:
        Af = np.full((P, P), np.nan); Ff = np.full(P, np.nan); ntr = np.zeros(P, dtype=int)
        tests = {}
        for j, t in enumerate(users):
            Xte, yte = splits[t][1], splits[t][3]
            tests[j] = (Xte, yte) if 0 < yte.sum() < len(yte) else None
            if tests[j] is None:
                n_nan += P + 1
        for i, s in enumerate(users):
            Xtr, ytr = splits[s][0], splits[s][2]; ntr[i] = len(ytr)
            prep = make_prep().fit(Xtr); clf = make_clf().fit(prep.transform(Xtr), ytr)
            for j in range(P):
                if tests[j] is not None:
                    Af[i, j] = roc_auc_score(tests[j][1], clf.predict_proba(prep.transform(tests[j][0]))[:, 1])
        Xg = np.vstack([splits[u][0] for u in users]); yg = np.concatenate([splits[u][2] for u in users])
        prep = make_prep().fit(Xg); clf = make_clf().fit(prep.transform(Xg), yg)
        for j in range(P):
            if tests[j] is not None:
                Ff[j] = roc_auc_score(tests[j][1], clf.predict_proba(prep.transform(tests[j][0]))[:, 1])
        A.append(Af); FULL.append(Ff); NTR.append(ntr)
        print(f"fold {fold}: transfer matrix done, diag mean={np.nanmean(np.diag(Af)):.3f}, offdiag mean={np.nanmean(offdiag(Af)):.3f}, FULL mean={np.nanmean(Ff):.3f}", flush=True)
    return np.nanmean(A, axis=0), np.nanmean(FULL, axis=0), np.mean(NTR, axis=0), n_nan


def offdiag(A):
    return A[~np.eye(len(A), dtype=bool)]


# ----------------------------------------------------------------------------- 2. metrics
def self_cross_gap(A):
    return np.nanmean(np.diag(A)) - np.nanmean(offdiag(A))


def source_variance_share(A):
    """Two-way additive decomposition of the off-diagonal cells: A[s,t] = mu + a_s + b_t + e."""
    M = A.copy().astype(float); np.fill_diagonal(M, np.nan)
    mu = np.nanmean(M); a = np.nanmean(M, axis=1) - mu; b = np.nanmean(M, axis=0) - mu
    E = M - mu - a[:, None] - b[None, :]
    tot = np.nanvar(M)
    return dict(source_share=np.var(a) / tot, target_share=np.var(b) / tot, residual_share=np.nanvar(E) / tot)


def source_sufficiency(A, FULL):
    M = A.copy().astype(float); np.fill_diagonal(M, np.nan)
    suff = np.nanmean(M - FULL[None, :], axis=1)          # per source: mean over t != s of A[s,t] - FULL[t]
    return suff


def boot_ci(fn, P, rng, n_boot=N_BOOT):
    """fn(idx) -> scalar computed on the participant subset idx (with replacement). Returns (lo, hi)."""
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, P, P)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def resample_matrix(A, idx):
    """Submatrix on resampled participants; duplicated participants' cross cells are masked
    (they are the same person) so that 'off-diagonal' keeps meaning 'different participants'."""
    M = A[np.ix_(idx, idx)].astype(float).copy()
    same = idx[:, None] == idx[None, :]
    M[same & ~np.eye(len(idx), dtype=bool)] = np.nan
    return M


def feature_space_separability(folds, users, make_prep, rng):
    """Per fold: standardized training rows (prep fit on the pooled rows). For each class c and
    participant p: within = mean pairwise distance among p's class-c rows, between = mean distance
    from p's class-c rows to other participants' class-c rows; ratio = between / within.
    Returns per-participant ratio (mean over classes and folds) and per-participant ID accuracy."""
    P = len(users); ratios = np.zeros((len(folds), P)); id_acc = np.zeros((len(folds), P))
    from sklearn.linear_model import LogisticRegression
    for k, (fold, splits) in enumerate(folds):
        Xg = np.vstack([splits[u][0] for u in users]); prep = make_prep().fit(Xg)
        Z = {u: prep.transform(splits[u][0]) for u in users}; y = {u: splits[u][2] for u in users}
        for c in (0, 1):
            rows = {}
            for u in users:
                m = np.where(y[u] == c)[0]
                if len(m) > MAX_ROWS_SEP:
                    m = rng.choice(m, MAX_ROWS_SEP, replace=False)
                rows[u] = Z[u][m]
            for i, u in enumerate(users):
                if len(rows[u]) < 2:
                    ratios[k, i] += np.nan; continue
                D_within = np.linalg.norm(rows[u][:, None, :] - rows[u][None, :, :], axis=-1)
                within = D_within[np.triu_indices(len(rows[u]), 1)].mean()
                others = np.vstack([rows[v] for v in users if v != u and len(rows[v])])
                between = np.linalg.norm(rows[u][:, None, :] - others[None, :, :], axis=-1).mean()
                ratios[k, i] += 0.5 * between / within
        # participant-ID classifier: equal rows per participant (chance = 1/P)
        Xid, yid = [], []
        for i, u in enumerate(users):
            m = np.arange(len(Z[u]))
            if len(m) > MAX_ROWS_ID:
                m = rng.choice(m, MAX_ROWS_ID, replace=False)
            Xid.append(Z[u][m]); yid += [i] * len(m)
        Xid = np.vstack(Xid); yid = np.array(yid)
        pred = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Xid, yid,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0))
        for i in range(P):
            id_acc[k, i] = (pred[yid == i] == i).mean()
        print(f"fold {fold}: separability ratio={np.nanmean(ratios[k]):.3f}, ID accuracy={id_acc[k].mean():.3f} (chance {1 / P:.3f})", flush=True)
    return np.nanmean(ratios, axis=0), id_acc.mean(axis=0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="extrasensory")
    ap.add_argument("--run", default="al_global_rolling_K10_baseline", help="CoreSet run providing Delta_p (al_query_sources.csv)")
    ap.add_argument("--plots_only", action="store_true", help="rebuild plots/table from the saved matrix and per-source CSVs")
    args = ap.parse_args()
    out = os.path.join(OUT_ROOT, args.dataset); os.makedirs(out, exist_ok=True); ds = args.dataset
    folds, users, mapping, delta, make_clf, make_prep = DATASETS[ds](args.run)
    pids = [mapping[u] for u in users]; P = len(users)
    delta_mean = delta.mean(axis=1).reindex(pids)
    order = delta_mean.sort_values(ascending=False).index.tolist()        # Fig. 6 order: most over-queried at top
    rng = np.random.default_rng(0)

    if args.plots_only:
        Adf = pd.read_csv(os.path.join(out, f"transfer_matrix_{ds}_values.csv"), index_col=0).reindex(index=pids, columns=pids)
        ps = pd.read_csv(os.path.join(out, f"per_source_{ds}.csv")).set_index("source").reindex(pids)
        A, FULL, ntr, sep_ratio, id_acc = Adf.values, ps.FULL.values, ps.n_train.values, ps.separability_ratio.values, ps.id_accuracy.values
        n_nan = int(pd.read_csv(os.path.join(out, f"homogeneity_{ds}.csv")).set_index("metric").loc["nan_cells", "value"])
    else:
        A, FULL, ntr, n_nan = transfer_matrix(folds, users, make_clf, make_prep)
        sep_ratio, id_acc = feature_space_separability(folds, users, make_prep, rng)
        pd.DataFrame(A, index=pids, columns=pids).to_csv(os.path.join(out, f"transfer_matrix_{ds}_values.csv"))

    # ---- metrics with bootstrap CIs over participants
    M = A.copy(); np.fill_diagonal(M, np.nan)
    offdiag_mean = np.nanmean(M, axis=1); suff = source_sufficiency(A, FULL); vs = source_variance_share(A)
    per_source = pd.DataFrame(dict(source=pids, delta_coreset=delta_mean.values, self_auroc=np.diag(A), mean_offdiag_auroc=offdiag_mean,
                                   FULL=FULL, sufficiency=suff, n_train=ntr, separability_ratio=sep_ratio, id_accuracy=id_acc))
    per_source.to_csv(os.path.join(out, f"per_source_{ds}.csv"), index=False)

    rows = []
    def add(metric, value, fn=None, note=""):
        lo, hi = boot_ci(fn, P, rng) if fn else (np.nan, np.nan)
        rows.append(dict(metric=metric, value=value, ci_lo=lo, ci_hi=hi, note=note))
    add("self_vs_cross_gap", self_cross_gap(A), lambda idx: self_cross_gap(resample_matrix(A, idx)), "mean diag - mean offdiag AUROC")
    add("source_variance_share", vs["source_share"], lambda idx: source_variance_share(resample_matrix(A, idx))["source_share"], "var(a_s)/var(offdiag A)")
    add("target_variance_share", vs["target_share"], lambda idx: source_variance_share(resample_matrix(A, idx))["target_share"], "var(b_t)/var(offdiag A)")
    add("residual_variance_share", vs["residual_share"], None, "var(e)/var(offdiag A)")
    add("source_sufficiency_median", np.median(suff), lambda idx: np.median(source_sufficiency(resample_matrix(A, idx), FULL[idx])), "median_s mean_{t!=s}(A[s,t]-FULL[t])")
    add(f"source_sufficiency_frac_within_{SUFF_TOL}", (suff >= -SUFF_TOL).mean(),
        lambda idx: (source_sufficiency(resample_matrix(A, idx), FULL[idx]) >= -SUFF_TOL).mean(), f"fraction of sources with sufficiency >= -{SUFF_TOL}")
    add("separability_ratio", np.nanmean(sep_ratio), lambda idx: np.nanmean(sep_ratio[idx]), "between/within participant distance, class-averaged")
    add("id_classifier_accuracy", id_acc.mean(), lambda idx: id_acc[idx].mean(), f"participant-ID LR, 5-fold CV; chance = 1/P = {1 / P:.3f}")
    add("id_classifier_chance", 1 / P, None, "1/P")
    r_a = spearmanr(delta_mean.values, offdiag_mean); r_b = spearmanr(delta_mean.values, ntr)
    add("spearman_delta_vs_offdiag_auroc", r_a.correlation, None, f"p = {r_a.pvalue:.3g}")
    add("spearman_delta_vs_n_train", r_b.correlation, None, f"p = {r_b.pvalue:.3g}")
    add("mean_diag_auroc", np.nanmean(np.diag(A)), None, "personal model, own test day")
    add("mean_offdiag_auroc", np.nanmean(M), None, "cross-participant transfer")
    add("mean_FULL_auroc", np.nanmean(FULL), None, "pooled model")
    add("nan_cells", n_nan, None, "target test day with one class (sum over folds)")
    add("n_participants", P, None, "")
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(out, f"homogeneity_{ds}.csv"), index=False)
    pd.set_option("display.width", 200); print(tab.round(4).to_string(index=False))

    # ---- LaTeX table
    fmt = lambda r: (f"{r.value:.3f}" if abs(r.value) < 100 else f"{r.value:.0f}") + (f" [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if np.isfinite(r.ci_lo) else "")
    def note_tex(n):
        return n.replace("_", " ").replace("!=", r"$\neq$").replace(">=", r"$\geq$")
    lines = ["%s & %s & %s \\\\" % (r.metric.replace("_", " "), fmt(r), note_tex(r.note)) for r in tab.itertuples()]
    tex = r"""\begin{table}[t]
\centering
\caption{Cross-participant homogeneity of %s (walking vs.\ not walking, %d participants, 3 walk-forward folds). Transfer matrix $A[s,t]$ = AUROC on target $t$'s test day of the logistic-regression model trained on 100\%% of source $s$'s training rows, mean over folds; FULL = pooled model. Brackets: 95\%% bootstrap CI over participants (%d resamples).}
\label{tab:homogeneity_%s}
\small
\begin{tabular}{llp{7cm}}
\toprule
Metric & Value [95\%% CI] & Definition \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (ds, P, N_BOOT, ds, "\n".join(lines))
    open(os.path.join(out, f"homogeneity_{ds}.tex"), "w").write(tex)
    sa = os.path.join(out, f"homogeneity_{ds}_standalone.tex")
    open(sa, "w").write("\\documentclass{article}\\usepackage{booktabs,amsmath}\\pagestyle{empty}\\begin{document}\n" + tex + "\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=batchmode", "-output-directory", out, sa], capture_output=True)
    for ext in (".aux", ".log"):
        q = sa.replace(".tex", ext); os.path.exists(q) and os.remove(q)
    print("table pdf", "compiled" if r.returncode == 0 else "FAILED")

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    Adf = pd.DataFrame(A, index=pids, columns=pids).loc[order, order]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(Adf.values, cmap="viridis", vmin=VMIN, vmax=VMAX, aspect="auto")
    for i in range(P):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", lw=1.5))
    ax.set_xticks(range(P)); ax.set_xticklabels(order, rotation=90, fontsize=8); ax.set_yticks(range(P)); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("target (test day)"); ax.set_ylabel(f"source (trained on 100% of its training rows); sorted by CoreSet Δ at round {FINAL_ROUND}, most over-queried at top")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("test AUROC (mean over 3 folds)")
    ax.set_title(f"Transfer matrix A[s, t], {ds}: diag mean {np.nanmean(np.diag(A)):.3f}, off-diag mean {np.nanmean(M):.3f}, FULL mean {np.nanmean(FULL):.3f}"
                 + (f"; {n_nan} NaN cells" if n_nan else ""))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"transfer_matrix_{ds}.{ext}"), dpi=150, facecolor="white")
    plt.close(fig)

    ps = per_source.set_index("source").loc[order]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(P)
    ax.bar(x, ps.mean_offdiag_auroc, color=["#c0392b" if d > 0 else "#2e6fa7" for d in ps.delta_coreset], width=0.7)
    ax.axhline(np.nanmean(FULL), color="black", ls="--", lw=1.2, label=f"FULL pooled model, mean over targets = {np.nanmean(FULL):.3f}")
    ax.scatter(x, ps.self_auroc, marker="_", s=200, color="black", label="self AUROC A[s, s]", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{p}\n{d:+.3f}" for p, d in zip(order, ps.delta_coreset)], fontsize=7)
    ax.set_xlabel(f"source, sorted by CoreSet Δ at round {FINAL_ROUND} (value under the ID; red = over-queried, blue = under-queried)")
    ax.set_ylabel("mean off-diagonal transfer AUROC"); ax.set_ylim(0.5, 1.0); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Per-source transfer, {ds}: Spearman(Δ, off-diag AUROC) = {r_a.correlation:+.2f} (p={r_a.pvalue:.2g}); Spearman(Δ, n_train) = {r_b.correlation:+.2f} (p={r_b.pvalue:.2g})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"source_transfer_{ds}.{ext}"), dpi=150, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
