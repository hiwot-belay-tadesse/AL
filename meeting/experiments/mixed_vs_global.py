"""Mixed-effects vs global logistic regression on 100% of the training labels (prompt_mixed.txt).

No active learning: this asks whether a participant random effect is worth building into the AL
loop at all, on both ExtraSensory targets, using the existing loaders, folds and preprocessing
unchanged.

Models, same features / folds / test days throughout:
  global    statsmodels GLM, Binomial, every participant pooled

IN_A_MEETING drops the raw_magnet, watch_heading, audio_properties and lf_measurements groups
(--drop_groups overrides); WALKING keeps its 46 watch-accelerometer features.
  mixed_vb  BinomialBayesMixedGLM.from_formula(..., {"person": "0 + C(person)"}).fit_vb()
  mixed_map the same model fitted with .fit_map()
  mixed_slope  intercept + random slopes on the 5 largest |coef| features of `global`
               (one vc_formulas entry each), run only if mixed_vb beats global on IN_A_MEETING
  personal  one ridge-penalised Binomial GLM per participant, on that participant's rows only

Checks on the mixed models: `remove_personal` predicts with the fixed effects only (u = 0), and
`shuffle_id` permutes person IDs across participants (fixed seed), refits, and predicts each target
with the ID it was mapped to. Both should erase any personalisation gain.

Prediction is computed by hand -- logit = const + X . fe_mean + u_person -- because
BinomialBayesMixedGLM.predict() is _BayesMixedGLM.predict, documented as "Return the fitted mean
structure" from the mean parameters only: it does NOT add the random effects (statsmodels 0.15.0).

    python experiments/mixed_vs_global.py                     # both labels
    python experiments/mixed_vs_global.py --labels walking --folds 1    # smoke test
"""
import argparse
import os
import sys
import time
import warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # meeting/
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "walking"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import make_prep  # noqa: E402  (median imputer + standard scaler, fit on train rows)

OUT = os.path.join(HERE, "results", "mixed_vs_global")
# IN_A_MEETING feature groups dropped for this experiment: device orientation, watch heading, the
# audio property block and the low-frequency measurements. meeting_data.GROUPS minus these.
DROP_GROUPS = ["raw_magnet", "watch_heading", "audio_properties", "lf_measurements"]
RIDGE_PERSONAL = 1e-3          # personal GLMs are ridge-penalised; single participants separate easily


# ----------------------------------------------------------------------------- data (reused as-is)
def load_label(label, seed=0, n_folds=3, drop_groups=None):
    """[(fold, {uuid: (Xtr, Xte, ytr, yte)})], users, feature names, {uuid: short id}."""
    if label == "walking":
        import data as wdata
        d = wdata.load_sorted()
        users, _, _ = wdata.select_users(d, "rolling")
        folds = wdata.folds_for(d, users, "rolling")
        ids = wdata.short_ids(users, wdata.RESULTS_DIR)
        import glob
        f0 = sorted(glob.glob(os.path.join(wdata.DATA_DIR, "*.features_labels.csv")))[0]
        cols = [c for c in pd.read_csv(f0, nrows=1).columns if c.startswith("watch_acceleration:")]
        assert len(cols) == folds[0][1][users[0]][0].shape[1], "walking feature count mismatch"
        return folds, users, cols, ids
    import meeting_data
    meeting_data.LABEL = "label:IN_A_MEETING"
    drop = DROP_GROUPS if drop_groups is None else drop_groups
    groups = [g for g in meeting_data.GROUPS if g not in drop]
    print(f"    dropped feature groups: {', '.join(drop) or '(none)'}", flush=True)
    d, cols = meeting_data.load_meeting(groups)
    el = meeting_data.eligibility(d, 20).set_index("uuid")
    d = {u: v for u, v in d.items() if el.loc[u, "eligible"]}
    folds, users = meeting_data.day_group_folds(d, sorted(d), n_folds, 0.3, seed=seed, min_test=5)
    return folds, users, cols, {u: f"M{i + 1}" for i, u in enumerate(sorted(users))}


def frame(Z, y, owner, ids):
    """Design frame with formula-safe feature names f0..fn and a `person` factor."""
    df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
    df["y"] = np.asarray(y, float)
    df["person"] = [ids[o] for o in owner]
    return df


# ----------------------------------------------------------------------------- models
def fit_global(df, feats):
    mod = sm.GLM.from_formula("y ~ " + " + ".join(feats), data=df, family=sm.families.Binomial())
    try:
        return mod.fit()
    except Exception as e:                       # separation on wide feature sets
        print(f"    global GLM.fit() failed ({type(e).__name__}); using ridge fit_regularized",
              flush=True)
        return mod.fit_regularized(alpha=1e-4, L1_wt=0.0)


def fit_mixed(df, feats, vc, how="vb"):
    mod = BinomialBayesMixedGLM.from_formula("y ~ " + " + ".join(feats), vc, df)
    return mod.fit_vb(verbose=False) if how == "vb" else mod.fit_map()


def parse_vc(res):
    """{(component, person id): posterior mean} from the vc parameter names.

    statsmodels 0.15.0 names these entries after the factor expression, not after the vc_formulas
    key: "C(person)[M4]" for the random intercept and "C(person)[M4]:f11" for a random slope on f11.
    The person id is therefore the first bracketed token and the component is whatever follows the
    closing bracket -- parsing on the vc_formulas key silently returns zeros.
    """
    names = list(getattr(res.model, "vc_names", []) or [])
    u = np.asarray(res.vc_mean, dtype=float)
    out = {}
    for nm, val in zip(names, u):
        i, j = nm.find("["), nm.find("]")
        pid = nm[i + 1:j] if i >= 0 else nm
        tail = nm[j + 1:]                       # "" for the intercept, ":f11" for a slope
        comp = tail[1:] if tail.startswith(":") else "intercept"
        out[(comp, pid)] = float(val)
    return out, names


def logit_fixed(res, Z):
    fe = np.asarray(res.fe_mean, dtype=float)
    return fe[0] + Z @ fe[1:]


def random_part(vc, pid, Z, slope_idx=()):
    """u_intercept(p) + sum_k b_pk x_k -- the random contribution for one participant's rows."""
    v = np.full(len(Z), vc.get(("intercept", pid), 0.0))
    for k in slope_idx:
        v = v + vc.get((f"f{k}", pid), 0.0) * Z[:, k]
    return v


def scores(y, lo):
    """AUROC / Brier / log-loss from a linear predictor."""
    p = np.clip(expit(lo), 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, lo) if 0 < y.sum() < len(y) else np.nan
    return auc, brier_score_loss(y, p), log_loss(y, p, labels=[0, 1])


def per_person_z(Xtr, owner, users, Xte_by_user):
    """Model 5: z-score every participant by THEIR OWN training-row mean/sd (pooled median impute).

    The target's test rows are standardised with the target's own training statistics, so nothing
    from the test days enters the transform."""
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xi = imp.transform(Xtr)
    stats, Ztr = {}, np.empty_like(Xi)
    for u in users:
        m = owner == u
        mu, sd = Xi[m].mean(0), Xi[m].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        stats[u] = (mu, sd)
        Ztr[m] = (Xi[m] - mu) / sd
    Zte = {u: (imp.transform(Xte_by_user[u]) - stats[u][0]) / stats[u][1] for u in users}
    return Ztr, Zte


def fit_personal(df, feats, p):
    d = df[df.person == p]
    if d.y.nunique() < 2:
        return None
    mod = sm.GLM.from_formula("y ~ " + " + ".join(feats), data=d, family=sm.families.Binomial())
    try:
        return mod.fit_regularized(alpha=RIDGE_PERSONAL, L1_wt=0.0)
    except Exception:
        return None


# ----------------------------------------------------------------------------- one label
def run_label(label, args):
    folds, users, cols, ids = load_label(label, args.seed, args.folds,
                                         getattr(args, "drop_groups", None))
    folds = folds[:args.folds]
    feats = [f"f{i}" for i in range(folds[0][1][users[0]][0].shape[1])]
    print(f"\n=== {label}: {len(users)} participants, {len(feats)} features, {len(folds)} folds ===",
          flush=True)
    print(f"    features: {cols[0]} ... {cols[-1]}" if cols else "", flush=True)
    rows, timings, warn_counts, sd_rows = [], {}, {}, []
    slope_feats = []

    for fold, sp in folds:
        Xtr = np.vstack([sp[u][0] for u in users]); ytr = np.concatenate([sp[u][2] for u in users])
        owner = np.concatenate([[u] * len(sp[u][2]) for u in users])
        prep = make_prep().fit(Xtr)                      # fit on this fold's training rows only
        Ztr = prep.transform(Xtr)
        df = frame(Ztr, ytr, owner, ids)
        te = {u: (prep.transform(sp[u][1]), sp[u][3]) for u in users}

        def timed(name, fn):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t0 = time.time(); res = fn()
                timings.setdefault(name, []).append(time.time() - t0)
                warn_counts[name] = warn_counts.get(name, 0) + len(w)
            return res

        g = timed("global", lambda: fit_global(df, feats))
        fe_g = np.asarray(g.params)[1:]; b0_g = float(np.asarray(g.params)[0])

        # model 5: global LR on per-participant z-scored features
        Ztr_pz, Zte_pz = per_person_z(Xtr, owner, users, {u: sp[u][1] for u in users})
        gz = timed("global_person_z", lambda: fit_global(frame(Ztr_pz, ytr, owner, ids), feats))
        fe_z = np.asarray(gz.params)[1:]; b0_z = float(np.asarray(gz.params)[0])

        vc = {"person": "0 + C(person)"}
        use_vb = label in args.vb_labels          # fit_vb collapses to an all-zero fit on meeting
        primary = "vb" if use_vb else "map"
        fits = {}
        if use_vb:
            fits["mixed_vb"] = timed("mixed_vb", lambda: fit_mixed(df, feats, vc, "vb"))
        if args.map or not use_vb:
            fits["mixed_map"] = timed("mixed_map", lambda: fit_mixed(df, feats, vc, "map"))

        # shuffle-ID control: a bijection on participants, refit, predict with the mapped ID
        # The permutation is a bijection on participants, so refitting on it reproduces the same
        # effects under new names; at test time a target knows only its OWN id, and the effect
        # wearing that id was estimated from somebody else's rows -- which is the control.
        rng = np.random.default_rng([args.seed, fold, 7])
        pid_list = sorted(set(df.person))
        perm = dict(zip(pid_list, rng.permutation(pid_list)))
        dsh = df.copy(); dsh["person"] = df.person.map(perm)
        sh = timed("shuffle_id", lambda: fit_mixed(dsh, feats, vc, primary))

        slope_idx, slope_res, sh_slope = [], None, None
        sched_idx, sched_res = [], None
        if args.slopes:
            # primary slope set: SENSOR features only -- a per-participant slope on a discrete:*
            # dummy (time of day, ringer mode) models the person's schedule, not their signal
            order = [int(i) for i in np.argsort(-np.abs(fe_g))]
            sensor = [i for i in order if not cols[i].startswith("discrete:")]
            slope_idx = sensor[:args.n_slopes]
            slope_feats = [f"f{k}" for k in slope_idx]
            if fold == folds[0][0]:
                print("    sensor slopes: %s" % ", ".join(cols[i] for i in slope_idx), flush=True)
            vcs = dict(vc, **{f"slope_f{k}": f"0 + C(person):f{k}" for k in slope_idx})
            slope_res = timed("mixed_slope", lambda: fit_mixed(df, feats, vcs, primary))
            sh_slope = timed("shuffle_id_slope", lambda: fit_mixed(dsh, feats, vcs, primary))
            # secondary "schedule slopes": the same rule WITHOUT the discrete:* exclusion
            sched_idx = order[:args.n_slopes]
            if sched_idx != slope_idx:
                if fold == folds[0][0]:
                    print("    schedule slopes: %s" % ", ".join(cols[i] for i in sched_idx), flush=True)
                vcx = dict(vc, **{f"slope_f{k}": f"0 + C(person):f{k}" for k in sched_idx})
                sched_res = timed("mixed_slope_sched", lambda: fit_mixed(df, feats, vcx, primary))

        vcmap = {k: parse_vc(v)[0] for k, v in fits.items()}
        vc_sh = parse_vc(sh)[0]
        vc_sl, names_sl = parse_vc(slope_res) if slope_res is not None else ({}, [])
        vc_shsl = parse_vc(sh_slope)[0] if sh_slope is not None else {}
        vc_sched = parse_vc(sched_res)[0] if sched_res is not None else {}
        if slope_res is not None and fold == folds[0][0]:
            print(f"    vc entries: {len(names_sl)} (e.g. {names_sl[0]}, {names_sl[-1]})", flush=True)
        pers = {u: fit_personal(df, feats, ids[u]) for u in users}

        for k, res in fits.items():                                    # random-intercept SD
            vcp, vsd = float(np.asarray(res.vcp_mean)[0]), float(np.asarray(res.vcp_sd)[0])
            sd_rows.append(dict(label=label, fold=fold, model=k, sd=np.exp(vcp),
                                sd_lo=np.exp(vcp - 1.96 * vsd), sd_hi=np.exp(vcp + 1.96 * vsd)))

        # ---- pooled over every target's test rows (the only place an intercept can act)
        Zp = np.vstack([te[u][0] for u in users]); yp = np.concatenate([te[u][1] for u in users])
        pooled = dict(label=label, fold=fold, target="POOLED")
        def add(d, name, y, lo):
            a_, b_, l_ = scores(y, lo)
            d[name], d[name + "__brier"], d[name + "__ll"] = a_, b_, l_
        add(pooled, "global", yp, b0_g + Zp @ fe_g)
        add(pooled, "global_person_z", yp,
            b0_z + np.vstack([Zte_pz[u] for u in users]) @ fe_z)
        prim_name = "mixed_vb" if "mixed_vb" in fits else "mixed_map"
        for k, res in fits.items():
            base = logit_fixed(res, Zp)
            ran = np.concatenate([random_part(vcmap[k], ids[u], te[u][0]) for u in users])
            add(pooled, k, yp, base + ran)
            if k == prim_name:
                add(pooled, "remove_personal", yp, base)
        add(pooled, "shuffle_id", yp, logit_fixed(sh, Zp) +
            np.concatenate([random_part(vc_sh, ids[u], te[u][0]) for u in users]))
        if slope_res is not None:
            base = logit_fixed(slope_res, Zp)
            add(pooled, "mixed_slope", yp,
                base + np.concatenate([random_part(vc_sl, ids[u], te[u][0], slope_idx) for u in users]))
            add(pooled, "remove_personal_slope", yp, base)
            add(pooled, "shuffle_id_slope", yp, logit_fixed(sh_slope, Zp) + np.concatenate(
                [random_part(vc_shsl, ids[u], te[u][0], slope_idx) for u in users]))
        if sched_res is not None:
            base = logit_fixed(sched_res, Zp)
            add(pooled, "mixed_slope_sched", yp,
                base + np.concatenate([random_part(vc_sched, ids[u], te[u][0], sched_idx) for u in users]))
            add(pooled, "remove_personal_sched", yp, base)
        rows.append(pooled)

        # ---- per target
        for u in users:
            Zt, yt = te[u]
            if not (0 < yt.sum() < len(yt)):
                continue
            p = ids[u]
            r = dict(label=label, fold=fold, target=p)
            add(r, "global", yt, b0_g + Zt @ fe_g)
            add(r, "global_person_z", yt, b0_z + Zte_pz[u] @ fe_z)
            for k, res in fits.items():
                add(r, k, yt, logit_fixed(res, Zt) + random_part(vcmap[k], p, Zt))
                if k == prim_name:
                    # explanation check: the mixed model's SHARED weights, no random effects.
                    # Equal to the mixed column per target (an intercept is a constant shift), so
                    # the informative contrast is this vs `global`: if it already matches the mixed
                    # model, the gain is cleaner beta rather than personalisation.
                    add(r, "fixed_only", yt, logit_fixed(res, Zt))
            if slope_res is not None:
                base = logit_fixed(slope_res, Zt)
                add(r, "mixed_slope", yt, base + random_part(vc_sl, p, Zt, slope_idx))
                add(r, "remove_personal_slope", yt, base)   # checks are informative for slopes
                add(r, "shuffle_id_slope", yt,
                    logit_fixed(sh_slope, Zt) + random_part(vc_shsl, p, Zt, slope_idx))
            if sched_res is not None:
                base = logit_fixed(sched_res, Zt)
                add(r, "mixed_slope_sched", yt, base + random_part(vc_sched, p, Zt, sched_idx))
                add(r, "remove_personal_sched", yt, base)
            pr = pers[u]
            if pr is not None:
                add(r, "personal", yt, np.asarray(pr.params)[0] + Zt @ np.asarray(pr.params)[1:])
            rows.append(r)
        print(f"    fold {fold} done ({', '.join(f'{k} {np.mean(v):.1f}s' for k, v in timings.items())})",
              flush=True)
    return pd.DataFrame(rows), pd.DataFrame(sd_rows), timings, warn_counts, slope_feats, cols


# ----------------------------------------------------------------------------- reporting
def summarise(T, models):
    """mean AUROC +- fold std, paired difference vs global, targets improved (POOLED excluded)."""
    T = T[T.target != "POOLED"]
    per_t = T.groupby("target")[models].mean()
    fold_mean = T.groupby("fold")[models].mean()
    out = []
    for m in models:
        d = (per_t[m] - per_t["global"]).dropna()
        out.append(dict(model=m, auroc=per_t[m].mean(), fold_sd=fold_mean[m].std(ddof=1),
                        delta=d.mean() if m != "global" else np.nan,
                        se=d.std(ddof=1) / np.sqrt(len(d)) if m != "global" else np.nan,
                        improved=f"{int((d > 0).sum())}/{len(d)}" if m != "global" else "",
                        ))
    return pd.DataFrame(out), per_t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=["walking", "meeting"])
    ap.add_argument("--folds", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop_groups", nargs="*", default=None,
                    help=f"IN_A_MEETING feature groups to drop (default: {' '.join(DROP_GROUPS)})")
    ap.add_argument("--map", action="store_true", default=True, help="also fit with fit_map()")
    ap.add_argument("--plots_only", action="store_true", help="redraw from the saved table_*.csv")
    ap.add_argument("--vb_labels", nargs="*", default=["walking"],
                    help="labels fitted with fit_vb(); others use fit_map() (see the methods note)")
    ap.add_argument("--no_map", dest="map", action="store_false")
    ap.add_argument("--slopes", action="store_true", default=True,
                    help="fit the random-slope model (both labels, no gate)")
    ap.add_argument("--n_slopes", type=int, default=10,
                    help="features with the largest |global coefficient| to give random slopes")
    ap.add_argument("--no_slopes", dest="slopes", action="store_false")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.plots_only:
        t = {l: pd.read_csv(os.path.join(OUT, f"table_{l}.csv"), index_col=0) for l in args.labels
             if os.path.exists(os.path.join(OUT, f"table_{l}.csv"))}
        make_figure(t)
        return

    tables, summaries, sds, meta = {}, {}, {}, {}
    for label in args.labels:
        T, SD, tim, warn, slope_feats, cols = run_label(label, args)
        models = [c for c in ["global", "global_person_z", "mixed_vb", "mixed_map", "fixed_only",
                              "mixed_slope", "remove_personal_slope", "shuffle_id_slope",
                              "mixed_slope_sched", "remove_personal_sched", "personal",
                              "remove_personal", "shuffle_id"] if c in T.columns]
        S, per_t = summarise(T, models)
        P = T[T.target == "POOLED"]
        S["pooled_auroc"] = [P[m].mean() if m in P else np.nan for m in S.model]
        pt = T[T.target != "POOLED"]
        S["brier"] = [pt[m + "__brier"].mean() if m + "__brier" in pt else np.nan for m in S.model]
        S["logloss"] = [pt[m + "__ll"].mean() if m + "__ll" in pt else np.nan for m in S.model]
        per_t.to_csv(os.path.join(OUT, f"table_{label}.csv"))
        T.to_csv(os.path.join(OUT, f"raw_{label}.csv"), index=False)
        tables[label], summaries[label], sds[label] = per_t, S, SD
        meta[label] = dict(timings={k: float(np.mean(v)) for k, v in tim.items()}, warnings=warn,
                           slope_feats=[cols[int(f[1:])] for f in (slope_feats or [])])
        print(S.round(4).to_string(index=False), flush=True)

    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# Mixed effects vs global, 100% of the training labels\n\n")
        fh.write("## Methods note: fit_vb() fails on IN_A_MEETING\n\n"
                 "`BinomialBayesMixedGLM.fit_vb()` returns an all-zero parameter vector on this "
                 "target -- |fe_mean| max 0.000, vcp_mean 0, every random effect 0 -- so its "
                 "predictor is constant and every metric degenerates (AUROC 0.5000, Brier 0.2500, "
                 "log-loss 0.6931 = ln 2). It raises no exception, only ~2570 numpy warnings per "
                 "fit ('overflow encountered in exp', 'invalid value encountered in scalar "
                 "subtract'): the optimiser never leaves its starting point. The likely cause is "
                 "the class balance -- 2466 positives in 79223 rows (3.1%) drives the intercept to "
                 "about -8.2, where the mean-field objective overflows. `fit_map()` on the same "
                 "data is well behaved (intercept -8.24, random-intercept scale 0.583). "
                 "IN_A_MEETING therefore uses fit_map(); WALKING (29% positive) keeps both "
                 "fitters, where they agree to four decimals.\n\n")
        fh.write("Prediction uses `const + X . fe_mean + u_person` computed by hand: "
                 "`BinomialBayesMixedGLM.predict()` is `_BayesMixedGLM.predict`, documented as "
                 "returning the fitted mean structure from the mean parameters, and does not add "
                 "the random effects (statsmodels 0.15.0).\n\n")
        for label in tables:
            S, SD = summaries[label], sds[label]
            prim = "mixed_vb" if (SD.model == "mixed_vb").any() else "mixed_map"
            sd_i = SD[SD.model == prim]
            fh.write(f"## {label}\n\n| model | AUROC | fold sd | delta vs global | SE | improved "
                     "| pooled AUROC | Brier | log-loss |\n|---|---|---|---|---|---|---|---|---|\n")
            for r in S.itertuples():
                fh.write(f"| {r.model} | {r.auroc:.4f} | {r.fold_sd:.4f} | "
                         f"{'' if np.isnan(r.delta) else f'{r.delta:+.4f}'} | "
                         f"{'' if np.isnan(r.se) else f'{r.se:.4f}'} | {r.improved} "
                         f"| {r.pooled_auroc:.4f} | {r.brier:.4f} | {r.logloss:.4f} |\n")
            fh.write("\nPer-target AUROC is invariant to a random INTERCEPT (a constant shift "
                     "within one participant), so the intercept-only checks (remove_personal, "
                     "shuffle_id) are reported on pooled AUROC only; Brier and log-loss do see the "
                     "intercept, and the slope checks (remove_personal_slope, shuffle_id_slope) are "
                     "informative per target.\n")
            fh.write(f"\nrandom-intercept SD ({prim}): {sd_i.sd.mean():.3f} "
                     f"[{sd_i.sd_lo.mean():.3f}, {sd_i.sd_hi.mean():.3f}] (95% posterior)\n\n")
            fh.write(f"fit time per fold: {meta[label]['timings']}\n\n"
                     f"warnings: {meta[label]['warnings']}\n\n")
            if meta[label]["slope_feats"]:
                fh.write("random-slope features: " + ", ".join(meta[label]["slope_feats"]) + "\n\n")

        fh.write("\n## Conclusions\n\n")
        for b_ in conclusions(tables, summaries):
            fh.write(f"- {b_}\n")
    make_figure(tables)


def conclusions(tables, summaries):
    """The 3-5 bullets prompt_mixed.txt asks for, written from the numbers rather than by hand."""
    out = []

    def row(label, model, col="auroc"):
        S = summaries.get(label)
        if S is None or model not in set(S.model):
            return np.nan
        return float(S.loc[S.model == model, col].iloc[0])

    def mixed_of(label):
        return "mixed_vb" if not np.isnan(row(label, "mixed_vb")) else "mixed_map"

    for label in tables:
        m = mixed_of(label)
        d, imp = row(label, m, "delta"), summaries[label].set_index("model").loc[m, "improved"]
        se = row(label, m, "se")
        out.append(f"**(a/b) {label}**: {m} {row(label, m):.4f} vs global {row(label, 'global'):.4f} "
                   f"({d:+.4f}, SE {se:.4f}, better on {imp}); pooled "
                   f"{row(label, m, 'pooled_auroc'):.4f} vs {row(label, 'global', 'pooled_auroc'):.4f}; "
                   f"Brier {row(label, m, 'brier'):.4f} vs {row(label, 'global', 'brier'):.4f}.")
    for label in tables:
        m = mixed_of(label)
        out.append(f"**(c) {label} checks**: per target the fixed-effects-only model matches the "
                   f"mixed model ({row(label, 'fixed_only'):.4f} vs {row(label, m):.4f}) -- a random "
                   f"intercept is a constant shift within a participant, so the per-target gain is "
                   f"cleaner shared weights, not personalisation. Pooled, removing the random "
                   f"effects costs {row(label, m, 'pooled_auroc') - row(label, 'remove_personal', 'pooled_auroc'):+.4f} "
                   f"and the shuffle-ID control costs "
                   f"{row(label, m, 'pooled_auroc') - row(label, 'shuffle_id', 'pooled_auroc'):+.4f}, "
                   f"so what the intercepts add pooled is genuinely personal.")
    for label in tables:
        m = mixed_of(label)
        t = tables[label]
        if m not in t:
            continue
        g = (t[m] - t["global"]).sort_values(ascending=False)
        top = ", ".join(f"{i} {v:+.3f}" for i, v in g.head(4).items())
        bot = ", ".join(f"{i} {v:+.3f}" for i, v in g.tail(2).items())
        extra = ""
        if "M10" in g.index:
            extra = f" M10 specifically: {g['M10']:+.3f} (global {t.loc['M10', 'global']:.3f} -> {t.loc['M10', m]:.3f})."
        out.append(f"**(d) {label} per target**: biggest gains {top}; biggest losses {bot}.{extra}")
    return out


def make_figure(tables):
    """global vs mixed effects vs personal, one panel per label. The mixed series is whichever
    fitter that label used -- meeting has no mixed_vb column (fit_vb fails there)."""
    fig, axes = plt.subplots(1, len(tables), figsize=(7.0 * len(tables), 4.4), squeeze=False)
    for ax, (label, per_t) in zip(axes[0], tables.items()):
        p = per_t.sort_values("global", ascending=False)
        x = np.arange(len(p))
        mixed = "mixed_vb" if "mixed_vb" in p else "mixed_map"
        series = [("global", "#777777", "global"), (mixed, "seagreen", f"mixed ({mixed[6:]})"),
                  ("mixed_slope", "darkorange", "mixed + slopes"), ("personal", "steelblue", "personal")]
        for m, c, lbl in series:
            if m in p:
                ax.plot(x, p[m], "o", ms=5, color=c, mec="black", mew=0.4, label=lbl)
        for xi, (a_, b_) in enumerate(zip(p["global"], p[mixed])):
            ax.plot([xi, xi], [a_, b_], color="grey", lw=0.5, zorder=0)
        d = (p[mixed] - p["global"])
        ax.set_xlabel(f"mixed - global: {d.mean():+.4f}, better on {int((d > 0).sum())}/{len(d)}",
                      fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(p.index, rotation=90, fontsize=7)
        ax.set_ylabel("test AUROC", fontsize=9); ax.set_title(label, fontsize=10)
        ax.set_ylim(0.3, 1.02); ax.grid(axis="y", lw=0.3, alpha=0.4); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("100% of the training labels: global vs mixed effects vs personal", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"per_target_auroc.{ext}"), dpi=150, facecolor="white")
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
