"""Which features do people differ on? Pre-specified random-slope sets on IN_A_MEETING.

No active learning: every model is fitted on 100% of the training labels, so this isolates where
between-participant variation lives. Reuses experiments/mixed_vs_global.py wholesale -- the
175-feature set, the 3 day-group folds, the median-imputer + standard-scaler preprocessing, and
fit_map() -- and never writes into another experiment's outputs.

Slope sets are fixed, label-free and identical in every fold (unlike the per-fold |coefficient|
ranking in personal_al_meeting.py), so the variance components are comparable across folds:

    audio            audio_naive:mfcc<k>:mean, k = 0..12                              13
    motion           magnitude_stats mean/std for raw_acc, proc_gyro, watch_accel      6
    location         every location:* and location_quick_features:*                   17
    location_no_alt  the same minus min_altitude and max_altitude                      15

Models: `shared` (random intercept only) is the reference; each set adds random slopes on top of
that intercept. Each slope model also reports remove_personal (fixed effects only, u = 0) and
shuffle_id (participant IDs permuted in training, refitted, predicted with the shuffled IDs).

Selection rule: the set used downstream is chosen on the VALIDATION split only -- ~25% of each
participant's training days, held out whole -- with test days touched once, at the end.

    python experiments/slope_sets_meeting.py --folds 1     # timing on one fold
    python experiments/slope_sets_meeting.py               # all three folds
"""
import argparse
import json
import os
import sys
import time
import warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(ROOT, "walking"), HERE, os.path.join(HERE, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mixed_vs_global import fit_mixed, frame, logit_fixed, parse_vc, random_part  # noqa: E402
from models import make_prep  # noqa: E402
from personal_al_meeting import load_folds  # noqa: E402

OUT = os.path.join(HERE, "results", "slope_sets_meeting")
VAL_FRAC, VAL_MIN_POS = 0.25, 2


def slope_sets(cols):
    """The four pre-specified sets, as column-index lists. Counts are asserted, not assumed."""
    loc = [c for c in cols if c.startswith("location:") or c.startswith("location_quick_features:")]
    sets = {
        "audio": [f"audio_naive:mfcc{k}:mean" for k in range(13)],
        "motion": [f"{g}:magnitude_stats:{s}" for g in ("raw_acc", "proc_gyro", "watch_acceleration")
                   for s in ("mean", "std")],
        "location": loc,
        "location_no_alt": [c for c in loc if c not in ("location:min_altitude", "location:max_altitude")],
    }
    want = dict(audio=13, motion=6, location=17, location_no_alt=15)
    for k, v in sets.items():
        missing = [c for c in v if c not in cols]
        assert not missing, f"{k}: columns absent from the 175-feature set: {missing}"
        assert len(v) == want[k], f"{k}: expected {want[k]} columns, got {len(v)}"
    return sets


def val_mask(day_tr, y_tr, rng, tries=40):
    """Whole-day validation split inside the training days: ~VAL_FRAC of days, >= 2 positives.

    Returns None when no draw satisfies it, which marks that participant's validation as NA."""
    days = np.unique(day_tr)
    n_val = max(1, int(round(VAL_FRAC * len(days))))
    if len(days) < 2:
        return None
    for _ in range(tries):
        vd = rng.choice(days, min(n_val, len(days) - 1), replace=False)
        m = np.isin(day_tr, vd)
        if y_tr[m].sum() >= VAL_MIN_POS and y_tr[~m].sum() >= 1 and 0 < m.sum() < len(m):
            return m
    return None


def fit_and_score(df, feats, vc, slope_idx, targets, Z_eval, y_eval, ids, tag=""):
    """One fit_map; per-target AUROC with that participant's random effects (u = 0 if unseen)."""
    t0 = time.time()
    status = "ok"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = fit_mixed(df, feats, vc, "map")
        except Exception as e:
            print(f"      {tag}: FIT FAILED ({type(e).__name__}: {e})", flush=True)
            return None, dict(status=f"fail:{type(e).__name__}", seconds=time.time() - t0, warnings=0)
        n_warn = len(w)
    if np.abs(np.asarray(res.fe_mean)).max() == 0:
        status = "all_zero"
        print(f"      {tag}: ALL-ZERO FIT -- logged, not substituted", flush=True)
    vcm = parse_vc(res)[0]
    out = {}
    for t in targets:
        Zt, yt = Z_eval[t], y_eval[t]
        if Zt is None or not (0 < yt.sum() < len(yt)):
            out[ids[t]] = np.nan
            continue
        out[ids[t]] = roc_auc_score(yt, logit_fixed(res, Zt)
                                    + random_part(vcm, ids[t], Zt, slope_idx))
    return (res, out), dict(status=status, seconds=time.time() - t0, warnings=n_warn)


def vc_sds(res, names):
    """{component: posterior-mean SD} from vcp_mean, which is on the log scale."""
    nm = list(getattr(res.model, "vcp_names", []) or [])
    sd = np.exp(np.asarray(res.vcp_mean, float))
    return {nm[i] if i < len(nm) else f"vc{i}": float(sd[i]) for i in range(len(sd))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", nargs="+", default=None, help="subset of the slope sets to run")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    folds, users, cols, days, ids = load_folds(args.seed, 3)
    feats = [f"f{i}" for i in range(len(cols))]
    SETS = slope_sets(cols)
    if args.sets:
        SETS = {k: v for k, v in SETS.items() if k in args.sets}
    json.dump({k: v for k, v in SETS.items()}, open(os.path.join(OUT, "slope_sets.json"), "w"), indent=2)
    print(f"IN_A_MEETING: {len(users)} participants, {len(cols)} features; sets: "
          + ", ".join(f"{k} ({len(v)})" for k, v in SETS.items()), flush=True)

    rows, sd_rows, meta = [], [], []
    for fold, sp in folds[:args.folds]:
        t_fold = time.time()
        ytr = np.concatenate([sp[u][2] for u in users])
        owner = np.concatenate([[u] * len(sp[u][2]) for u in users])
        Xtr = np.vstack([sp[u][0] for u in users])
        day_tr = np.concatenate([days[fold][u] for u in users])

        # validation: whole training days per participant, >= 2 positives
        vm = np.zeros(len(ytr), bool)
        na = []
        for u in users:
            m_u = owner == u
            rng = np.random.default_rng([args.seed, fold, users.index(u), 31])
            m = val_mask(day_tr[m_u], ytr[m_u], rng)
            if m is None:
                na.append(ids[u]); continue
            idx = np.where(m_u)[0]
            vm[idx[m]] = True
        print(f"  fold {fold}: validation {vm.sum()} rows ({vm.mean():.1%}), "
              f"NA for {na if na else 'nobody'}", flush=True)

        prep_sub = make_prep().fit(Xtr[~vm])              # val stage: fit on sub-train only
        prep_all = make_prep().fit(Xtr)                   # test stage: fit on all training rows
        stages = {
            "val": (prep_sub, ~vm,
                    {u: (prep_sub.transform(Xtr[(owner == u) & vm]) if ((owner == u) & vm).any() else None)
                     for u in users},
                    {u: ytr[(owner == u) & vm] for u in users}),
            "test": (prep_all, np.ones(len(ytr), bool),
                     {u: prep_all.transform(sp[u][1]) for u in users},
                     {u: sp[u][3] for u in users}),
        }
        for split, (prep, fit_mask, Z_eval, y_eval) in stages.items():
            Z = prep.transform(Xtr[fit_mask])
            df = frame(Z, ytr[fit_mask], owner[fit_mask], ids)
            rng = np.random.default_rng([args.seed, fold, 7])
            pid_list = sorted(set(df.person))
            perm = dict(zip(pid_list, rng.permutation(pid_list)))
            dsh = df.copy(); dsh["person"] = df.person.map(perm)

            base = {"person": "0 + C(person)"}
            got, info = fit_and_score(df, feats, base, [], users, Z_eval, y_eval, ids,
                                      tag=f"{split}/shared")
            meta.append(dict(fold=fold, split=split, model="shared", **info))
            if got:
                res, auc = got
                rows += [dict(fold=fold, split=split, model="shared", target=k, auroc=v)
                         for k, v in auc.items()]
                sd_rows.append(dict(fold=fold, split=split, model="shared", **vc_sds(res, [])))

            for name, columns in SETS.items():
                sidx = [cols.index(c) for c in columns]
                vc = dict(base, **{f"slope_f{k}": f"0 + C(person):f{k}" for k in sidx})
                got, info = fit_and_score(df, feats, vc, sidx, users, Z_eval, y_eval, ids,
                                          tag=f"{split}/{name}")
                meta.append(dict(fold=fold, split=split, model=name, **info))
                if got:
                    res, auc = got
                    rows += [dict(fold=fold, split=split, model=name, target=k, auroc=v)
                             for k, v in auc.items()]
                    sd_rows.append(dict(fold=fold, split=split, model=name, **vc_sds(res, columns)))
                    # remove_personal is free: the same fit with u = 0
                    vcm = parse_vc(res)[0]
                    for t in users:
                        Zt, yt = Z_eval[t], y_eval[t]
                        if Zt is not None and 0 < yt.sum() < len(yt):
                            rows.append(dict(fold=fold, split=split, model=f"{name}:remove_personal",
                                             target=ids[t], auroc=roc_auc_score(yt, logit_fixed(res, Zt))))
                # The permutation is a bijection, so the refit reproduces the same effects under
                # new names; the target knows only its OWN id, and the effect wearing that id was
                # estimated from somebody else's rows -- so look up ids[t], not perm[ids[t]].
                got, info = fit_and_score(dsh, feats, vc, sidx, users, Z_eval, y_eval, ids,
                                          tag=f"{split}/{name}:shuffle_id")
                meta.append(dict(fold=fold, split=split, model=f"{name}:shuffle_id", **info))
                if got:
                    rows += [dict(fold=fold, split=split, model=f"{name}:shuffle_id", target=k, auroc=v)
                             for k, v in got[1].items()]
            print(f"    {split} done ({time.time() - t_fold:.0f}s cumulative)", flush=True)
        print(f"  fold {fold} total {time.time() - t_fold:.0f}s", flush=True)

    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "per_target.csv"), index=False)
    pd.DataFrame(sd_rows).to_csv(os.path.join(OUT, "variance_components.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(OUT, "fit_log.csv"), index=False)
    report(T, pd.DataFrame(sd_rows), pd.DataFrame(meta), SETS)


def report(T, SD, META, SETS):
    """summary.md + the per-target delta figure."""
    per = (T.groupby(["split", "model", "target"]).auroc.mean().unstack("model"))
    lines = []
    for name in SETS:
        row = dict(slope_set=name, n_slopes=len(SETS[name]))
        for split in ("val", "test"):
            if split not in per.index.get_level_values(0):
                continue
            p = per.loc[split]
            for lab, col in (("", name), ("_rm", f"{name}:remove_personal"), ("_sh", f"{name}:shuffle_id")):
                if col in p and "shared" in p:
                    d = (p[col] - p["shared"]).dropna()
                    row[f"{split}{lab}"] = d.mean()
                    if lab == "":
                        row[f"{split}_se"] = d.std(ddof=1) / np.sqrt(len(d))
                        row[f"{split}_won"] = f"{int((d > 0).sum())}/{len(d)}"
        s = SD[(SD.model == name)]
        slope_cols = [c for c in s.columns if c not in ("fold", "split", "model")]
        vals = s[slope_cols].mean(numeric_only=True)
        row["mean_slope_sd"] = float(vals.drop(labels=[c for c in vals.index if "person" == c],
                                               errors="ignore").mean())
        lines.append(row)
    S = pd.DataFrame(lines)
    S.to_csv(os.path.join(OUT, "summary_table.csv"), index=False)

    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("# Pre-specified random-slope sets on IN_A_MEETING (100% of training labels)\n\n")
        bad = META[META.status != "ok"]
        fh.write(f"{len(META)} fits, {len(bad)} not ok"
                 + (f" ({bad.model.tolist()})" if len(bad) else "") + "; "
                 f"total fit time {META.seconds.sum() / 60:.1f} min\n\n")
        fh.write(S.round(4).to_markdown(index=False) + "\n\n")
        fh.write("Columns: `val`/`test` are the paired difference vs the random-intercept-only "
                 "`shared` model; `_rm` removes the random effects at prediction time; `_sh` is the "
                 "shuffled-ID control.\n\n## Conclusions\n\n")
        if "val" in S:
            best = S.loc[S.val.idxmax()]
            fh.write(f"- Largest validation gain: **{best.slope_set}** ({best.val:+.4f}); "
                     f"its shuffled-ID control moves {best.get('val_sh', float('nan')):+.4f}.\n")
        if {"location", "location_no_alt"}.issubset(set(S.slope_set)):
            a = S.set_index("slope_set")
            fh.write(f"- Location without altitude: test {a.loc['location_no_alt'].get('test', np.nan):+.4f} "
                     f"vs {a.loc['location'].get('test', np.nan):+.4f} with it.\n")
    if "test" in per.index.get_level_values(0):
        p = per.loc["test"]
        names = [n for n in SETS if n in p]
        fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 4.4), sharey=True)
        for ax, n in zip(np.atleast_1d(axes), names):
            d = (p[n] - p["shared"]).dropna().sort_values()
            ax.barh(range(len(d)), d.values, color=np.where(d.values > 0, "seagreen", "indianred"))
            ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=7)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"{n} ({len(SETS[n])})\nmean {d.mean():+.3f}", fontsize=9)
            ax.set_xlabel("test AUROC - shared", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "per_target_delta.png"), dpi=150, facecolor="white")
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
