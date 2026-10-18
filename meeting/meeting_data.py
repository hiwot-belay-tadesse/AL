"""ExtraSensory "in a meeting vs not" on the handcrafted features -- data, split and screen.

Lives in meeting/ so nothing about this target is mixed into walking/; the only things imported
from there are the shared loaders (data.py), metrics (homogeneity.py) and model builders
(models.py), which are read unchanged. The active-learning loop is meeting/run_al.py.

A second prediction target alongside walking, chosen because it is a *context* rather than a
physical act: walking is the same motion on everyone's wrist, which is why the dataset came out
homogeneous (source variance share 0.19, headroom 0.009 -- see ../globem/screen_results/). Whether
you are in a meeting depends on where you work, who you work with and how your office sounds, so
there is at least a mechanism for participants to differ in how the sensors map to the label.

Target (label:IN_A_MEETING, one of the 51 ExtraSensory labels):
    y = 1  the minute is reported as in a meeting
    y = 0  the minute is explicitly reported as NOT in a meeting
    excluded: the label is missing for that minute (NaN)
That is the dataset's own convention, so no activity whitelist is needed as it was for walking.

Features: the handcrafted columns only -- `--groups` picks which of the 11 sensor groups to use
(default: all 225 columns). Everything starting with `label` is dropped, as is `timestamp`, so no
label information reaches X. watch_acceleration alone is the walking experiment's feature set and
is kept selectable for a like-for-like comparison, but meetings are an audio/location/phone-state
phenomenon, so the default is all groups.

Layout is the usual {uuid: (X, y, day)} with rows in time order, so data.folds_for,
homogeneity.transfer_matrix and ../globem/screen_dataset.py all apply unchanged.

    python meeting_data.py                              # eligibility + homogeneity screen, all groups
    python meeting_data.py --groups watch_acceleration  # the walking feature set, for comparison
    python meeting_data.py --min_per_class 20 --n_folds 2 --train_days 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WALKING = os.path.join(os.path.dirname(HERE), "walking")
if WALKING not in sys.path:
    sys.path.insert(0, WALKING)

from data import DATA_DIR, folds_for, select_users          # noqa: E402  (shared loaders only)
from homogeneity import feature_space_separability, source_variance_share, transfer_matrix  # noqa: E402
from models import make_clf, make_prep                      # noqa: E402
LABEL = "label:IN_A_MEETING"   # default target; main() --label overrides it for the whole module
GROUPS = ["raw_acc", "proc_gyro", "raw_magnet", "watch_acceleration", "watch_heading",
          "location", "location_quick_features", "audio_naive", "audio_properties",
          "discrete", "lf_measurements"]

# --drop_context: columns that predict the label without sensing the activity. The cut is at
# sub-group level, finer than --groups can express, because `discrete` mixes both kinds.
# Two reasons to drop, kept separate so the list can be read as an argument:
DROP_CONTEXT = [
    # (a) context, not evidence -- these say where/when the person is, not what they are doing
    "location", "location_quick_features",   # "has not moved in an hour": a negative-class filter,
                                             #   and its signature identifies home vs office
    "raw_magnet",                            # mostly device orientation
    "discrete:time_of_day",                  # weekday 10am
    "discrete:wifi_status",                  # on the office network
    "discrete:battery_plugged", "discrete:battery_state",   # at a desk with a charger
    "discrete:ringer_mode", "discrete:on_the_phone",
    # (b) artifacts -- these are not about the person's situation at all
    "discrete:app_state",                    # the ExtraSensory app's own state = when they LABELED
    "lf_measurements:relative_humidity",     # present for 4 of 60 participants; median imputation
    "lf_measurements:temperature_ambient",   #   turns them into a per-person constant = device id
    "lf_measurements:battery_level",         # monotone discharge within a day = a clock
    "audio_properties",                      # recording gain: hardware metadata, not sound content
]


def feature_columns(df, groups, drop=None):
    drop = drop or []
    keep = []
    for c in df.columns:
        if c.startswith("label") or c == "timestamp":
            continue
        g = c.split(":")[0]
        if g in groups and not any(c == d or c.startswith(d + ":") for d in drop):
            keep.append(c)
    return keep


def load_meeting(groups=None, data_dir=DATA_DIR, label=None, drop=None):
    """-> ({uuid: (X, y, day)}, feature names). Rows = minutes with the label present.
    `label` defaults to the module-level LABEL, which main() rewrites from --label, so every
    consumer (run_al.py, subset_table.py, targeted_coverage.py) follows without a signature change."""
    groups = set(groups or GROUPS)
    LABEL_ = label or LABEL
    data, cols = {}, None
    for f in sorted(glob.glob(os.path.join(data_dir, "*.features_labels.csv"))):
        uuid = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        if LABEL_ not in df.columns:
            continue
        if cols is None:
            cols = feature_columns(df, groups, drop)
        s = df[LABEL_]
        use = s.notna().values
        if use.sum() == 0:
            continue
        t = df.loc[use, "timestamp"].values
        o = np.argsort(t, kind="stable")
        day = pd.to_datetime(t[o], unit="s").date
        dmap = {d: i + 1 for i, d in enumerate(sorted(set(day)))}
        data[uuid] = (df.loc[use, cols].to_numpy(float)[o], s[use].to_numpy(int)[o],
                      np.array([dmap[d] for d in day]))
    return data, cols


def day_group_folds(data, users, n_folds=3, test_frac=0.3, seed=0, min_test=5):
    """Hold out whole DAYS at random instead of a walk-forward window.

    Meetings cluster on two to five workdays per participant, so a specific test day (4, 5, 6)
    usually holds no meeting minutes at all and the rolling split drops everyone. Splitting by
    whole days keeps the no-leak property -- adjacent minutes of one meeting never straddle the
    split -- without requiring the meetings to fall on the particular days the walk-forward
    schedule happens to test. Days are redrawn until both sides carry >= min_test of each class.
    Mirrors ../globem/screen_dataset.py week_group_folds, which exists for the same reason.
    """
    out = []
    for f in range(n_folds):
        splits = {}
        for ui, u in enumerate(users):
            X, y, d = data[u]
            days = np.unique(d)
            if len(days) < 3:
                continue
            rng = np.random.default_rng([seed, f, ui, 17])
            n_te = max(1, int(round(test_frac * len(days))))
            for _ in range(40):
                te_d = rng.choice(days, min(n_te, len(days) - 1), replace=False)
                te = np.isin(d, te_d); tr = ~te
                if (np.bincount(y[te], minlength=2).min() >= min_test
                        and np.bincount(y[tr], minlength=2).min() >= min_test):
                    splits[u] = (X[tr], X[te], y[tr], y[te]); break
        out.append((f, splits))
    keep = [u for u in users if all(u in sp for _, sp in out)]
    return [(f, {u: sp[u] for u in keep}) for f, sp in out], keep


def eligibility(data, min_per_class):
    rows = []
    for u, (X, y, d) in data.items():
        rows.append(dict(uuid=u, n=len(y), pos=int(y.sum()), neg=int((y == 0).sum()),
                         pos_rate=float(y.mean()), days=len(np.unique(d)),
                         days_with_pos=len(np.unique(d[y == 1])),
                         eligible=min(int(y.sum()), int((y == 0).sum())) >= min_per_class))
    return pd.DataFrame(rows).sort_values("pos", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="IN_A_MEETING",
                    help="ExtraSensory label to predict, without the 'label:' prefix "
                         "(meeting/results/label_census.csv ranks all 51)")
    ap.add_argument("--groups", nargs="+", default=None, choices=GROUPS, help="sensor groups to use (default: all)")
    ap.add_argument("--drop_context", action="store_true",
                    help="drop the DROP_CONTEXT columns (context proxies and artifacts), "
                         "leaving only features that sense the activity")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. _dropped")
    ap.add_argument("--min_per_class", type=int, default=20, help="minutes of each class a participant needs")
    ap.add_argument("--train_days", type=int, default=3); ap.add_argument("--test_days", type=int, default=1)
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--split", choices=["day_group", "rolling"], default="day_group",
                    help="day_group: hold out whole days at random (see day_group_folds)")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--min_test", type=int, default=None, help="override data.MIN_TEST (rows per class a test day needs)")
    ap.add_argument("--no_sep", action="store_true")
    ap.add_argument("--plots_only", action="store_true", help="re-plot from the saved CSVs, no refit")
    ap.add_argument("--out_dir", default=None,
                    help="default: results/ for IN_A_MEETING (keeps the existing files), "
                         "results/<label>/ otherwise")
    args = ap.parse_args()
    global LABEL
    LABEL = "label:" + args.label
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "results" + args.tag) if args.label == "IN_A_MEETING" \
            else os.path.join(HERE, "results", args.label + args.tag)
    os.makedirs(args.out_dir, exist_ok=True)

    import data as D
    if args.min_test:
        D.MIN_TEST = args.min_test
    D.TRAIN_DAYS, D.TEST_DAYS, D.N_FOLDS = args.train_days, args.test_days, args.n_folds

    if args.plots_only:
        plots(args.out_dir); return

    drop = DROP_CONTEXT if args.drop_context else None
    data, cols = load_meeting(args.groups, drop=drop)
    groups = args.groups or GROUPS
    print(f"{len(data)} participants with the label, {len(cols)} features from {len(groups)} group(s)"
          + (f", {len(DROP_CONTEXT)} context sub-groups dropped" if drop else ""))
    el = eligibility(data, args.min_per_class)
    el.to_csv(os.path.join(args.out_dir, "label_availability_meeting.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n{int(el.eligible.sum())} of {len(el)} participants have >= {args.min_per_class} minutes of each class")
    print(el.head(15).round(3).to_string(index=False))

    data = {u: v for u, v in data.items() if el.set_index('uuid').loc[u, 'eligible']}
    if args.split == "day_group":
        folds, users = day_group_folds(data, sorted(data), args.n_folds, args.test_frac,
                                       min_test=args.min_test or D.MIN_TEST)
        print(f"\nday-group split ({args.n_folds} folds, {args.test_frac:.0%} of days held out): "
              f"{len(users)} usable, {len(data) - len(users)} dropped")
    else:
        users, dropped, reason = select_users(data, "rolling", args.train_days, args.test_days, args.n_folds)
        print(f"\nrolling split (train {args.train_days} d -> test {args.test_days} d, {args.n_folds} folds): "
              f"{len(users)} usable, {len(dropped)} dropped ({reason})")
        folds = folds_for(data, users, "rolling", 0, args.train_days, args.test_days, args.n_folds)
    if len(users) < 3:
        raise SystemExit("too few participants survive the split; try --min_test 3 or --test_frac 0.4")

    A, FULL, ntr, n_nan = transfer_matrix(folds, users, make_clf, make_prep)
    M = A.copy(); np.fill_diagonal(M, np.nan)
    vs = source_variance_share(A)
    full, best = float(np.nanmean(FULL)), float(np.nanmax(np.nanmean(M, axis=1)))
    sep = idacc = np.nan
    if not args.no_sep:
        s, i = feature_space_separability(folds, users, make_prep, np.random.default_rng(0))
        sep, idacc = float(np.nanmean(s)), float(np.mean(i))

    hom = dict(target=args.label, groups=",".join(groups),
               dropped=",".join(drop) if drop else "", n_features=len(cols),
               split=args.split, n_participants=len(users), source_variance_share=vs["source_share"],
               target_variance_share=vs["target_share"], residual_variance_share=vs["residual_share"],
               self_vs_cross_gap=float(np.nanmean(np.diag(A)) - np.nanmean(M)),
               mean_diag_auroc=float(np.nanmean(np.diag(A))), mean_offdiag_auroc=float(np.nanmean(M)),
               full_auroc=full, best_single_source_auroc=best, headroom=full - best,
               separability_ratio=sep, id_classifier_accuracy=idacc, chance_id=1 / len(users))
    pd.DataFrame([dict(metric=k, value=v) for k, v in hom.items()]).to_csv(
        os.path.join(args.out_dir, "homogeneity_meeting.csv"), index=False)
    pd.DataFrame(A, index=users, columns=users).to_csv(os.path.join(args.out_dir, "transfer_matrix_meeting.csv"))
    pd.DataFrame(dict(participant=users, n_train=ntr, self_auroc=np.diag(A),
                      mean_offdiag_auroc=np.nanmean(M, axis=1), FULL=FULL)).to_csv(
        os.path.join(args.out_dir, "per_source_meeting.csv"), index=False)

    print(f"\n=== homogeneity, {args.label} (walking, for reference: source 0.190, target 0.533, "
          "headroom 0.009, ID 0.336; IN_A_MEETING: source 0.250, target 0.140, headroom 0.035) ===")
    for k, v in hom.items():
        print(f"  {k:26} {v if isinstance(v, str) else round(v, 3)}")
    plots(args.out_dir)
    print("\nsaved", args.out_dir)


def plots(out):
    """The homogeneity figures for this target, from the saved CSVs: the source x target transfer
    grid and the per-source summary. Sources and targets are ordered by mean off-diagonal transfer
    (best source at the top), since there is no CoreSet run for this target to order by."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    A = pd.read_csv(os.path.join(out, "transfer_matrix_meeting.csv"), index_col=0)
    ps = pd.read_csv(os.path.join(out, "per_source_meeting.csv"))
    hom = pd.read_csv(os.path.join(out, "homogeneity_meeting.csv")).set_index("metric").value
    tgt = str(hom.get("target", "IN_A_MEETING"))
    try:
        from data import RESULTS_DIR, short_ids
        m = short_ids(list(A.index), RESULTS_DIR)
    except Exception:
        m = {}
    lab = {u: m.get(u, "M" + str(i + 1)) for i, u in enumerate(A.index)}
    A.index = [lab[u] for u in A.index]; A.columns = [lab[u] for u in A.columns]
    ps["pid"] = [lab[u] for u in ps.participant]
    M = A.copy().to_numpy(); np.fill_diagonal(M, np.nan)
    order = list(pd.Series(np.nanmean(M, axis=1), index=A.index).sort_values(ascending=False).index)
    Adf = A.loc[order, order]
    P = len(order)

    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(Adf.values, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(P):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor="white", lw=1.5))
    ax.set_xticks(range(P)); ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_yticks(range(P)); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("target (its held-out days)")
    ax.set_ylabel("source (trained on 100% of its training rows); sorted by mean off-diagonal transfer")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.set_label("test AUROC (mean over 3 folds)")
    ax.set_title("Transfer matrix A[s, t], %s: diag %.3f, off-diag %.3f, FULL %.3f\n"
                 "boxed = self; source share %.2f vs target share %.2f (walking: 0.19 vs 0.53)"
                 % (tgt, float(hom.mean_diag_auroc), float(hom.mean_offdiag_auroc), float(hom.full_auroc),
                    float(hom.source_variance_share), float(hom.target_variance_share)), fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"transfer_matrix_meeting.{ext}"), dpi=150, facecolor="white")
    plt.close(fig)

    q = ps.set_index("pid").loc[order]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(P)
    ax.bar(x, q.mean_offdiag_auroc, color="#2e6fa7", width=.7, label="mean off-diagonal transfer")
    ax.axhline(float(hom.full_auroc), color="black", ls="--", lw=1.2,
               label="FULL pooled model = %.3f" % float(hom.full_auroc))
    ax.axhline(float(hom.best_single_source_auroc), color="#c0392b", ls=":", lw=1.2,
               label="best single source = %.3f (headroom %.3f)"
                     % (float(hom.best_single_source_auroc), float(hom.headroom)))
    ax.scatter(x, q.self_auroc, marker="_", s=200, color="black", zorder=3, label="self AUROC A[s, s]")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8, rotation=90)
    ax.set_xlabel("source"); ax.set_ylabel("test AUROC"); ax.set_ylim(.4, 1.0)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Per-source transfer, %s: self-vs-cross gap %.3f (walking: 0.035)"
                 % (tgt, float(hom.self_vs_cross_gap)))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"source_transfer_meeting.{ext}"), dpi=150, facecolor="white")
    plt.close(fig)
    print("plots saved")


if __name__ == "__main__":
    main()
