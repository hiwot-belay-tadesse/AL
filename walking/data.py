"""Data access for the ExtraSensory walking experiments, organised by functionality:

  1. Rolling (walk-forward) temporal split     rolling_fold, rolling_fold_ids, rolling_users
  2. Single temporal split                     temporal_split, temporal_ok
  3. Random stratified split                   random_split
  4. Split dispatch shared by all experiments  valid_split, split_user, select_users, folds_for, describe_split
  5. Loading and target construction           participant_files, load_participant, build_target, usable_mask, load_sorted
  6. Participant ids                           eligible_participants, short_ids, pid_order

Target:
  1 = label:FIX_walking == 1
  0 = one of LYING_DOWN / SITTING / OR_standing / FIX_running / BICYCLING == 1
  excluded = no main activity reported, or all watch_acceleration features NaN

Data layout used throughout: {uuid: (X, y, day)} with rows in time order, X the 46
watch-accelerometer features, y in {0, 1}, day the 1-based calendar-day index (UTC).
A split returns (Xtr, Xte, ytr, yte) per user.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "DATA", "ExtraSensory")
RESULTS_DIR = os.path.join(HERE, "results")

POS_LABEL = "label:FIX_walking"
NEG_LABELS = ["label:LYING_DOWN", "label:SITTING", "label:OR_standing", "label:FIX_running", "label:BICYCLING"]
MAIN_LABELS = [POS_LABEL] + NEG_LABELS

MIN_PER_CLASS = 10   # eligibility: at least this many usable rows of each class per user
MIN_TRAIN = 2        # split validity: train side needs >= this many rows of each class
MIN_TEST = 5         # split validity: test side needs >= this many rows of each class
TEST_FRAC = 0.2      # random / temporal splits
TRAIN_DAYS, TEST_DAYS, N_FOLDS = 3, 1, 3   # rolling split defaults


# =============================================================================
# 1. Rolling (walk-forward) temporal split
#
# Day-based with FIXED sizes for every user: N_FOLDS folds, fold k uses
#   train = days [k-TRAIN_DAYS, k-1],  test = days [k, k+TEST_DAYS-1],
#   k = TRAIN_DAYS+1, ..., TRAIN_DAYS+N_FOLDS  (sliding one day at a time).
# A user is kept only if ALL folds are valid (see valid_split), so every fold has the
# same users, the same number of train days and the same number of test days.
# =============================================================================
def rolling_fold_ids(train_days=TRAIN_DAYS, n_folds=N_FOLDS):
    """Fold ids = first test day of each fold."""
    return list(range(1 + train_days, 1 + train_days + n_folds))


def rolling_fold(X, y, d, k, train_days=TRAIN_DAYS, test_days=TEST_DAYS):
    """(Xtr, Xte, ytr, yte) for fold k, or None if the fold is not valid for this user."""
    tr = (d >= k - train_days) & (d < k)
    te = (d >= k) & (d < k + test_days)
    if tr.sum() and te.sum() and valid_split(y[tr], y[te]):
        return X[tr], X[te], y[tr], y[te]
    return None


def rolling_users(data, users, train_days=TRAIN_DAYS, test_days=TEST_DAYS, n_folds=N_FOLDS):
    """Users for whom ALL n_folds walk-forward folds are valid."""
    ks = rolling_fold_ids(train_days, n_folds)
    return [u for u in users if all(rolling_fold(*data[u], k, train_days, test_days) is not None for k in ks)]


# =============================================================================
# 2. Single temporal split: first (1-TEST_FRAC) of the user's minutes in time order
#    -> train/pool, last TEST_FRAC -> test. Fixed (no seed).
# =============================================================================
def temporal_split(X, y):
    n_te = int(np.ceil(TEST_FRAC * len(y)))
    return X[:-n_te], X[-n_te:], y[:-n_te], y[-n_te:]


def temporal_ok(X, y):
    _, _, ytr, yte = temporal_split(X, y)
    return valid_split(ytr, yte)


# =============================================================================
# 3. Random stratified split: (1-TEST_FRAC)/TEST_FRAC per user, re-drawn per seed.
#    Note: adjacent minutes of one walk land on both sides (optimistic for personal models).
# =============================================================================
def random_split(X, y, seed):
    return train_test_split(X, y, test_size=TEST_FRAC, stratify=y, random_state=seed)


# =============================================================================
# 4. Split dispatch shared by main.py (active learning) and sanity_check.py
# =============================================================================
def valid_split(ytr, yte):
    """Test side needs >= MIN_TEST rows of each class, train side >= MIN_TRAIN of each class."""
    return min(yte.sum(), len(yte) - yte.sum()) >= MIN_TEST and min(ytr.sum(), len(ytr) - ytr.sum()) >= MIN_TRAIN


def split_user(X, y, split, seed=0):
    """Single-fold splits (random / temporal) for one user."""
    return random_split(X, y, seed) if split == "random" else temporal_split(X, y)


def select_users(data, split, train_days=TRAIN_DAYS, test_days=TEST_DAYS, n_folds=N_FOLDS):
    """(kept users, dropped users, reason) for a split."""
    users = sorted(data)
    if split == "rolling":
        keep = rolling_users(data, users, train_days, test_days, n_folds)
        reason = (f"not all {n_folds} folds valid: test days {1+train_days}..{train_days+n_folds}, "
                  f"each needing >= {MIN_TEST} rows per class and a train window with both classes")
    elif split == "temporal":
        keep = [u for u in users if temporal_ok(data[u][0], data[u][1])]
        reason = f"temporal test set has < {MIN_TEST} rows of a class"
    else:
        keep, reason = users, ""
    return keep, [u for u in users if u not in keep], reason


def folds_for(data, users, split, seed=0, train_days=TRAIN_DAYS, test_days=TEST_DAYS, n_folds=N_FOLDS):
    """List of (fold_id, {user: (Xtr, Xte, ytr, yte)}): n_folds folds with identical user
    sets for rolling; a single fold (id 0) for random / temporal."""
    if split == "rolling":
        return [(k, {u: rolling_fold(*data[u], k, train_days, test_days) for u in users})
                for k in rolling_fold_ids(train_days, n_folds)]
    return [(0, {u: split_user(data[u][0], data[u][1], split, seed) for u in users})]


def describe_split(split, train_days=TRAIN_DAYS, test_days=TEST_DAYS, n_folds=N_FOLDS):
    if split == "rolling":
        return f"rolling (train {train_days} d -> test {test_days} d, {n_folds} folds, identical users in every fold)"
    pct_tr, pct_te = int(100 * (1 - TEST_FRAC)), int(100 * TEST_FRAC)
    return {"random": f"random stratified {pct_tr}/{pct_te} per seed",
            "temporal": f"temporal first {pct_tr}% -> last {pct_te}%"}[split]


# =============================================================================
# 5. Loading and target construction
# =============================================================================
def participant_files(data_dir=DATA_DIR):
    return sorted(glob.glob(os.path.join(data_dir, "*.features_labels.csv")))


def load_participant(path):
    df = pd.read_csv(path)
    uuid = os.path.basename(path).split(".")[0]
    watch_cols = [c for c in df.columns if c.startswith("watch_acceleration:")]
    return uuid, df, watch_cols


def build_target(df):
    """Return y (1/0/-1 for excluded-unknown) and the negative activity name per row."""
    assert (df[MAIN_LABELS] == 1).sum(axis=1).max() <= 1, "main activities not mutually exclusive"
    y = np.full(len(df), -1, dtype=int)
    y[(df[POS_LABEL] == 1).values] = 1
    neg_any = (df[NEG_LABELS] == 1).any(axis=1).values
    y[neg_any] = 0
    neg_name = df[NEG_LABELS].idxmax(axis=1).where(neg_any, None)
    return y, neg_name


def usable_mask(df, y, watch_cols):
    """Rows with a known main activity and watch accelerometer features present."""
    return (y != -1) & df[watch_cols].notna().any(axis=1).values


def load_sorted(files=None, data_dir=DATA_DIR):
    """{uuid: (X, y, day)} for eligible users, rows in time order;
    day = 1-based calendar-day index (UTC) within the user's usable data."""
    elig = set(eligible_participants())
    data = {}
    for f in files or participant_files(data_dir):
        uuid, df, watch = load_participant(f)
        if uuid not in elig:
            continue
        y, _ = build_target(df)
        use = usable_mask(df, y, watch)
        ts = df.loc[use, "timestamp"].values
        o = np.argsort(ts, kind="stable")
        day = pd.to_datetime(ts[o], unit="s").date
        dmap = {d: i + 1 for i, d in enumerate(sorted(set(day)))}
        data[uuid] = (df.loc[use, watch].to_numpy(float)[o], y[use][o], np.array([dmap[d] for d in day]))
    return data


# =============================================================================
# 6. Participant ids (fixed P1..P55 labels, identical in every plot)
# =============================================================================
def eligible_participants(results_dir=RESULTS_DIR):
    """UUIDs flagged eligible in label_availability.csv (written by es_walking_feasibility.py)."""
    avail = pd.read_csv(os.path.join(results_dir, "label_availability.csv"))
    return sorted(avail.loc[avail.eligible, "participant"])


def short_ids(participants, results_dir=RESULTS_DIR):
    """Fixed UUID -> P1..P55 mapping over ALL eligible participants (sorted by UUID).
    Saves the mapping to participant_ids.csv; returns it restricted to `participants`."""
    uuids = eligible_participants(results_dir)
    mapping = {u: f"P{i+1}" for i, u in enumerate(uuids)}
    pd.DataFrame(dict(participant=uuids, pid=[mapping[u] for u in uuids])).to_csv(
        os.path.join(results_dir, "participant_ids.csv"), index=False)
    return {u: mapping[u] for u in participants}


def pid_order(pids):
    return sorted(pids, key=lambda p: int(p[1:]))
