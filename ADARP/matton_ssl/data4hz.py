"""Raw 4 Hz ADARP EDA for the Matton-augmentation encoder, plus window banks.

The pipeline's exported stream (`processed/signals/*_eda.csv`) is EDA averaged
onto HR's 1 Hz grid. Matton et al. work on the E4's native 4 Hz, so this module
reads `EDA.csv` straight from the archives with `load_adarp.read_signal` --
the same reader the pipeline uses -- and keeps it at 4 Hz, unfiltered, in
microsiemens. One cache file per participant under `cache/` makes later runs
fast; delete it to re-read.

Session names are the recording folder names, identical to the `session`
column everywhere else in the pipeline, so the pipeline's session split applies
directly.

Also here:
  * `WindowBank`     window starts over session arrays, gathered per batch with
                     context buffers for the time-shift augmentation
  * `labelled_eda_windows`  the saved labelled windows (`processed/adarp_windows.csv`)
                     cut out of the 4 Hz stream by their timestamps, 240 samples
                     each, for encoding with a 4 Hz encoder
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ADARP_DIR = _HERE.parent
for p in (_HERE, _ADARP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import load_adarp  # noqa: E402  (reused unchanged)
from load_adarp import EDA_HZ, read_signal, session_dirs  # noqa: E402

CACHE_DIR = _HERE / "cache"
FS_EDA = float(EDA_HZ)        # 4 Hz


# ------------------------------------------------------------------ loading

def load_eda_4hz(pid, cache_dir=CACHE_DIR, sensor_dir=load_adarp.SENSOR_DIR):
    """One participant's raw 4 Hz EDA: DataFrame indexed by UTC time, columns eda, session."""
    pid = str(pid).rstrip("C")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{pid}_eda4hz.npz"

    if cache.exists() and cache.stat().st_size > 0:
        z = np.load(cache, allow_pickle=False)
        idx = pd.to_datetime(z["t_ns"], utc=True)
        sessions = np.asarray(z["sessions"]).astype(str)[z["session_code"]]
        return pd.DataFrame({"eda": z["eda"].astype(np.float32), "session": sessions}, index=idx)

    parts = []
    for path in session_dirs(f"{pid}C", sensor_dir):
        eda_path = path / "EDA.csv"
        if not eda_path.exists() or eda_path.stat().st_size == 0:
            continue
        s = read_signal(eda_path, "eda")
        if s.empty:
            continue
        frame = s.to_frame()
        frame["session"] = path.name
        parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=["eda", "session"],
                            index=pd.DatetimeIndex([], tz="UTC"))

    out = pd.concat(parts).sort_index()
    out.index.name = "timestamp"

    sessions, codes = np.unique(out["session"].to_numpy(), return_inverse=True)
    np.savez_compressed(cache, t_ns=out.index.asi8, eda=out["eda"].to_numpy(np.float32),
                        session_code=codes.astype(np.int32), sessions=sessions.astype(str))
    return out


def session_arrays_4hz(frames, sessions_by_pid, participants):
    """[(pid, session, eda values)] over the training sessions of `participants`."""
    out = []
    for pid in participants:
        frame = frames.get(pid)
        if frame is None:
            continue
        for sess in sessions_by_pid.get(pid, []):
            vals = frame.loc[frame["session"] == sess, "eda"].to_numpy(dtype=float)
            if len(vals):
                out.append((pid, sess, vals))
    return out


# ---------------------------------------------------------------- windows

class WindowBank:
    """Window starts over a list of session arrays, gathered per batch.

    Keeps the sessions themselves and (session_idx, start) pairs, so buffers for
    the time shift can be sliced on demand without materialising them for every
    window. A window is kept only if it holds no NaN.
    """

    def __init__(self, sessions, window, step, buffer):
        self.sessions = [np.asarray(v, dtype=np.float32) for _, _, v in sessions]
        self.window, self.buffer = int(window), int(buffer)
        idx = []
        for si, v in enumerate(self.sessions):
            ok = ~np.isnan(v)
            for s in range(0, len(v) - self.window + 1, int(step)):
                if ok[s:s + self.window].all():
                    idx.append((si, s))
        self.index = np.asarray(idx, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return len(self.index)

    def gather(self, rows, with_buffers=False):
        X = np.empty((len(rows), self.window), dtype=np.float32)
        L = R = None
        if with_buffers:
            L = np.full((len(rows), self.buffer), np.nan, dtype=np.float32)
            R = np.full((len(rows), self.buffer), np.nan, dtype=np.float32)
        for k, (si, s) in enumerate(self.index[rows]):
            v = self.sessions[si]
            X[k] = v[s:s + self.window]
            if with_buffers:
                left = v[max(0, s - self.buffer):s]
                right = v[s + self.window:s + self.window + self.buffer]
                if len(left):
                    L[k, -len(left):] = left
                if len(right):
                    R[k, :len(right)] = right
        return X, L, R


def zscore_rows(X):
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return np.divide(X - mu, sd, out=np.zeros_like(X), where=sd > 0)


def labelled_eda_windows(meta, frames, fs=FS_EDA, window_sec=60):
    """(n_windows, fs*window_sec) raw 4 Hz EDA for each row of the pipeline's window table.

    Rows whose stretch is not fully present on the 4 Hz stream are NaN; the
    caller drops and counts them.
    """
    n = int(round(fs * window_sec))
    out = np.full((len(meta), n), np.nan, dtype=np.float32)
    # int64 nanoseconds throughout, so tz-aware and tz-naive inputs compare alike
    starts = pd.DatetimeIndex(pd.to_datetime(meta["window_start"], utc=True)).asi8
    ends = pd.DatetimeIndex(pd.to_datetime(meta["window_end"], utc=True)).asi8
    pids = meta["participant"].astype(str).to_numpy()

    for pid in np.unique(pids):
        frame = frames.get(pid)
        if frame is None:
            continue
        ts = frame.index.asi8
        vals = frame["eda"].to_numpy(dtype=np.float32)
        rows = np.flatnonzero(pids == pid)
        a = np.searchsorted(ts, starts[rows], "left")
        b = np.searchsorted(ts, ends[rows], "left")
        for r, i0, i1 in zip(rows, a, b):
            if i1 - i0 == n:
                seg = vals[i0:i1]
                if not np.isnan(seg).any():
                    out[r] = seg
    return out
