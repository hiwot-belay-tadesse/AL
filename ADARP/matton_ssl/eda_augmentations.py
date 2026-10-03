"""EDA data augmentations from Matton, Lewis, Guttag & Picard (CHIL 2023).

    "Contrastive Learning of Electrodermal Activity Representations for Stress
    Detection", PMLR 209:410-426. Code: github.com/kmatton/contrastive-learning-for-eda

Their "All DAs" recipe, which this module reproduces: for each training window
make two views, each by applying ONE transform sampled uniformly from the 17
below (`n_transforms=1`, `stochastic_choice=true` in their pretraining config).
Parameter ranges are the ones in that config, not the slightly different ones
quoted in the paper text, because the config is what their runs used.

EDA-specific (9)                      Generic time-series (8)
  low-pass filter                       amplitude constant scale
  high-pass filter                      amplitude warp
  band-pass filter                      gaussian noise
  band-stop filter                      time shift
  high-frequency noise                  temporal cutout
  jump (motion) artifact                time warp
  loose-sensor artifact                 permutation
  tonic constant scale                  flip
  tonic amplitude warp

Inputs are raw microsiemens, one window at a time, with the signal to either
side of the window available as `left`/`right` buffers (the time shift needs
them; everything else ignores them). Several transforms zero negative values or
mimic µS-scale artifacts, so standardise AFTER augmenting, never before.

Adapting to this pipeline's rate. Their EDA is 4 Hz in 240-sample (60 s)
windows; ADARP's SSL stream here is 1 Hz. Parameters are therefore kept in
physical units (seconds, Hz, µS) and converted with `fs`:
  * durations in seconds -> samples, then clipped so they fit the window;
  * the 0.05 Hz tonic/phasic split, the high-pass range (0.05-0.25 Hz) and the
    band-pass range (0.05-0.25 Hz) are physiological and are kept as published;
  * ranges that would exceed the new Nyquist (low-pass upper cutoff 1 Hz,
    band-stop 0.75-1 Hz, high-frequency-noise band 1-2 Hz) are scaled by
    fs/4 so they occupy the same fraction of the spectrum as at 4 Hz.
`paper_params(fs)` returns the resulting table so a run can log exactly what
it used; at fs=4 it is the authors' config unchanged.
"""

from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

import numpy as np
import scipy.signal
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt, iirnotch, iirpeak

PAPER_FS = 4.0

TRANSFORM_NAMES = [
    "low_pass", "high_pass", "band_pass", "band_stop", "hf_noise",
    "jump_artifact", "loose_sensor", "tonic_scale", "tonic_warp",
    "amplitude_scale", "amplitude_warp", "gaussian_noise", "time_shift",
    "temporal_cutout", "time_warp", "permute", "flip",
]
EDA_SPECIFIC = TRANSFORM_NAMES[:9]


def paper_params(fs=PAPER_FS):
    """The authors' stochastic-transform ranges, expressed for sampling rate `fs`.

    Values are in seconds / Hz / µS; conversion to samples happens inside each
    transform. See the module docstring for which ranges are rescaled.
    """
    nyq_scale = fs / PAPER_FS          # 1.0 at the paper's 4 Hz
    return {
        # --- filters (Hz) -------------------------------------------------
        "low_pass":   {"cutoff_hz": (0.25 * nyq_scale, 1.0 * nyq_scale), "order": 4},
        "high_pass":  {"cutoff_hz": (0.05, 0.25), "order": 4},
        "band_pass":  {"center_hz": (0.05, 0.25), "Q": 0.707},
        "band_stop":  {"center_hz": (0.75 * nyq_scale, 1.0 * nyq_scale), "Q": 0.707},
        # bins 60..120 of a 240-point FFT at 4 Hz are 1.0-2.0 Hz
        "hf_noise":   {"band_hz": (1.0 * nyq_scale, 2.0 * nyq_scale), "sigma_scale": (0.1, 1.0)},
        # --- artifacts -------------------------------------------------------
        # shift factor is µS per second; smoothing ramp 2-12 samples at 4 Hz
        "jump_artifact": {"max_jumps": 2, "shift_us_per_s": (0.01, 0.2),
                          "ramp_s": (2 / PAPER_FS, 12 / PAPER_FS)},
        # width 40-80 samples at 4 Hz, edge smoothing 2-20 samples
        "loose_sensor":  {"width_s": (40 / PAPER_FS, 80 / PAPER_FS),
                          "smooth_s": (2 / PAPER_FS, 20 / PAPER_FS)},
        # --- tonic / thermoregulation ---------------------------------------
        "tonic_scale": {"scale": (0.25, 2.0), "split_hz": 0.05, "order": 4},
        "tonic_warp":  {"sigma": (0.01, 0.05), "knots": (0, 4), "split_hz": 0.05, "order": 4},
        # --- generic -----------------------------------------------------------
        "amplitude_scale": {"scale": (0.25, 2.0)},
        "amplitude_warp":  {"sigma": (0.01, 0.05), "knots": (0, 4)},
        "gaussian_noise":  {"sigma_scale": (0.0, 0.5)},
        "time_shift":      {"shift_s": (60 / PAPER_FS, 240 / PAPER_FS)},
        "temporal_cutout": {"size_s": (20 / PAPER_FS, 100 / PAPER_FS)},
        "time_warp":       {"sigma": (0.01, 0.1), "knots": (1, 4)},
        "permute":         {"max_splits": 6},
        "flip":            {},
    }


# ----------------------------------------------------------------- helpers

def _u(rng, lo_hi):
    return rng.uniform(*lo_hi)


def _samples(seconds, fs, n, lo=1):
    """seconds -> whole samples, clipped into [lo, n]."""
    return int(np.clip(round(seconds * fs), lo, n))


@lru_cache(maxsize=None)
def _butter(order, cutoff, btype, fs):
    return scipy.signal.butter(order, float(cutoff), btype=btype, output="ba", fs=fs)


def _safe_filtfilt(b, a, x):
    """filtfilt needs > 3*max(len(a),len(b)) samples; pad by reflection when short."""
    need = 3 * max(len(a), len(b))
    if len(x) > need:
        return filtfilt(b, a, x)
    pad = need - len(x) + 1
    xp = np.pad(x, pad, mode="reflect")
    return filtfilt(b, a, xp)[pad:-pad]


def _spline_warper(rng, n, sigma, knots):
    """Smooth multiplicative factor over n samples: knots+2 points ~ N(1, sigma)."""
    heights = rng.normal(1.0, sigma, size=knots + 2)
    steps = np.linspace(0, n - 1, num=knots + 2)
    return CubicSpline(steps, heights)(np.arange(n))


def _tonic_phasic(x, fs, split_hz, order):
    b_t, a_t = _butter(order, split_hz, "lowpass", fs)
    b_p, a_p = _butter(order, split_hz, "highpass", fs)
    return _safe_filtfilt(b_t, a_t, x), _safe_filtfilt(b_p, a_p, x)


# -------------------------------------------------------------- transforms
# Every transform: (x, left, right, fs, p, rng) -> array of len(x).
# `x` is a 1-D float array in µS. `left`/`right` are the adjacent signal
# (may be empty). `p` is that transform's entry from paper_params(fs).

def low_pass(x, left, right, fs, p, rng):
    hi = min(_u(rng, p["cutoff_hz"]), 0.95 * fs / 2)
    b, a = _butter(p["order"], float(hi), "lowpass", fs)
    return _safe_filtfilt(b, a, x)


def high_pass(x, left, right, fs, p, rng):
    lo = min(_u(rng, p["cutoff_hz"]), 0.95 * fs / 2)
    b, a = _butter(p["order"], float(lo), "highpass", fs)
    return _safe_filtfilt(b, a, x)


def band_pass(x, left, right, fs, p, rng):
    f0 = min(_u(rng, p["center_hz"]), 0.95 * fs / 2)
    b, a = iirpeak(f0, p["Q"], fs=fs)
    return _safe_filtfilt(b, a, x)


def band_stop(x, left, right, fs, p, rng):
    f0 = min(_u(rng, p["center_hz"]), 0.95 * fs / 2)
    b, a = iirnotch(f0, p["Q"], fs=fs)
    return _safe_filtfilt(b, a, x)


def hf_noise(x, left, right, fs, p, rng):
    """Gaussian noise added to the FFT bins inside `band_hz`, mirrored, then iFFT."""
    n = len(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    lo, hi = p["band_hz"]
    hi = min(hi, fs / 2)
    bins = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if not len(bins):
        return x.copy()
    X = fft(x)
    sigma = _u(rng, p["sigma_scale"]) * np.mean(np.abs(X))
    noise = rng.normal(scale=sigma, size=len(bins))
    X[bins] += noise
    neg = (n - bins) % n          # each bin's mirrored negative-frequency bin
    X[neg] += noise               # same real noise there keeps the spectrum Hermitian
    return np.abs(ifft(X))        # as in the reference implementation


def jump_artifact(x, left, right, fs, p, rng):
    """Up to `max_jumps` abrupt rises/drops, each ramped over a short spline."""
    n = len(x)
    y = x.copy()
    flip_time = rng.choice([-1, 1]) == -1
    if flip_time:
        y = y[::-1].copy()

    ramp_lo = _samples(p["ramp_s"][0], fs, n, lo=1)
    ramp_hi = max(ramp_lo, _samples(p["ramp_s"][1], fs, n, lo=1))
    max_start = n - ramp_lo - 2
    if max_start < 1:
        return x.copy()

    n_jumps = int(rng.integers(1, p["max_jumps"] + 1))
    n_jumps = min(n_jumps, max_start)
    starts = np.sort(rng.choice(np.arange(1, max_start + 1), size=n_jumps, replace=False))
    rates = rng.uniform(*p["shift_us_per_s"], size=n_jumps) * rng.choice([-1, 1], size=n_jumps)

    for s, rate in zip(starts, rates):
        ramp_max = min(ramp_hi, n - s - 2)
        ramp = int(rng.integers(ramp_lo, ramp_max + 1)) if ramp_max >= ramp_lo else ramp_lo
        y[s + ramp:] += rate * (ramp / fs)             # jump = rate × ramp duration
        keep = np.r_[np.arange(s), np.arange(s + ramp, n)]
        spline = CubicSpline(keep, y[keep])
        y[s:s + ramp] = spline(np.arange(s, s + ramp))
        y[y < 0] = 0
    return y[::-1].copy() if flip_time else y


def loose_sensor(x, left, right, fs, p, rng):
    """Signal drops to ~0 for width_s seconds, with smooth edges, residual kept."""
    n = len(x)
    width = _samples(_u(rng, p["width_s"]), fs, n, lo=3)
    width = min(width, n)
    start = int(rng.integers(0, n - width + 1))
    end = start + width - 1

    smooth_left, smooth_right = start != 0, end != n - 1
    avail = (width - 2) // 2
    s_max = min(_samples(p["smooth_s"][1], fs, n, lo=0), avail)
    s_min = min(_samples(p["smooth_s"][0], fs, n, lo=0), s_max)
    w1 = int(rng.integers(s_min, s_max + 1)) if smooth_left else 0
    w2 = int(rng.integers(s_min, s_max + 1)) if smooth_right else 0

    y = x.copy()
    d0, d1 = start + w1, end - w2
    y[d0:d1 + 1] -= np.mean(y[d0:d1 + 1])
    y[y < 0] = 0

    keep = np.r_[np.arange(start), np.arange(d0, d1 + 1), np.arange(end + 1, n)]
    if len(keep) >= 2 and (w1 or w2):
        spline = CubicSpline(keep, y[keep])
        if w1:
            y[start:d0] = spline(np.arange(start, d0))
        if w2:
            y[d1 + 1:end + 1] = spline(np.arange(d1 + 1, end + 1))
    return y


def tonic_scale(x, left, right, fs, p, rng):
    tonic, phasic = _tonic_phasic(x, fs, p["split_hz"], p["order"])
    return tonic * _u(rng, p["scale"]) + phasic


def tonic_warp(x, left, right, fs, p, rng):
    tonic, phasic = _tonic_phasic(x, fs, p["split_hz"], p["order"])
    k = int(rng.integers(p["knots"][0], p["knots"][1] + 1))
    return tonic * _spline_warper(rng, len(x), _u(rng, p["sigma"]), k) + phasic


def amplitude_scale(x, left, right, fs, p, rng):
    return x * _u(rng, p["scale"])


def amplitude_warp(x, left, right, fs, p, rng):
    k = int(rng.integers(p["knots"][0], p["knots"][1] + 1))
    return x * _spline_warper(rng, len(x), _u(rng, p["sigma"]), k)


def gaussian_noise(x, left, right, fs, p, rng):
    sigma = _u(rng, p["sigma_scale"]) * np.mean(np.abs(x - np.mean(x)))
    return x + rng.normal(scale=sigma, size=len(x))


def time_shift(x, left, right, fs, p, rng):
    """Slide the window into its neighbours; shift is capped by the buffer available."""
    n = len(x)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left[~np.isnan(left)]
    right = right[~np.isnan(right)]
    if not len(left) and not len(right):
        return x.copy()
    shift = _samples(_u(rng, p["shift_s"]), fs, 10 ** 9, lo=1)
    choices = []
    if len(left):
        choices.append(-min(shift, len(left)))
    if len(right):
        choices.append(min(shift, len(right)))
    s = int(rng.choice(choices))
    sig = np.concatenate([left, x, right])
    i0 = len(left) + s
    return sig[i0:i0 + n].copy()


def temporal_cutout(x, left, right, fs, p, rng):
    n = len(x)
    size = _samples(_u(rng, p["size_s"]), fs, n, lo=1)
    start = int(rng.integers(0, n - size + 1))
    y = x.copy()
    y[start:start + size] = 0
    return y


def time_warp(x, left, right, fs, p, rng):
    n = len(x)
    k = int(rng.integers(p["knots"][0], p["knots"][1] + 1))
    sigma = _u(rng, p["sigma"])
    steps = np.arange(n)
    warps = rng.normal(1.0, sigma, size=k + 2)
    knots = np.linspace(0, n - 1, num=k + 2)
    warped = CubicSpline(knots, knots * warps)(steps)
    scale = (n - 1) / warped[-1]
    return np.interp(steps, np.clip(scale * warped, 0, n - 1), x)


def permute(x, left, right, fs, p, rng):
    n_splits = int(rng.integers(2, p["max_splits"]))
    parts = np.array_split(np.arange(len(x)), n_splits)
    order = rng.permutation(len(parts))
    return x[np.concatenate([parts[i] for i in order])]


def flip(x, left, right, fs, p, rng):
    return -x + 2 * np.mean(x)


TRANSFORMS = {
    "low_pass": low_pass, "high_pass": high_pass, "band_pass": band_pass,
    "band_stop": band_stop, "hf_noise": hf_noise, "jump_artifact": jump_artifact,
    "loose_sensor": loose_sensor, "tonic_scale": tonic_scale, "tonic_warp": tonic_warp,
    "amplitude_scale": amplitude_scale, "amplitude_warp": amplitude_warp,
    "gaussian_noise": gaussian_noise, "time_shift": time_shift,
    "temporal_cutout": temporal_cutout, "time_warp": time_warp,
    "permute": permute, "flip": flip,
}
assert list(TRANSFORMS) == TRANSFORM_NAMES


# ------------------------------------------------------------------ sampler

@dataclass
class MattonAugmenter:
    """One transform per call, sampled uniformly from `names` (the All-DAs recipe).

    `fs` is the sampling rate of the windows being augmented. `params` defaults
    to `paper_params(fs)`; pass a dict to override individual ranges.
    """
    fs: float = 1.0
    names: tuple = tuple(TRANSFORM_NAMES)
    params: dict = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        base = paper_params(self.fs)
        for k, v in self.params.items():
            base[k] = {**base.get(k, {}), **v}
        self.params = base
        self.rng = np.random.default_rng(self.seed)
        unknown = set(self.names) - set(TRANSFORMS)
        if unknown:
            raise ValueError(f"unknown transforms: {sorted(unknown)}")

    def one(self, x, left=(), right=(), name=None):
        """Augment one window. Returns (augmented, transform_name)."""
        x = np.asarray(x, dtype=float).ravel()
        name = name or self.rng.choice(self.names)
        y = TRANSFORMS[name](x, left, right, self.fs, self.params[name], self.rng)
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):            # filters on a flat window can misbehave
            y = x.copy()
        return y, name

    def batch(self, X, L=None, R=None):
        """Augment each row of X independently. L/R: per-row buffers (NaN-padded) or None."""
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        names = []
        for i in range(len(X)):
            left = () if L is None else L[i]
            right = () if R is None else R[i]
            out[i], nm = self.one(X[i], left, right)
            names.append(nm)
        return out, names


# ------------------------------------------------------------------ preview

def plot_examples(x, left=(), right=(), fs=1.0, seed=0, path=None, subtitle=None):
    """One panel per transform on a single window, like the paper's Figure 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aug = MattonAugmenter(fs=fs, seed=seed)
    x = np.asarray(x, dtype=float).ravel()
    t = np.arange(len(x)) / fs

    fig, axes = plt.subplots(3, 6, figsize=(18, 7.5))
    axes = axes.ravel()
    axes[0].plot(t, x, color="k", lw=1.2)
    axes[0].set_title("original (µS)")
    for ax, name in zip(axes[1:], TRANSFORM_NAMES):
        y, _ = aug.one(x, left, right, name=name)
        ax.plot(t, x, color="0.75", lw=0.8)
        ax.plot(t, y, color="#d95f02" if name in EDA_SPECIFIC else "#1f77b4", lw=1.2)
        ax.set_title(name.replace("_", " "))
    for ax in axes:
        ax.tick_params(labelsize=7)
    for ax in axes[len(TRANSFORM_NAMES) + 1:]:
        ax.set_visible(False)
    title = f"Matton et al. augmentations at fs={fs:g} Hz (orange: EDA-specific, blue: generic)"
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    return fig


def real_example(pid="101", window_id=None, buffer_sec=60):
    """One real labelled ADARP window at 4 Hz, with its context: (x, left, right, label)."""
    import sys
    here = Path(__file__).resolve().parent
    for q in (here, here.parent):
        if str(q) not in sys.path:
            sys.path.insert(0, str(q))
    import pandas as pd
    from data4hz import FS_EDA, load_eda_4hz
    from preprocess_adarp_data import PROCESSED_DIR, load_windows

    _, _, meta = load_windows(PROCESSED_DIR)
    meta = meta[meta["participant"].astype(str) == str(pid)]
    if window_id is None:                      # a stress window from the middle of the first event
        stress = meta[meta["label"] == 1]
        row = stress.iloc[min(40, len(stress) - 1)]
    else:
        row = meta[meta["window_id"] == window_id].iloc[0]

    frame = load_eda_4hz(pid)
    ts, vals = frame.index.asi8, frame["eda"].to_numpy(dtype=float)   # int64 ns, tz-safe
    t0 = pd.Timestamp(row["window_start"]).tz_convert("UTC").value
    t1 = pd.Timestamp(row["window_end"]).tz_convert("UTC").value
    i0, i1 = np.searchsorted(ts, t0), np.searchsorted(ts, t1)
    buf = int(buffer_sec * FS_EDA)
    return (vals[i0:i1], vals[max(0, i0 - buf):i0], vals[i1:i1 + buf],
            f"{pid} {row['window_id']} ({'stress' if row['label'] == 1 else 'non-stress'}, "
            f"{pd.Timestamp(row['window_start']).strftime('%Y-%m-%d %H:%M')} UTC)")


if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    pid = sys.argv[1] if len(sys.argv) > 1 else "101"
    window_id = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        x, left, right, label = real_example(pid, window_id)
        fs = 4.0
    except (FileNotFoundError, ImportError, IndexError) as exc:   # no processed data here
        print(f"real window unavailable ({exc}); using a synthetic one")
        fs, n, buf = 4.0, 240, 240
        tt = np.arange(-buf, n + buf) / fs
        sig = 1.5 + 0.004 * tt + 0.3 * np.exp(-((tt - 8) / 3.0) ** 2) + 0.2 * np.exp(-((tt - 20) / 2.5) ** 2)
        x, left, right, label = sig[buf:buf + n], sig[:buf], sig[buf + n:], "synthetic"

    out = plot_examples(x, left, right, fs=fs, seed=1,
                        path=str(_P(__file__).resolve().parent / "matton_augmentations_example.png"),
                        subtitle=label)
    print(f"wrote {out}  [{label}]")
