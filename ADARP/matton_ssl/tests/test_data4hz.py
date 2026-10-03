"""Window bank and timestamp-slicing checks for the 4 Hz data path."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data4hz as d4  # noqa: E402


def test_window_bank_skips_nan_windows_and_gathers_buffers():
    rng = np.random.default_rng(0)
    s1 = 1.0 + rng.normal(0, 0.01, 2400)             # 10 min at 4 Hz
    s2 = 2.0 + rng.normal(0, 0.01, 1200)
    s2[500:505] = np.nan
    bank = d4.WindowBank([("p", "a", s1), ("p", "b", s2)], window=240, step=120, buffer=240)
    # s1: (2400-240)/120+1 = 19 windows; s2: 9 candidates minus those touching 500..504
    assert len(bank) == 19 + 9 - sum(1 for s in range(0, 1200 - 240 + 1, 120)
                                     if s <= 504 and s + 240 > 500)
    X, L, R = bank.gather(np.arange(len(bank)), with_buffers=True)
    assert X.shape == (len(bank), 240) and L.shape == R.shape == (len(bank), 240)
    assert np.isfinite(X).all()
    # first window of a session has an all-NaN left buffer, the last an all-NaN right buffer
    assert np.isnan(L[0]).all()
    assert np.isnan(R[18]).all()


def test_zscore_rows_handles_flat_rows():
    X = np.vstack([np.arange(10.0), np.full(10, 3.0)])
    Z = d4.zscore_rows(X)
    np.testing.assert_allclose(Z[0].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(Z[0].std(), 1, atol=1e-6)
    assert (Z[1] == 0).all()


def test_labelled_windows_are_cut_by_timestamp_at_4hz():
    t0 = pd.Timestamp("2019-05-01 10:00:00", tz="UTC")
    idx = pd.date_range(t0, periods=4 * 600, freq="250ms", tz="UTC")       # 10 min
    vals = np.arange(len(idx), dtype=np.float32)
    frames = {"101": pd.DataFrame({"eda": vals, "session": "s"}, index=idx)}

    meta = pd.DataFrame({
        "participant": ["101", "101", "101"],
        "window_start": [t0 + pd.Timedelta(seconds=30), t0 + pd.Timedelta(seconds=60),
                         t0 + pd.Timedelta(seconds=570)],          # last one runs past the end
    })
    meta["window_end"] = meta["window_start"] + pd.Timedelta(seconds=60)

    X = d4.labelled_eda_windows(meta, frames, fs=4.0, window_sec=60)
    assert X.shape == (3, 240)
    np.testing.assert_array_equal(X[0], vals[120:360])      # 30 s * 4 Hz = sample 120
    np.testing.assert_array_equal(X[1], vals[240:480])
    assert np.isnan(X[2]).all()                             # incomplete -> NaN, not a short window
