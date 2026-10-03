"""Shape, finiteness and sampling checks for the Matton et al. augmentations.

    python -m pytest ADARP/matton_ssl/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import eda_augmentations as ea  # noqa: E402


def _window(fs, n_sec=30, buf_sec=60, seed=0):
    rng = np.random.default_rng(seed)
    n, buf = int(n_sec * fs), int(buf_sec * fs)
    t = np.arange(-buf, n + buf) / fs
    x = (1.5 + 0.004 * t + 0.3 * np.exp(-((t - 8) / 3.0) ** 2)
         + 0.2 * np.exp(-((t - 20) / 2.5) ** 2) + rng.normal(0, 0.005, len(t)))
    return x[buf:buf + n], x[:buf], x[buf + n:]


@pytest.mark.parametrize("fs", [1.0, 4.0])
@pytest.mark.parametrize("name", ea.TRANSFORM_NAMES)
def test_every_transform_preserves_shape_and_is_finite(fs, name):
    x, left, right = _window(fs)
    aug = ea.MattonAugmenter(fs=fs, seed=3)
    for _ in range(20):                       # stochastic: hit many parameter draws
        y, used = aug.one(x, left, right, name=name)
        assert used == name
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))


def test_seventeen_transforms_and_nine_eda_specific():
    assert len(ea.TRANSFORM_NAMES) == 17
    assert len(ea.EDA_SPECIFIC) == 9
    assert set(ea.TRANSFORMS) == set(ea.TRANSFORM_NAMES)


def test_sampler_uses_every_transform():
    x, left, right = _window(1.0)
    aug = ea.MattonAugmenter(fs=1.0, seed=0)
    seen = {aug.one(x, left, right)[1] for _ in range(600)}
    assert seen == set(ea.TRANSFORM_NAMES)


def test_seed_makes_it_reproducible():
    x, left, right = _window(1.0)
    a = ea.MattonAugmenter(fs=1.0, seed=11)
    b = ea.MattonAugmenter(fs=1.0, seed=11)
    ya, na = a.batch(np.tile(x, (8, 1)))
    yb, nb = b.batch(np.tile(x, (8, 1)))
    assert na == nb
    np.testing.assert_allclose(ya, yb)


def test_paper_params_match_config_at_4hz_and_rescale_above_nyquist():
    p4 = ea.paper_params(4.0)
    assert p4["low_pass"]["cutoff_hz"] == (0.25, 1.0)
    assert p4["band_stop"]["center_hz"] == (0.75, 1.0)
    assert p4["time_shift"]["shift_s"] == (15.0, 60.0)       # 60-240 samples at 4 Hz
    assert p4["loose_sensor"]["width_s"] == (10.0, 20.0)      # 40-80 samples
    assert p4["temporal_cutout"]["size_s"] == (5.0, 25.0)     # 20-100 samples
    p1 = ea.paper_params(1.0)
    assert p1["low_pass"]["cutoff_hz"] == (0.0625, 0.25)      # scaled by fs/4
    assert p1["band_stop"]["center_hz"][1] <= 0.5             # below the 1 Hz Nyquist
    assert p1["high_pass"] == p4["high_pass"]                 # physiological: unchanged
    assert p1["tonic_scale"]["split_hz"] == 0.05


def test_time_shift_without_buffers_returns_input():
    x, _, _ = _window(1.0)
    aug = ea.MattonAugmenter(fs=1.0, seed=0)
    y, _ = aug.one(x, (), (), name="time_shift")
    np.testing.assert_allclose(y, x)


def test_time_shift_with_buffers_moves_the_window():
    x, left, right = _window(1.0)
    aug = ea.MattonAugmenter(fs=1.0, seed=0)
    y, _ = aug.one(x, left, right, name="time_shift")
    assert not np.allclose(y, x)


def test_flip_mirrors_about_mean():
    x, _, _ = _window(1.0)
    y, _ = ea.MattonAugmenter(fs=1.0).one(x, name="flip")
    np.testing.assert_allclose(y.mean(), x.mean())
    np.testing.assert_allclose(y, 2 * x.mean() - x)


def test_tonic_scale_leaves_phasic_unchanged():
    x, _, _ = _window(4.0)
    aug = ea.MattonAugmenter(fs=4.0, seed=0, params={"tonic_scale": {"scale": (2.0, 2.0)}})
    y, _ = aug.one(x, name="tonic_scale")
    tonic, phasic = ea._tonic_phasic(x, 4.0, 0.05, 4)
    np.testing.assert_allclose(y, 2 * tonic + phasic, atol=1e-8)
