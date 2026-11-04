"""Week-1 merge pieces (src/data/merge.py): normalisation matches the
one-window path, the NOISE_FLOOR SNR fix, and mixture labels / power."""
import numpy as np
import pytest

from src.config import CFG, DATASET_CLASSES
from src.data import merge
from src.data.composite import active_power
from src.data.preprocess import preprocess_window

L = CFG["signal"]["window_len"]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _tone(rng, f=0.05):
    return (np.exp(2j * np.pi * f * np.arange(L)) * rng.uniform(0.1, 5)).astype(np.complex64)


def test_normalize_batch_matches_preprocess_window(rng):
    iq = (rng.standard_normal((4, L)) + 1j * rng.standard_normal((4, L))).astype(np.complex64) * 3 + 1
    batch = merge.normalize_batch(iq)
    assert batch.shape == (4, 2, L) and batch.dtype == np.float32
    for b, w in zip(batch, iq):
        np.testing.assert_allclose(b, preprocess_window(w), atol=1e-5)


def test_fill_noise_floor_snr_only_touches_noise_floor(rng):
    y = np.zeros((4, len(DATASET_CLASSES)), dtype=np.uint8)
    y[[0, 1], DATASET_CLASSES.index("NOISE_FLOOR")] = 1
    y[[2, 3], DATASET_CLASSES.index("BPSK")] = 1
    snr = np.array([np.nan, np.nan, 4, -2], dtype=np.float32)
    out = merge.fill_noise_floor_snr(snr, y, rng)
    assert np.isfinite(out).all()
    assert set(out[:2]) <= set(CFG["dataset_snr_bins_db"])
    assert list(out[2:]) == [4, -2]
    assert np.isnan(snr[0]), "input must not be modified"


def test_fill_noise_floor_snr_rejects_nan_on_a_real_signal(rng):
    y = np.zeros((1, len(DATASET_CLASSES)), dtype=np.uint8)
    y[0, DATASET_CLASSES.index("QPSK")] = 1
    with pytest.raises(ValueError):
        merge.fill_noise_floor_snr(np.array([np.nan], dtype=np.float32), y, rng)


@pytest.mark.parametrize("kind", merge.MIX_KINDS)
def test_pick_mixture_classes_are_distinct_and_valid(rng, kind):
    for _ in range(200):
        classes = merge.pick_mixture_classes(rng, kind)
        assert len(set(classes)) == len(classes) == (3 if kind == "three" else 2)
        assert "NOISE_FLOOR" not in classes
        assert sum(c not in merge.JAMMERS for c in classes) >= 2 - (kind == "signal+jammer")
        assert sum(c in merge.JAMMERS for c in classes) <= 1


def test_pick_mixture_classes_without_jammers(rng):
    for kind in merge.MIX_KINDS:
        classes = merge.pick_mixture_classes(rng, kind, jammers=[])
        assert not set(classes) & merge.JAMMERS


def test_mix_windows_hits_requested_snr(rng):
    """With a high SNR the noise is small, so the mixture's power is close to
    the composite's; at a low SNR noise dominates by the requested amount."""
    comps = [("BPSK", _tone(rng, 0.05)), ("2FSK", _tone(rng, -0.1))]
    noisy = merge.mix_windows(comps, -10, np.random.default_rng(1))
    clean = merge.mix_windows(comps, 60, np.random.default_rng(1))
    assert noisy.dtype == np.complex64 and noisy.shape == (L,)
    noise_db = 10 * np.log10(np.mean(np.abs(noisy - clean) ** 2) / active_power(clean))
    assert noise_db == pytest.approx(10, abs=1)


def test_mix_windows_needs_an_emitter(rng):
    with pytest.raises(ValueError):
        merge.mix_windows([("JAM_TONE", _tone(rng))], 10, rng)


def test_combined_version_ignores_order():
    a = [{"shard": "x.npz", "sha256": "1"}, {"shard": "y.npz", "sha256": "2"}]
    assert merge.combined_version(a) == merge.combined_version(a[::-1])
    assert merge.combined_version(a) != merge.combined_version([a[0], {"shard": "y.npz", "sha256": "3"}])
