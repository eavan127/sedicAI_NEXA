"""
Tests for the week-1 multicarrier and APSK generators.

As with test_digital.py, these check INTERNAL CONSISTENCY -- that each class is
built the way its generator says it is -- not realism.
"""
import numpy as np
import pytest

from src.config import DATASET_CLASSES
from src.data import impairments
from src.generators import apsk, multicarrier

N = multicarrier.WINDOW_LEN
ALL = [(multicarrier, c) for c in multicarrier.CLASSES] + [(apsk, c) for c in apsk.CLASSES]


def clean(cls, seed=0):
    return multicarrier.generate_window(cls, None, np.random.default_rng(seed))


def prefix_match(sig, nfft, cp):
    """Best correlation between a cp-long stretch and the same stretch one FFT
    later, over every start. Close to 1 only if a cyclic prefix sits there."""
    best = 0.0
    for s in range(nfft + cp):
        a, b = sig[s:s + cp], sig[s + nfft:s + nfft + cp]
        if len(b) == cp:
            best = max(best, np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return best


def test_every_generator_class_is_a_dataset_class():
    assert set(multicarrier.CLASSES) | set(apsk.CLASSES) <= set(DATASET_CLASSES)


@pytest.mark.parametrize("gen,cls", ALL)
def test_shape_dtype_and_unit_power(gen, cls):
    sig, _ = gen.generate_window(cls, None, np.random.default_rng(0))
    assert sig.shape == (N,) and sig.dtype == np.complex64
    assert np.isfinite(sig).all()
    assert np.mean(np.abs(sig) ** 2) == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("gen,cls", ALL)
def test_same_seed_same_window(gen, cls):
    a, _ = gen.generate_window(cls, 4, np.random.default_rng(7))
    b, _ = gen.generate_window(cls, 4, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("gen,cls", [(multicarrier, "NR_LIKE"), (apsk, "32APSK")])
@pytest.mark.parametrize("snr_db", [-10, 0, 20])
def test_noise_is_added_at_the_requested_snr(gen, cls, snr_db):
    ratios = []
    for seed in range(20):
        sig, _ = gen.generate_window(cls, None, np.random.default_rng(seed))
        noisy, _ = gen.generate_window(cls, snr_db, np.random.default_rng(seed))
        noise = noisy.astype(complex) - sig
        ratios.append(np.mean(np.abs(sig) ** 2) / np.mean(np.abs(noise) ** 2))
    assert 10 * np.log10(np.mean(ratios)) == pytest.approx(snr_db, abs=0.5)


@pytest.mark.parametrize("gen,cls", ALL)
def test_generate_matches_the_contract(gen, cls):
    iq, meta = gen.generate(cls, 3, 6, np.random.default_rng(0))
    assert iq.shape == (3, N) and iq.dtype == np.complex64
    assert all(np.isfinite(v).all() and len(v) == 3 for v in meta.values())


@pytest.mark.parametrize("gen", [multicarrier, apsk])
def test_unknown_class_is_rejected(gen):
    with pytest.raises(ValueError):
        gen.generate_window("QPSK", None, np.random.default_rng(0))


@pytest.mark.parametrize("gen,cls", ALL)
def test_runs_through_the_impairment_module(gen, cls):
    iq, info = impairments.generate_impaired(gen, cls, 6, np.random.default_rng(0))
    assert iq.shape == (N,) and np.isfinite(iq).all()
    assert np.isfinite(info["bandwidth_hz"])


# --- multicarrier -----------------------------------------------------------

@pytest.mark.parametrize("cls", multicarrier.CLASSES)
def test_window_carries_the_cyclic_prefix(cls):
    for seed in range(5):
        sig, p = clean(cls, seed)
        assert prefix_match(sig, int(p["nfft"]), int(p["cp"])) > 0.99


@pytest.mark.parametrize("cls", multicarrier.CLASSES)
def test_signal_stays_inside_its_subcarriers(cls):
    for seed in range(5):
        sig, p = clean(cls, seed)
        spec = np.abs(np.fft.fftshift(np.fft.fft(sig * np.hanning(N)))) ** 2
        f = np.fft.fftshift(np.fft.fftfreq(N, 1 / multicarrier.FS))
        inside = np.abs(f) <= 0.6 * p["bandwidth_hz"]
        assert spec[inside].sum() > 0.95 * spec.sum()


def test_spacing_separates_the_three_classes():
    spacing = {c: {round(clean(c, s)[1]["scs_hz"], -2) for s in range(40)} for c in multicarrier.CLASSES}
    assert spacing["LTE_LIKE"] == {15000}
    assert all(s >= 29000 for s in spacing["NR_LIKE"])
    assert spacing["OFDM"].isdisjoint(spacing["LTE_LIKE"] | spacing["NR_LIKE"])


def test_lte_leaves_dc_empty_and_nr_does_not():
    assert 0 not in multicarrier._centred_bins(72, multicarrier.LTE_NFFT, skip_dc=True)
    assert 0 in multicarrier._centred_bins(72, 107)
    assert len(multicarrier._centred_bins(72, multicarrier.LTE_NFFT, skip_dc=True)) == 72


def test_nr_fills_whole_resource_blocks():
    for seed in range(20):
        assert clean("NR_LIKE", seed)[1]["n_subcarriers"] % multicarrier.NR_RB_SUBCARRIERS == 0


def test_pilots_land_on_the_stated_bins():
    bins = np.arange(12)
    calls = lambda i: np.array([0, 6]) if i == 0 else np.empty(0, dtype=int)
    sig = multicarrier._ofdm_symbols(64, 4, bins, 2, "64QAM", calls, np.random.default_rng(0))
    first = np.fft.fft(sig[4:68]) / np.sqrt(64)
    pilots = multicarrier.CONSTELLATIONS["QPSK"]
    assert np.min(np.abs(pilots - first[6])) < 1e-9
    assert np.max(np.abs(first[12:])) < 1e-9


# --- apsk -------------------------------------------------------------------

@pytest.mark.parametrize("cls,m", [("16APSK", 16), ("32APSK", 32), ("64APSK", 64), ("128APSK", 128)])
def test_constellation_has_the_named_number_of_points(cls, m):
    pts = apsk.CONSTELLATIONS[cls]
    assert len(np.unique(np.round(pts, 6))) == m
    assert np.mean(np.abs(pts) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize("cls", apsk.CLASSES)
def test_points_sit_on_the_stated_rings(cls):
    counts, radii = apsk.RINGS[cls]
    mags = np.round(np.abs(apsk.CONSTELLATIONS[cls]), 6)
    values, found = np.unique(mags, return_counts=True)
    assert tuple(found) == counts
    np.testing.assert_allclose(values / values[0], radii, atol=1e-4)
