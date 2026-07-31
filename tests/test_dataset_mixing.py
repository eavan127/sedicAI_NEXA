"""
Tests for mixing real (RadChar) and synthetic radar in one dataset.

These run without RadChar present — only P2 downloads it, so the suite must
stay green for the rest of the team.
"""
import numpy as np
import pytest

from src.config import CFG, CLASS_TO_IDX
from src.data import build_dataset as bd


@pytest.fixture
def small_dataset(monkeypatch):
    """Shrink the per-class quota so these tests run in seconds, not minutes."""
    monkeypatch.setitem(CFG["dataset"], "examples_per_class_per_snr", 8)
    return CFG["dataset"]["examples_per_class_per_snr"]


class TestBalance:
    def test_real_examples_displace_synthetic_rather_than_adding_to_them(
            self, small_dataset):
        """LFM_RADAR must end up the same size as the other classes. If real
        examples were added on top of a full synthetic quota, radar would be
        double everything else and the class weighting would be wrong."""
        n_per = small_dataset
        n_bins = len(CFG["snr_bins_db"])

        for n_real in (0, n_per // 4, n_per // 2):
            synth = bd.build_synthetic_examples(n_real_radar=n_real,
                                                 rng=np.random.default_rng(0))
            radar = [e for e in synth if e[1] == "LFM_RADAR"]
            fhss = [e for e in synth if e[1] == "FHSS"]

            assert len(radar) == (n_per - n_real) * n_bins
            assert len(radar) + n_real * n_bins == len(fhss)

    def test_synthetic_count_never_goes_negative(self):
        """A radchar_fraction above 1 would otherwise request negative examples."""
        synth = bd.build_synthetic_examples(n_real_radar=10 ** 6,
                                             rng=np.random.default_rng(0))
        assert [e for e in synth if e[1] == "LFM_RADAR"] == []


class TestNoiseHandling:
    def test_synthetic_examples_are_noisy(self, small_dataset):
        """Synthetic signals are generated clean, so build_synthetic_examples
        must add noise — otherwise the SNR labels are fiction."""
        synth = bd.build_synthetic_examples(rng=np.random.default_rng(1))
        low = [e for e in synth if e[1] == "LFM_RADAR" and e[2] == min(CFG["snr_bins_db"])]
        assert low, "no low-SNR radar examples generated"

        # A clean chirp has constant envelope; noise breaks that
        sig = low[0][0]
        active = np.abs(sig)[np.abs(sig) > 1e-9]
        assert active.std() > 1e-3, "signal looks noiseless — was add_awgn skipped?"

    def test_radchar_examples_never_reach_add_awgn(self, small_dataset, monkeypatch):
        """RadChar waveforms already carry noise at their labelled SNR. Passing
        them through add_awgn would leave them noisier than the label claims,
        making every real SNR label wrong.

        Checked behaviourally by spying on add_awgn — an earlier version of this
        test inspected source text and failed on a comment mentioning the name.
        """
        sentinel = np.full(CFG["signal"]["window_len"], 7.0 + 7.0j)
        monkeypatch.setattr(bd, "load_real_radar",
                            lambda: [(sentinel.copy(), "LFM_RADAR", 2.0)])

        seen = []
        real_add_awgn = bd.add_awgn

        def spy(signal, snr_db, **kw):
            seen.append(np.asarray(signal).copy())
            return real_add_awgn(signal, snr_db, **kw)

        monkeypatch.setattr(bd, "add_awgn", spy)
        bd.build_full_dataset()

        assert seen, "add_awgn was never called — synthetic path is broken"
        for arr in seen:
            if arr.shape == sentinel.shape and np.allclose(arr, sentinel):
                pytest.fail("a RadChar waveform was passed through add_awgn")


class TestGracefulFallback:
    def test_missing_radchar_returns_empty_not_error(self, monkeypatch):
        """Only P2 has the dataset. Everyone else must still be able to build."""
        def boom(*a, **k):
            raise FileNotFoundError("no RadChar here")

        monkeypatch.setattr("src.data.radchar.load_radchar_lfm", boom)
        assert bd.load_real_radar() == []

    def test_zero_fraction_skips_radchar_entirely(self, monkeypatch):
        monkeypatch.setitem(CFG["dataset"], "radchar_fraction", 0.0)
        called = []
        monkeypatch.setattr("src.data.radchar.load_radchar_lfm",
                            lambda *a, **k: called.append(1) or [])
        assert bd.load_real_radar() == []
        assert not called, "loader was called despite radchar_fraction=0"


def test_radchar_fraction_is_a_sane_proportion():
    f = CFG["dataset"]["radchar_fraction"]
    assert 0.0 <= f <= 1.0, "radchar_fraction must be a proportion"
