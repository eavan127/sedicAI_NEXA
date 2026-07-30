"""
The format contract: every class must be indistinguishable by shape, dtype or
statistics — only by signal content.

WHY: if any property correlates with class, the model learns that property
instead of the signal. "512 samples means radar" or "half the window is flat
means radar" scores brilliantly on our own split and collapses on the
organisers' stream. These tests make that impossible to introduce by accident.
"""
import numpy as np
import pytest

from src.config import CFG
from src.data.preprocess import add_awgn, preprocess_window
from src.generators.fhss import random_fhss_example
from src.generators.jamming import random_jamming_example
from src.generators.radar import random_radar_example

GENERATORS = {
    "LFM_RADAR": random_radar_example,
    "FHSS": random_fhss_example,
    "JAMMING": random_jamming_example,
}
WINDOW = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]


def _windows(gen_fn, n=6, snr_db=10, seed=0):
    rng = np.random.default_rng(seed)
    return [preprocess_window(add_awgn(gen_fn(rng=rng), snr_db, rng=rng)) for _ in range(n)]


class TestUniformShape:
    @pytest.mark.parametrize("name", list(GENERATORS))
    def test_every_class_produces_the_same_shape_and_dtype(self, name):
        for arr in _windows(GENERATORS[name]):
            assert arr.shape == (2, WINDOW)
            assert arr.dtype == np.float32

    @pytest.mark.parametrize("name", list(GENERATORS))
    def test_no_nan_or_inf(self, name):
        for arr in _windows(GENERATORS[name]):
            assert np.isfinite(arr).all(), f"{name} produced non-finite values"


class TestNoLeakage:
    """Properties that must NOT differ between classes."""

    @pytest.mark.parametrize("name", list(GENERATORS))
    def test_normalisation_is_identical_across_classes(self, name):
        """Post-normalisation every class has mean 0 and std 1, so amplitude
        carries no class information."""
        for arr in _windows(GENERATORS[name]):
            assert arr.mean() == pytest.approx(0.0, abs=1e-4)
            assert arr.std() == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("name", list(GENERATORS))
    def test_generators_fill_the_window_without_padding(self, name):
        """A generator producing fewer samples than the window gets zero-padded,
        leaving a flat tail the model can key on. Every generator must supply at
        least a full window before preprocessing."""
        rng = np.random.default_rng(1)
        raw = GENERATORS[name](rng=rng)
        assert len(raw) >= WINDOW, (
            f"{name} produced {len(raw)} samples for a {WINDOW}-sample window — "
            "the remainder would be zero-padded and become a class fingerprint"
        )

    @pytest.mark.parametrize("name", list(GENERATORS))
    def test_no_long_flat_run_in_any_window(self, name):
        """Catches padding or dead-air artefacts: a long constant stretch is a
        giveaway the model would learn instead of the signal."""
        for arr in _windows(GENERATORS[name]):
            magnitude = np.abs(arr[0] + 1j * arr[1])
            flat = magnitude < 1e-6
            # longest consecutive run of near-zero samples
            longest, run = 0, 0
            for f in flat:
                run = run + 1 if f else 0
                longest = max(longest, run)
            assert longest < WINDOW // 4, (
                f"{name} has a flat run of {longest} samples in a {WINDOW} window"
            )


class TestSignalContentIsDistinguishable:
    """The classes must differ by content — the thing the model is supposed to learn."""

    def test_fhss_shows_multiple_frequencies_within_one_window(self):
        """The bug this was written for: if dwell time exceeds the window, every
        FHSS example is one tone and is indistinguishable from tone jamming."""
        rng = np.random.default_rng(2)
        seg = WINDOW // 4
        axis = np.fft.fftfreq(seg, d=1 / FS)

        multi = 0
        for _ in range(8):
            sig = random_fhss_example(rng=rng)[:WINDOW]
            peaks = {
                round(float(axis[np.argmax(np.abs(np.fft.fft(sig[i*seg:(i+1)*seg])))]))
                for i in range(4)
            }
            if len(peaks) > 1:
                multi += 1

        assert multi >= 6, (
            f"only {multi}/8 FHSS windows contained more than one frequency — "
            "hop rate is too slow for this window length"
        )

    def test_radar_window_contains_a_silent_gap(self):
        """Radar transmits then listens. That gap is what separates it from a
        continuously-sweeping jammer, so it must survive into the window."""
        rng = np.random.default_rng(3)
        with_gap = 0
        for _ in range(8):
            sig = random_radar_example(rng=rng)[:WINDOW]
            if (np.abs(sig) < 1e-9).mean() > 0.05:
                with_gap += 1
        assert with_gap >= 6, f"only {with_gap}/8 radar windows showed a listening gap"
