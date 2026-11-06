"""The maths behind scripts/calibrate_week1.py (SED-39 item 2), on synthetic
scores where the right answer is known."""
import numpy as np
import pytest

from scripts.calibrate_week1 import ece, fit_temperature, noise_rule, sigmoid, vote_curve


def _labels_from(z_true, rng):
    """Draw labels exactly as calibrated probabilities sigmoid(z_true) say."""
    return rng.random(z_true.shape) < sigmoid(z_true)


def test_temperature_recovers_how_overconfident_the_logits_are():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 2, size=(20000, 5))
    y = _labels_from(z_true, rng)
    assert fit_temperature(z_true * 2.0, y) == pytest.approx(2.0, rel=0.05)
    assert fit_temperature(z_true, y) == pytest.approx(1.0, rel=0.05)


def test_ece_is_zero_for_honest_confidence_and_large_for_overconfidence():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 50000)
    hit = rng.random(50000) < p
    assert ece(p, hit) < 0.01
    assert ece(np.full(50000, 0.95), rng.random(50000) < 0.5) == pytest.approx(0.45, abs=0.01)


def test_noise_rule_moves_only_unsure_windows_to_noise():
    # 3 classes, NOISE_FLOOR is index 2. Window 0: sure signal. Window 1: unsure
    # signal. Window 2: noise the model called class 0, unsure.
    p = np.array([[0.9, 0.1, 0.05], [0.12, 0.1, 0.05], [0.08, 0.02, 0.03]])
    y = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], bool)
    snr = np.array([20.0, 20.0, 0.0])
    off = noise_rule(p, y, snr, 2, 0.0)
    assert off["noise_found"] == 0 and off["signals_called_noise_high"] == 0
    on = noise_rule(p, y, snr, 2, 0.1)
    assert on["noise_found"] == 1 and on["signals_called_noise_high"] == 0
    strict = noise_rule(p, y, snr, 2, 0.2)
    assert strict["signals_called_noise_high"] == 0.5 and strict["signal_top1_high"] == 0.5


def test_votes_help_when_windows_are_noisy_but_unbiased():
    rng = np.random.default_rng(2)
    n, c = 4000, 4
    cls = rng.integers(0, c, n)
    y = np.eye(c, dtype=bool)[cls]
    z = rng.normal(0, 1.5, (n, c))
    z[np.arange(n), cls] += 1.0
    snr = np.where(np.arange(n) % 2 == 0, 0.0, 12.0)
    curve = vote_curve(z, y, snr, noise_idx=3, rng=np.random.default_rng(0), reps=4000)
    assert curve[1]["top1"] < curve[8]["top1"] < curve[64]["top1"]
    assert curve[64]["top1"] > 0.95
