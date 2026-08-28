"""
Tests for the experimental `model.stft_freq_summary` flag on STFTBranch.

Why this flag exists (see src/models/amc_cnn.py's STFTBranch docstring for
the full story): STFTBranch's final line averages the frequency axis away
(`f.mean(dim=2)`), so "peak at bin 2 then bin 6" (FHSS hopping) and "peak at
bin 4 then bin 4" (tone jamming) collapse to identical features. Measured
consequences: FHSS recall 0.960 -> 0.040 (0dB -> +10dB jammer-to-signal
ratio), 46.5% of held-out jamming predicted as FHSS, jamming's seed spread
10.8 points against FHSS's 1.1.

These tests protect two things at once:
  1. Flag OFF (the default) must be pixel-for-pixel the existing behaviour --
     the five checkpoints in results/ must keep loading.
  2. Flag ON must add the three physically-motivated features this file's
     sibling docstring describes, without touching AttentionPool1d.
"""
import numbers

import pytest
import torch

from src.config import CFG, CLASSES
from src.models.amc_cnn import (
    AMC_CNN,
    STFTBranch,
    _frequency_max,
    _peak_freq_delta,
    _spectral_flatness,
)

WINDOW_LEN = 512  # matches configs/default.yaml signal.window_len


def _random_iq_batch(batch=3, window_len=WINDOW_LEN):
    return torch.randn(batch, 2, window_len)


# ---------------------------------------------------------------------------
# Flag off == today's behaviour, exactly.
# ---------------------------------------------------------------------------

def test_stft_branch_default_flag_is_off():
    branch = STFTBranch()
    assert branch.freq_summary is False


def test_stft_branch_flag_off_shape_matches_original_formula():
    """n_fft=16, hop_length=4, window_len=512, center=False ->
    time_frames = floor((512-16)/4) + 1 = 125, freq_bins = 16 (full complex
    STFT, non-onesided). The original 2x2 MaxPool halves both -> freq' = 8,
    time' = 62. conv2 -> 64 channels. mean over freq -> (batch, 64, 62).
    This is the exact shape STFTBranch has always returned; the flag must not
    change it."""
    branch = STFTBranch(freq_summary=False)
    out = branch(_random_iq_batch(batch=5))
    assert out.shape == (5, 64, 62)


def test_stft_branch_flag_off_is_deterministic_given_fixed_weights():
    """Two branches built from the same seed, flag off, must produce
    byte-identical output -- there is no hidden state-dependent branching
    left in the flag-off path."""
    torch.manual_seed(0)
    branch_a = STFTBranch(freq_summary=False)
    torch.manual_seed(0)
    branch_b = STFTBranch(freq_summary=False)
    branch_a.eval()
    branch_b.eval()

    x = _random_iq_batch(batch=4)
    with torch.no_grad():
        out_a = branch_a(x)
        out_b = branch_b(x)
    assert torch.equal(out_a, out_b)


def test_existing_checkpoint_still_loads_with_flag_off():
    """The guard that protects the submission: results/best_model.pt was
    trained against today's architecture. Flag off must keep AMC_CNN's
    state_dict shape-compatible with it."""
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN)
    state_dict = torch.load("results/best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)  # must not raise


def test_flag_off_is_the_config_default():
    """Verified from the config, not the function signature -- a default
    baked only into STFTBranch's kwarg would not protect a caller that reads
    the flag from CFG."""
    assert CFG.get("model", {}).get("stft_freq_summary") is False


# ---------------------------------------------------------------------------
# Flag on: channel count and time-axis invariants.
# ---------------------------------------------------------------------------

def test_flag_on_adds_exactly_three_channels_same_time_axis():
    x = _random_iq_batch(batch=2)
    off = STFTBranch(freq_summary=False)
    on = STFTBranch(freq_summary=True)
    out_off = off(x)
    out_on = on(x)

    assert out_on.shape[0] == out_off.shape[0]
    assert out_on.shape[1] == out_off.shape[1] + 3
    assert out_on.shape[2] == out_off.shape[2]


def test_amc_cnn_fused_channels_computed_not_hardcoded():
    """fused_channels must reflect whichever branch config was actually
    used, in both modes -- not a hardcoded 192."""
    model_off = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                         stft_freq_summary=False)
    model_on = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                        stft_freq_summary=True)

    assert model_off.fc1.in_features == 128 + 64
    assert model_on.fc1.in_features == 128 + 64 + 3

    # And the forward pass actually runs end to end in both modes.
    x = _random_iq_batch(batch=2)
    assert model_off(x).shape == (2, len(CLASSES))
    assert model_on(x).shape == (2, len(CLASSES))


# ---------------------------------------------------------------------------
# The three physical features, tested directly against hand-built spectra.
# ---------------------------------------------------------------------------

def test_spectral_flatness_near_zero_for_pure_tone_near_one_for_white_noise():
    """Near 1 for broadband noise, near 0 for tonal/narrowband -- geometric
    mean over frequency divided by arithmetic mean over frequency."""
    n_bins, n_frames = 32, 5

    # Pure tone: all energy in one bin, ~nothing elsewhere.
    tone = torch.full((1, n_bins, n_frames), 1e-6)
    tone[:, 7, :] = 1.0

    # Idealized white noise: (near-)flat magnitude across every bin.
    noise = torch.full((1, n_bins, n_frames), 1.0) + torch.rand(1, n_bins, n_frames) * 0.01

    flatness_tone = _spectral_flatness(tone, dim=1)
    flatness_noise = _spectral_flatness(noise, dim=1)

    assert flatness_tone.max().item() < 0.05, flatness_tone
    assert flatness_noise.min().item() > 0.9, flatness_noise
    # Clear margin between the two regimes.
    assert (flatness_noise.min() - flatness_tone.max()).item() > 0.5


def test_peak_freq_delta_zero_for_stationary_tone():
    n_bins, n_frames = 16, 4
    mag = torch.full((1, n_bins, n_frames), 1e-6)
    mag[:, 5, :] = 1.0  # same peak bin every frame -> stationary tone

    delta = _peak_freq_delta(mag, dim=1)
    assert delta.shape == (1, n_frames)
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_peak_freq_delta_nonzero_and_signed_for_a_hop():
    n_bins, n_frames = 16, 4
    mag = torch.full((1, n_bins, n_frames), 1e-6)
    mag[:, 2, 0] = 1.0  # frame 0: bin 2
    mag[:, 2, 1] = 1.0  # frame 1: bin 2 (still)
    mag[:, 9, 2] = 1.0  # frame 2: hop UP to bin 9
    mag[:, 9, 3] = 1.0  # frame 3: bin 9 (still)

    delta = _peak_freq_delta(mag, dim=1)
    assert delta[0, 0].item() == 0.0  # first frame always defined as 0
    assert delta[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert delta[0, 2].item() > 0  # hopped UP -> positive
    expected = (9 - 2) / n_bins
    assert delta[0, 2].item() == pytest.approx(expected, abs=1e-6)
    assert delta[0, 3].item() == pytest.approx(0.0, abs=1e-6)
    assert delta.abs().max().item() <= 1.0


def test_frequency_max_separates_narrowband_under_broadband_floor_where_mean_does_not():
    """This is the feature's entire justification: a narrowband victim
    survives a broadband floor in the max and is erased in the mean."""
    n_bins, n_frames = 64, 3
    base_level = 1.0

    broadband_only = torch.full((1, n_bins, n_frames), base_level)

    # Narrowband tone ~10dB above the broadband floor (power ratio 10 ->
    # amplitude ratio sqrt(10)) added on top of the same floor, in ONE bin.
    signal_plus_floor = broadband_only.clone()
    signal_plus_floor[:, 30, :] = base_level * (10 ** 0.5)

    max_noise_only = _frequency_max(broadband_only, dim=1)
    max_with_signal = _frequency_max(signal_plus_floor, dim=1)
    mean_noise_only = broadband_only.mean(dim=1)
    mean_with_signal = signal_plus_floor.mean(dim=1)

    max_gap = (max_with_signal - max_noise_only).abs().mean().item()
    mean_gap = (mean_with_signal - mean_noise_only).abs().mean().item()

    assert max_gap > 1.0  # the max feature clearly sees the spike
    assert mean_gap < 0.1  # one bin out of 64 barely moves the mean
    assert max_gap > 10 * mean_gap


# ---------------------------------------------------------------------------
# Gradients flow through the new path.
# ---------------------------------------------------------------------------

def test_gradients_flow_through_flag_on_path():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     stft_freq_summary=True)
    x = _random_iq_batch(batch=3)
    x.requires_grad_(True)

    out = model(x)
    loss = out.sum()
    loss.backward()  # must not raise

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    any_param_grad = False
    for p in model.stft_branch.parameters():
        if p.grad is not None:
            any_param_grad = True
            assert torch.isfinite(p.grad).all()
    assert any_param_grad
