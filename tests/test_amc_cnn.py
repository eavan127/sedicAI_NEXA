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

import numpy as np
import pytest
import torch

from src.config import CFG, CLASSES
from src.cumulants import normalized_c40, normalized_c42, normalized_c63
from src.models.amc_cnn import (
    AMC_CNN,
    CumulantFeatures,
    IFFeatures,
    STFTBranch,
    _frequency_max,
    _if_spikiness_ratio,
    _peak_freq_delta,
    _spectral_flatness,
    _torch_normalized_cumulants,
    _torch_unwrap,
)
from tests.test_measure import _qam_constellation

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


# ---------------------------------------------------------------------------
# `model.cumulant_features`: expert-feature branch (RRC matched filter +
# |C40|/|C42|/|C63|) fixing the model's demonstrated inability to tell
# 16QAM from 64QAM apart. See configs/default.yaml's `cumulant_features`
# comment for the measured numbers motivating this (51.4% chance-level
# single-window accuracy, 47.0% on true 16QAM -- worse than chance -- and
# the AUC improvement matched filtering buys: 0.576 raw -> 0.609 filtered).
# ---------------------------------------------------------------------------

def test_cumulant_flag_default_is_off_in_config():
    assert CFG.get("model", {}).get("cumulant_features") is False


def test_cumulant_flag_off_fc1_unchanged_and_forward_shape_unchanged():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     cumulant_features=False)
    assert model.fc1.in_features == 128 + 64
    x = _random_iq_batch(batch=2)
    assert model(x).shape == (2, len(CLASSES))


def test_existing_checkpoint_still_loads_with_cumulant_flag_off():
    """The guard that protects the submission: results/best_model.pt was
    trained against today's architecture (no cumulant branch). Flag off
    must keep AMC_CNN's state_dict shape-compatible with it -- the
    CumulantFeatures branch must not even be constructed, let alone add
    parameters/buffers, when the flag is off."""
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     cumulant_features=False)
    state_dict = torch.load("results/best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)  # must not raise


def test_cumulant_flag_on_adds_exactly_three_fc1_inputs():
    model_off = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                         cumulant_features=False)
    model_on = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                        cumulant_features=True)
    assert model_on.fc1.in_features == model_off.fc1.in_features + 3

    x = _random_iq_batch(batch=2)
    assert model_on(x).shape == (2, len(CLASSES))


def test_torch_cumulants_match_numpy_cumulants():
    """The test that matters: two independent implementations of the same
    formula (numpy in src/cumulants.py, batched torch in
    src/models/amc_cnn.py's _torch_normalized_cumulants) WILL drift apart
    silently unless something checks them against each other directly."""
    rng = np.random.default_rng(0)
    batch = 5
    n = 200
    signals = (rng.normal(size=(batch, n)) + 1j * rng.normal(size=(batch, n))
               ) * (1.0 + rng.uniform(0, 3, size=(batch, 1)))  # varied power

    z = torch.tensor(signals, dtype=torch.complex64)
    torch_feats = _torch_normalized_cumulants(z).numpy()  # (batch, 3): c40,c42,c63

    for i in range(batch):
        pts = signals[i]
        expected = np.array([
            normalized_c40(pts), normalized_c42(pts), normalized_c63(pts),
        ])
        np.testing.assert_allclose(torch_feats[i], expected, atol=1e-4, rtol=1e-4)


def test_cumulant_feature_separates_16qam_from_64qam():
    """The feature actually separates the classes it exists to separate.
    Theoretical noiseless |C42|: 16QAM 0.680, 64QAM 0.619 (C42_THEORY,
    src/measure.py) -- 16QAM's constellation is "spikier" (more low-amplitude
    inner points relative to its power) so its normalised 4th moment sits
    higher. The torch path must reproduce that ordering on bare
    constellations (no matched filtering needed here -- these are already
    symbol-rate points, not an oversampled window)."""
    const_16 = _qam_constellation(16)
    const_64 = _qam_constellation(64)

    z16 = torch.tensor(const_16, dtype=torch.complex64).unsqueeze(0)
    z64 = torch.tensor(const_64, dtype=torch.complex64).unsqueeze(0)

    feats_16 = _torch_normalized_cumulants(z16)[0]
    feats_64 = _torch_normalized_cumulants(z64)[0]

    c42_16 = feats_16[1].item()
    c42_64 = feats_64[1].item()
    assert c42_16 > c42_64
    assert c42_16 == pytest.approx(0.680, abs=1e-3)
    assert c42_64 == pytest.approx(0.619, abs=1e-3)


def test_cumulant_branch_gradients_finite_with_flag_on():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     cumulant_features=True)
    x = _random_iq_batch(batch=3)
    x.requires_grad_(True)

    out = model(x)
    loss = out.sum()
    loss.backward()  # must not raise

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    any_param_grad = False
    for p in model.parameters():
        if p.grad is not None:
            any_param_grad = True
            assert torch.isfinite(p.grad).all()
    assert any_param_grad

    # The matched-filter kernel is fixed, not learned.
    assert model.cumulant_branch.mf_kernel.requires_grad is False
    for p in model.cumulant_branch.parameters():
        pytest.fail(f"CumulantFeatures must have no learned parameters, got {p}")


def test_cumulant_features_degenerate_all_zero_window_is_finite():
    branch = CumulantFeatures()
    x = torch.zeros(2, 2, WINDOW_LEN)
    feats = branch(x)
    assert torch.isfinite(feats).all()
    assert feats.shape == (2, 3)


# ---------------------------------------------------------------------------
# `model.if_features`: expert-feature branch (instantaneous-frequency
# spikiness ratio) targeting the measured LFM_RADAR <-> FHSS confusion. See
# configs/default.yaml's `if_features` comment for the full measured story:
# half of LFM_RADAR's false positives are genuinely FHSS (50.7%) and 35.5%
# of FHSS's are genuinely radar, and the two classes trade against each
# other across fix iterations (FHSS recall 82.5 -> 89.7 -> 92.2 while
# JAMMING fell 80.0 -> 73.3 -> 67.5 in the same runs). A chirp sweeps
# frequency linearly and continuously; a hopper holds a channel then jumps.
# The second derivative of unwrapped phase -- ratio of its max absolute
# value to its median absolute value -- captures exactly that difference:
# measured pooled AUC 0.887 (0.611 at -10dB rising to 0.964 at +10dB, see
# the module-level docstring on IFFeatures for the full per-SNR table).
# ---------------------------------------------------------------------------

def _synthetic_chirp(window_len=WINDOW_LEN, f0=0.02, f1=0.15):
    """A linear chirp: phase quadratic in time (instantaneous frequency
    sweeps linearly from f0 to f1, both in cycles/sample). This is the
    textbook model of an LFM pulse -- constant-rate frequency sweep, no
    holds and no jumps."""
    t = np.arange(window_len)
    k = (f1 - f0) / window_len   # chirp rate, cycles/sample^2
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    z = np.exp(1j * phase)
    return torch.tensor(np.stack([z.real, z.imag]), dtype=torch.float32)


def _synthetic_hopper(window_len=WINDOW_LEN, n_hops=8, freqs=(0.05, 0.35, 0.15, 0.42)):
    """A frequency hopper: piecewise-CONSTANT instantaneous frequency,
    holding each channel for a dwell then jumping (phase discontinuous in
    slope, not in value -- each segment's phase starts where the last one's
    left off, so there's no amplitude glitch, only a kink in frequency)."""
    dwell = window_len // n_hops
    phase = np.zeros(window_len)
    t_local = np.arange(dwell)
    running_phase = 0.0
    for i in range(n_hops):
        start = i * dwell
        end = window_len if i == n_hops - 1 else start + dwell
        seg_len = end - start
        f = freqs[i % len(freqs)]
        seg_t = np.arange(seg_len)
        seg_phase = running_phase + 2 * np.pi * f * seg_t
        phase[start:end] = seg_phase
        running_phase = seg_phase[-1] + 2 * np.pi * f   # continue phase across the jump
    z = np.exp(1j * phase)
    return torch.tensor(np.stack([z.real, z.imag]), dtype=torch.float32)


def test_if_flag_default_is_off_in_config():
    assert CFG.get("model", {}).get("if_features") is False


def test_if_flag_off_fc1_unchanged_and_forward_shape_unchanged():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     if_features=False)
    assert model.fc1.in_features == 128 + 64
    x = _random_iq_batch(batch=2)
    assert model(x).shape == (2, len(CLASSES))


def test_existing_checkpoint_still_loads_with_if_flag_off():
    """The guard that protects the submission: results/best_model.pt was
    trained against today's architecture (no IF branch). Flag off must
    keep AMC_CNN's state_dict shape-compatible with it -- the IFFeatures
    branch must not even be constructed, let alone add parameters/buffers,
    when the flag is off."""
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     if_features=False)
    state_dict = torch.load("results/best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)  # must not raise


def test_if_flag_on_adds_exactly_one_fc1_input():
    model_off = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                         if_features=False)
    model_on = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                        if_features=True)
    assert model_on.fc1.in_features == model_off.fc1.in_features + 1

    x = _random_iq_batch(batch=2)
    assert model_on(x).shape == (2, len(CLASSES))


def test_both_expert_flags_compose_fc1_in_features_all_four_combinations():
    base = 128 + 64
    combos = {
        (False, False): base,
        (True, False): base + 3,
        (False, True): base + 1,
        (True, True): base + 4,
    }
    for (cum, ifb), expected in combos.items():
        model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                         cumulant_features=cum, if_features=ifb)
        assert model.fc1.in_features == expected, (cum, ifb, model.fc1.in_features)
        x = _random_iq_batch(batch=2)
        assert model(x).shape == (2, len(CLASSES))


def test_torch_unwrap_matches_numpy_unwrap_on_wrapping_signal():
    """The test that matters most for correctness: torch has no built-in
    unwrap on this version (torch 2.13 -- see IFFeatures' docstring), so
    _torch_unwrap is a hand-rolled cumulative-correction implementation.
    Two independent implementations of the same algorithm WILL drift apart
    silently unless something checks them directly against numpy.unwrap,
    the reference. A fast chirp wraps the raw angle several times over the
    window, which is exactly the regime where a broken unwrap shows up."""
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW_LEN)
    # Fast sweep so raw phase wraps many times over the window.
    phase = 2 * np.pi * (0.01 * t + 0.0008 * t ** 2) + rng.normal(0, 0.05, WINDOW_LEN)

    expected = np.unwrap(phase)
    got = _torch_unwrap(torch.tensor(phase, dtype=torch.float64)).numpy()

    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_if_spikiness_ratio_separates_chirp_from_hopper():
    """The feature actually separates the classes it exists to separate.
    Measured at +2dB SNR (the report's reference table): radar 4.20 +/-
    0.35, FHSS 5.27 +/- 0.50 -- the hopper's ratio is clearly higher
    because its instantaneous frequency sits still then spikes, while the
    chirp's changes at a near-constant rate. The synthetic case here is
    noiseless, so the absolute numbers won't match, but the DIRECTION
    (hopper > chirp) must."""
    chirp = _synthetic_chirp().unsqueeze(0)
    hopper = _synthetic_hopper().unsqueeze(0)

    z_chirp = torch.complex(chirp[:, 0, :], chirp[:, 1, :])
    z_hopper = torch.complex(hopper[:, 0, :], hopper[:, 1, :])

    ratio_chirp = _if_spikiness_ratio(z_chirp).item()
    ratio_hopper = _if_spikiness_ratio(z_hopper).item()

    assert ratio_hopper > ratio_chirp


def test_if_spikiness_ratio_degenerate_inputs_are_finite():
    zero = torch.zeros(2, WINDOW_LEN, dtype=torch.complex64)
    tone_t = torch.arange(WINDOW_LEN, dtype=torch.float32)
    tone_phase = 2 * torch.pi * 0.1 * tone_t
    tone = torch.polar(torch.ones(WINDOW_LEN), tone_phase).unsqueeze(0).repeat(2, 1)

    ratio_zero = _if_spikiness_ratio(zero)
    ratio_tone = _if_spikiness_ratio(tone)

    assert torch.isfinite(ratio_zero).all()
    assert torch.isfinite(ratio_tone).all()


def test_if_features_module_output_shape_and_finite():
    branch = IFFeatures()
    x = _random_iq_batch(batch=4)
    feats = branch(x)
    assert feats.shape == (4, 1)
    assert torch.isfinite(feats).all()


def test_if_branch_gradients_finite_with_flag_on():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     if_features=True)
    x = _random_iq_batch(batch=3)
    x.requires_grad_(True)

    out = model(x)
    loss = out.sum()
    loss.backward()  # must not raise

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    any_param_grad = False
    for p in model.parameters():
        if p.grad is not None:
            any_param_grad = True
            assert torch.isfinite(p.grad).all()
    assert any_param_grad

    # IFFeatures has no learned parameters at all.
    for p in model.if_branch.parameters():
        pytest.fail(f"IFFeatures must have no learned parameters, got {p}")
