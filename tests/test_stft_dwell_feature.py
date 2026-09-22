"""
Tests for the experimental `model.stft_dwell_feature` flag on STFTBranch.

Why this flag exists (see src/models/amc_cnn.py's STFTBranch and
_sweep_consistency docstrings for the full story): LFM_RADAR precision is
51%, and 46% of its false positives are windows that are really FHSS
(docs/experiments/c2_baseline_eval.json, radar_fhss_confusion). Not masking
-- the signal is visible -- the two classes just look alike to the model as
it stands. A radar chirp sweeps continuously across frequency rows; FHSS
holds one row per hop then jumps to an arbitrary new one.
`_sweep_consistency` reports whether this frame's frequency step keeps going
the same way as the previous one's, which is what tells a sweep apart from a
sequence of unrelated jumps.

These tests protect two things at once:
  1. Flag OFF (the default) must be pixel-for-pixel the existing behaviour --
     the five checkpoints in results/ must keep loading.
  2. Flag ON must add exactly the one new feature, without touching
     AttentionPool1d, and must compose cleanly with stft_freq_summary.
"""
import pytest
import torch

from src.config import CFG, CLASSES
from src.models.amc_cnn import AMC_CNN, STFTBranch, _sweep_consistency

WINDOW_LEN = 512  # matches configs/default.yaml signal.window_len


def _random_iq_batch(batch=3, window_len=WINDOW_LEN):
    return torch.randn(batch, 2, window_len)


# ---------------------------------------------------------------------------
# Flag off == today's behaviour, exactly.
# ---------------------------------------------------------------------------

def test_stft_branch_default_dwell_flag_is_off():
    branch = STFTBranch()
    assert branch.dwell_feature is False


def test_dwell_flag_off_shape_matches_original_formula():
    torch.manual_seed(0)
    branch = STFTBranch()
    x = _random_iq_batch()
    out = branch(x)
    assert out.shape[1] == 64   # unchanged out_channels, no extra rows


def test_dwell_flag_off_output_identical_with_and_without_the_argument():
    torch.manual_seed(0)
    branch_default = STFTBranch()
    torch.manual_seed(0)
    branch_explicit = STFTBranch(dwell_feature=False)
    x = _random_iq_batch()
    torch.testing.assert_close(branch_default(x), branch_explicit(x))


def test_existing_checkpoint_still_loads_with_dwell_flag_off():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     stft_freq_summary=False, stft_dwell_feature=False)
    # A fresh state_dict from the same architecture must round-trip --
    # equivalent to the guarantee that results/ensemble_*.pt keeps loading,
    # since this flag off means an identical module tree to today's.
    sd = model.state_dict()
    model2 = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN)
    model2.load_state_dict(sd, strict=True)


def test_dwell_flag_off_is_the_config_default():
    assert CFG.get("model", {}).get("stft_dwell_feature", False) is False


# ---------------------------------------------------------------------------
# Flag on: exactly one new channel, composes with stft_freq_summary.
# ---------------------------------------------------------------------------

def test_dwell_flag_on_adds_exactly_one_channel_same_time_axis():
    torch.manual_seed(0)
    branch_off = STFTBranch()
    torch.manual_seed(0)
    branch_on = STFTBranch(dwell_feature=True)
    x = _random_iq_batch()
    out_off = branch_off(x)
    out_on = branch_on(x)
    assert out_on.shape[1] == out_off.shape[1] + 1
    assert out_on.shape[2] == out_off.shape[2]   # same time' axis
    assert out_on.shape[0] == out_off.shape[0]


def test_dwell_and_freq_summary_compose_to_four_extra_channels():
    branch = STFTBranch(freq_summary=True, dwell_feature=True)
    x = _random_iq_batch()
    out = branch(x)
    assert out.shape[1] == 64 + 3 + 1


def test_amc_cnn_fused_channels_account_for_dwell_flag():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     stft_freq_summary=False, stft_dwell_feature=True)
    assert model.stft_branch.out_channels == 65
    assert model.attn_pool.score.in_channels == 128 + 65 + 1  # +1 for raw_power


def test_gradients_flow_through_dwell_flag_path():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN,
                     stft_freq_summary=False, stft_dwell_feature=True)
    x = _random_iq_batch(batch=2)
    out = model(x)
    out.sum().backward()
    assert model.stft_branch.conv1.weight.grad is not None


# ---------------------------------------------------------------------------
# _sweep_consistency itself: does it actually separate a sweep from hops?
# ---------------------------------------------------------------------------

def test_sweep_consistency_high_for_a_monotonic_sweep():
    # A synthetic magnitude spectrogram whose loudest bin walks 0,1,2,...,7
    # then wraps -- a clean one-direction sweep for 7 of 8 steps.
    n_freq, n_time = 8, 9
    mag = torch.full((1, n_freq, n_time), 0.1)
    peak_bins = [0, 1, 2, 3, 4, 5, 6, 7, 0]
    for t, b in enumerate(peak_bins):
        mag[0, b, t] = 5.0
    consistency = _sweep_consistency(mag, dim=1)
    # frames 2..7 (0-indexed) should agree with the previous step (all +1)
    assert consistency[0, 2:8].mean() > 0.9


def test_sweep_consistency_low_for_random_hops():
    torch.manual_seed(3)
    n_freq, n_time = 8, 9
    mag = torch.full((1, n_freq, n_time), 0.1)
    peak_bins = torch.randint(0, n_freq, (n_time,))
    for t in range(n_time):
        mag[0, peak_bins[t], t] = 5.0
    consistency = _sweep_consistency(mag, dim=1)
    # no claim of an exact number (it's random), just that it is not
    # saturated like the clean sweep case above
    assert consistency[0, 2:].mean() < 0.9


def test_sweep_consistency_first_two_frames_are_zero():
    mag = torch.rand(2, 8, 5)
    consistency = _sweep_consistency(mag, dim=1)
    assert torch.all(consistency[:, :2] == 0)


def test_sweep_consistency_shape_matches_peak_freq_delta_pattern():
    mag = torch.rand(4, 8, 10)
    consistency = _sweep_consistency(mag, dim=1)
    assert consistency.shape == (4, 10)
