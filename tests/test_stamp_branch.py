"""
Tests for the experimental `model.stamp_branch` option (StampBranch in src/models/amc_cnn.py).

StampBranch slides 120 fixed clean LFM chirps along the raw IQ -- a matched-filter bank -- so the
model gets a chirp detector that survives noise. The defining property: in heavy noise a real LFM
pulse still produces a sharp spike and FHSS does not.

Groups:
  1. Off by default, and flag-off is exactly the model that shipped (checkpoints load strict).
  2. The stamps are fixed physics: never trained, never saved, unit energy.
  3. It is a correct matched filter (right stamp wins, right place, right direction).
  4. It separates radar from FHSS in noise, and ignores overall loudness.
  5. Plumbing: shapes, gradients, and the ONNX copy computes the same network.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import CFG, CLASSES
from src.data.preprocess import add_awgn, preprocess_window
from src.generators.fhss import random_fhss_example
from src.generators.radar import embed_pulse_train, generate_lfm_chirp_iq
from src.models.amc_cnn import AMC_CNN, StampBranch
from src.models.onnx_export import AMC_CNN_ONNX, compute_stft_mag

WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]
CHECKPOINT = Path(__file__).resolve().parents[1] / "results" / "ensemble_0.pt"


def _iq(batch=4, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).standard_normal((batch, 2, WINDOW_LEN)).astype(np.float32))


def _tensor(iq_complex):
    return torch.from_numpy(preprocess_window(iq_complex)).unsqueeze(0)


def _pulse(pw=40e-6, bw=800e3, start=100e-6 / 2, up=True):
    """One LFM pulse in an otherwise silent window."""
    b = bw if up else -bw
    chirp = generate_lfm_chirp_iq(FS, pw, b, -b / 2)
    return embed_pulse_train(chirp, 1e-3, FS, WINDOW_LEN / FS, start, n_pulses=1), chirp


# ---------------------------------------------------------------------------
# 1. Off by default; flag-off is the shipped model.
# ---------------------------------------------------------------------------

def test_stamp_branch_is_off_in_the_config():
    assert CFG.get("model", {}).get("stamp_branch") is False


def test_flag_off_builds_no_stamp_branch_and_the_same_state_dict():
    torch.manual_seed(0)
    default = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN)
    torch.manual_seed(0)
    off = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=False)
    assert off.stamp_branch is None
    assert default.state_dict().keys() == off.state_dict().keys()
    assert not any(k.startswith("stamp_branch") for k in off.state_dict())


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="shipped checkpoint not present")
def test_shipped_checkpoint_still_loads_strict_with_the_flag_off():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"), strict=True)


# ---------------------------------------------------------------------------
# 2. The stamps are fixed physics.
# ---------------------------------------------------------------------------

def test_there_are_120_unit_energy_stamps():
    stamps = StampBranch.build_stamps(FS)
    assert len(stamps) == 120
    for s in stamps:
        assert np.linalg.norm(s) == pytest.approx(1.0, abs=1e-9)


def test_stamps_are_never_trained_and_never_saved():
    branch = StampBranch()
    param_ids = {id(p) for p in branch.parameters()}
    assert id(branch.stamps) not in param_ids
    assert not branch.stamps.requires_grad
    assert "stamps" not in branch.state_dict()
    # the only learned part is the 120 -> 8 mix and its BatchNorm
    assert sum(p.numel() for p in branch.parameters()) == 120 * 8 + 8 + 2 * 8


def test_training_does_not_move_the_stamps():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=True)
    before = model.stamp_branch.stamps.clone()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    model(_iq()).sum().backward()
    opt.step()
    assert torch.equal(before, model.stamp_branch.stamps)


def test_stamps_come_back_after_a_checkpoint_round_trip():
    trained = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=True)
    fresh = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=True)
    fresh.load_state_dict(trained.state_dict(), strict=True)
    assert torch.equal(trained.stamp_branch.stamps, fresh.stamp_branch.stamps)


# ---------------------------------------------------------------------------
# 3. It is a correct matched filter.
# ---------------------------------------------------------------------------

def _stamp_index(pw, bw, up):
    widths, bws = StampBranch.STAMP_PULSE_WIDTHS_S, StampBranch.STAMP_BANDWIDTHS_HZ
    i, j = int(np.argmin(np.abs(np.log(np.array(widths) / pw)))), int(np.argmin(np.abs(np.log(np.array(bws) / bw))))
    return (i * len(bws) + j) * 2 + (0 if up else 1)


@pytest.mark.parametrize("up", [True, False])
def test_a_pulse_identical_to_a_stamp_matches_that_stamp_best(up):
    pw, bw = StampBranch.STAMP_PULSE_WIDTHS_S[2], StampBranch.STAMP_BANDWIDTHS_HZ[6]
    window, chirp = _pulse(pw=pw, bw=bw, up=up)
    x = torch.from_numpy(np.stack([window.real, window.imag]).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        mag = StampBranch().match_scores(x)[0]       # (120, time)
    peak_per_stamp = mag.max(dim=1).values
    k = _stamp_index(pw, bw, up)
    assert int(peak_per_stamp.argmax()) == k
    # unit-energy stamp against the same chirp at amplitude 1: peak = sqrt(energy of the chirp)
    assert float(peak_per_stamp[k]) == pytest.approx(np.linalg.norm(chirp), rel=1e-3)
    # the opposite sweep direction matches far worse (a sign error in the complex split fails here)
    opposite = k + (1 if up else -1)
    assert float(peak_per_stamp[opposite]) < 0.5 * float(peak_per_stamp[k])


def test_the_spike_lands_at_the_centre_of_the_pulse():
    pw, bw = StampBranch.STAMP_PULSE_WIDTHS_S[3], StampBranch.STAMP_BANDWIDTHS_HZ[5]
    start = 150
    window, chirp = _pulse(pw=pw, bw=bw, start=start / FS)
    x = torch.from_numpy(np.stack([window.real, window.imag]).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        mag = StampBranch().match_scores(x)[0, _stamp_index(pw, bw, True)]
    assert abs(int(mag.argmax()) - (start + len(chirp) // 2)) <= 2


# ---------------------------------------------------------------------------
# 4. Radar vs FHSS in noise; loudness does not matter.
# ---------------------------------------------------------------------------

def _best_spike(branch, iq_complex):
    with torch.no_grad():
        mag = branch.match_scores(_tensor(iq_complex))[0]
        return float((mag / mag.mean(dim=1, keepdim=True)).max())


def test_radar_spikes_higher_than_fhss_at_minus_6_db():
    """The whole point of the branch. Measured AUC 0.92 on the generator's full radar range;
    this seeded check uses one mid-range pulse and asks for clear separation."""
    rng = np.random.default_rng(7)
    branch = StampBranch()
    radar = [_best_spike(branch, add_awgn(_pulse()[0], -6, rng=rng)) for _ in range(40)]
    fhss = [_best_spike(branch, add_awgn(np.asarray(random_fhss_example(rng=rng))[:WINDOW_LEN], -6, rng=rng))
            for _ in range(40)]
    auc = (np.asarray(radar)[:, None] > np.asarray(fhss)[None, :]).mean()
    assert auc > 0.85, f"AUC {auc:.2f}"


def test_output_ignores_overall_loudness():
    branch = StampBranch().eval()
    x = _iq(batch=2)
    with torch.no_grad():
        assert torch.allclose(branch(x), branch(3.0 * x), atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Plumbing.
# ---------------------------------------------------------------------------

def test_shapes_and_fused_width():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=True).eval()
    x = _iq(batch=3)
    with torch.no_grad():
        assert model.stamp_branch(x).shape == (3, 8, WINDOW_LEN)
        assert model(x).shape == (3, len(CLASSES))
    assert model.fc1.in_features == 128 + 64 + 8


def test_gradients_reach_the_learned_mix():
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, stamp_branch=True)
    model(_iq()).sum().backward()
    g = model.stamp_branch.mix.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    dict(stamp_branch=True),
    dict(stamp_branch=True, stft_keep_rows=True),
    dict(stamp_branch=True, stft_freq_summary=True),
])
def test_onnx_wrapper_matches_the_model(kwargs):
    torch.manual_seed(3)
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN, **kwargs).eval()
    x = _iq(batch=2)
    with torch.no_grad():
        mag = compute_stft_mag(x, n_fft=model.stft_branch.n_fft,
                               hop_length=model.stft_branch.hop_length,
                               window=model.stft_branch.window)
        logits, _attention = AMC_CNN_ONNX(model)(x, mag)
        assert torch.allclose(logits, model(x), atol=1e-5)
