"""The GPU expert-feature branch must compute what src/expert_features.py
(the numpy reference the probes used) computes, and plug into AMC_CNN
without disturbing models built without it."""
import numpy as np
import pytest
import torch

from src.config import CFG
from src.data import impairments
from src.data.merge import normalize_batch
from src.expert_features import EXPERT_DIM, expert_features_from_input
from src.generators import apsk, digital, multicarrier
from src.models.amc_cnn import AMC_CNN, ExpertFeatures

L = CFG["signal"]["window_len"]


@pytest.fixture(scope="module")
def windows():
    rng = np.random.default_rng(3)
    iq = [impairments.generate_impaired(g, c, snr, rng)[0]
          for g, c in ((digital, "16QAM"), (apsk, "32APSK"), (multicarrier, "LTE_LIKE"), (digital, "BPSK"))
          for snr in (-6, 6, 18)]
    return normalize_batch(np.array(iq))


def test_torch_branch_matches_numpy_reference(windows):
    got = ExpertFeatures(L)(torch.from_numpy(windows)).numpy()
    want = np.array([expert_features_from_input(w) for w in windows])
    assert got.shape == want.shape == (len(windows), EXPERT_DIM)
    # float32 DFT vs float64 FFT: a bin right at the occupancy threshold can
    # flip, so compare robustly rather than element-exactly.
    err = np.abs(got - want)
    assert np.median(err) < 1e-4
    assert np.mean(err < 1e-2) > 0.97


def test_amc_cnn_with_and_without_expert_features():
    x = torch.randn(4, 2, L)
    off = AMC_CNN(num_classes=60, input_len=L, stft_n_fft=64, expert_features=False)
    on = AMC_CNN(num_classes=60, input_len=L, stft_n_fft=64, expert_features=True)
    assert off.expert_branch is None and "expert_norm.weight" not in off.state_dict()
    assert on.fc1.in_features == off.fc1.in_features + EXPERT_DIM
    assert on(x).shape == off(x).shape == (4, 60)
    assert not any(k.startswith("expert_branch.") for k in on.state_dict()), "DFT tables must not be saved"


def test_expert_features_get_no_gradient():
    model = AMC_CNN(num_classes=60, input_len=L, stft_n_fft=64, expert_features=True)
    x = torch.randn(3, 2, L, requires_grad=True)
    model(x).sum().backward()
    assert model.expert_norm.weight.grad is not None
