"""The physics branch (AMC_CNN(physics_features=True)) must compute the same
numbers as the numpy reference src/physics_features.py, stay out of the
gradient, add no saved weights beyond its BatchNorm, and leave models built
without it unchanged."""
import numpy as np
import pytest
import torch

from src import physics_features as pf
from src.models.amc_cnn import AMC_CNN, PhysicsFeatures

L = 512


def _windows(n=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    out = []
    sym = rng.choice([1, -1, 1j, -1j], L // 8).repeat(8)                 # QPSK, 8 samples / symbol
    out.append(sym * np.exp(2j * np.pi * 0.1 * t))
    out.append(np.exp(2j * np.pi * (0.05 * t + 0.3 * np.sin(2 * np.pi * t / 90))))   # FM
    out.append(rng.normal(size=L) + 1j * rng.normal(size=L))              # noise
    burst = np.zeros(L, complex)
    burst[200:] = np.exp(2j * np.pi * 0.2 * t[200:])                      # switch-on
    out.append(burst)
    out.append(np.exp(2j * np.pi * 0.15 * t) + 0.5 * np.exp(2j * np.pi * -0.2 * t))
    out.append(rng.choice([1, -1], L // 4).repeat(4) * 1.0 + 0j)          # BPSK at 0 Hz
    W = np.array(out[:n])
    W = W + 0.05 * (rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape))
    return np.stack([W.real, W.imag], 1).astype(np.float32)


def test_torch_matches_numpy_reference():
    x = _windows()
    ref = np.array([pf.physics_features_from_input(w) for w in x])
    got = PhysicsFeatures(L)(torch.from_numpy(x)).numpy()
    assert got.shape == ref.shape == (len(x), pf.PHYSICS_DIM)
    assert np.isfinite(got).all()
    # float32 vs float64: compare with a tolerance scaled to each feature
    tol = 0.05 * np.maximum(1.0, np.abs(ref))
    bad = np.argwhere(np.abs(got - ref) > tol)
    # argmax-type features (symbol-rate position, echo delay) may land on a
    # neighbouring tie in float32; allow at most a couple of such flips
    assert len(bad) <= 2, f"mismatches at (window, feature): {bad.tolist()}"


def test_branch_is_gradient_free_and_has_no_saved_buffers():
    m = PhysicsFeatures(L)
    assert len(list(m.parameters())) == 0
    assert m.state_dict() == {} or all("chan" not in k for k in m.state_dict())
    x = torch.from_numpy(_windows(2)).requires_grad_(True)
    assert not m(x).requires_grad


def test_model_with_physics_branch_runs_and_trains_the_rest():
    m = AMC_CNN(60, input_len=L, expert_features=True, physics_features=True)
    x = torch.from_numpy(_windows(4))
    out = m(x)
    assert out.shape == (4, 60)
    out.sum().backward()
    assert m.physics_norm.weight.grad is not None
    keys = m.state_dict().keys()
    assert any(k.startswith("physics_norm.") for k in keys)
    assert not any(k.startswith("physics_branch.") for k in keys)


def test_default_model_is_unchanged_by_the_new_flag():
    a = AMC_CNN(60, input_len=L, expert_features=True)
    assert a.physics_branch is None and a.physics_norm is None
    assert not any("physics" in k for k in a.state_dict())


@pytest.mark.skipif(not __import__("pathlib").Path("results/classifier/old/seed4_best.pt").exists(),
                    reason="seed4 checkpoint not on this machine")
def test_seed4_checkpoint_still_loads():
    from src.pipeline.classify import Classifier
    clf = Classifier(["results/classifier/old/seed4_best.pt"])
    p = clf(_windows(2)[:, 0] + 1j * _windows(2)[:, 1])
    assert p.shape[-1] == 60
