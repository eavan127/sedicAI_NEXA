"""Inference must follow the MODEL's device, not a module-level constant.

Regression guard for a bug that was invisible on every CPU-only machine and
only appeared on Colab's GPU: classify_capture and predict_probs take `model`
as a parameter but moved the INPUT to their own module-level DEVICE, which
silently assumed the caller had already moved the model there. src/train.py,
src/evaluate.py and src/ui/app_models.py all do (they construct with
.to(DEVICE)), so production was fine -- but a caller-constructed model, which
is exactly what the test fixtures build, sat on the CPU while its input went
to cuda:

    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
    (torch.FloatTensor) should be the same

On a CPU-only box DEVICE resolves to cpu and the mismatch cannot occur, so
33 tests passed locally and failed on GPU. These tests reproduce the split
WITHOUT needing a GPU by pointing the module DEVICE at 'meta' -- a real
device whose tensors carry no data, so any code still routing input through
it fails loudly instead of silently working.
"""
import numpy as np
import pytest
import torch

import src.breakdown as breakdown
import src.timeline as timeline
from src.config import CLASSES
from src.models.amc_cnn import AMC_CNN


@pytest.fixture
def cpu_model():
    """Built WITHOUT .to(DEVICE), the way a caller (and every fixture in this
    suite) constructs one."""
    return AMC_CNN(num_classes=len(CLASSES), input_len=512).eval()


@pytest.fixture
def device_mismatch(monkeypatch):
    """Point both module DEVICEs somewhere the model is not."""
    monkeypatch.setattr(timeline, "DEVICE", torch.device("meta"))
    monkeypatch.setattr(breakdown, "DEVICE", torch.device("meta"))


def test_classify_capture_uses_the_model_device(cpu_model, device_mismatch):
    iq = np.random.randn(2048) + 1j * np.random.randn(2048)
    result = timeline.classify_capture(iq, cpu_model, hop=512)
    assert result.probs.shape[1] == len(CLASSES)
    assert np.isfinite(result.probs).all()


def test_predict_probs_uses_the_model_device(cpu_model, device_mismatch):
    X = np.random.randn(8, 2, 512).astype(np.float32)
    probs = breakdown.predict_probs(cpu_model, X)
    assert probs.shape == (8, len(CLASSES))
    assert np.isfinite(probs).all()


def test_single_vs_multi_uses_the_model_device(cpu_model, device_mismatch):
    """The public entry point the Performance page calls."""
    n = 12
    X = np.random.randn(n, 2, 512).astype(np.float32)
    y = np.zeros((n, len(CLASSES)), dtype=np.float32)
    y[:, CLASSES.index("LFM_RADAR")] = 1.0
    snr = np.full(n, -6.0)
    r = breakdown.single_vs_multi(cpu_model, X, y, snr)
    assert "single" in r.recall and "multi" in r.recall


def test_inference_still_works_when_model_and_device_agree(cpu_model):
    """The ordinary path must be unaffected by the fix."""
    iq = np.random.randn(1024) + 1j * np.random.randn(1024)
    result = timeline.classify_capture(iq, cpu_model, hop=512)
    assert result.n_windows == 2
