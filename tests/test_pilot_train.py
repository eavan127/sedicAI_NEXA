"""
Tests for the pre-merge pilot script: the batch normalisation must match the
real pipeline, and the scorecard must count what it says it counts.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import DATASET_CLASSES
from src.data.preprocess import preprocess_window

path = Path(__file__).resolve().parents[1] / "scripts" / "pilot_train_shards.py"
spec = importlib.util.spec_from_file_location("pilot_train_shards", path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_normalise_matches_preprocess_window():
    rng = np.random.default_rng(0)
    iq = (rng.normal(size=(3, 512)) + 1j * rng.normal(size=(3, 512))).astype(np.complex64)
    batch = pilot.normalise(iq)
    for window, expected in zip(iq, batch):
        np.testing.assert_allclose(preprocess_window(window), expected, atol=1e-5)


def test_present_classes_finds_only_used_columns():
    y = np.zeros((2, len(DATASET_CLASSES)), np.uint8)
    y[0, 3] = y[1, 7] = 1
    cols, names = pilot.present_classes(y)
    assert list(cols) == [3, 7] and names == [DATASET_CLASSES[3], DATASET_CLASSES[7]]


def test_scorecard_counts_recall_false_alarm_and_confusion():
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], np.uint8)
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.8]])
    snr = np.array([0, 10, 0, 10], np.float32)
    card = pilot.scorecard(probs, y, snr, ["A", "B"])
    a = card["per_class"]["A"]
    assert a["recall"] == 0.5 and a["false_alarm"] == 0.5 and a["top1_recall"] == 0.5
    assert a["recall_by_snr"] == {"0": 1.0, "10": 0.0}
    assert a["confused_with"] == {"B": 1}
    assert card["macro_recall"] == pytest.approx(0.75)


def test_missing_shards_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        pilot.load_split(tmp_path, ["analog"], "train")


def test_wide_branch_keeps_length_and_sees_the_whole_window():
    branch = pilot.WideIQBranch().eval()
    x = torch.zeros(1, 2, 512, requires_grad=True)
    out = branch(x)
    assert out.shape == (1, 128, 512)
    out[0, :, 256].sum().backward()
    seen = torch.nonzero(x.grad.abs().sum(dim=(0, 1))).squeeze()
    assert seen.max() - seen.min() > 450


def test_fraction_keeps_the_same_share_of_each_shard(tmp_path):
    (tmp_path / "analog").mkdir()
    for k in range(2):
        y = np.zeros((100, len(DATASET_CLASSES)), np.uint8)
        y[:, k] = 1
        iq = (np.ones((100, 512)) + np.arange(100).reshape(-1, 1) * 1j).astype(np.complex64)
        np.savez(tmp_path / "analog" / f"analog_train_seed{k}.npz", iq=iq, y=y,
                 snr_db=np.zeros(100, np.float32), source=np.array(["t"] * 100), seed=np.arange(100))
    run = lambda: pilot.load_split(tmp_path, ["analog"], "train", manifest_dir=tmp_path, fraction=0.25)
    X, y, _ = run()
    assert len(X) == 50 and list(y.sum(axis=0)[:2]) == [25, 25]
    np.testing.assert_array_equal(X, run()[0])
