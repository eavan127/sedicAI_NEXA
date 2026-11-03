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


def test_noise_floor_nan_snr_gives_valid_sampling_weights():
    y = np.array([[1, 0], [0, 1], [0, 1]], np.uint8)
    snr = np.array([np.nan, -10, 10], np.float32)
    weights = pilot.sampling_weights(snr, y, ["NOISE_FLOOR", "BPSK"]).numpy()
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert weights[0] == 1 and weights[1] > weights[2]


def test_calibrate_picks_the_best_f1_threshold_per_class():
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], np.uint8)
    # A sits at 0.3 for its own windows and 0.1 elsewhere; B is never found.
    probs = np.array([[0.3, 0.0], [0.3, 0.0], [0.1, 0.0], [0.1, 0.0]])
    thresholds = pilot.calibrate(probs, y)
    assert 0.09 < thresholds[0] < 0.3
    assert thresholds[1] == pilot.THRESHOLD


def test_scorecard_reports_calibrated_recall():
    y = np.array([[1, 0], [0, 1]], np.uint8)
    probs = np.array([[0.3, 0.1], [0.1, 0.9]])
    card = pilot.scorecard(probs, y, np.zeros(2, np.float32), ["A", "B"], np.array([0.2, 0.5]))
    assert card["per_class"]["A"]["recall"] == 0
    assert card["per_class"]["A"]["calibrated"]["recall"] == 1
    assert card["calibrated_macro_recall"] == 1 and card["calibrated_macro_false_alarm"] == 0


def test_polar_features_give_amplitude_and_frequency():
    n = torch.arange(512, dtype=torch.float32)
    tone = 2 * torch.exp(1j * 2 * torch.pi * 0.1 * n)        # 0.1 fs = 0.2 of fs/2
    feats = pilot.polar_features(torch.stack([tone.real, tone.imag]).unsqueeze(0))
    assert feats.shape == (1, 2, 512)
    torch.testing.assert_close(feats[0, 0], torch.full((512,), 2.0))
    torch.testing.assert_close(feats[0, 1, 1:], torch.full((511,), 0.2), atol=1e-4, rtol=0)


@pytest.mark.parametrize("options", [{"features": True}, {"stft_nfft": 64},
                                     {"features": True, "stft_nfft": 64}])
def test_build_model_options_run_end_to_end(options):
    model = pilot.build_model(5, 512, **options).eval()
    assert model(torch.randn(3, 2, 512)).shape == (3, 5)
    if "stft_nfft" in options:
        assert model.stft_branch.n_fft == 64


def test_build_model_rejects_two_iq_branches():
    with pytest.raises(ValueError):
        pilot.build_model(5, 512, wide_iq=True, features=True)


def test_scorecard_skips_nan_snr():
    y = np.array([[1, 0], [0, 1]], np.uint8)
    card = pilot.scorecard(np.array([[0.9, 0.1], [0.1, 0.9]]), y,
                           np.array([np.nan, 0], np.float32), ["NOISE_FLOOR", "BPSK"])
    assert card["per_class"]["NOISE_FLOOR"]["recall_by_snr"] == {}
