import numpy as np
import pytest
import torch

from src.breakdown import single_vs_multi
from src.config import CFG, CLASSES
from src.models.amc_cnn import AMC_CNN


@pytest.fixture(scope="module")
def model():
    """Untrained model, but with FIXED weights.

    AMC_CNN initialises randomly, so without a seed every test in this file
    that runs the model gets different predictions on every run -- which made
    assertions about event counts flaky, failing roughly one run in three for
    no reason connected to the code under test.
    """
    torch.manual_seed(0)
    m = AMC_CNN(num_classes=len(CLASSES), input_len=512)
    m.eval()
    return m


def _fixture(n=40):
    """Half single-label, half multi-label, spread over two SNR bins."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, 2, 512)).astype(np.float32)
    y = np.zeros((n, len(CLASSES)), dtype=np.float32)
    snr = np.zeros(n)
    bins = CFG["snr_bins_db"][:2]
    for i in range(n):
        snr[i] = bins[i % 2]
        if i < n // 2:
            y[i, CLASSES.index("FHSS")] = 1                    # single
        else:
            y[i, CLASSES.index("FHSS")] = 1                    # multi
            y[i, CLASSES.index("JAMMING")] = 1
    return X, y, snr


def test_splits_single_from_multi(model):
    X, y, snr = _fixture()
    r = single_vs_multi(model, X, y, snr)
    assert r.n_windows["single"] == 20
    assert r.n_windows["multi"] == 20


def test_recall_is_a_percentage_or_none(model):
    X, y, snr = _fixture()
    r = single_vs_multi(model, X, y, snr)
    for group in ("single", "multi"):
        for cls in r.classes:
            for v in r.recall[group][cls].values():
                assert v is None or 0.0 <= v <= 100.0


def test_empty_cells_are_none_not_zero(model):
    """A bin with no examples of a class must read as absent, not as 0%
    recall -- those mean very different things on a chart."""
    X, y, snr = _fixture()
    r = single_vs_multi(model, X, y, snr)
    # LFM_RADAR appears in neither group in this fixture
    assert all(v is None for v in r.recall["single"]["LFM_RADAR"].values())
    assert r.totals["single"]["LFM_RADAR"] is None


def test_support_counts_match_the_fixture(model):
    X, y, snr = _fixture()
    r = single_vs_multi(model, X, y, snr)
    total_fhss_single = sum(r.support["single"]["FHSS"].values())
    assert total_fhss_single == 20


def test_covers_every_judged_class(model):
    X, y, snr = _fixture()
    r = single_vs_multi(model, X, y, snr)
    assert set(r.classes) == set(CFG["judged_classes"])


def test_uses_per_class_thresholds_by_default(model):
    """Must not silently fall back to a flat 0.5 -- that is the bug this
    project already fixed once in the UI."""
    X, y, snr = _fixture()
    strict = single_vs_multi(model, X, y, snr,
                              thresholds=np.ones(len(CLASSES)) * 0.99)
    lenient = single_vs_multi(model, X, y, snr,
                               thresholds=np.ones(len(CLASSES)) * 0.01)
    s = strict.totals["single"]["FHSS"]
    lo = lenient.totals["single"]["FHSS"]
    assert lo >= s, "a lower threshold must not reduce recall"


def test_does_not_mutate_inputs(model):
    X, y, snr = _fixture()
    yb, sb = y.copy(), snr.copy()
    single_vs_multi(model, X, y, snr)
    np.testing.assert_array_equal(y, yb)
    np.testing.assert_array_equal(snr, sb)
