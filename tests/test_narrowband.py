"""src/pipeline/narrowband.py: the slow-path model names slow narrowband
signals the 0.16 ms classifier cannot see (RTTY, PSK31, AFSK, SSB, CW, empty
band), and refine() only replaces the zoom rules where it is the better judge."""
import numpy as np
import pytest

from src.pipeline import analog, narrowband as nb

pytestmark = pytest.mark.skipif(not nb.MODEL_PATH.exists(), reason="slow-path model not trained")


@pytest.mark.parametrize("kind,need", [("NOISE", 4), ("CW", 4), ("RTTY", 4), ("PSK31", 4), ("AFSK", 4),
                                       # voice: the zoom rules keep the final say (refine), the model is a helper
                                       ("USB", 3), ("LSB", 3), ("NBFM", 3)])
def test_names_clear_signals(kind, need):
    rng = np.random.default_rng(2024)
    right = 0
    for _ in range(5):
        y, _ = nb.make(kind, rng, snr_db=20)
        right += nb.classify(y)[0] == kind
    assert right >= need, f"{kind}: {right}/5"


def test_features_are_finite_and_named():
    y, _ = nb.make("RTTY", np.random.default_rng(1), snr_db=10)
    f = nb.features(y)
    assert len(f) == len(nb.FEATURES) and np.all(np.isfinite(f))


def test_refine_names_what_rules_cannot():
    rng = np.random.default_rng(5)
    y, _ = nb.make("RTTY", rng, snr_db=20)
    kind, why, demod_kind, info = nb.refine("CW", "rules said CW", y)
    assert kind == "RTTY" and demod_kind == "USB" and "slow-path model" in why and info["kind"] == "RTTY"


def test_refine_leaves_am_to_the_rules():
    rng = np.random.default_rng(9)
    y, _ = nb.make("AM", rng, snr_db=25)
    rules = analog.classify(y, nb.FA)[0]
    kind = nb.refine(rules, "rules", y)[0]
    assert kind == rules or kind not in ("AM", "CARRIER", "NOISE")    # never swaps AM <-> CARRIER


def test_refine_without_model_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.setattr(nb, "MODEL_PATH", tmp_path / "missing.joblib")
    y, _ = nb.make("RTTY", np.random.default_rng(3), snr_db=20)
    assert nb.refine("CW", "why", y) == ("CW", "why", "CW", None)
