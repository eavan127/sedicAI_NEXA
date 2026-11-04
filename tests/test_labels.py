"""Checks for the week-1 label rules (src/data/labels.py): DRFM windows carry
the radar they copied, ambiguous groups are labelled and scored as one."""
import numpy as np
import pytest

from src.config import DATASET_CLASS_TO_IDX, DATASET_CLASSES
from src.data import impairments, labels
from src.generators import jam, radar

C = len(DATASET_CLASSES)
IDX = DATASET_CLASS_TO_IDX


def onehot(*names):
    y = np.zeros(C, dtype=np.uint8)
    for n in names:
        y[IDX[n]] = 1
    return y


def test_config_groups_cw_with_tone():
    assert ("RADAR_CW", "JAM_TONE") in labels.AMBIGUOUS_GROUPS


def test_drfm_reports_a_pulsed_radar_it_copied():
    seen = set()
    for seed in range(40):
        _, p = jam.generate_window("JAM_DRFM", None, np.random.default_rng(seed))
        seen.add(radar.CLASSES[int(p["copied_radar"])])
    assert seen == set(jam.DRFM_RADAR_CLASSES)


def test_other_jammers_have_no_copied_radar():
    _, p = jam.generate_window("JAM_TONE", None, np.random.default_rng(0))
    assert np.isnan(p["copied_radar"])


def test_window_labels():
    _, p = jam.generate_window("JAM_DRFM", None, np.random.default_rng(3))
    copied = radar.CLASSES[int(p["copied_radar"])]
    assert labels.window_labels("JAM_DRFM", p) == {"JAM_DRFM", copied}
    assert labels.window_labels("FHSS") == {"FHSS"}
    assert labels.window_labels("JAM_TONE") == {"JAM_TONE", "RADAR_CW"}
    assert labels.window_labels("JAM_TONE", ambiguous=False) == {"JAM_TONE"}
    with pytest.raises(ValueError):
        labels.window_labels("JAM_DRFM")


@pytest.mark.parametrize("impaired", [False, True])
def test_copied_class_is_recovered_from_the_seed(impaired):
    for seed in range(15):
        rng = np.random.default_rng(seed)
        if impaired:
            _, info = impairments.generate_impaired(jam, "JAM_DRFM", 0, rng)
            copied = info["gen_copied_radar"]
        else:
            _, p = jam.generate_window("JAM_DRFM", 0, rng)
            copied = p["copied_radar"]
        assert labels.drfm_copied_class(seed) == radar.CLASSES[int(copied)]


def test_apply_ambiguous_fills_the_group_only():
    y = np.stack([onehot("RADAR_CW"), onehot("JAM_TONE"), onehot("FHSS"), onehot("FHSS", "JAM_TONE")])
    out = labels.apply_ambiguous(y)
    assert labels.class_names(out[0]) == {"RADAR_CW", "JAM_TONE"}
    assert labels.class_names(out[1]) == {"RADAR_CW", "JAM_TONE"}
    assert labels.class_names(out[2]) == {"FHSS"}
    assert labels.class_names(out[3]) == {"FHSS", "RADAR_CW", "JAM_TONE"}
    assert y[0, IDX["JAM_TONE"]] == 0                      # input untouched


def test_standalone_pairs():
    y = np.stack([onehot("FHSS"),                          # single
                  onehot("JAM_DRFM", "RADAR_BARKER"),      # DRFM pair
                  onehot("RADAR_CW", "JAM_TONE"),          # ambiguous group
                  onehot("FHSS", "DSSS"),                  # mixture
                  onehot("JAM_DRFM", "RADAR_BARKER", "FHSS")])  # DRFM inside a mixture
    rows, cls = labels.standalone_pairs(y)
    got = sorted((int(r), DATASET_CLASSES[c]) for r, c in zip(rows, cls))
    assert got == [(0, "FHSS"), (1, "JAM_DRFM"), (2, "JAM_TONE"), (2, "RADAR_CW")]


def test_scored_prediction_merges_the_group():
    true = np.array([IDX["RADAR_CW"], IDX["RADAR_CW"], IDX["FHSS"]])
    top = np.array([IDX["JAM_TONE"], IDX["DSSS"], IDX["JAM_TONE"]])
    out = labels.scored_prediction(true, top)
    assert out.tolist() == [IDX["RADAR_CW"], IDX["DSSS"], IDX["JAM_TONE"]]


def _old_style_shard(impaired, n=6, snr=10):
    """A jam shard as made before the fix: JAM_DRFM rows labelled alone."""
    iq, y, seeds = [], [], []
    for i in range(n):
        cls = "JAM_DRFM" if i % 2 == 0 else "JAM_TONE"
        s = 9000 + i
        rng = np.random.default_rng(s)
        x = (impairments.generate_impaired(jam, cls, snr, rng)[0] if impaired
             else jam.generate_window(cls, snr, rng)[0])
        iq.append(x)
        y.append(onehot(cls))
        seeds.append(s)
    return {"iq": np.stack(iq), "y": np.stack(y), "seed": np.array(seeds),
            "snr_db": np.full(n, float(snr)), "source": np.array(["gen"] * n)}


@pytest.mark.parametrize("impaired", [False, True])
def test_fix_shard_labels_adds_the_copied_radar(impaired):
    data = _old_style_shard(impaired)
    y, n_fixed = labels.fix_shard_labels(data, {"notes": {"impairments": impaired}})
    assert n_fixed == 3
    for r in range(len(y)):
        names = labels.class_names(y[r])
        if r % 2 == 0:
            assert names == {"JAM_DRFM", labels.drfm_copied_class(data["seed"][r])}
        else:
            assert names == {"JAM_TONE", "RADAR_CW"}
    # Already fixed rows are left alone on a second pass.
    data["y"] = y
    assert labels.fix_shard_labels(data)[1] == 0


def test_fix_shard_labels_refuses_a_window_it_cannot_replay():
    data = _old_style_shard(impaired=True)
    data["iq"][0] = data["iq"][0] * 1.01
    with pytest.raises(RuntimeError, match="regenerate"):
        labels.fix_shard_labels(data, {"notes": {"impairments": True}})
