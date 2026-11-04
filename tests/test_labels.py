"""Checks for the week-1 label rules (src/data/labels.py): DRFM windows carry
the radar they copied, ambiguous groups are labelled and scored as one."""
import numpy as np
import pytest

from src.config import DATASET_CLASS_TO_IDX, DATASET_CLASSES
from src.data import impairments, labels
from src.generators import jam, radar, spread

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
    both = {"LORA_CSS", "JAM_SWEPT"}
    assert labels.window_labels("LORA_CSS", {"lora_like_sweep": 1.0}) == both
    assert labels.window_labels("LORA_CSS", {"lora_like_sweep": 0.0}) == {"LORA_CSS"}
    assert labels.window_labels("JAM_SWEPT", {"lora_like_sweep": 1.0}) == both
    assert labels.window_labels("JAM_SWEPT", {"lora_like_sweep": 0.0}) == {"JAM_SWEPT"}
    for cls in ("JAM_DRFM", "LORA_CSS", "JAM_SWEPT"):
        with pytest.raises(ValueError):
            labels.window_labels(cls)


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
                  onehot("JAM_DRFM", "RADAR_BARKER", "FHSS"),  # DRFM inside a mixture
                  onehot("LORA_CSS", "JAM_SWEPT")])        # straight sweep
    rows, cls = labels.standalone_pairs(y)
    got = sorted((int(r), DATASET_CLASSES[c]) for r, c in zip(rows, cls))
    assert got == [(0, "FHSS"), (1, "JAM_DRFM"), (2, "JAM_TONE"), (2, "RADAR_CW"),
                   (5, "JAM_SWEPT"), (5, "LORA_CSS")]


def test_scored_prediction_merges_only_the_windows_own_labels():
    y = np.stack([onehot("RADAR_CW", "JAM_TONE"), onehot("RADAR_CW", "JAM_TONE"), onehot("FHSS"),
                  onehot("LORA_CSS", "JAM_SWEPT"), onehot("LORA_CSS")])
    true = np.array([IDX[c] for c in ("RADAR_CW", "RADAR_CW", "FHSS", "LORA_CSS", "LORA_CSS")])
    top = np.array([IDX[c] for c in ("JAM_TONE", "DSSS", "JAM_TONE", "JAM_SWEPT", "JAM_SWEPT")])
    out = labels.scored_prediction(true, top, y)
    # A LoRa window that shows its wrap, called JAM_SWEPT, is still a mistake.
    assert [DATASET_CLASSES[i] for i in out] == ["RADAR_CW", "DSSS", "JAM_TONE", "LORA_CSS", "JAM_SWEPT"]


def test_generated_class():
    y = np.stack([onehot("FHSS"), onehot("JAM_DRFM", "RADAR_PULSE"), onehot("LORA_CSS", "JAM_SWEPT"),
                  onehot("FHSS", "DSSS")])
    assert [DATASET_CLASSES[i] if i >= 0 else None for i in labels.generated_class(y, "spread")] == \
        ["FHSS", "JAM_DRFM", "LORA_CSS", None]
    assert DATASET_CLASSES[labels.generated_class(y, "jam")[2]] == "JAM_SWEPT"


def _old_style_shard(impaired, module=jam, classes=("JAM_DRFM", "JAM_TONE", "JAM_SWEPT"), n=24, snr=10):
    """A shard as made before the rules: every row labelled with its class alone."""
    iq, y, seeds, made = [], [], [], []
    for i in range(n):
        cls = classes[i % len(classes)]
        s = 9000 + i
        rng = np.random.default_rng(s)
        x = (impairments.generate_impaired(module, cls, snr, rng)[0] if impaired
             else module.generate_window(cls, snr, rng)[0])
        iq.append(x)
        y.append(onehot(cls))
        seeds.append(s)
        made.append(cls)
    return {"iq": np.stack(iq), "y": np.stack(y), "seed": np.array(seeds),
            "snr_db": np.full(n, float(snr)), "source": np.array(["gen"] * n)}, made


@pytest.mark.parametrize("impaired", [False, True])
def test_fix_shard_labels_on_an_old_jam_shard(impaired):
    data, made = _old_style_shard(impaired)
    y, fixed = labels.fix_shard_labels(data, {"notes": {"impairments": impaired}})
    n_swept_both = sum(labels.class_names(y[r]) == {"JAM_SWEPT", "LORA_CSS"}
                       for r, c in enumerate(made) if c == "JAM_SWEPT")
    assert fixed == {"JAM_DRFM": 8, **({"JAM_SWEPT": n_swept_both} if n_swept_both else {})}
    for r, cls in enumerate(made):
        names = labels.class_names(y[r])
        if cls == "JAM_DRFM":
            assert names == {"JAM_DRFM", labels.drfm_copied_class(data["seed"][r])}
        elif cls == "JAM_TONE":
            assert names == {"JAM_TONE", "RADAR_CW"}
        else:
            _, p = jam.generate_window(cls, None, np.random.default_rng(int(data["seed"][r])))
            assert names == labels.window_labels(cls, p)
    # Already fixed rows are left alone on a second pass (sweeps that stayed
    # single are replayed again, harmlessly).
    data["y"] = y
    y2, again = labels.fix_shard_labels(data, {"notes": {"impairments": impaired}})
    assert "JAM_DRFM" not in again and np.array_equal(y2, y)


def test_fix_shard_labels_on_an_old_spread_shard():
    data, made = _old_style_shard(True, spread, ("LORA_CSS", "FHSS"), n=40)
    y, fixed = labels.fix_shard_labels(data, {"notes": {"impairments": True}})
    both = [labels.class_names(y[r]) == {"LORA_CSS", "JAM_SWEPT"} for r, c in enumerate(made) if c == "LORA_CSS"]
    assert 0 < sum(both) < len(both)
    assert fixed == {"LORA_CSS": sum(both)}


def test_fix_shard_labels_refuses_a_window_it_cannot_replay():
    data, _ = _old_style_shard(impaired=True)
    data["iq"][0] = data["iq"][0] * 1.01
    with pytest.raises(RuntimeError, match="regenerate"):
        labels.fix_shard_labels(data, {"notes": {"impairments": True}})
