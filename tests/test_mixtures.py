"""Multi-emitter mixtures: label correctness and power invariants.

Companion to the overlay_jamming tests in test_generators.py. Avoids RadioML
(21 GB, gitignored, absent on CI) by exercising the synthetic classes only --
the civilian path is a pool lookup, whereas the parts that can silently
produce WRONG LABELS are all on the mixing side.
"""
import numpy as np
import pytest

from src.config import CFG, CLASSES
from src.data.build_dataset import build_mixture_examples
from src.data.composite import active_power, mix_components, unit_power
from src.generators.fhss import random_fhss_example
from src.generators.jamming import random_jamming_example
from src.generators.radar import random_radar_example


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_unit_power_normalises_on_active_samples(rng):
    """A low-duty radar and a continuous carrier must come out at the same
    power -- otherwise a 0 dB SIR silently favours the continuous one."""
    radar = unit_power(random_radar_example(rng=rng))
    fhss = unit_power(random_fhss_example(rng=rng))
    assert active_power(radar) == pytest.approx(1.0, rel=1e-6)
    assert active_power(fhss) == pytest.approx(1.0, rel=1e-6)


def test_mix_labels_every_component(rng):
    for combo in (["LFM_RADAR", "FHSS"],
                  ["LFM_RADAR", "FHSS", "JAMMING"],
                  ["FHSS", "JAMMING"]):
        gens = {"LFM_RADAR": random_radar_example, "FHSS": random_fhss_example,
                "JAMMING": random_jamming_example}
        components = [(c, gens[c](rng=rng)) for c in combo]
        mixed, class_set = mix_components(components, rng=rng)
        assert class_set == set(combo)
        assert len(mixed) > 0
        assert np.isfinite(mixed).all()


def test_mixture_keeps_both_emitters_present(rng):
    """Removing either component must change the window materially -- a
    mixture where one side is inaudible is a mislabelled single."""
    components = [("LFM_RADAR", random_radar_example(rng=rng)),
                  ("FHSS", random_fhss_example(rng=rng))]
    mixed, _ = mix_components(components, rng=rng)
    total = np.mean(np.abs(mixed) ** 2)
    for _, iq in components:
        assert np.mean(np.abs(mixed - unit_power(iq[:len(mixed)])) ** 2) > 0.05 * total


def test_jamming_only_mixture_rejected(rng):
    with pytest.raises(ValueError, match="non-JAMMING"):
        mix_components([("JAMMING", random_jamming_example(rng=rng))], rng=rng)


def test_noise_floor_mixture_rejected(monkeypatch):
    ds = dict(CFG["dataset"]); ds["mixture_combos"] = [["NOISE_FLOOR", "FHSS"]]
    monkeypatch.setitem(CFG, "dataset", ds)
    with pytest.raises(ValueError, match="NOISE_FLOOR"):
        list(build_mixture_examples({}))


def test_unknown_class_rejected(monkeypatch):
    ds = dict(CFG["dataset"]); ds["mixture_combos"] = [["FHSS", "NOT_A_CLASS"]]
    monkeypatch.setitem(CFG, "dataset", ds)
    with pytest.raises(ValueError, match="unknown"):
        list(build_mixture_examples({}))


def test_civilian_combos_skipped_without_radioml(monkeypatch, capsys):
    """No RadioML must mean 'skip those combos', not a crash -- teammates
    without the 21 GB file still need the rest of the dataset to build."""
    ds = dict(CFG["dataset"])
    ds["mixture_combos"] = [["BPSK", "FHSS"], ["LFM_RADAR", "FHSS"]]
    monkeypatch.setitem(CFG, "dataset", ds)
    out = list(build_mixture_examples({}))
    assert "skipping BPSK+FHSS" in capsys.readouterr().out
    assert out and all(cs == {"LFM_RADAR", "FHSS"} for _, cs, _ in out)


def test_mixture_examples_span_every_snr_bin(monkeypatch):
    ds = dict(CFG["dataset"])
    ds["mixture_combos"] = [["LFM_RADAR", "FHSS"]]
    monkeypatch.setitem(CFG, "dataset", ds)
    snrs = {snr for _, _, snr in build_mixture_examples({})}
    assert snrs == set(CFG["snr_bins_db"])
