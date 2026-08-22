"""
Composite ("jammer overlaid on a real signal") example generation.

Real jamming targets something -- standalone JAMMING examples alone never
teach the model that shape. These pair a jammer with a legitimate "victim"
signal so the model sees jamming-on-top-of-traffic during training, not only
jamming in isolation. Reuses apply_jamming() (src/generators/jamming.py),
already tested for JSR accuracy in
tests/test_generators.py::TestJamming::test_applied_jsr_matches_request.

Operates on CLEAN (or already-appropriately-noisy, e.g. RadioML) victim
signals and a clean jammer -- the caller decides whether/when to add_awgn,
same division of responsibility build_dataset.py already uses for every
other source.
"""
import numpy as np

from src.config import CFG
from src.generators.jamming import apply_jamming, random_jamming_example

# NOISE_FLOOR is never a victim (nothing to jam), and JAMMING is never its
# own victim (that's just standalone jamming, already covered elsewhere).
VICTIM_CLASSES = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS"]


def overlay_jamming(victim_iq, victim_class, rng=None, fs=None):
    """Overlay a randomly-chosen jammer onto `victim_iq` at a randomly drawn
    JSR (jamming.jsr_db config range -- the same range standalone jamming
    examples use, so overlay strength isn't a new, untested parameter).

    Returns (jammed_iq, class_set) where class_set = {victim_class, "JAMMING"}.
    """
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    if victim_class not in VICTIM_CLASSES:
        raise ValueError(f"{victim_class!r} is not a valid overlay victim "
                          f"(expected one of {VICTIM_CLASSES})")

    n_samples = len(victim_iq)
    # Small margin so rounding in random_jamming_example's duration->sample
    # conversion can never leave the jammer shorter than the victim;
    # apply_jamming truncates jammer[:len(signal)] itself, so a longer
    # jammer is harmless.
    margin = 8
    jammer = random_jamming_example(fs=fs, total_duration=(n_samples + margin) / fs, rng=rng)

    jsr_db = rng.uniform(*CFG["jamming"]["jsr_db"])
    jammed = apply_jamming(victim_iq, jammer, jsr_db)
    return jammed, {victim_class, "JAMMING"}


# ---------------------------------------------------------------------------
# Multi-emitter mixtures (extends overlay_jamming above to combinations that
# are not "something + JAMMING").
#
# overlay_jamming covers the hostile case: a jammer on top of a victim. It does
# not cover two LEGITIMATE emitters sharing a window, which is the ordinary
# state of a contested band:
#
#     military x military   LFM_RADAR + FHSS
#     military x civilian   a tactical emitter alongside civilian traffic
#     three-way             either of the above, jammed
#
# Same label representation, same apply_jamming, one extra rule: components are
# summed at a random SIR so the model has to find the weaker emitter instead of
# only ever seeing equal-power blends.
# ---------------------------------------------------------------------------

def active_power(x):
    """Mean power over the samples actually carrying signal.

    Same "active" convention as preprocess.add_awgn, and for the same reason:
    a pulsed radar window is mostly silence, so a whole-window mean understates
    its power several-fold. Normalising two components by their whole-window
    means would put a low-duty radar far below a continuous carrier despite an
    SIR of 0 dB -- the SIR would then be wrong for exactly the class whose
    detection we are judged on.
    """
    mag_sq = np.abs(x) ** 2
    peak = mag_sq.max()
    if peak == 0:
        return 0.0
    active = mag_sq > 0.01 * peak      # -20 dB below peak counts as "on"
    return float(mag_sq[active].mean())


def unit_power(x):
    p = active_power(x)
    return x if p == 0 else x / np.sqrt(p)


def mix_components(components, rng=None):
    """Sum several CLEAN emitters into one window.

    `components` is [(class_name, clean_iq), ...]. The first non-JAMMING entry
    is the power reference; every other one is scaled by an SIR drawn from
    dataset.mixture_sir_db. A JAMMING component is applied last, at a JSR
    against the composite of everything else -- jammer-to-SIGNAL ratio is
    defined against the victim, not against one of several co-present emitters.

    Returns (mixed_iq, class_set). The caller adds noise, same division of
    responsibility as overlay_jamming and the rest of build_dataset.
    """
    rng = rng or np.random.default_rng()
    sir_lo, sir_hi = CFG["dataset"]["mixture_sir_db"]

    non_jam = [(name, iq) for name, iq in components if name != "JAMMING"]
    if not non_jam:
        raise ValueError("a mixture needs at least one non-JAMMING component; "
                         "standalone jamming is already covered elsewhere")

    n = min(len(iq) for _, iq in components)
    mixed = unit_power(non_jam[0][1][:n])
    for _, iq in non_jam[1:]:
        mixed = mixed + unit_power(iq[:n]) * 10 ** (rng.uniform(sir_lo, sir_hi) / 20)

    if any(name == "JAMMING" for name, _ in components):
        jammer = next(iq for name, iq in components if name == "JAMMING")
        mixed = apply_jamming(mixed, unit_power(jammer[:n]),
                              rng.uniform(*CFG["jamming"]["jsr_db"]))

    return mixed, {name for name, _ in components}
