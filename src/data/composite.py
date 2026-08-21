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
