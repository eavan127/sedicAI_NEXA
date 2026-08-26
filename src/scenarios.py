"""Synthesized multi-emitter captures, long enough to drive the replay deck.

Why generate rather than stitch stored dataset windows: every stored window
went through preprocess_window, which normalizes it to zero mean and unit
variance. Concatenating them yields a capture with no amplitude dynamics (so
no usable noise floor and no SNR), and a phase discontinuity at every splice
-- which at hop 256 would corrupt roughly half of all sliding windows.

The generators already accept total_duration. build_dataset asks for 0.002 s
(6,400 samples) and then keeps only the first 512, discarding 92% of it. Here
we simply ask for more and keep all of it.
"""
from dataclasses import dataclass

import numpy as np

from src.config import CFG
from src.generators.fhss import random_fhss_example
from src.generators.jamming import random_jamming_example
from src.generators.radar import random_radar_example

GENERATORS = {
    "LFM_RADAR": random_radar_example,
    "FHSS": random_fhss_example,
    "JAMMING": random_jamming_example,
}


@dataclass
class ScenarioSegment:
    """Ground truth for one emitter's active period. TRUTH provenance -- the
    UI renders these in outline styling, never as if they were detections,
    and never at all for uploaded captures."""
    class_name: str
    start_s: float
    end_s: float


def raised_cosine_ramp(n_samples, ramp_len=256):
    """Envelope that fades in and out with a raised-cosine shape.

    Switching an emitter on with a hard edge produces broadband splatter
    across the whole band -- which would show on the waterfall as a vertical
    stripe and could plausibly trigger a JAMMING detection that nothing in the
    scenario put there. Real transmitters ramp; so do these.
    """
    env = np.ones(n_samples)
    ramp_len = int(min(ramp_len, n_samples // 2))
    if ramp_len < 1:
        return env
    rise = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
    env[:ramp_len] = rise
    env[-ramp_len:] = rise[::-1]
    return env


# Fractions of total_duration. Reads as: quiet, radar sweeping, FHSS joins,
# jammer lands on top of both, then quiet again.
DEFAULT_SCRIPT = [
    ("LFM_RADAR", 0.10, 0.45),
    ("FHSS", 0.30, 0.70),
    ("JAMMING", 0.55, 0.85),
]


def build_scenario(fs=None, total_duration=0.1, snr_db=-6, seed=0,
                    script=None):
    """Build one continuous capture with known ground truth.

    Returns (iq, segments). The capture is NOT normalized: absolute amplitude
    has to survive so the waterfall, the noise floor and the SNR readout mean
    something.

    Noise is added once at the end, scaled against the mean of the PER-EMITTER
    powers -- not against the pooled power of every active sample.

    Pooling was a real measurement bug. Where two emitters overlap, the summed
    power is higher, so the pooled mean rose with the number of emitters and
    the noise scaled up with it. A two-emitter scenario at "-6 dB" was
    therefore markedly harder than a one-emitter scenario at "-6 dB", and any
    comparison across cases at a fixed nominal SNR was measuring the scenario
    builder rather than the model.

    Referencing the per-emitter mean keeps each emitter at roughly the
    requested SNR regardless of how many others share the capture, which is
    what makes case-to-case comparison meaningful.
    """
    fs = fs or CFG["signal"]["fs"]
    script = script if script is not None else DEFAULT_SCRIPT
    rng = np.random.default_rng(seed)

    n_total = int(round(total_duration * fs))
    iq = np.zeros(n_total, dtype=np.complex128)
    segments = []
    active = np.zeros(n_total, dtype=bool)
    emitter_powers = []

    for class_name, start_frac, end_frac in script:
        start = int(start_frac * n_total)
        end = min(int(end_frac * n_total), n_total)
        if end - start < 2:
            continue
        length = end - start

        emitter = np.asarray(GENERATORS[class_name](
            fs=fs, total_duration=length / fs, rng=rng))
        if len(emitter) < length:
            emitter = np.pad(emitter, (0, length - len(emitter)))
        emitter = emitter[:length] * raised_cosine_ramp(length)

        iq[start:end] += emitter
        active[start:end] = True
        emitter_powers.append(float(np.mean(np.abs(emitter) ** 2)))
        segments.append(ScenarioSegment(class_name, start / fs, end / fs))

    reference_power = float(np.mean(emitter_powers)) if emitter_powers else 1.0
    noise_power = reference_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0, 1, n_total) + 1j * rng.normal(0, 1, n_total)
    iq += noise * np.sqrt(noise_power / 2.0)

    return iq, segments
