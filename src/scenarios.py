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
from src.data.composite import unit_power
from src.generators.fhss import random_fhss_example
from src.generators.jamming import random_jamming_example
from src.generators.radar import random_radar_example

GENERATORS = {
    "LFM_RADAR": random_radar_example,
    "FHSS": random_fhss_example,
    "JAMMING": random_jamming_example,
}

# Civilian classes have no generator -- BPSK/QPSK/16QAM/64QAM in this project
# are real RadioML captures, not synthesised. A scenario that wants civilian
# traffic therefore has to draw from the dataset, which is what `library`
# below is for.
CIVILIAN = ("BPSK", "QPSK", "16QAM", "64QAM")


@dataclass
class ScenarioSegment:
    """Ground truth for one emitter. TRUTH provenance -- the UI renders these
    in outline styling, never as if they were detections, and never at all for
    uploaded captures.

    Two different truths, because two different questions:

    start_s/end_s is when the emitter is SCHEDULED -- operationally "a radar is
    operating here", true even between pulses.

    radiating_spans is when it is actually PUT ENERGY IN THE AIR. A pulsed
    radar at max_duty_cycle 0.15 in burst mode can be scheduled across half the
    capture while transmitting in under 2% of it. Scoring a detector against
    the scheduled span then counts every silent gap as a miss, which measures
    the duty cycle rather than the model -- and does so identically at every
    SNR, since the pulse pattern is fixed for a seed.

    Use radiating_spans to evaluate detection. Use start_s/end_s to describe
    the scenario.
    """
    class_name: str
    start_s: float
    end_s: float
    radiating_spans: list = None

    @property
    def duty(self):
        """Fraction of the scheduled span in which the emitter transmits."""
        if not self.radiating_spans:
            return 1.0
        on = sum(b - a for a, b in self.radiating_spans)
        return on / max(self.end_s - self.start_s, 1e-12)


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

# Named cases, single emitter through fully contested band. Ordered so that
# stepping down the list adds one emitter at a time, which is how the
# behaviour is worth demonstrating: isolated emitters and overlapping ones
# fail in different ways, and only one of the two improves with SNR.
CASES = {
    "Radar only": [("LFM_RADAR", 0.25, 0.75)],
    "FHSS only": [("FHSS", 0.25, 0.75)],
    "Jamming only": [("JAMMING", 0.25, 0.75)],
    "Radar + FHSS": [("LFM_RADAR", 0.15, 0.70), ("FHSS", 0.40, 0.85)],
    "FHSS + Jamming": [("FHSS", 0.15, 0.70), ("JAMMING", 0.40, 0.85)],
    "All three": DEFAULT_SCRIPT,
    # Civilian cases need a library (see load_scenario) -- they draw real
    # RadioML captures from the dataset rather than a generator.
    "Civilian only": [("QPSK", 0.25, 0.75)],
    "Civilian + Jamming": [("QPSK", 0.15, 0.70), ("JAMMING", 0.40, 0.85)],
    "Civilian + Radar": [("BPSK", 0.15, 0.70), ("LFM_RADAR", 0.35, 0.85)],
    "Contested band": [("QPSK", 0.05, 0.60), ("LFM_RADAR", 0.20, 0.55),
                        ("FHSS", 0.35, 0.80), ("JAMMING", 0.55, 0.95)],
}


def _from_library(class_name, length, library, rng):
    """Assemble one emitter of `length` samples from real captured windows.

    Used for civilian classes, which have no generator. The dataset stores
    independent 512-sample captures, so a longer stretch has to be built by
    concatenating several -- and consecutive captures are unrelated, so each
    join is a phase discontinuity.

    Joins are crossfaded over a raised-cosine ramp to keep that discontinuity
    from radiating broadband splatter across the display. The signal is still
    a concatenation of separate recordings rather than one continuous
    transmission: honest for demonstrating civilian traffic in a scene, and
    NOT a basis for measuring civilian detection performance. Use the held-out
    test split for that.
    """
    pool = library.get(class_name)
    if pool is None or not len(pool):
        raise ValueError(
            f"{class_name} has no generator and no library entry -- civilian "
            f"classes must be supplied from the dataset")
    win = pool.shape[-1]
    fade = max(win // 16, 8)
    out = np.zeros(length + win, dtype=np.complex128)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
    pos = 0
    while pos < length:
        w = pool[rng.integers(len(pool))]
        seg = (w[0] + 1j * w[1]).astype(np.complex128)
        if pos:                       # crossfade into whatever is already there
            seg[:fade] *= ramp
            out[pos:pos + fade] *= ramp[::-1]
        out[pos:pos + win] += seg
        pos += win - fade
    return out[:length]


def _radiating_spans(emitter, offset, fs, rel_threshold=0.05, min_gap_s=2e-5):
    """Time spans where a CLEAN emitter is actually transmitting.

    Computed before noise is added, so this is genuine ground truth rather
    than an energy detector's opinion. The threshold is relative to the
    emitter's own peak, so it does not depend on absolute scaling.

    Gaps shorter than min_gap_s are bridged: a pulse's own envelope dips
    between cycles, and splitting on those would produce thousands of
    one-sample spans describing a single continuous transmission.
    """
    power = np.abs(emitter) ** 2
    if power.max() <= 0:
        return []
    on = power > rel_threshold * power.max()
    if not on.any():
        return []

    edges = np.diff(on.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        ends.append(len(on))

    spans, min_gap = [], int(min_gap_s * fs)
    for a, b in zip(starts, ends):
        if spans and a - spans[-1][1] <= min_gap:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return [((offset + a) / fs, (offset + b) / fs) for a, b in spans]


def build_scenario(fs=None, total_duration=0.1, snr_db=-6, seed=0,
                    script=None, library=None, library_snr_db=None):
    """Build one continuous capture with known ground truth.

    Returns (iq, segments). The capture is NOT normalized: absolute amplitude
    has to survive so the waterfall, the noise floor and the SNR readout mean
    something.

    library_snr_db is the SNR bin `library` was drawn from (civilian_library()
    in src/ui/session.py always draws from the cleanest available bin). A
    civilian recording already carries noise at that SNR, so noising it again
    on top of the scenario noise would double-count -- a scene labelled
    "+10 dB" would really be about +6.9 dB. When library_snr_db is None, or
    the script has no civilian emitter, behaviour is exactly what it was
    before this parameter existed: same values, same RNG draws.

    Noise is added once at the end, scaled against the mean of the PER-EMITTER
    powers -- not against the pooled power of every active sample.

    Pooling was a real measurement bug. Where two emitters overlap, the summed
    power is higher, so the pooled mean rose with the number of emitters and
    the noise scaled up with it. A two-emitter scenario at "-6 dB" was
    therefore markedly harder than a one-emitter scenario at "-6 dB", and any
    comparison across cases at a fixed nominal SNR was measuring the scenario
    builder rather than the model.

    Referencing the first non-jamming emitter keeps the victim at roughly the
    requested SNR regardless of how many others share the capture, which is
    what makes case-to-case comparison meaningful.

    Emitters are mixed at the same relative levels the training composites
    use -- unit power, SIR from dataset.mixture_sir_db, jammer last at JSR
    from jamming.jsr_db. Mixing any other way puts the overlap regions out of
    distribution, and measurements taken on them describe this function
    rather than the model.
    """
    fs = fs or CFG["signal"]["fs"]
    script = script if script is not None else DEFAULT_SCRIPT
    rng = np.random.default_rng(seed)

    n_total = int(round(total_duration * fs))
    iq = np.zeros(n_total, dtype=np.complex128)
    segments = []
    active = np.zeros(n_total, dtype=bool)
    emitter_powers = []
    civilian_spans = []

    for class_name, start_frac, end_frac in script:
        start = int(start_frac * n_total)
        end = min(int(end_frac * n_total), n_total)
        if end - start < 2:
            continue
        length = end - start

        if class_name in GENERATORS:
            emitter = np.asarray(GENERATORS[class_name](
                fs=fs, total_duration=length / fs, rng=rng))
        else:
            emitter = _from_library(class_name, length, library or {}, rng)
            civilian_spans.append((start, end))
        if len(emitter) < length:
            emitter = np.pad(emitter, (0, length - len(emitter)))
        emitter = emitter[:length] * raised_cosine_ramp(length)

        # Power-normalise, then place at the same relative level the training
        # composites use. Summing raw generator output instead put emitters at
        # whatever ratio the generators happened to produce -- and in
        # particular put JAMMING at a level the model never saw.
        #
        # mix_components (src/data/composite.py) normalises every component to
        # unit power, scales non-primary emitters by an SIR drawn from
        # dataset.mixture_sir_db, and applies a jammer LAST at a JSR from
        # jamming.jsr_db, i.e. 0-20 dB ABOVE the victim. A scenario that mixes
        # any other way is out of distribution, and measurements taken on it
        # describe the scenario builder rather than the model.
        emitter = unit_power(emitter)
        if class_name == "JAMMING":
            jsr = rng.uniform(*CFG["jamming"]["jsr_db"])
            emitter = emitter * (10 ** (jsr / 20.0))
        elif emitter_powers:
            sir = rng.uniform(*CFG["dataset"]["mixture_sir_db"])
            emitter = emitter * (10 ** (sir / 20.0))

        iq[start:end] += emitter
        active[start:end] = True
        emitter_powers.append((class_name, float(np.mean(np.abs(emitter) ** 2))))
        segments.append(ScenarioSegment(
            class_name, start / fs, end / fs,
            radiating_spans=_radiating_spans(emitter, start, fs)))

    # Noise references the FIRST NON-JAMMING emitter, exactly as
    # mix_components does ("the first non-JAMMING entry is the power
    # reference"). Averaging over all emitters instead would let a jammer --
    # deliberately 0-20 dB hot -- drag the reference up and raise the noise
    # floor, so adding a jammer would quietly make the victim harder to see.
    non_jam = [p for name, p in emitter_powers if name != "JAMMING"]
    reference_power = (non_jam[0] if non_jam
                        else (emitter_powers[0][1] if emitter_powers else 1.0))
    target = reference_power / (10 ** (snr_db / 10.0))

    # Two draws of exactly n_total values, in exactly this order, regardless
    # of what follows -- every synthetic (non-civilian) scenario in the
    # project depends on this exact RNG draw sequence for a given seed, and
    # drawing per-segment or drawing a different count would move all of
    # them. The per-sample scaling below is applied as a multiplication
    # AFTER both draws, never by changing what or how much is drawn.
    noise = rng.normal(0, 1, n_total) + 1j * rng.normal(0, 1, n_total)

    if library_snr_db is None or not civilian_spans:
        # No civilian emitter (or no library SNR given): behaviour must be
        # bit-identical to before this parameter existed.
        iq += noise * np.sqrt(target / 2.0)
    else:
        # The civilian recording already carries noise at library_snr_db, so
        # a target SNR better than that bin is not achievable -- you can add
        # noise to a recording but never remove it. The floor actually used
        # is therefore the noisier (lower-SNR, i.e. higher-power) of the two.
        carried = reference_power / (10 ** (library_snr_db / 10.0))
        floor = max(target, carried)

        # Everywhere gets `floor`, except inside a civilian span, which
        # already has `carried` baked into the recording and only needs
        # `floor - carried` added on top. Noise powers add, so the added
        # component's AMPLITUDE scale is sqrt(floor - carried), not
        # sqrt(floor) - sqrt(carried).
        added_power = np.full(n_total, floor, dtype=np.float64)
        topped_up = max(floor - carried, 0.0)
        for start, end in civilian_spans:
            added_power[start:end] = topped_up
        iq += noise * np.sqrt(added_power / 2.0)

    return iq, segments
