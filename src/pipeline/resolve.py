"""
Settle the look-alike pairs from what a track does over time.

One 160 us window cannot tell these apart, so the classifier is trained to
name both (src/data/labels.py). A track is many windows long and carries the
behaviour features of src/pipeline/extract.py, which can:

  LORA_CSS vs JAM_SWEPT   occupied width and wraps. LoRa occupies at most
                          0.5 MHz and wraps at every symbol (<= 512 us); a
                          swept jammer spans 0.9-1.4 MHz, and a short burst
                          that shows no wrap for > 1.2 ms is too slow for LoRa.
                          Wraps only count at >= 6 dB in-band; weaker sweeps
                          stay undecided. On 400 random generator bursts
                          (-5..20 dB, 0.6-10 ms, 2026-11-05): 0 wrong, 27
                          (7%, all weak) undecided.
  pure tone vs modulated  sidebands in one long FFT. A pure tone's strongest
                          off-carrier line is noise (7-13.5 dB above the
                          floor); NBFM's is >= 16 dB. On 360 bursts: every
                          NBFM found, 2 of 240 pure tones called modulated.
  RADAR_CW vs JAM_TONE    no signal feature can (both are the same pure tone),
                          so context decides: a tone that appears on top of a
                          signal that was already there is a jammer; a tone
                          alone for a long time (cw_min_s) is a CW radar.
                          Otherwise the answer stays "RADAR_CW / JAM_TONE".
  JAM_DRFM + radar        both are true: a radar with a DRFM jammer copying it.

Each rule only fires when the classifier's own answer points at that pair, and
every answer carries a plain-language reason for the UI.
"""
from dataclasses import dataclass

import numpy as np

from src.config import DATASET_CLASS_TO_IDX, DATASET_CLASSES
from src.pipeline import extract
from src.pipeline.tracker import FS, freq_iou

TONES = ("RADAR_CW", "JAM_TONE")
SWEEPS = ("LORA_CSS", "JAM_SWEPT")
DRFM_RADARS = ("RADAR_PULSE", "LFM_RADAR", "RADAR_BARKER")
# Carriers that can look like a pure tone in one window when lightly modulated.
TONE_LIKE = ("NBFM", "PM", "WBFM", "AFSK", "AM_DSB_WC")

SWEPT_MIN_WIDTH_HZ = 700e3      # occupied band: LoRa <= ~0.48 MHz, a full jammer sweep >= ~0.86 MHz
LORA_MIN_WRAPS = 2
NO_WRAP_MIN_S = 1.2e-3          # > 2 of the longest LoRa symbols with no wrap: not LoRa
SIDEBAND_DB = 15.0              # strongest off-carrier line above the floor
LINE_DB_FOR_PURITY = 35.0       # the line must be strong enough that missing sidebands mean something
CW_MIN_S = 1.0                  # a lone tone this long is taken as a CW radar
CO_CHANNEL_LEAD_S = 1e-3        # the other signal must have been there this long before the tone


@dataclass
class Answer:
    label: str                  # final answer, or "A / B" when it cannot be decided
    classes: tuple              # the class(es) the answer stands for
    confidence: float           # track probability of the answer (best of a pair)
    reason: str


def _p(probs, c):
    return float(probs[DATASET_CLASS_TO_IDX[c]])


def _undecided(probs, pair, reason):
    return Answer(" / ".join(pair), tuple(pair), max(_p(probs, c) for c in pair), reason)


def _one(probs, c, reason):
    return Answer(c, (c,), _p(probs, c), reason)


def _tone(track, probs, feats, others, cw_min_s):
    start = track.t_first
    for o in others:
        if o is track or o.t_first > start - CO_CHANNEL_LEAD_S * FS or o.t_last < start:
            continue
        if freq_iou(track.f_lo, track.f_hi, o.f_lo, o.f_hi) > 0 or (o.f_lo <= track.f_lo and track.f_hi <= o.f_hi):
            return _one(probs, "JAM_TONE",
                        f"pure tone that appeared on top of track {o.id}, which was already transmitting")
    if feats["duration_s"] >= cw_min_s:
        return _one(probs, "RADAR_CW", f"pure tone, steady and alone for {feats['duration_s']:.1f} s")
    return _undecided(probs, TONES, f"pure tone for {feats['duration_s'] * 1e3:.1f} ms with no other signal "
                                    "under it: CW radar or tone jammer cannot be told apart yet")


def resolve(track, others=(), cw_min_s=CW_MIN_S):
    """Final Answer for one Track (others: every track of the capture)."""
    probs = track.probs()
    feats = track.features()
    order = np.argsort(-probs)
    top = DATASET_CLASSES[order[0]]
    second = DATASET_CLASSES[order[1]]

    if top == "JAM_DRFM" or (second == "JAM_DRFM" and top in DRFM_RADARS):
        radar = max(DRFM_RADARS, key=lambda c: _p(probs, c))
        return Answer("JAM_DRFM", ("JAM_DRFM", radar), _p(probs, "JAM_DRFM"),
                      f"DRFM jammer copying a {radar} (both are on the air)")

    if top in SWEEPS:
        w, wraps, dur = feats["width_hz"], feats["wraps"], feats["duration_s"]
        if w >= SWEPT_MIN_WIDTH_HZ:
            return _one(probs, "JAM_SWEPT", f"sweep {w / 1e6:.2f} MHz wide, wider than any LoRa signal")
        if wraps >= LORA_MIN_WRAPS:
            return _one(probs, "LORA_CSS", f"{wraps} chirp wraps in a {w / 1e3:.0f} kHz band: LoRa symbols")
        if feats["snr_db"] < extract.WRAP_MIN_SNR_DB:
            return _undecided(probs, SWEEPS, f"sweep {feats['snr_db']:.0f} dB above the noise: "
                                             "too weak to see whether it wraps")
        if dur >= NO_WRAP_MIN_S:
            return _one(probs, "JAM_SWEPT", f"no wrap in {dur * 1e3:.1f} ms: too slow for LoRa")
        return _undecided(probs, SWEEPS, f"straight sweep for only {dur * 1e3:.1f} ms")

    sb, line = feats["sideband_db"], feats["line_db"]
    if top in TONES:
        if sb >= SIDEBAND_DB:
            rest = [DATASET_CLASSES[i] for i in order if DATASET_CLASSES[i] not in TONES]
            return _one(probs, rest[0], f"carrier with sidebands {sb:.0f} dB above the noise: "
                                        f"modulated, not a pure tone")
        return _tone(track, probs, feats, others, cw_min_s)

    if top in TONE_LIKE and sb < SIDEBAND_DB and line >= LINE_DB_FOR_PURITY:
        return _tone(track, probs, feats, others, cw_min_s)

    return _one(probs, top, "classifier answer, averaged over the track")
