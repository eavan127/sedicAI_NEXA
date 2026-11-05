"""
Synthetic captures longer than one detector chunk, with known emitters: the
test bench for the tracker (scripts/eval_pipeline.py, tests/test_rf_pipeline.py).

Built like src/detect/scenes.py (same generators, placement, SNR convention
and receiver impairments) but n_chunks x 10.24 ms long, so an emitter can
cross chunk boundaries and the tracker has to join its boxes.

  make_capture(rng, n_chunks=4, classes=("LORA_CSS", "JAM_SWEPT"))
  -> iq complex64, truth: list of {cls, group, t_start, t_end, f_lo, f_hi, snr_db}

Limitation: emitters start at random times and frequencies. Behaviour such as
"a tone jammer switches on over a signal it targets" is not simulated, so the
RADAR_CW / JAM_TONE context rule mostly answers "cannot tell" here.
"""
import numpy as np

from src.config import CFG
from src.data import impairments
from src.detect import scenes

FS = CFG["signal"]["fs"]
CLASS_GROUP = {c: g for g, (_, cs) in scenes.GROUPS.items() for c in cs}


def make_capture(rng, n_chunks=4, n_emitters=None, snr_db=None, classes=None, impair=True,
                 min_len=scenes.MIN_BURST):
    n = n_chunks * scenes.SCENE_SAMPLES
    pool = list(classes) if classes else list(CLASS_GROUP)
    unknown = set(pool) - set(CLASS_GROUP)
    if unknown:
        raise ValueError(f"no burst generator for {sorted(unknown)}")
    if n_emitters is None:
        n_emitters = int(rng.integers(1, scenes.MAX_EMITTERS + 1))
    iq = np.zeros(n, dtype=complex)
    truth = []
    for _ in range(n_emitters):
        cls = str(rng.choice(pool))
        group = CLASS_GROUP[cls]
        length = int(rng.integers(min_len, n + 1))
        t0 = int(rng.integers(0, n - length + 1))
        sig, _ = scenes._burst(group, cls, length, rng)
        lo, hi = scenes.band_edges(sig)
        room = scenes.BAND_FRACTION * FS / 2 - max(abs(lo), abs(hi))
        offset = rng.uniform(-room, room) if room > 0 else 0.0
        sig = impairments.frequency_shift(sig, offset)
        snr = float(rng.uniform(*scenes.EMITTER_SNR_DB)) if snr_db is None else float(snr_db)
        sig = sig * np.sqrt(10 ** (snr / 10))
        iq[t0:t0 + length] += sig
        on = np.flatnonzero(np.abs(sig) > 1e-3 * np.abs(sig).max())
        f_lo, f_hi = scenes.band_edges(sig[on[0]:on[-1] + 1])
        truth.append(dict(cls=cls, group=group, t_start=t0 + int(on[0]), t_end=t0 + int(on[-1]),
                          f_lo=f_lo, f_hi=f_hi, snr_db=snr))
    iq += impairments.noise(n, 1.0, rng)
    if impair:
        iq, _ = impairments.apply(iq, None, rng, bandwidth_hz=FS, skip=scenes.SCENE_SKIP)
    return np.asarray(iq, dtype=np.complex64), truth
