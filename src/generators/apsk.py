"""
APSK generators for four of Eileen's week-1 classes: concentric rings of points,
root-raised-cosine shaped and randomised the same way as digital.py's linear
classes.

  class    points per ring, inner to outer    ring radii
  16APSK   4, 12                              1, 2.85 (DVB-S2)
  32APSK   4, 12, 16                          1, 2.84, 5.27 (DVB-S2)
  64APSK   4, 12, 20, 28                      1, 2.4, 4.3, 7.0 (DVB-S2X)
  128APSK  16, 16, 16, 16, 16, 48             1, 1.6, 2.2, 2.8, 3.4, 4.2

128APSK's ring sizes follow DVB-S2X, but its radii are this project's choice:
the standard's depend on code rate. Output follows the shard contract the same
way digital.py does, and impairments stay in the shared module.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn
from src.dsp import RRC_SPAN_SYMBOLS, rrc_taps
from src.generators.digital import LINEAR_SPS, RRC_BETA_RANGE

WINDOW_LEN = CFG["signal"]["window_len"]

RINGS = {
    #          points per ring            radii
    "16APSK": ((4, 12), (1, 2.85)),
    "32APSK": ((4, 12, 16), (1, 2.84, 5.27)),
    "64APSK": ((4, 12, 20, 28), (1, 2.4, 4.3, 7.0)),
    "128APSK": ((16, 16, 16, 16, 16, 48), (1, 1.6, 2.2, 2.8, 3.4, 4.2)),
}
CLASSES = tuple(RINGS)


def _constellation(counts, radii):
    rings = []
    for ring, (n, r) in enumerate(zip(counts, radii)):
        # Half a step of turn per ring stops points lining up along the spokes.
        offset = np.pi / n * (ring % 2)
        rings.append(r * np.exp(1j * (2 * np.pi * np.arange(n) / n + offset)))
    points = np.concatenate(rings)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


CONSTELLATIONS = {name: _constellation(*spec) for name, spec in RINGS.items()}


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit average power) -- used by the
    tests to check the physics without noise in the way.
    """
    if cls not in CONSTELLATIONS:
        raise ValueError(f"{cls!r} is not an apsk.py class; choose from {CLASSES}")
    points = CONSTELLATIONS[cls]
    sps = int(rng.choice(LINEAR_SPS))
    beta = rng.uniform(*RRC_BETA_RANGE)
    taps = rrc_taps(sps, beta=beta)
    n_sym = WINDOW_LEN // sps + 2 * RRC_SPAN_SYMBOLS + 2
    up = np.zeros(n_sym * sps, dtype=complex)
    up[::sps] = points[rng.integers(0, len(points), n_sym)]
    sig = np.convolve(up, taps, mode="same")
    # Drop the filter start-up and tail before choosing the window.
    edge = RRC_SPAN_SYMBOLS * sps
    sig = sig[edge:-edge]
    start = rng.integers(0, len(sig) - WINDOW_LEN + 1)
    sig = sig[start:start + WINDOW_LEN] * np.exp(1j * rng.uniform(0, 2 * np.pi))
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng, reference="mean")
    return sig.astype(np.complex64), {"sps": sps, "beta": beta}


def generate(cls, n, snr_db, rng):
    """The shared generator signature: `n` windows as (complex64 (n, WINDOW_LEN),
    meta) where meta holds one array per randomised parameter."""
    iq = np.empty((n, WINDOW_LEN), dtype=np.complex64)
    meta = {k: np.empty(n) for k in ("sps", "beta")}
    for i in range(n):
        iq[i], params = generate_window(cls, snr_db, rng)
        for k, v in params.items():
            meta[k][i] = v
    return iq, meta
