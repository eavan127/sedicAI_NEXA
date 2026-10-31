"""
Digital communication generators for the week-1 FSK / PSK / QAM classes.

Two generators cover all 18 classes:

  * LINEAR (BPSK ... 32PSK, 16QAM ... 256QAM, OQPSK): random constellation
    points, root-raised-cosine pulse shaping, random carrier phase. What the
    class name means is the constellation -- the set of points a symbol can
    land on.
  * FREQUENCY (2FSK, 4FSK, 8FSK, CPFSK, MSK, GMSK, GFSK): random symbols move
    the instantaneous frequency between M levels. What separates the classes
    is how many levels (M), how far apart they are (modulation index h), how
    the frequency moves between them (instant jump vs Gaussian smoothing) and
    whether the phase stays continuous across a symbol change.

The FSK-family names are used loosely in the literature, so the definitions
below are this project's choice, written down so every teammate means the
same thing:

  class   M  phase        frequency pulse   h
  2FSK    2  jumps        rectangular       0.8 - 1.5
  4FSK    4  jumps        rectangular       0.8 - 1.2
  8FSK    8  jumps        rectangular       0.8 - 1.2
  CPFSK   2  continuous   rectangular       0.6 - 1.0
  MSK     2  continuous   rectangular       0.5 exactly
  GMSK    2  continuous   Gaussian BT 0.25-0.35   0.5 exactly
  GFSK    2  continuous   Gaussian BT 0.5   0.28 - 0.35 (Bluetooth-like)

"Phase jumps" means each symbol starts a fresh tone at a random phase, as a
transmitter switching between independent oscillators does.

Output follows the shard contract (README "Shared data workflow"): complex64
windows of CFG["signal"]["window_len"] samples, signal at unit average power,
plus complex AWGN at the requested SNR. Symbol rate is randomised through the
samples-per-symbol (sps), so the model cannot key on one bandwidth.

Deliberately NOT done here: carrier frequency offset, clock drift, IQ
imbalance, fading and the other receiver impairments. Those belong to the
shared impairment module, applied the same way to every team's generator.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn
from src.dsp import RRC_SPAN_SYMBOLS, rrc_taps

WINDOW_LEN = CFG["signal"]["window_len"]

# Even values only: OQPSK delays Q by half a symbol, and rrc_taps needs
# span * sps even for its centre tap.
LINEAR_SPS = (4, 6, 8, 10, 12, 14, 16)
RRC_BETA_RANGE = (0.2, 0.5)
FSK_SPS_RANGE = (4, 16)
# Highest instantaneous frequency allowed, in cycles per sample. Nyquist is
# 0.5; keeping a margin stops the outer tones of 8FSK folding over.
FSK_MAX_FREQ = 0.4


def _psk(m, offset=0.0):
    k = np.arange(m)
    return np.exp(1j * (2 * np.pi * k / m + offset))


def _square_qam(m):
    side = int(round(np.sqrt(m)))
    levels = np.arange(-(side - 1), side, 2)
    i, q = np.meshgrid(levels, levels)
    return (i + 1j * q).ravel()


def _cross_qam(m):
    """32QAM and 128QAM: the square grid one size up with its corners cut off,
    which is how both are defined in practice (no square grid has 32 or 128
    points)."""
    side = {32: 6, 128: 12}[m]
    corner = {32: 1, 128: 2}[m]
    levels = np.arange(-(side - 1), side, 2)
    i, q = np.meshgrid(levels, levels)
    pts = (i + 1j * q).ravel()
    edge = side - 1 - 2 * (corner - 1)
    keep = ~((np.abs(pts.real) >= edge) & (np.abs(pts.imag) >= edge))
    return pts[keep]


def _unit_energy(points):
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


CONSTELLATIONS = {name: _unit_energy(pts) for name, pts in {
    "BPSK": _psk(2),
    "QPSK": _psk(4, np.pi / 4),
    "OQPSK": _psk(4, np.pi / 4),
    "8PSK": _psk(8),
    "16PSK": _psk(16),
    "32PSK": _psk(32),
    "16QAM": _square_qam(16),
    "32QAM": _cross_qam(32),
    "64QAM": _square_qam(64),
    "128QAM": _cross_qam(128),
    "256QAM": _square_qam(256),
}.items()}

FSK_PARAMS = {
    #         M  continuous  gaussian_bt        h range
    "2FSK": (2, False, None, (0.8, 1.5)),
    "4FSK": (4, False, None, (0.8, 1.2)),
    "8FSK": (8, False, None, (0.8, 1.2)),
    "CPFSK": (2, True, None, (0.6, 1.0)),
    "MSK": (2, True, None, (0.5, 0.5)),
    "GMSK": (2, True, (0.25, 0.35), (0.5, 0.5)),
    "GFSK": (2, True, (0.5, 0.5), (0.28, 0.35)),
}

LINEAR_CLASSES = tuple(CONSTELLATIONS)
FSK_CLASSES = tuple(FSK_PARAMS)
CLASSES = LINEAR_CLASSES + FSK_CLASSES


def _crop(sig, rng, n_samples=WINDOW_LEN):
    """A random n_samples slice from the steady middle, so the window never
    starts on a symbol boundary in a fixed place and never contains the
    filter's start-up or tail."""
    start = rng.integers(0, len(sig) - n_samples + 1)
    return sig[start:start + n_samples]


def _linear(cls, rng, n_samples=WINDOW_LEN):
    points = CONSTELLATIONS[cls]
    sps = int(rng.choice(LINEAR_SPS))
    beta = rng.uniform(*RRC_BETA_RANGE)
    taps = rrc_taps(sps, beta=beta)
    n_sym = n_samples // sps + 2 * RRC_SPAN_SYMBOLS + 2
    symbols = points[rng.integers(0, len(points), n_sym)]

    def shape(sym):
        up = np.zeros(n_sym * sps, dtype=complex)
        up[::sps] = sym
        return np.convolve(up, taps, mode="same")

    if cls == "OQPSK":
        # Offset QPSK: Q runs half a symbol behind I, so I and Q never change
        # together and the envelope never passes through zero.
        i = shape(symbols.real)
        q = np.roll(shape(symbols.imag), sps // 2)
        sig = i + 1j * q
    else:
        sig = shape(symbols)
    # Drop the filter start-up and tail before choosing the window.
    edge = RRC_SPAN_SYMBOLS * sps
    sig = sig[edge:-edge]
    sig = sig * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return _crop(sig, rng, n_samples), {"sps": sps, "beta": beta, "h": np.nan, "bt": np.nan}


def _gaussian_pulse(bt, sps, span=4):
    """Gaussian frequency-smoothing filter (as in GSM / Bluetooth), unit sum
    so the total frequency deviation of a symbol is unchanged."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    g = np.exp(-(np.pi * t / alpha) ** 2)
    return g / g.sum()


def _fsk(cls, rng, n_samples=WINDOW_LEN):
    m, continuous, bt_range, h_range = FSK_PARAMS[cls]
    h = rng.uniform(*h_range)
    # Outer tone sits at (M-1) * h / (2 * sps) cycles/sample; pick sps so it
    # stays under FSK_MAX_FREQ.
    sps_min = max(FSK_SPS_RANGE[0], int(np.ceil((m - 1) * h / (2 * FSK_MAX_FREQ))))
    sps = int(rng.integers(sps_min, FSK_SPS_RANGE[1] + 1))
    n_sym = n_samples // sps + 8
    levels = 2 * rng.integers(0, m, n_sym) - (m - 1)  # -(M-1) ... +(M-1)
    freq = np.repeat(levels * h / (2 * sps), sps)    # cycles per sample
    bt = np.nan
    if bt_range is not None:
        bt = rng.uniform(*bt_range)
        freq = np.convolve(freq, _gaussian_pulse(bt, sps), mode="same")

    if continuous:
        phase = 2 * np.pi * np.cumsum(freq)
    else:
        # Independent oscillator per symbol: fresh random phase at each
        # symbol start, then the tone runs from there.
        n = np.arange(len(freq))
        sym_start = (n // sps) * sps
        start_phase = np.repeat(rng.uniform(0, 2 * np.pi, n_sym), sps)
        phase = 2 * np.pi * freq * (n - sym_start) + start_phase
    sig = np.exp(1j * (phase + rng.uniform(0, 2 * np.pi)))
    edge = 2 * sps  # skip the Gaussian filter's start-up
    return _crop(sig[edge:-edge], rng, n_samples), {"sps": sps, "beta": np.nan, "h": h, "bt": bt}


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit average power) -- used by the
    tests to check the physics without noise in the way.
    """
    sig, params = _build(cls, rng, WINDOW_LEN)
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng, reference="mean")
    return sig.astype(np.complex64), params


def _build(cls, rng, n_samples):
    if cls in CONSTELLATIONS:
        sig, params = _linear(cls, rng, n_samples)
    elif cls in FSK_PARAMS:
        sig, params = _fsk(cls, rng, n_samples)
    else:
        raise ValueError(f"{cls!r} is not a digital.py class; choose from {CLASSES}")
    return sig / np.sqrt(np.mean(np.abs(sig) ** 2)), params


def generate_burst(cls, n_samples, rng):
    """A clean emitter of class `cls` lasting n_samples with its parameters
    fixed for the whole burst (unit power, centred on 0 Hz) -- for the
    detector's multi-millisecond scenes (src/detect/scenes.py)."""
    return _build(cls, rng, n_samples)


def generate(cls, n, snr_db, rng):
    """The shared generator signature: `n` windows as (complex64 (n, WINDOW_LEN),
    meta) where meta holds one array per randomised parameter."""
    iq = np.empty((n, WINDOW_LEN), dtype=np.complex64)
    meta = {k: np.empty(n) for k in ("sps", "beta", "h", "bt")}
    for i in range(n):
        iq[i], params = generate_window(cls, snr_db, rng)
        for k, v in params.items():
            meta[k][i] = v
    return iq, meta
