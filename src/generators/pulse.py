"""
Pulse-comms generators for the week-1 PULSE classes: PPM, SSR_IFF, AFSK.

Same contract as src/generators/spread.py: complex64 windows of WINDOW_LEN
samples at FS, unit average power, AWGN at the requested SNR.

generate_burst makes the same emitters lasting longer than one window, for the
detector's multi-millisecond scenes: PPM sends message after message, SSR_IFF
repeats its interrogation, AFSK keeps sending random bits.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn

WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]

CLASSES = ("PPM", "SSR_IFF", "AFSK")
PARAM_KEYS = ("bandwidth_hz", "n_pulses")

US = 1e-6  # one microsecond

# Bursts only (longer than one window).
PPM_GAP_US = (10.0, 300.0)               # silence between two ADS-B messages
SSR_PERIOD_US = (200.0, 2000.0)          # time between two interrogations


def _envelope(n, starts_us, width_us):
    """0/1 envelope of n samples with a pulse of width_us at each start (us)."""
    env = np.zeros(n)
    for s in starts_us:
        i0, i1 = int(np.ceil(s * US * FS)), int(np.ceil((s + width_us) * US * FS))
        env[max(i0, 0):min(i1, n)] = 1.0
    return env


def _ppm_pulses(rng):
    """Pulse start times (us) of one ADS-B message, starting at 0."""
    pulses = [0.0, 1.0, 3.5, 4.5]                         # preamble, in us
    bits = rng.integers(0, 2, 112)
    pulses += [8.0 + i + (0.0 if b else 0.5) for i, b in enumerate(bits)]
    return pulses


def _ppm(rng, n):
    """ADS-B: 8 us preamble (pulses at 0, 1, 3.5, 4.5 us) then 112 bits of
    1 us each. Bit 1 = pulse in the first half, bit 0 = pulse in the second."""
    if n <= WINDOW_LEN:
        pulses = _ppm_pulses(rng)
        start = rng.uniform(0, 30) * US                   # message not always at sample 0
        t = np.arange(n) / FS
        envelope = np.zeros(n)
        for p in pulses:
            envelope[(t >= start + p * US) & (t < start + (p + 0.5) * US)] = 1.0
    else:
        # Message after message (many aircraft squittering), each 120 us long.
        starts, t0 = [], rng.uniform(0, 30)
        while t0 + 120.0 <= n / FS / US:
            starts += [t0 + p for p in _ppm_pulses(rng)]
            t0 += 120.0 + rng.uniform(*PPM_GAP_US)
        if not starts:
            starts = _ppm_pulses(rng)
        envelope = _envelope(n, starts, 0.5)
        pulses = starts

    sig = envelope * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return sig, dict(bandwidth_hz=2e6, n_pulses=len(pulses))


def _ssr_pulses(rng):
    """Pulse start times (us) of one interrogation, P1 at 0."""
    gap = rng.choice([8.0, 21.0])                         # Mode A or Mode C, in us
    pulses = [0.0, gap]                                   # P1, P3
    if rng.random() < 0.5:
        pulses.append(2.0)                                # P2
    return pulses


def _ssr_iff(rng, n):
    """SSR / IFF interrogation at 1030 MHz: P1 and P3, each 0.8 us wide,
    8 us apart (Mode A, identity) or 21 us apart (Mode C, altitude). Half the
    interrogations also carry P2, the sidelobe-suppression pulse 2 us after P1."""
    if n <= WINDOW_LEN:
        pulses = _ssr_pulses(rng)
        start = rng.uniform(0, 130) * US                  # whole group stays in the window
        t = np.arange(n) / FS
        envelope = np.zeros(n)
        for p in pulses:
            envelope[(t >= start + p * US) & (t < start + (p + 0.8) * US)] = 1.0
    else:
        # The interrogator repeats, interlacing Mode A and Mode C.
        period = rng.uniform(*SSR_PERIOD_US)
        starts, t0 = [], rng.uniform(0, min(period, n / FS / US - 22.0))
        while t0 + 22.0 <= n / FS / US:
            starts += [t0 + p for p in _ssr_pulses(rng)]
            t0 += period
        envelope = _envelope(n, starts, 0.8)
        pulses = starts

    sig = envelope * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return sig, dict(bandwidth_hz=2e6, n_pulses=len(pulses))


def _afsk(rng, n):
    """APRS AFSK (Bell 202): an FM carrier whose audio tone is 1200 Hz (mark)
    or 2200 Hz (space), 1200 baud. One bit lasts 833 us, longer than the
    160 us window, so most windows hold one tone; some catch a bit change."""
    tone = np.full(n, rng.choice([1200.0, 2200.0]))
    bit_len = FS / 1200                                   # samples per bit
    edge = rng.uniform(0, bit_len)                        # where the next bit starts
    if n <= WINDOW_LEN:
        if edge < n:                                      # bit change inside the window
            tone[int(edge):] = 3400.0 - tone[0]           # 1200 <-> 2200
    else:
        # Random data: a new bit every bit_len samples after the first edge.
        edges = np.arange(edge, n, bit_len).astype(int)
        for e, b in zip(edges, rng.integers(0, 2, len(edges))):
            tone[e:] = 1200.0 if b else 2200.0

    audio_phase = 2 * np.pi * np.cumsum(tone / FS) + rng.uniform(0, 2 * np.pi)
    dev = rng.uniform(2e3, 5e3)                           # FM deviation, Hz
    freq = dev * np.sin(audio_phase)                      # instantaneous frequency, Hz
    phase = 2 * np.pi * np.cumsum(freq / FS) + rng.uniform(0, 2 * np.pi)
    return np.exp(1j * phase), dict(bandwidth_hz=2 * (dev + 2200.0), n_pulses=np.nan)


_BUILDERS = {"PPM": _ppm, "SSR_IFF": _ssr_iff, "AFSK": _afsk}


def _build(cls, rng, n_samples):
    if cls not in _BUILDERS:
        raise ValueError(f"{cls!r} is not a pulse class; choose from {CLASSES}")
    sig, built = _BUILDERS[cls](rng, n_samples)
    params = dict.fromkeys(PARAM_KEYS, np.nan)
    params.update(built)
    return sig / np.sqrt(np.mean(np.abs(sig) ** 2)), params


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).
    snr_db=None returns the clean signal (the impairment module adds noise)."""
    sig, params = _build(cls, rng, WINDOW_LEN)
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng)              # "active": pulsed signal
    return sig.astype(np.complex64), params


def generate_burst(cls, n_samples, rng):
    """A clean emitter of class `cls` lasting n_samples (unit average power,
    centred on 0 Hz) -- for the detector's multi-millisecond scenes
    (src/detect/scenes.py). With n_samples == WINDOW_LEN it is exactly
    generate_window(cls, None, rng)."""
    return _build(cls, rng, n_samples)
