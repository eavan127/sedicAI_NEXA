"""
Pulse-comms generators for the week-1 PULSE classes: PPM, SSR_IFF, AFSK.

Same contract as src/generators/spread.py: complex64 windows of WINDOW_LEN
samples at FS, unit average power, AWGN at the requested SNR.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn

WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]

CLASSES = ("PPM", "SSR_IFF", "AFSK")
PARAM_KEYS = ("bandwidth_hz", "n_pulses")

US = 1e-6  # one microsecond


def _ppm(rng):
    """ADS-B: 8 us preamble (pulses at 0, 1, 3.5, 4.5 us) then 112 bits of
    1 us each. Bit 1 = pulse in the first half, bit 0 = pulse in the second."""
    pulses = [0.0, 1.0, 3.5, 4.5]                         # preamble, in us
    bits = rng.integers(0, 2, 112)
    pulses += [8.0 + i + (0.0 if b else 0.5) for i, b in enumerate(bits)]

    start = rng.uniform(0, 30) * US                       # message not always at sample 0
    t = np.arange(WINDOW_LEN) / FS
    envelope = np.zeros(WINDOW_LEN)
    for p in pulses:
        envelope[(t >= start + p * US) & (t < start + (p + 0.5) * US)] = 1.0

    sig = envelope * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return sig, dict(bandwidth_hz=2e6, n_pulses=len(pulses))


def _ssr_iff(rng):
    """SSR / IFF interrogation at 1030 MHz: P1 and P3, each 0.8 us wide,
    8 us apart (Mode A, identity) or 21 us apart (Mode C, altitude). Half the
    windows also carry P2, the sidelobe-suppression pulse 2 us after P1."""
    gap = rng.choice([8.0, 21.0])                         # Mode A or Mode C, in us
    pulses = [0.0, gap]                                   # P1, P3
    if rng.random() < 0.5:
        pulses.append(2.0)                                # P2

    start = rng.uniform(0, 130) * US                      # whole group stays in the window
    t = np.arange(WINDOW_LEN) / FS
    envelope = np.zeros(WINDOW_LEN)
    for p in pulses:
        envelope[(t >= start + p * US) & (t < start + (p + 0.8) * US)] = 1.0

    sig = envelope * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return sig, dict(bandwidth_hz=2e6, n_pulses=len(pulses))


def _afsk(rng):
    """APRS AFSK (Bell 202): an FM carrier whose audio tone is 1200 Hz (mark)
    or 2200 Hz (space), 1200 baud. One bit lasts 833 us, longer than the
    160 us window, so most windows hold one tone; some catch a bit change."""
    tone = np.full(WINDOW_LEN, rng.choice([1200.0, 2200.0]))
    bit_len = FS / 1200                                   # samples per bit
    edge = rng.uniform(0, bit_len)                        # where the next bit starts
    if edge < WINDOW_LEN:                                 # bit change inside the window
        tone[int(edge):] = 3400.0 - tone[0]               # 1200 <-> 2200

    audio_phase = 2 * np.pi * np.cumsum(tone / FS) + rng.uniform(0, 2 * np.pi)
    dev = rng.uniform(2e3, 5e3)                           # FM deviation, Hz
    freq = dev * np.sin(audio_phase)                      # instantaneous frequency, Hz
    phase = 2 * np.pi * np.cumsum(freq / FS) + rng.uniform(0, 2 * np.pi)
    return np.exp(1j * phase), dict(bandwidth_hz=2 * (dev + 2200.0), n_pulses=np.nan)



_BUILDERS = {"PPM": _ppm, "SSR_IFF": _ssr_iff, "AFSK": _afsk}


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).
    snr_db=None returns the clean signal (the impairment module adds noise)."""
    if cls not in _BUILDERS:
        raise ValueError(f"{cls!r} is not a pulse class; choose from {CLASSES}")
    sig, built = _BUILDERS[cls](rng)
    params = dict.fromkeys(PARAM_KEYS, np.nan)
    params.update(built)
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng)              # "active": pulsed signal
    return sig.astype(np.complex64), params
