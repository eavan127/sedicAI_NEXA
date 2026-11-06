"""
Voice-like test signals for the analog rules and the demodulator (tests and
scripts only): speech-shaped audio, then NBFM / AM / SSB / CW / carrier at the
50 kS/s zoom, with a small tuning offset and white noise at a given SNR.

The audio is band-limited noise (300-3000 Hz) tilted toward low frequencies
like speech, with syllables (~4 per second) and pauses: close enough to speech
for these measurements, without shipping any recording.
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from src.intel import morse

FA = 50e3
KINDS = ("NBFM", "AM", "AM_SC", "USB", "LSB", "USB_WC", "LSB_WC", "CW", "CARRIER")


def speech_like(n, fs, rng):
    x = rng.standard_normal(n)
    x = sosfiltfilt(butter(4, [300, 3000], btype="band", fs=fs, output="sos"), x)
    x = x + 2.0 * sosfiltfilt(butter(2, [300, 900], btype="band", fs=fs, output="sos"), rng.standard_normal(n))
    t = np.arange(n) / fs
    syll = np.clip(np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6)), 0, None) ** 2
    pause = (np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 6)) > -0.5).astype(float)
    a = x * syll * pause
    return a / (np.abs(a).max() + 1e-12)


def make(kind, seconds=1.0, snr_db=15.0, rng=None, offset_hz=None):
    rng = rng or np.random.default_rng(0)
    n = int(seconds * FA)
    t = np.arange(n) / FA
    off = rng.uniform(-800, 800) if offset_hz is None else offset_hz
    a = speech_like(n, FA, rng)
    if kind == "NBFM":
        s = np.exp(1j * 2 * np.pi * np.cumsum(2500 * a) / FA)
    elif kind == "AM":
        s = (1 + 0.8 * a).astype(complex)
    elif kind == "AM_SC":
        s = a.astype(complex)
    elif kind in ("USB", "LSB"):
        s = hilbert(a)
        s = s if kind == "USB" else np.conj(s)
    elif kind in ("USB_WC", "LSB_WC"):
        h = hilbert(a)
        s = 0.7 + (h if kind == "USB_WC" else np.conj(h))
    elif kind == "CW":
        dot = 1.2 / 18
        key = []
        for ch in "CQ CQ DE YB1ABC K":
            if ch == " ":
                key += [0] * 4
                continue
            for el in morse.TO_CODE[ch]:
                key += [1] * (1 if el == "." else 3) + [0]
            key += [0] * 2
        gate = np.repeat(np.asarray(key, float), int(dot * FA))
        ramp = int(0.005 * FA)                       # 5 ms soft edges, as a real keyer (no key clicks)
        gate = np.convolve(gate, np.hanning(2 * ramp) / np.hanning(2 * ramp).sum(), mode="same")
        s = np.resize(gate, n).astype(complex)
    elif kind == "CARRIER":
        s = np.ones(n, complex)
    else:
        raise ValueError(kind)
    s = s * np.exp(2j * np.pi * off * t)
    on = np.abs(s) > 1e-6
    p = np.mean(np.abs(s[on]) ** 2) if on.any() else 1.0
    # SNR over the signal's own ~3 kHz band; the zoomed band is 50 kHz wide
    noise_p = p / 10 ** (snr_db / 10) * (FA / 3000)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_p / 2)
    return s + noise, a


def wideband(parts, seconds=1.0, rng=None, fs=3.2e6):
    """A 3.2 MS/s capture with unit-power white noise and narrowband voice/Morse
    emitters in it. parts: [(kind, offset_hz, snr_db_in_3khz), ...]. Returns
    (iq complex64, truth list)."""
    from scipy.signal import resample_poly
    rng = rng or np.random.default_rng(0)
    up = int(round(fs / FA))
    n = int(seconds * fs)
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    noise_density = 1.0 / fs                       # unit power over fs
    truth = []
    for kind, offset, snr in parts:
        s, _ = make(kind, seconds, snr_db=200, rng=rng, offset_hz=0.0)   # clean
        on = np.abs(s) > 1e-6
        p = np.mean(np.abs(s[on]) ** 2) if on.any() else 1.0
        s = s / np.sqrt(p) * np.sqrt(10 ** (snr / 10) * noise_density * 3000)
        w = resample_poly(s, up, 1)[:n]
        w = w * np.exp(2j * np.pi * offset * np.arange(len(w)) / fs)
        iq[:len(w)] += w
        truth.append({"cls": kind, "offset_hz": offset, "snr_db": snr})
    return iq.astype(np.complex64), truth
