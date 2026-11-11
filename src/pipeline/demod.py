"""
Turn a zoomed narrowband track back into the sound an operator would hear.

Input: complex baseband at 50 kS/s (src/pipeline/analog.zoom), the kind from
analog.classify and its features. Output: float audio at 16 kHz in [-1, 1].

  NBFM   frequency discriminator, 750 us de-emphasis, 300 Hz - 3.5 kHz voice band
  WBFM   the same at 400 kS/s, 50 us de-emphasis, 15 kHz low-pass
  AM     carrier to 0 Hz, envelope, DC removed
  AM_SC  carrier phase from the squared signal, real part
  USB    lower band edge (or the carrier) to 0 Hz, positive frequencies, real part
  LSB    upper band edge (or the carrier) to 0 Hz, mirrored, real part
  CW / CARRIER   shifted to a 700 Hz beep (what a receiver's CW mode does)
"""
import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfiltfilt

AUDIO_FS = 16000
BEEP_HZ = 700.0


def _lowpass(x, fs, hz):
    return sosfiltfilt(butter(6, min(hz, 0.45 * fs), fs=fs, output="sos"), x)


def _deemphasis(x, fs, tau):
    a = np.exp(-1 / (fs * tau))
    return lfilter([1 - a], [1, -a], x)


def _shift(y, fs, hz):
    return y * np.exp(-2j * np.pi * hz * np.arange(len(y)) / fs)


def _to_audio(a, fs):
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean()
    from fractions import Fraction
    r = Fraction(AUDIO_FS / fs).limit_denominator(200)
    a = resample_poly(a, r.numerator, r.denominator)
    peak = np.percentile(np.abs(a), 99.9) + 1e-12
    return np.clip(0.9 * a / peak, -1, 1).astype(np.float32)


def fm(y, fs, tau=750e-6, audio_hz=3500, low_hz=None):
    d = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi)
    d = _deemphasis(d - np.median(d), fs, tau)
    d = _lowpass(d, fs, audio_hz)
    if low_hz:
        d = sosfiltfilt(butter(4, low_hz, btype="high", fs=fs, output="sos"), d)
    return _to_audio(d, fs)


# A voice radio's receiver passes 300-3000 Hz (below that sit sub-audible
# squelch tones). Without it, half the NBFM audio power was rumble under
# 300 Hz from the residual offset and drift after de-emphasis; filtering it
# took Whisper's word error on a 21 dB demo clip from 67% to 33% (SED-44,
# en_cv_001).
VOICE_LOW_HZ = 300.0


def demodulate(y, fs, kind, feats):
    """-> 16 kHz float32 audio, or None for kinds without sound."""
    if len(y) < fs * 0.05:
        return None
    lo, hi = feats.get("band", (-1500.0, 1500.0))
    carrier = feats.get("carrier_hz", 0.0)
    has_carrier = feats.get("carrier_frac", 0) >= 0.25
    if kind == "NBFM":
        return fm(_shift(y, fs, (lo + hi) / 2), fs, low_hz=VOICE_LOW_HZ)
    if kind == "WBFM":
        return fm(y, fs, tau=50e-6, audio_hz=15000)
    if kind == "AM":
        z = _lowpass(_shift(y, fs, carrier), fs, 5000)
        return _to_audio(np.abs(z), fs)
    if kind == "AM_SC":
        z = _lowpass(_shift(y, fs, (lo + hi) / 2), fs, 5000)
        blk = int(0.02 * fs)
        out = np.empty(len(z))
        for s in range(0, len(z), blk):                 # carrier phase from the squared signal, per 20 ms
            seg = z[s:s + blk]
            ph = 0.5 * np.angle(np.sum(seg ** 2))
            out[s:s + blk] = np.real(seg * np.exp(-1j * ph))
        return _to_audio(out, fs)
    if kind in ("USB", "LSB"):
        edge = carrier if has_carrier else (lo if kind == "USB" else hi)
        z = _shift(y, fs, edge)
        if kind == "LSB":
            z = np.conj(z)
        Z = np.fft.fft(z)
        f = np.fft.fftfreq(len(z), 1 / fs)
        Z[(f < 100) | (f > 3500)] = 0                   # keep the voice band on the positive side only
        return _to_audio(np.real(np.fft.ifft(Z)), fs)
    if kind in ("CW", "CARRIER"):
        z = _lowpass(_shift(y, fs, carrier), fs, 300)
        return _to_audio(np.real(z * np.exp(2j * np.pi * BEEP_HZ * np.arange(len(z)) / fs)), fs)
    return None
