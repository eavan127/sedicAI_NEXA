"""
Cut one detector box out of the wideband capture and measure it.

The classifier was trained on windows of the whole 3.2 MHz band: one emitter
somewhere in it plus white noise everywhere (src/data/impairments.py). A box
from a busy capture has other emitters around it, so extract() keeps only the
box's band and refills every other frequency with white noise at the measured
noise floor. The classifier then sees the conditions it was trained on: one
emitter, flat noise, nothing else. The emitter is also moved to 0 Hz, where
the generators make it before placement.

Features are measured on the same cut with the out-of-band part left empty
(only the box's own noise), over the whole box, which is many windows long:

  snr_db          in-band power above the in-band noise
  occupied_hz     the signal's own width (2-98% of its power above the noise
                  floor); the detector's box is usually wider
  wraps           sweep wraps: the frequency jumps by more than half the
                  occupied width (a LoRa chirp or a sawtooth sweep restarting)
  sweep_rate_hz   median frequency change per sample of the smoothed track
  sideband_db     strongest spectral line away from the main one, above the
                  noise floor, in one long FFT: ~0 for a pure tone, high when
                  the carrier is modulated (FM/AM sidebands, data)
  line_db         the main line above the noise floor
  power_cv        how much the power varies between 0.32 ms blocks
"""
from dataclasses import dataclass, field

import numpy as np

from src.config import CFG
from src.detect import scenes

FS = CFG["signal"]["fs"]
WINDOW_LEN = CFG["signal"]["window_len"]
MIN_MARGIN_HZ = 25e3            # extra band kept on each side of the box
CUT_MARGIN_FRAC = 0.1           # ... or this share of the box width, whichever is larger
SMOOTH = 32                     # samples averaged in the frequency track
SIDEBAND_GUARD_HZ = 2e3         # around the main line, not counted as a sideband
BLOCK = 1024                    # power blocks for power_cv (0.32 ms)
MAX_WINDOWS = 64                # windows scored per box (evenly spread)
OCC_SEG = 1024                  # averaged-spectrum segment for the occupied band (3.1 kHz bins)
WRAP_MIN_SNR_DB = 6.0           # in-band SNR below which noise hides real wraps and fakes others
MIN_WRAP_HZ = 20e3              # never call a jump smaller than this a wrap
LN2 = float(np.log(2))
# Since the FSK/PSK data fix (f993d8ebc417) every training window sits anywhere
# in the band and none at exactly 0 Hz, so the classifier gets the box back at
# its own frequency. Moving it to 0 Hz turns a pure tone into a DC offset the
# model never saw (RADAR_CW / JAM_TONE 80% -> 0% at 15 dB). The features still
# use the centred copy.
KEEP_POSITION = True


def noise_variance(chunk_iq):
    """Per-sample noise power of a chunk from its spectrogram pixels. The
    median pixel is noise for any realistic occupancy; for exponential pixel
    power the mean is median / ln 2."""
    px = scenes.pixel_power(chunk_iq)
    return float(np.median(px) / LN2 / scenes.NOISE_PIXEL_POWER)


@dataclass
class Cut:
    """One box, cut out: `filled` (noise refilled outside the band, for the
    classifier) and `inband` (zero outside the band, for the features)."""
    t0: int
    t1: int
    f_lo: float
    f_hi: float
    filled: np.ndarray
    inband: np.ndarray
    keep_frac: float
    noise_var: float
    features: dict = field(default_factory=dict)

    @property
    def centre_hz(self):
        return (self.f_lo + self.f_hi) / 2

    @property
    def width_hz(self):
        return self.f_hi - self.f_lo


def cut(iq, t0, t1, f_lo, f_hi, noise_var, rng, margin_hz=None):
    """Box [t0, t1) samples x [f_lo, f_hi] Hz of `iq` -> Cut. Boxes shorter
    than one window grow to one window around their middle."""
    n = len(iq)
    t0, t1 = int(max(0, t0)), int(min(n, t1))
    if t1 - t0 < WINDOW_LEN:
        mid = (t0 + t1) // 2
        t0 = max(0, min(mid - WINDOW_LEN // 2, n - WINDOW_LEN))
        t1 = min(n, t0 + WINDOW_LEN)
    seg = np.asarray(iq[t0:t1], dtype=complex)
    fc = (f_lo + f_hi) / 2
    half = (f_hi - f_lo) / 2 + (margin_hz if margin_hz is not None else max(MIN_MARGIN_HZ, CUT_MARGIN_FRAC * (f_hi - f_lo)))
    seg = seg * np.exp(-2j * np.pi * fc * np.arange(t0, t1) / FS)
    X = np.fft.fft(seg)
    keep = np.abs(np.fft.fftfreq(len(seg), 1 / FS)) <= half
    inband = np.fft.ifft(np.where(keep, X, 0))
    fill = (rng.standard_normal(len(seg)) + 1j * rng.standard_normal(len(seg))) * np.sqrt(noise_var / 2)
    filled = np.fft.ifft(np.where(keep, X, np.fft.fft(fill)))
    c = Cut(t0, t1, float(f_lo), float(f_hi), filled.astype(np.complex64), inband.astype(np.complex64),
            float(keep.mean()), float(noise_var))
    c.features = features(c)
    return c


def _window(c, s):
    """Classifier window starting at sample s of the cut: the noise-filled cut,
    moved back to the box's own frequency when KEEP_POSITION is on."""
    w = c.filled[s:s + WINDOW_LEN]
    if KEEP_POSITION:
        w = w * np.exp(2j * np.pi * c.centre_hz * np.arange(c.t0 + s, c.t0 + s + len(w)) / FS)
    return w.astype(np.complex64)


def windows(c, max_windows=MAX_WINDOWS):
    """The cut's classifier windows: complex (k, WINDOW_LEN), back to back,
    at most max_windows of them spread evenly over the box."""
    n = len(c.filled) // WINDOW_LEN
    starts = np.arange(n) * WINDOW_LEN
    if n > max_windows:
        starts = starts[np.linspace(0, n - 1, max_windows).round().astype(int)]
    return np.stack([_window(c, s) for s in starts])


def active_windows(c, max_windows=MAX_WINDOWS, min_snr_db=3.0):
    """Like windows(), but only windows where the emitter is actually on:
    in-band power at least min_snr_db above the in-band noise. A pulsed radar
    or a keyed signal leaves many windows between pulses that hold only noise;
    voting with them drags the track's answer towards whatever the classifier
    says about noise. Falls back to windows(c) when fewer than 2 are active."""
    n = len(c.filled) // WINDOW_LEN
    if n == 0:
        return windows(c, max_windows)
    seg = np.asarray(c.inband[:n * WINDOW_LEN]).reshape(n, WINDOW_LEN)
    power = np.mean(np.abs(seg) ** 2, axis=1)
    noise = c.noise_var * c.keep_frac
    on = np.flatnonzero(power >= noise * (1 + 10 ** (min_snr_db / 10)))
    if len(on) < 2:
        return windows(c, max_windows)
    if len(on) > max_windows:
        on = on[np.linspace(0, len(on) - 1, max_windows).round().astype(int)]
    return np.stack([_window(c, s * WINDOW_LEN) for s in on])


def _smooth(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) > k else x


def occupied_band(x, noise_var, half_hz, seg=OCC_SEG):
    """(lo, hi) Hz holding 2..98% of the signal's power above the noise floor,
    from an averaged spectrum of x (already centred and band-limited). The
    detector's box is usually wider than the signal; this is the signal."""
    n = len(x) // seg
    if n == 0:
        return -half_hz, half_hz
    w = np.hanning(seg)
    P = np.mean(np.abs(np.fft.fftshift(np.fft.fft(x[:n * seg].reshape(n, seg) * w, axis=1), axes=1)) ** 2, axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(seg, 1 / FS))
    excess = np.where(np.abs(f) <= half_hz, np.maximum(P - noise_var * np.sum(w ** 2), 0), 0)
    if excess.sum() <= 0:
        return -half_hz, half_hz
    cum = np.cumsum(excess) / excess.sum()
    return float(f[np.searchsorted(cum, 0.02)]), float(f[min(np.searchsorted(cum, 0.98), seg - 1)])


def features(c):
    x = c.inband.astype(complex)
    p_in = float(np.mean(np.abs(x) ** 2))
    p_noise = c.noise_var * c.keep_frac
    snr_db = 10 * np.log10(max(p_in - p_noise, 1e-12) / max(p_noise, 1e-12))
    occ_lo, occ_hi = occupied_band(x, c.noise_var, c.width_hz / 2)
    occupied = max(occ_hi - occ_lo, 1.0)

    # Frequency track of the smoothed signal; a wrap is a fall or rise of more
    # than half the occupied band within SMOOTH samples. Below WRAP_MIN_SNR_DB
    # the track is mostly noise and its jumps mean nothing.
    f = _smooth(np.diff(np.unwrap(np.angle(_smooth(x, 4)))) * FS / (2 * np.pi), SMOOTH)
    if len(f) > 2 * SMOOTH and snr_db >= WRAP_MIN_SNR_DB:
        jump = np.abs(f[SMOOTH:] - f[:-SMOOTH])
        big = jump > max(0.5 * occupied, MIN_WRAP_HZ)
        # Count each run of samples over the threshold once.
        wraps = int(np.count_nonzero(big[1:] & ~big[:-1]) + int(big[0]))
        sweep_rate = float(np.median(np.abs(np.diff(f))))
    else:
        wraps, sweep_rate = 0, 0.0

    # One long FFT: a pure tone is one line; modulation puts power in lines
    # (or a hump) away from it.
    P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS))
    inside = np.abs(freqs) <= c.width_hz / 2 + MIN_MARGIN_HZ
    floor = float(np.median(P[inside])) + 1e-20
    k0 = int(np.argmax(np.where(inside, P, 0)))
    away = inside & (np.abs(freqs - freqs[k0]) > SIDEBAND_GUARD_HZ)
    side = float(P[away].max()) if away.any() else floor
    line_db = 10 * np.log10(P[k0] / floor)
    sideband_db = 10 * np.log10(side / floor)

    blocks = len(x) // BLOCK
    if blocks >= 3:
        bp = np.mean(np.abs(x[:blocks * BLOCK].reshape(blocks, BLOCK)) ** 2, axis=1)
        power_cv = float(bp.std() / (bp.mean() + 1e-20))
    else:
        power_cv = float("nan")
    return dict(snr_db=float(snr_db), occupied_hz=float(occupied), wraps=wraps, sweep_rate_hz=sweep_rate,
                line_db=float(line_db), sideband_db=float(sideband_db), power_cv=power_cv,
                duration_s=(c.t1 - c.t0) / FS)
