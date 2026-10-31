"""
Scenes with automatic boxes for the emitter detector (src/detect/model.py).

A scene is SCENE_SAMPLES of the 3.2 MHz band (10.24 ms, 64x one classifier
window) holding 0-6 emitters, each one a burst at its own time, frequency and
SNR. Every emitter comes from the team's own generators (radar, spread,
digital comms, the jammers), so the detector learns the same signals the
classifier does. Its box is measured from the emitter itself before it is
mixed in, so labels are exact by construction:

  time       first to last sample where the emitter is on (all pulses of a
             pulse train sit inside one box: one box per emitter)
  frequency  the band holding 99% of the emitter's power after placement

The scene then gets unit-power noise and the receiver impairments (phase
noise, IQ imbalance, DC offset, front-end filter, AGC jump, ADC) from
src/data/impairments.py, and becomes a spectrogram image:

  IMG_H x IMG_W = 128 frequency rows (25 kHz each, row 0 = -1.6 MHz)
                x 256 time columns (40 us each)

Pixels are dB above the scene's median (the noise floor), 0-40 dB scaled to
0-1. Boxes are [x0, y0, x1, y1] in pixels: x along time, y along frequency.

Emitter SNR is its power while on against the noise power in the whole
3.2 MHz band, so a narrow signal looks stronger in the image than its SNR
label suggests (its power sits in few rows), as it does on a real receiver.

Visibility: some emitters are genuinely invisible in the image (a 1 us pulse
every few ms at low SNR spreads over many rows and few columns). Each emitter
gets a visibility score -- the energy-detection deflection of its own pixels,
sum(pixel SNR) / sqrt(number of pixels) -- and boxes below MIN_VISIBILITY are
flagged `ignore`: training neither rewards nor penalises the detector there,
and scoring counts them as neither found nor missed (COCO's "ignore" regions).
"""
import numpy as np

from src.config import CFG
from src.data import impairments
from src.generators import digital, radar, spread
from src.generators.jamming import random_jamming_example

FS = CFG["signal"]["fs"]
N_FFT = 128                       # 25 kHz rows, 40 us columns, no overlap
IMG_H = N_FFT
IMG_W = 256
SCENE_SAMPLES = IMG_W * N_FFT     # 32,768 samples = 10.24 ms
MAX_EMITTERS = 6
EMPTY_FRACTION = 0.1
EMITTER_SNR_DB = (-10.0, 20.0)
MIN_BURST = 2048                  # 0.64 ms, 16 image columns
FLOOR_DB, RANGE_DB = 0.0, 40.0
BAND_FRACTION = 0.95              # keep emitters inside 95% of Nyquist
MIN_VISIBILITY = 3.0
# The Hann window smears any single frequency over ~+/-2 rows (its main lobe),
# so a box drawn at the signal's true band would cut off what the image shows
# -- a pure tone would get a 1-row box sitting inside a 4-row line.
FREQ_PAD_PX = 1.5
_WINDOW = np.hanning(N_FFT)
NOISE_PIXEL_POWER = float(np.sum(_WINDOW ** 2))   # mean |X|^2 of unit complex noise

GROUPS = {
    "radar": (radar, radar.CLASSES),
    "spread": (spread, spread.CLASSES),
    "comms": (digital, digital.CLASSES),
    "jamming": (None, ("JAMMING",)),
}
GROUP_NAMES = tuple(GROUPS)

# Receiver effects only: the channel ones are per emitter, and interference
# would add signals nobody labelled.
SCENE_SKIP = ("fading", "placement", "drift", "clock", "interference")


def _active_power(x):
    mag_sq = np.abs(x) ** 2
    return float(mag_sq[mag_sq > 0.01 * mag_sq.max()].mean())


def _burst(group, cls, n_samples, rng):
    """Clean emitter lasting n_samples: (signal, already_placed). Jammers come
    from the old generator, which places them in the band itself."""
    if group == "jamming":
        sig = random_jamming_example(fs=FS, total_duration=(n_samples + 1) / FS, rng=rng)
        sig = np.resize(sig, n_samples) if len(sig) < n_samples else sig[:n_samples]
        return sig / np.sqrt(_active_power(sig)), True
    gen = GROUPS[group][0]
    sig, _ = gen.generate_burst(cls, n_samples, rng)
    return np.asarray(sig, dtype=complex), False


def band_edges(x, fraction=0.99):
    """(f_lo, f_hi) in Hz holding `fraction` of x's power."""
    spec = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS))
    cum = np.cumsum(spec) / spec.sum()
    tail = (1 - fraction) / 2
    lo = f[min(np.searchsorted(cum, tail), len(f) - 1)]
    hi = f[min(np.searchsorted(cum, 1 - tail), len(f) - 1)]
    return float(lo), float(hi)


def to_pixels(t_start, t_end, f_lo, f_hi):
    """Physical box (samples, Hz) -> [x0, y0, x1, y1] in image pixels, padded
    in frequency by the window's smear (FREQ_PAD_PX), at least one pixel wide
    and tall, clipped to the image."""
    x0, x1 = t_start / N_FFT, (t_end + 1) / N_FFT
    y0 = (f_lo + FS / 2) / FS * IMG_H - FREQ_PAD_PX
    y1 = (f_hi + FS / 2) / FS * IMG_H + FREQ_PAD_PX
    if x1 - x0 < 1:
        mid = (x0 + x1) / 2
        x0, x1 = mid - 0.5, mid + 0.5
    if y1 - y0 < 1:
        mid = (y0 + y1) / 2
        y0, y1 = mid - 0.5, mid + 0.5
    return np.clip([x0, y0, x1, y1], 0, [IMG_W, IMG_H, IMG_W, IMG_H]).astype(np.float32)


def pixel_power(iq):
    """Linear power per pixel, (IMG_H, len(iq) // N_FFT), rows in frequency order."""
    n_cols = len(iq) // N_FFT
    frames = np.asarray(iq[:n_cols * N_FFT]).reshape(n_cols, N_FFT) * _WINDOW
    return (np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)) ** 2).T


def spectrogram(iq):
    """Image of a capture: (IMG_H, len(iq) // N_FFT) float32 in 0..1."""
    db = 10 * np.log10(pixel_power(iq) + 1e-12)
    img = (db - np.median(db) - FLOOR_DB) / RANGE_DB
    return np.clip(img, 0, 1).astype(np.float32)


def visibility(emitter_power):
    """Energy-detection deflection of one emitter's own pixels against unit
    noise: sum(pixel SNR) / sqrt(n pixels), over pixels holding >= 10% of its
    peak. ~3 is the edge of what an energy detector can see at all."""
    occupied = emitter_power[emitter_power >= 0.1 * emitter_power.max()]
    snr = occupied / NOISE_PIXEL_POWER
    return float(snr.sum() / np.sqrt(len(snr)))


def make_scene(rng, n_emitters=None, snr_db=None, impair=True, groups=GROUP_NAMES):
    """One scene: (iq complex64 (SCENE_SAMPLES,), image (IMG_H, IMG_W),
    boxes (k, 4) pixels, meta dict of per-emitter lists)."""
    if n_emitters is None:
        n_emitters = 0 if rng.random() < EMPTY_FRACTION else int(rng.integers(1, MAX_EMITTERS + 1))
    scene = np.zeros(SCENE_SAMPLES, dtype=complex)
    boxes = []
    meta = {"group": [], "cls": [], "snr_db": [], "t_start": [], "t_end": [], "f_lo": [], "f_hi": [],
            "visibility": [], "ignore": []}
    for _ in range(n_emitters):
        group = str(rng.choice(groups))
        cls = str(rng.choice(GROUPS[group][1]))
        length = int(rng.integers(MIN_BURST, SCENE_SAMPLES + 1))
        t0 = int(rng.integers(0, SCENE_SAMPLES - length + 1))
        sig, placed = _burst(group, cls, length, rng)
        if not placed:
            lo, hi = band_edges(sig)
            room = BAND_FRACTION * FS / 2 - max(abs(lo), abs(hi))
            offset = rng.uniform(-room, room) if room > 0 else 0.0
            sig = impairments.frequency_shift(sig, offset)
        snr = float(rng.uniform(*EMITTER_SNR_DB)) if snr_db is None else float(snr_db)
        sig = sig * np.sqrt(10 ** (snr / 10))
        scene[t0:t0 + length] += sig

        on = np.flatnonzero(np.abs(sig) > 1e-3 * np.abs(sig).max())
        t_start, t_end = t0 + int(on[0]), t0 + int(on[-1])
        f_lo, f_hi = band_edges(sig[on[0]:on[-1] + 1])
        boxes.append(to_pixels(t_start, t_end, f_lo, f_hi))
        alone = np.zeros(SCENE_SAMPLES, dtype=complex)
        alone[t0:t0 + length] = sig
        vis = visibility(pixel_power(alone))
        for k, v in zip(meta, (group, cls, snr, t_start, t_end, f_lo, f_hi, vis, vis < MIN_VISIBILITY)):
            meta[k].append(v)

    scene += impairments.noise(SCENE_SAMPLES, 1.0, rng)
    if impair:
        scene, _ = impairments.apply(scene, None, rng, bandwidth_hz=FS, skip=SCENE_SKIP)
    iq = np.asarray(scene, dtype=np.complex64)
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return iq, spectrogram(iq), boxes, meta
