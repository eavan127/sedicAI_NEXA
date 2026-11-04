"""
Jamming generators for the week-1 JAM classes, plus NOISE_FLOOR.

Same contract as src/generators/spread.py: complex64 windows of WINDOW_LEN
samples at FS, centred on 0 Hz, unit average power, AWGN at the requested SNR.
No receiver impairments here -- frequency offset, drift, fading etc. come from
the shared impairment module, which also moves each signal around the band
(that is why every builder reports bandwidth_hz).

What each class means INSIDE ONE 160 us WINDOW (this project's definitions):

  class              what the window shows
  JAM_BARRAGE        noise filling 0.9-1.4 MHz (most of the band)
  JAM_PARTIAL_BAND   noise filling 200-600 kHz
  JAM_SPOT           noise filling 10-60 kHz
  JAM_TONE           one continuous tone
  JAM_MULTITONE      2-8 continuous tones at once
  JAM_SWEPT          a slow sweep: one piece of a sweep much longer than the window
  JAM_CHIRP          a fast sawtooth chirp, repeating every 16-56 samples
  JAM_PULSED         bursts of noise switching on and off
  JAM_REACTIVE       silence, then noise switches on once and stays on
  JAM_DRFM           a radar pulse plus a delayed, Doppler-shifted, stronger copy
  NOISE_FLOOR        receiver noise only, nobody transmitting

The bandwidth gaps between SPOT (<= 60 kHz), PARTIAL_BAND (200-600 kHz) and
BARRAGE (>= 900 kHz) are on purpose: if the ranges touched, the model could
not tell where one class ends and the next begins.

Kept apart from look-alike classes in other generators:
  * JAM_CHIRP repeats every 16-56 samples; RADAR_FMCW (radar.py) repeats every
    64-512 samples. Same sawtooth shape, separated by repetition period.
  * JAM_PULSED bursts are noise; RADAR_PULSE pulses are a clean carrier.
  * JAM_TONE is the same shape as RADAR_CW (one tone). Within one window the
    two cannot be told apart, so both carry both labels (configs week1.
    ambiguous_labels, src/data/labels.py).
  * A JAM_SWEPT window that is one straight sweep at a LoRa chirp's rate looks
    like a piece of a LoRa chirp: it is labelled JAM_SWEPT + LORA_CSS
    (lora_like_sweep, src/generators/spread.py looks_like_lora_sweep).
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn
from src.generators import radar, spread

WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]

CLASSES = ("JAM_BARRAGE", "JAM_SPOT", "JAM_TONE", "JAM_MULTITONE", "JAM_SWEPT",
           "JAM_CHIRP", "JAM_PULSED", "JAM_PARTIAL_BAND", "JAM_REACTIVE",
           "JAM_DRFM", "NOISE_FLOOR")

MAX_BANDWIDTH_HZ = 1.4e6

# --- noise jammers (two-sided width of the noise band) ----------------------
BARRAGE_BW_HZ = (0.9e6, MAX_BANDWIDTH_HZ)
PARTIAL_BW_HZ = (200e3, 600e3)
SPOT_BW_HZ = (10e3, 60e3)

# --- tone jammers -------------------------------------------------------------
TONE_SPREAD_HZ = 600e3           # multi-tones sit within +/- this of 0 Hz
MULTITONE_N = (2, 8)
MULTITONE_MIN_SPACING_HZ = 20e3

# --- sweeps -------------------------------------------------------------------
SWEPT_BW_HZ = (0.9e6, MAX_BANDWIDTH_HZ)
SWEPT_PERIOD_SAMPLES = (2 * WINDOW_LEN, 20 * WINDOW_LEN)   # slower than one window
CHIRP_BW_HZ = (0.6e6, MAX_BANDWIDTH_HZ)
CHIRP_PERIOD_SAMPLES = (16, 56)                            # RADAR_FMCW starts at 64

# --- on/off jammers -----------------------------------------------------------
PULSED_PERIOD_SAMPLES = (32, WINDOW_LEN // 3)
PULSED_DUTY = (0.1, 0.5)
PULSED_BW_HZ = (200e3, MAX_BANDWIDTH_HZ)
REACTIVE_ONSET_FRAC = (0.15, 0.6)  # when in the window the jammer switches on

# --- DRFM -----------------------------------------------------------------------
DRFM_RADAR_CLASSES = ("RADAR_PULSE", "LFM_RADAR", "RADAR_BARKER")
DRFM_MAX_PULSE_SAMPLES = 256
DRFM_DELAY_SAMPLES = (2, 64)       # range-gate pull-off, frozen at one instant
DRFM_DOPPLER_HZ = (1e3, 50e3)      # velocity-gate pull-off, frozen at one instant
DRFM_GAIN_DB = (3, 12)             # the false target is louder than the skin return
DRFM_PHASE_BITS = (2, 3, 4)        # DRFMs store the pulse with few bits
# Bursts longer than one window (detector scenes) repeat the pulse every PRI;
# over the burst the copy's delay grows by DRFM_PULL_SAMPLES, and each PRI
# leaves up to DRFM_PRI_EXTRA_SAMPLES of silence after the copy.
DRFM_PULL_SAMPLES = (16, 128)
DRFM_PRI_EXTRA_SAMPLES = 1024

# copied_radar: for JAM_DRFM, the index in radar.CLASSES of the pulse the jammer
# copied. The window holds that radar's own pulse too, so it is labelled with
# both classes (src/data/labels.py).
# lora_like_sweep: for JAM_SWEPT, 1 when the window is one straight sweep at a
# LoRa chirp's rate, so it is labelled JAM_SWEPT + LORA_CSS.
PARAM_KEYS = ("bandwidth_hz", "n_tones", "period_samples", "duty", "onset_sample",
              "delay_samples", "doppler_hz", "copied_radar", "lora_like_sweep")


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def band_noise(rng, bandwidth_hz, n=WINDOW_LEN):
    """Complex Gaussian noise filling `bandwidth_hz` around 0 Hz, unit power."""
    white = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    freqs = np.fft.fftfreq(n, d=1 / FS)
    mask = np.abs(freqs) <= bandwidth_hz / 2
    mask[0] = True                       # always keep at least the centre bin
    x = np.fft.ifft(np.fft.fft(white) * mask)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def _noise_jammer(bw_range):
    def build(rng, n):
        bw = _log_uniform(rng, *bw_range)
        return band_noise(rng, bw, n), dict(bandwidth_hz=bw)
    return build


def _tone(rng, n):
    sig = np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.ones(n)
    return sig, dict(bandwidth_hz=FS / WINDOW_LEN, n_tones=1)


def _multitone(rng, n):
    k = int(rng.integers(MULTITONE_N[0], MULTITONE_N[1] + 1))
    # Draw tones until they are all at least MULTITONE_MIN_SPACING_HZ apart.
    while True:
        f = np.sort(rng.uniform(-TONE_SPREAD_HZ, TONE_SPREAD_HZ, k))
        if np.all(np.diff(f) >= MULTITONE_MIN_SPACING_HZ):
            break
    f -= (f[0] + f[-1]) / 2                           # centre the comb on 0 Hz
    t = np.arange(n)
    amp = rng.uniform(0.5, 1.0, k)
    phase = rng.uniform(0, 2 * np.pi, k)
    sig = (amp[:, None] * np.exp(1j * (2 * np.pi * f[:, None] / FS * t + phase[:, None]))).sum(0)
    return sig, dict(bandwidth_hz=f[-1] - f[0] + MULTITONE_MIN_SPACING_HZ, n_tones=k)


def _sawtooth(rng, bw, period, n):
    """Sawtooth frequency sweep across `bw`, repeating every `period` samples,
    at a random point in its cycle."""
    t = np.arange(n)
    pos = ((t + rng.uniform(0, period)) % period) / period      # 0 .. 1
    if rng.random() < 0.5:
        pos = 1 - pos                                           # sweep down
    freq = (pos - 0.5) * bw / FS                                # cycles per sample
    return np.exp(1j * (2 * np.pi * np.cumsum(freq) + rng.uniform(0, 2 * np.pi)))


def _swept(rng, n):
    bw = rng.uniform(*SWEPT_BW_HZ)
    period = _log_uniform(rng, *SWEPT_PERIOD_SAMPLES)
    sig = _sawtooth(rng, bw, period, n)
    # One straight sweep at a LoRa chirp's rate looks like LoRa: labelled both
    # (src/data/labels.py). Measured on the signal, so no extra random draws.
    freq = np.diff(np.unwrap(np.angle(sig))) / (2 * np.pi) * FS
    return sig, dict(bandwidth_hz=bw, period_samples=period,
                     lora_like_sweep=float(spread.looks_like_lora_sweep(freq)))


def _chirp(rng, n):
    bw = rng.uniform(*CHIRP_BW_HZ)
    period = rng.uniform(*CHIRP_PERIOD_SAMPLES)
    return _sawtooth(rng, bw, period, n), dict(bandwidth_hz=bw, period_samples=period)


def _pulsed(rng, n):
    bw = _log_uniform(rng, *PULSED_BW_HZ)
    # Period at most a third of the window, so every window shows at least two
    # bursts -- one burst alone would look like JAM_REACTIVE switching on.
    period = int(rng.integers(PULSED_PERIOD_SAMPLES[0], PULSED_PERIOD_SAMPLES[1] + 1))
    burst = max(4, int(round(period * rng.uniform(*PULSED_DUTY))))
    t = np.arange(n)
    gate = ((t + rng.integers(0, period)) % period) < burst
    return band_noise(rng, bw, n) * gate, dict(bandwidth_hz=bw, period_samples=period,
                                               duty=burst / period)


def _reactive(rng, n):
    bw = _log_uniform(rng, *PULSED_BW_HZ)
    onset = int(rng.uniform(*REACTIVE_ONSET_FRAC) * n)
    gate = np.arange(n) >= onset
    return band_noise(rng, bw, n) * gate, dict(bandwidth_hz=bw, onset_sample=onset)


def _drfm(rng, n):
    # One radar pulse, reusing the radar generator so the pulse shapes match.
    while True:
        copied = str(rng.choice(DRFM_RADAR_CLASSES))
        pulse, p = radar._pulse(copied, rng)
        if len(pulse) <= DRFM_MAX_PULSE_SAMPLES:
            break
    delay = int(rng.integers(DRFM_DELAY_SAMPLES[0], DRFM_DELAY_SAMPLES[1] + 1))
    doppler = rng.uniform(*DRFM_DOPPLER_HZ) * rng.choice((-1, 1))
    gain = 10 ** (rng.uniform(*DRFM_GAIN_DB) / 20)
    bits = int(rng.choice(DRFM_PHASE_BITS))

    # The jammer's copy: phase quantised to `bits`, Doppler shifted, louder.
    levels = 2 ** bits
    q_phase = np.round(np.angle(pulse) / (2 * np.pi / levels)) * (2 * np.pi / levels)
    m = np.arange(len(pulse))
    copy = gain * np.abs(pulse) * np.exp(1j * (q_phase + 2 * np.pi * doppler / FS * m))

    sig = np.zeros(n, dtype=complex)
    if n <= WINDOW_LEN:
        # One window: the pulse and its copy, frozen at one instant.
        start = int(rng.integers(0, n - len(pulse) - delay + 1))
        sig[start:start + len(pulse)] += pulse
        sig[start + delay:start + delay + len(pulse)] += copy
    else:
        # A burst spans many pulses: the pulse repeats every PRI and the copy
        # walks away from it (range-gate pull-off), its delay growing pulse by pulse.
        pull = int(rng.integers(DRFM_PULL_SAMPLES[0], DRFM_PULL_SAMPLES[1] + 1))
        busy = len(pulse) + delay + pull
        pri = int(rng.integers(busy + 8, busy + DRFM_PRI_EXTRA_SAMPLES + 1))
        starts = np.arange(int(rng.integers(0, pri)), n - busy + 1, pri)
        if len(starts) == 0:
            starts = np.array([0])
        for j, start in enumerate(starts):
            d = delay + int(round(pull * j / max(len(starts) - 1, 1)))
            seg = min(len(pulse), n - start)
            sig[start:start + seg] += pulse[:seg]
            seg = min(len(pulse), n - start - d)
            if seg > 0:
                sig[start + d:start + d + seg] += copy[:seg]
    sig *= np.exp(1j * rng.uniform(0, 2 * np.pi))
    return sig, dict(bandwidth_hz=p["bandwidth_hz"] + abs(doppler), delay_samples=delay,
                     doppler_hz=doppler, copied_radar=float(radar.CLASSES.index(copied)))


def _noise_floor(rng, n):
    return band_noise(rng, FS, n), {}


_BUILDERS = {
    "JAM_BARRAGE": _noise_jammer(BARRAGE_BW_HZ),
    "JAM_SPOT": _noise_jammer(SPOT_BW_HZ),
    "JAM_PARTIAL_BAND": _noise_jammer(PARTIAL_BW_HZ),
    "JAM_TONE": _tone,
    "JAM_MULTITONE": _multitone,
    "JAM_SWEPT": _swept,
    "JAM_CHIRP": _chirp,
    "JAM_PULSED": _pulsed,
    "JAM_REACTIVE": _reactive,
    "JAM_DRFM": _drfm,
    "NOISE_FLOOR": _noise_floor,
}


def _build(cls, rng, n_samples):
    if cls not in _BUILDERS:
        raise ValueError(f"{cls!r} is not a jam class; choose from {CLASSES}")
    sig, built = _BUILDERS[cls](rng, n_samples)
    params = dict.fromkeys(PARAM_KEYS, np.nan)
    params.update(built)
    return sig / np.sqrt(np.mean(np.abs(sig) ** 2)), params


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit average power) -- used by the
    tests and by the impairment module. NOISE_FLOOR ignores snr_db: it is
    noise already.
    """
    sig, params = _build(cls, rng, WINDOW_LEN)
    if snr_db is not None and cls != "NOISE_FLOOR":
        # "active": pulsed / reactive / DRFM windows are partly silent.
        sig = add_awgn(sig, snr_db, rng=rng)
    return sig.astype(np.complex64), params


def generate_burst(cls, n_samples, rng):
    """A clean emitter of class `cls` lasting n_samples with its parameters
    fixed for the whole burst (unit average power, centred on 0 Hz) -- for the
    detector's multi-millisecond scenes (src/detect/scenes.py). With
    n_samples == WINDOW_LEN it is exactly generate_window(cls, None, rng)."""
    return _build(cls, rng, n_samples)


def generate(cls, n, snr_db, rng):
    """The shared generator signature: `n` windows as (complex64 (n, WINDOW_LEN),
    meta) where meta holds one array per parameter in PARAM_KEYS."""
    iq = np.empty((n, WINDOW_LEN), dtype=np.complex64)
    meta = {k: np.empty(n) for k in PARAM_KEYS}
    for i in range(n):
        iq[i], params = generate_window(cls, snr_db, rng)
        for k, v in params.items():
            meta[k][i] = v
    return iq, meta
