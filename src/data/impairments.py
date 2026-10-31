"""
Shared impairment ("realism") module: what a real channel and receiver do to
a clean generated window, applied the same way to every team's classes.

Every generator in src/generators/ produces a clean window centred on 0 Hz
(generate_window(cls, None, rng)). This module turns it into what a receiver
would record, in the order things happen physically:

  channel   fading / multipath -> placement in the band -> Doppler drift ->
            sample-clock error
  noise     AWGN at the labelled SNR (white or coloured), impulsive bursts,
            optional weak neighbouring interference
  receiver  phase noise -> IQ imbalance -> DC offset -> front-end filter ->
            AGC gain jump -> ADC quantisation and clipping

Use it like this (the shard scripts' one call per window):

    iq, info = impairments.generate_impaired(radar, "LFM_RADAR", snr_db=6, rng=rng)

Each impairment is applied with its own probability (PROFILE), so the data
still contains clean-ish windows. `info` records every value drawn; with the
same rng seed the window is reproduced exactly.

SNR: noise is set against the clean signal's power WHILE ON (the "active"
convention of src/data/preprocess.add_awgn), measured BEFORE fading, so the
label is the average SNR and a deep fade makes a window genuinely harder.

Scale honesty for a 160 us window (512 samples at 3.2 MS/s): a 50 ppm clock
error moves the last sample by 0.03 of a sample, and a 5 kHz/s Doppler drift
changes the frequency by under 1 Hz. Both are kept because the plan lists them
and they start to matter for longer clips (the detector + extraction design);
at 160 us they are close to no-ops, and the tests check them on purpose-built
signals rather than pretending otherwise.

External windows (RadioML, RadChar ...) already carry their own channel
effects and noise: pass snr_db=None and skip=("fading",) so they are not
double-impaired (README "Shared data workflow").
"""
import numpy as np
from scipy.signal import firwin, lfilter

from src.config import CFG

FS = CFG["signal"]["fs"]
WINDOW_LEN = CFG["signal"]["window_len"]
NYQUIST = FS / 2

# Probability that each impairment is applied to a window, and its range.
PROFILE = {
    "fading":          dict(p=0.5, kinds=("rician", "rayleigh"), k_db=(0, 10),
                            n_taps=(1, 4), max_delay_samples=10),
    "placement":       dict(p=1.0, band_fraction=0.95),
    "drift":           dict(p=0.3, rate_hz_per_s=(-5e3, 5e3)),
    "clock":           dict(p=0.5, ppm=(-50, 50)),
    "coloured_noise":  dict(p=0.3, ar=(-0.7, 0.7)),
    "impulsive":       dict(p=0.1, rate=(1e-3, 1e-2), level_db=(10, 20)),
    "interference":    dict(p=0.1, level_db=(-30, -15), bandwidth_hz=(20e3, 200e3)),
    "phase_noise":     dict(p=0.5, linewidth_hz=(10, 5e3)),
    "iq_imbalance":    dict(p=0.5, gain_db=(0, 1), phase_deg=(0, 5)),
    "dc_offset":       dict(p=0.5, level=(0, 0.3)),
    "frontend_filter": dict(p=0.5, cutoff=(0.8, 0.98), n_taps=(15, 63)),
    "agc_jump":        dict(p=0.1, step_db=(-10, 10)),
    "adc":             dict(p=1.0, bits=(8, 12, 14), backoff_db=(-3, 15)),
}


# --- helpers ------------------------------------------------------------------

def active_power(x):
    """Mean power over samples within 20 dB of the peak (preprocess.add_awgn's
    "active" rule), so a pulsed signal's power is its power while on."""
    mag_sq = np.abs(x) ** 2
    peak = mag_sq.max()
    return float(mag_sq[mag_sq > 0.01 * peak].mean()) if peak > 0 else 0.0


def occupied_bandwidth(x, fraction=0.99):
    """Two-sided bandwidth (Hz) around 0 Hz holding `fraction` of the power --
    for generators that do not report their own bandwidth."""
    spec = np.abs(np.fft.fft(x)) ** 2
    f = np.abs(np.fft.fftfreq(len(x), 1 / FS))
    order = np.argsort(f)
    cum = np.cumsum(spec[order]) / spec.sum()
    return 2 * float(f[order][min(np.searchsorted(cum, fraction), len(cum) - 1)])


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _pad_filter(x, taps):
    """Zero-phase FIR (odd symmetric taps), reflect-padded so the window edges
    are not tapered by imaginary silence outside the window."""
    half = len(taps) // 2
    padded = np.pad(x, half, mode="reflect")
    return np.convolve(padded, taps, mode="valid")


# --- channel ------------------------------------------------------------------

def multipath(x, taps, delays):
    """Sum of delayed, complex-weighted copies (a tapped delay line). Over 160 us
    a channel is effectively static, so the taps are constant in the window."""
    out = np.zeros(len(x), dtype=complex)
    for h, d in zip(taps, delays):
        out[d:] += h * x[:len(x) - d]
    return out


def draw_fading(rng, kind, k_db, n_taps, max_delay):
    """Taps and delays with E[sum |h|^2] = 1. Rician: the first tap holds a
    fixed line-of-sight part with K-factor k_db over the scattered power."""
    delays = np.sort(np.concatenate(([0], rng.choice(np.arange(1, max_delay + 1),
                                                       n_taps - 1, replace=False))))
    power = np.exp(-delays / max(max_delay / 3, 1))                 # exponential profile
    power = power / power.sum()
    taps = np.sqrt(power / 2) * (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps))
    if kind == "rician":
        k = 10 ** (k_db / 10)
        los = np.sqrt(power[0] * k / (k + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        taps[0] = los + taps[0] * np.sqrt(1 / (k + 1))
    return taps, delays


def frequency_shift(x, offset_hz, drift_hz_per_s=0.0):
    """Move the signal to offset_hz, optionally drifting linearly (Doppler)."""
    t = np.arange(len(x)) / FS
    return x * np.exp(2j * np.pi * (offset_hz * t + 0.5 * drift_hz_per_s * t ** 2))


def clock_error(x, ppm):
    """Resample as a receiver clock running ppm parts per million fast would:
    every frequency in the window scales by (1 + ppm * 1e-6)."""
    t = np.arange(len(x)) * (1 + ppm * 1e-6)
    return np.interp(t, np.arange(len(x)), x.real) + 1j * np.interp(t, np.arange(len(x)), x.imag)


# --- noise --------------------------------------------------------------------

def noise(n, power, rng, ar=0.0):
    """Complex Gaussian noise of the given mean power. ar != 0 colours it with a
    first-order filter (same power, tilted spectrum)."""
    extra = 64 if ar else 0
    w = (rng.standard_normal(n + extra) + 1j * rng.standard_normal(n + extra)) / np.sqrt(2)
    if ar:
        w = lfilter([np.sqrt(1 - ar ** 2)], [1, -ar], w)[extra:]   # drop the start-up
    return np.sqrt(power) * w


def impulses(n, noise_power, rng, rate, level_db):
    """Sparse bursts (switching transients, ignition, lightning) level_db above
    the noise floor, each one sample long."""
    hit = rng.random(n) < rate
    amp = np.sqrt(noise_power * 10 ** (level_db / 10) / 2)
    return hit * amp * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def interference(n, signal_power, rng, level_db, bandwidth_hz, centre_hz):
    """A weak band of noise somewhere else in the band: a crowded spectrum's
    background, deliberately not shaped like any class."""
    x = noise(n, 1.0, rng)
    taps = firwin(31, bandwidth_hz / 2, fs=FS)
    x = _pad_filter(x, taps)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2)) * np.sqrt(signal_power * 10 ** (level_db / 10))
    return frequency_shift(x, centre_hz)


# --- receiver -----------------------------------------------------------------

def phase_noise(x, linewidth_hz, rng):
    """Oscillator phase noise as a random walk (Wiener process) of the given
    3 dB linewidth: per-sample step std sqrt(2 pi linewidth / FS)."""
    steps = rng.standard_normal(len(x)) * np.sqrt(2 * np.pi * linewidth_hz / FS)
    return x * np.exp(1j * np.cumsum(steps))


def iq_imbalance(x, gain_db, phase_deg):
    """Q branch with gain error g and phase error phi relative to I:
    Q' = g * (Q cos phi + I sin phi). A tone at +f gains an image at -f."""
    g, phi = 10 ** (gain_db / 20), np.deg2rad(phase_deg)
    return x.real + 1j * g * (x.imag * np.cos(phi) + x.real * np.sin(phi))


def image_rejection_db(gain_db, phase_deg):
    """Image-to-signal power ratio produced by iq_imbalance (negative dB)."""
    g, phi = 10 ** (gain_db / 20), np.deg2rad(phase_deg)
    return 10 * np.log10((1 + g ** 2 - 2 * g * np.cos(phi)) / (1 + g ** 2 + 2 * g * np.cos(phi)))


def frontend_filter(x, cutoff_fraction, n_taps):
    """Anti-alias low-pass with its edge at cutoff_fraction of Nyquist: the
    band edges droop as they do on a real front end."""
    n_taps = n_taps | 1                                 # odd -> zero phase
    return _pad_filter(x, firwin(n_taps, cutoff_fraction * NYQUIST, fs=FS))


def agc_jump(x, at, step_db, ramp=4):
    """The AGC re-settles mid-window: gain steps by step_db from sample `at`."""
    gain = np.ones(len(x))
    target = 10 ** (step_db / 20)
    end = min(at + ramp, len(x))
    gain[at:end] = np.linspace(1, target, end - at + 1)[1:]
    gain[end:] = target
    return x * gain


def adc(x, bits, backoff_db):
    """Quantise I and Q to `bits` with full scale backoff_db above the signal's
    RMS. Negative backoff overdrives the converter, which clips."""
    full_scale = np.sqrt(np.mean(np.abs(x) ** 2)) * 10 ** (backoff_db / 20)
    step = 2 * full_scale / 2 ** bits

    def q(v):
        return np.round(np.clip(v, -full_scale, full_scale) / step) * step

    return q(x.real) + 1j * q(x.imag)


# --- the one call -------------------------------------------------------------

def apply(clean, snr_db, rng, bandwidth_hz=None, profile=None, skip=()):
    """Impair one clean window. Returns (complex64 window, info dict).

    snr_db=None adds no noise (external data that already has its own).
    bandwidth_hz: the signal's two-sided width, so placement keeps it inside
    the band; measured from the clean window when not given.
    skip: impairment names to leave out for this window.
    """
    prof = profile or PROFILE
    x = np.asarray(clean, dtype=complex)
    info = {}

    def on(name):
        return name not in skip and rng.random() < prof[name]["p"]

    ref_power = active_power(x)
    bw = bandwidth_hz if bandwidth_hz is not None and np.isfinite(bandwidth_hz) \
        else occupied_bandwidth(x)
    info["bandwidth_hz"] = bw

    # channel
    if on("fading"):
        c = prof["fading"]
        kind = str(rng.choice(c["kinds"]))
        n_taps = int(rng.integers(c["n_taps"][0], c["n_taps"][1] + 1))
        taps, delays = draw_fading(rng, kind, rng.uniform(*c["k_db"]), n_taps,
                                   c["max_delay_samples"])
        x = multipath(x, taps, delays)
        info.update(fading=kind, fading_taps=n_taps, fading_gain_db=10 * np.log10(np.sum(np.abs(taps) ** 2)))
    offset = drift = 0.0
    if on("placement"):
        room = max(0.0, prof["placement"]["band_fraction"] * NYQUIST - bw / 2)
        offset = rng.uniform(-room, room)
    if on("drift"):
        drift = rng.uniform(*prof["drift"]["rate_hz_per_s"])
    if offset or drift:
        x = frequency_shift(x, offset, drift)
    info.update(offset_hz=offset, drift_hz_per_s=drift)
    if on("clock"):
        ppm = rng.uniform(*prof["clock"]["ppm"])
        x = clock_error(x, ppm)
        info["clock_ppm"] = ppm

    # noise
    if snr_db is not None:
        noise_power = ref_power / 10 ** (snr_db / 10)
        ar = rng.uniform(*prof["coloured_noise"]["ar"]) if on("coloured_noise") else 0.0
        x = x + noise(len(x), noise_power, rng, ar)
        info.update(snr_db=snr_db, noise_ar=ar)
        if on("impulsive"):
            c = prof["impulsive"]
            rate, level = _log_uniform(rng, *c["rate"]), rng.uniform(*c["level_db"])
            x = x + impulses(len(x), noise_power, rng, rate, level)
            info.update(impulse_rate=rate, impulse_db=level)
    if on("interference"):
        c = prof["interference"]
        level, ibw = rng.uniform(*c["level_db"]), rng.uniform(*c["bandwidth_hz"])
        centre = rng.uniform(-(0.95 * NYQUIST - ibw / 2), 0.95 * NYQUIST - ibw / 2)
        x = x + interference(len(x), ref_power, rng, level, ibw, centre)
        info.update(interference_db=level, interference_hz=centre)

    # receiver
    if on("phase_noise"):
        lw = _log_uniform(rng, *prof["phase_noise"]["linewidth_hz"])
        x = phase_noise(x, lw, rng)
        info["phase_noise_linewidth_hz"] = lw
    if on("iq_imbalance"):
        gdb, pdeg = rng.uniform(*prof["iq_imbalance"]["gain_db"]), rng.uniform(*prof["iq_imbalance"]["phase_deg"])
        x = iq_imbalance(x, gdb, pdeg)
        info.update(iq_gain_db=gdb, iq_phase_deg=pdeg)
    if on("dc_offset"):
        dc = rng.uniform(*prof["dc_offset"]["level"]) * np.sqrt(ref_power) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        x = x + dc
        info["dc_offset"] = abs(dc)
    if on("frontend_filter"):
        c = prof["frontend_filter"]
        cut, nt = rng.uniform(*c["cutoff"]), int(rng.integers(c["n_taps"][0], c["n_taps"][1] + 1))
        x = frontend_filter(x, cut, nt)
        info.update(filter_cutoff=cut, filter_taps=nt | 1)
    if on("agc_jump"):
        at, step = int(rng.integers(len(x) // 8, len(x) - len(x) // 8)), rng.uniform(*prof["agc_jump"]["step_db"])
        x = agc_jump(x, at, step)
        info.update(agc_at=at, agc_step_db=step)
    if on("adc"):
        bits, backoff = int(rng.choice(prof["adc"]["bits"])), rng.uniform(*prof["adc"]["backoff_db"])
        x = adc(x, bits, backoff)
        info.update(adc_bits=bits, adc_backoff_db=backoff)

    return x.astype(np.complex64), info


def generate_impaired(gen, cls, snr_db, rng, profile=None, skip=()):
    """Clean window from a generator module (radar, spread, digital ...), then
    apply(). Returns (complex64 window, info) with the generator's own params
    merged in under "gen_<name>"."""
    clean, params = gen.generate_window(cls, None, rng)
    bw = params.get("bandwidth_hz")
    if bw is None and np.isfinite(params.get("sps", np.nan)):
        bw = FS / params["sps"] * (1 + (params.get("beta") if np.isfinite(params.get("beta", np.nan)) else 1.0))
    iq, info = apply(clean, snr_db, rng, bandwidth_hz=bw, profile=profile, skip=skip)
    info.update({f"gen_{k}": v for k, v in params.items()})
    return iq, info
