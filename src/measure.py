"""DSP measurement over a raw IQ capture.

Everything here is MEASURED -- computed from the samples themselves, with no
model involved. Under the UI's provenance rule these values are rendered in
neutral instrument styling and are never coloured as detections. Keeping them
in a separate module from src/timeline.py makes that separation structural.

Requires a capture that has NOT been through preprocess_window: that function
normalizes to zero mean and unit variance, which destroys the absolute
amplitude every function here depends on.
"""
import numpy as np
from scipy.signal import stft

from src.config import CFG

_EPS = 1e-20


def _frame_powers(iq, frame_len=512):
    """Mean power per non-overlapping frame."""
    n_frames = max(len(iq) // frame_len, 1)
    frames = np.asarray(iq)[:n_frames * frame_len].reshape(n_frames, frame_len)
    return np.mean(np.abs(frames) ** 2, axis=1)


def noise_floor_power(iq, percentile=10.0, frame_len=512):
    """Noise power estimated from the quietest frames of the capture.

    Uses a low percentile rather than the minimum so a single anomalously
    quiet frame cannot set the floor, and so a capture that is mostly quiet
    still yields a stable estimate. Assumes the capture contains SOME region
    without a strong emitter -- true for scenario captures by construction,
    and typical of real recordings.
    """
    return float(max(np.percentile(_frame_powers(iq, frame_len), percentile),
                     _EPS))


def estimate_snr_db(window_iq, noise_power):
    """Estimated SNR of one window, in dB.

    Subtracts the noise floor before taking the ratio: total window power is
    signal PLUS noise, so 10*log10(total / noise) reads +3 dB for a window
    that actually sits at 0 dB SNR. Subtracting first gives an unbiased
    estimate.

    ALWAYS an estimate. The UI must render the result with a visible `est.`
    prefix -- the classifier does not produce SNR, and this is not a
    calibrated receiver measurement.
    """
    total = float(np.mean(np.abs(np.asarray(window_iq)) ** 2))
    signal = max(total - noise_power, _EPS)
    return float(10.0 * np.log10(signal / max(noise_power, _EPS)))


def occupancy(iq, nperseg=256, margin_db=6.0):
    """Fraction of time-frequency cells sitting above the noise floor.

    This is a MEASURED spectrum-occupancy figure. It deliberately replaces the
    "Channel Load" readout an OmniSIG-style console would show: the obvious
    implementation of that name here (fraction of windows where a class fired)
    would be MODEL output wearing a measurement's name, which the provenance
    rule forbids.
    """
    _, _, Z = stft(np.asarray(iq), nperseg=nperseg, return_onesided=False)
    power = np.abs(Z) ** 2
    # Median, not a low percentile. Noise power per cell is exponentially
    # distributed, so the 10th percentile sits far below the mean and most
    # noise cells clear a 6 dB margin above it -- which reported ~66%
    # occupancy on pure noise. Against the median, P(noise > 4x median) is
    # exp(-4 * ln2) ~ 6%, which is the right order for an empty band.
    floor = max(np.median(power), _EPS)
    return float(np.mean(power > floor * 10 ** (margin_db / 10.0)))


def power_spectrum_db(iq, fs=None, nperseg=1024):
    """Average power spectrum in dB, over the full complex band.

    Returns (freqs_hz, spectrum_db), both fftshifted so frequency runs
    -fs/2 .. +fs/2 -- the capture is complex baseband, so the negative half is
    real signal, not a mirror.
    """
    fs = fs or CFG["signal"]["fs"]
    f, _, Z = stft(np.asarray(iq), fs=fs, nperseg=nperseg,
                    return_onesided=False)
    mean_power = np.mean(np.abs(Z) ** 2, axis=1)
    freqs = np.fft.fftshift(f)
    spectrum = 10.0 * np.log10(np.fft.fftshift(mean_power) + _EPS)
    return freqs, spectrum
