"""DSP measurement over a raw IQ capture.

Everything here is MEASURED -- computed from the samples themselves, with no
model involved. Under the UI's provenance rule these values are rendered in
neutral instrument styling and are never coloured as detections. Keeping them
in a separate module from src/timeline.py makes that separation structural.

Requires a capture that has NOT been through preprocess_window: that function
normalizes to zero mean and unit variance, which destroys the absolute
amplitude every function here depends on.
"""
from dataclasses import dataclass

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


# --- Constellation order (16QAM vs 64QAM) via the fourth-order cumulant ----
#
# The classifier cannot make this call at all, not just poorly: choosing the
# larger of its two probabilities is right 51.4% of the time on one window,
# still a coin flip at 49.7% after averaging 64 windows at SNR >= +2 dB, and
# the error is a BIAS, not noise -- 16QAM is called correctly only 47.0% of
# the time (worse than chance) while 64QAM gets 56.9% -- which is exactly why
# averaging more windows of the model's own output cannot help. The reported
# per-class recalls for these two classes are therefore measuring "dense QAM
# detected, split by a coin flip", not two distinguishable classes.
#
# |C42|, the normalised fourth-order cumulant of the recovered symbols, DOES
# carry the distinction the model cannot make. For a zero-mean, unit-average
# -power complex constellation,
#
#     C42 = E[|s|^4] - 2 * (E[|s|^2])^2
#
# (E[s^2] drops out: every square QAM constellation is symmetric enough that
# it is exactly zero). C42 is the excess of the constellation's own 4th
# moment over what a circularly-symmetric Gaussian of the same power would
# have -- zero for Gaussian noise, and a class-specific non-zero constant for
# a real digital constellation, which is what makes it useful here: unlike
# the model's output, its error is VARIANCE (finite-sample estimation noise
# from a limited number of recovered symbols), not bias, so it averages down
# over windows the way the model's guess never could.
C42_THEORY = {"16QAM": 0.619, "64QAM": 0.5745}

# The decision boundary a caller compares one window's (or a pooled average
# of many windows') |C42| against, to say "nearer 16QAM" or "nearer 64QAM".
# This is the MIDPOINT OF THE TWO THEORY VALUES ABOVE, and nothing else --
# in particular, it is NOT fit to this project's own measured means (0.617
# +/- 0.093 for 16QAM, 0.569 +/- 0.104 for 64QAM, 450 windows/class at SNR
# >= +2 dB, 56 recovered symbols/window). Those measured means independently
# average to an empirical midpoint of 0.593, against this theoretical
# midpoint of 0.597 -- close, but NOT identical, because they were never
# forced to be. That agreement is the evidence the estimator is calibrated:
# a boundary FITTED to the measured 0.593 would have destroyed exactly the
# check that this agreement provides, by construction making the "boundary
# matches the data" observation true by definition rather than by
# measurement. Do not "improve" this by plugging in the measured means --
# see test_constellation_order_boundary_is_the_theoretical_midpoint, which
# pins the arithmetic so a well-meaning refactor cannot quietly do that.
C42_BOUNDARY = (C42_THEORY["16QAM"] + C42_THEORY["64QAM"]) / 2

# Below this many pooled windows, constellation_order refuses to decide
# rather than guess. 8 is not an arbitrary round number: measured accuracy
# of the midpoint rule at exactly 8 pooled windows is 0.759 (SNR >= +2 dB,
# see C42_POOLED_ACCURACY below) -- "only just useful". One window alone is
# 0.595, barely better than the coin flip this estimator exists to replace;
# refusing below 8 keeps every decision this function returns backed by at
# least the first pooling level that is actually worth reporting.
MIN_WINDOWS_FOR_C42_DECISION = 8

# Measured accuracy of the C42-midpoint decision rule after pooling N
# windows' |C42| by simple averaging. 450 windows/class, 56 recovered
# symbols/window, SNR >= +2 dB -- the regime this estimator is meant for.
# Keyed by exact pooled-window counts that were actually measured;
# _pooled_accuracy below reports the accuracy of the largest measured count
# that does not exceed the number actually pooled, never extrapolating past
# what was measured.
C42_POOLED_ACCURACY = {
    1: 0.595, 2: 0.641, 4: 0.698, 8: 0.759, 16: 0.845, 32: 0.914, 64: 0.975,
}

# Same measurement, but over EVERY SNR in the test split rather than only
# SNR >= +2 dB -- kept alongside the table above for a caller or reader who
# wants the honest floor rather than the best case. Not used to compute
# C42_POOLED_ACCURACY's values above; a low-SNR window's |C42| is closer to
# 0 (noise dilutes the constellation's own moment toward the Gaussian value)
# than to either theory constant, so pooling windows that span every SNR is
# a strictly harder problem than pooling only the ones at SNR >= +2 dB.
C42_POOLED_ACCURACY_ALL_SNR = {
    1: 0.544, 2: 0.549, 4: 0.590, 8: 0.610, 16: 0.647, 32: 0.725, 64: 0.798,
}


@dataclass
class ConstellationOrderEstimate:
    """Result of constellation_order: enough for a caller to judge the
    answer, not just take it on faith.

    decision:   "16QAM", "64QAM", or None (refused -- see
                MIN_WINDOWS_FOR_C42_DECISION).
    mean_c42:   the pooled |C42| average the decision (or refusal) was based
                on. NaN when zero usable windows were pooled.
    n_windows:  how many windows actually contributed a usable |C42| --
                degenerate windows (recover_symbols returns them unchanged,
                see that function's docstring) are silently excluded, so
                this can be smaller than len(windows).
    margin:     abs(mean_c42 - C42_BOUNDARY). Distance from the boundary,
                not a probability -- useful for telling a comfortable call
                apart from one that landed a hair on the right side of the
                line.
    accuracy:   the measured accuracy of the midpoint rule at this many
                pooled windows (C42_POOLED_ACCURACY, SNR >= +2 dB), or None
                if n_windows is 0.
    """
    decision: object
    mean_c42: float
    n_windows: int
    margin: float
    accuracy: object


def _normalized_c42(points):
    """|C42| of a set of recovered symbol points, normalised to unit average
    power so absolute amplitude (an AGC/scaling artefact, not information
    about the constellation) cannot move the result. Returns None for an
    empty set or a set with zero power -- both mean there is nothing here to
    measure, not a C42 of 0."""
    points = np.asarray(points)
    if len(points) == 0:
        return None
    m2 = float(np.mean(np.abs(points) ** 2))
    if m2 <= 0:
        return None
    p = points / np.sqrt(m2)
    m2n = float(np.mean(np.abs(p) ** 2))
    m4n = float(np.mean(np.abs(p) ** 4))
    return abs(m4n - 2.0 * m2n ** 2)


def _pooled_accuracy(n_windows):
    measured = [k for k in C42_POOLED_ACCURACY if k <= n_windows]
    if not measured:
        return None
    return C42_POOLED_ACCURACY[max(measured)]


def constellation_order(windows, min_windows=MIN_WINDOWS_FOR_C42_DECISION):
    """Resolve 16QAM vs 64QAM by pooling the fourth-order cumulant across
    `windows` -- a distinction the classifier cannot make at all (see the
    module-level comment above C42_THEORY for the measured numbers proving
    that: 51.4% single-window accuracy picking the larger probability,
    49.7% even after averaging 64 windows, and biased rather than noisy so
    averaging the model's own output cannot fix it).

    For each window: recover symbols with the existing recover_symbols
    (src/ui/plots.py -- unit-power scale, matched filter, de-rotate,
    decimate; imported inside this function, not at module level, because
    src.ui.plots itself imports from this module and a top-level import
    here would be circular), then take the normalised |C42| of the
    recovered points (_normalized_c42). Windows recover_symbols could not
    process (too short, or zero power -- see that function's docstring) are
    silently skipped rather than counted as a 0.0 measurement, which would
    quietly bias the pooled average toward "noise-like".

    The per-window |C42| values are then averaged (simple mean -- the whole
    point of pooling is that this estimator's error is VARIANCE, not the
    model's BIAS, so a plain average is exactly what should shrink that
    error as more windows are added) and compared to C42_BOUNDARY, the
    midpoint of the two THEORETICAL constants in C42_THEORY. See that
    constant's own comment for why the boundary is theoretical and must
    never be fit to this project's own measured means -- the empirical
    midpoint of real measured data (0.593) landing close to but not
    exactly on the theoretical one (0.597) is the calibration check a
    fitted boundary would destroy.

    Refuses rather than guesses: below `min_windows` (default
    MIN_WINDOWS_FOR_C42_DECISION, 8 -- see that constant's comment for why)
    `decision` in the returned ConstellationOrderEstimate is None, even
    though mean_c42/n_windows/margin/accuracy are still populated so the
    caller can see what WAS measured. A caller must never be able to dress
    up a 2-window guess (measured 64.1% accuracy -- barely better than the
    64.1%-vs-59.5% gap over a single window) as an answer.

    MEASURED, not MODEL: every number this function touches comes from the
    capture's own recovered samples, with no model involvement anywhere in
    the chain, and it fits nothing to an expected answer -- the boundary is
    theory, not a fit. It must never take a tier colour on screen; TEXT_DIM
    /INSTRUMENT styling only, exactly like everything else in src/measure.py.
    """
    from src.ui.plots import recover_symbols

    per_window_c42 = []
    for window in windows:
        points, _, _ = recover_symbols(window)
        c42 = _normalized_c42(points)
        if c42 is not None:
            per_window_c42.append(c42)

    n = len(per_window_c42)
    if n == 0:
        return ConstellationOrderEstimate(
            decision=None, mean_c42=float("nan"), n_windows=0,
            margin=float("nan"), accuracy=None)

    mean_c42 = float(np.mean(per_window_c42))
    margin = abs(mean_c42 - C42_BOUNDARY)
    accuracy = _pooled_accuracy(n)

    decision = None
    if n >= min_windows:
        dist_16 = abs(mean_c42 - C42_THEORY["16QAM"])
        dist_64 = abs(mean_c42 - C42_THEORY["64QAM"])
        decision = "16QAM" if dist_16 <= dist_64 else "64QAM"

    return ConstellationOrderEstimate(
        decision=decision, mean_c42=mean_c42, n_windows=n, margin=margin,
        accuracy=accuracy)
