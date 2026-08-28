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
C42_THEORY = {"16QAM": 0.680, "64QAM": 0.619}

# The decision boundary a caller compares one window's (or a pooled
# average of many windows') |C42| against, to say "16QAM" or "64QAM".
#
# This is a CALIBRATED constant, not the midpoint of C42_THEORY -- that was
# tried first and is wrong for this estimator. Loaded from
# configs/default.yaml (constellation_order.c42_boundary) exactly the way
# every other calibrated decision threshold in this project is loaded (see
# resolve_multilabel_thresholds in src/config.py), so a recalibration is a
# one-line config edit, not a source hunt. See that config entry's own
# comment for the full validation/test numbers and for why a boundary
# derived from the noiseless theory constants would misclassify almost
# everything: channel noise pulls |C42| toward the Gaussian limit of zero,
# so real recovered symbols sit BELOW their noiseless value, and at the SNR
# this estimator operates at BOTH classes measure under the theoretical
# midpoint of 0.6495 -- a theory-derived boundary would call everything
# 64QAM. C42_THEORY above stays useful as a sanity check against physics
# (see test_c42_theory_matches_noiseless_synthetic_constellations), it is
# just not where the boundary comes from.
C42_BOUNDARY = float(CFG["constellation_order"]["c42_boundary"])

# Below this many pooled windows, constellation_order refuses to decide
# rather than guess. 8 is not an arbitrary round number: measured accuracy
# of the boundary rule at exactly 8 pooled windows is 0.754 (SNR >= +2 dB,
# see C42_POOLED_ACCURACY below) -- "only just useful". One window alone is
# 0.593, barely better than the coin flip this estimator exists to replace;
# refusing below 8 keeps every decision this function returns backed by at
# least the first pooling level that is actually worth reporting.
MIN_WINDOWS_FOR_C42_DECISION = 8

# C42_BOUNDARY was calibrated at SNR >= +2 dB (see its own comment and
# configs/default.yaml) and is only valid there: measured directly, the
# SAME boundary applied to windows spanning every SNR in the test split
# (not just >= +2 dB) gets 0.548 accuracy pooling a single window and 0.815
# even pooling 64 -- both far below the >= +2 dB numbers at the same
# pooling level, because a low-SNR window's |C42| sits closer to 0 (the
# Gaussian limit) than to either class. constellation_order refuses the
# decision outright below this SNR rather than silently extrapolating past
# what the boundary was ever validated against.
C42_MIN_SNR_DB = 2.0

# Measured accuracy of the C42-boundary decision rule after pooling N
# windows' |C42| by simple averaging. 450 windows/class, 56 recovered
# symbols/window, SNR >= +2 dB -- the regime C42_BOUNDARY was calibrated
# for and C42_MIN_SNR_DB gates on. Keyed by exact pooled-window counts that
# were actually measured; _pooled_accuracy below reports the accuracy of
# the largest measured count that does not exceed the number actually
# pooled, never extrapolating past what was measured.
C42_POOLED_ACCURACY = {
    1: 0.593, 8: 0.754, 16: 0.843, 32: 0.921, 64: 0.980,
}


@dataclass
class ConstellationOrderEstimate:
    """Result of constellation_order: enough for a caller to judge the
    answer, not just take it on faith.

    decision:   "16QAM", "64QAM", or None (refused -- either fewer than
                MIN_WINDOWS_FOR_C42_DECISION windows pooled, or the pooled
                estimated SNR is below C42_MIN_SNR_DB, the regime
                C42_BOUNDARY was calibrated for).
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
    accuracy:   the measured accuracy of the boundary rule at this many
                pooled windows (C42_POOLED_ACCURACY, SNR >= +2 dB), or None
                if n_windows is 0.
    snr_db:     estimate_snr_db's estimate over every sample pooled into
                this measurement (an ESTIMATE, like everywhere else that
                function is used). None only when n_windows is 0. The
                reason for a None decision when it sits below
                C42_MIN_SNR_DB despite n_windows clearing the minimum.
    """
    decision: object
    mean_c42: float
    n_windows: int
    margin: float
    accuracy: object
    snr_db: object


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


def constellation_order(windows, noise_power,
                          min_windows=MIN_WINDOWS_FOR_C42_DECISION,
                          min_snr_db=C42_MIN_SNR_DB):
    """Resolve 16QAM vs 64QAM by pooling the fourth-order cumulant across
    `windows` -- a distinction the classifier cannot make at all (see the
    module-level comment above C42_THEORY for the measured numbers proving
    that: 51.4% single-window accuracy picking the larger probability,
    49.7% even after averaging 64 windows, and biased rather than noisy so
    averaging the model's own output cannot fix it).

    `noise_power` is the capture's own noise floor (e.g.
    CaptureSession.noise_power, or noise_floor_power(iq) for a raw
    capture) -- required, not optional, because the SNR gate below cannot
    be honest without it.

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
    error as more windows are added) and compared to C42_BOUNDARY: at or
    above reads 16QAM (the larger-cumulant class), below reads 64QAM.
    C42_BOUNDARY is CALIBRATED (see its own comment for the full story and
    why a theory-derived boundary is wrong for this estimator), not
    theoretical -- the theory constants in C42_THEORY are checked
    separately, against physics, in
    test_c42_theory_matches_noiseless_synthetic_constellations.

    Refuses rather than guesses, in two independent ways:

    1. Below `min_windows` (default MIN_WINDOWS_FOR_C42_DECISION, 8 -- see
       that constant's comment for why) -- not enough pooling for the
       estimator's own variance to have shrunk to something usable.
    2. Below `min_snr_db` (default C42_MIN_SNR_DB, +2 dB) -- outside the
       SNR regime C42_BOUNDARY was actually calibrated against (see
       C42_MIN_SNR_DB's comment for the measured accuracy collapse outside
       that regime).

    Either way `decision` in the returned ConstellationOrderEstimate is
    None, even though mean_c42/n_windows/margin/accuracy/snr_db are still
    populated so the caller can see what WAS measured and why it refused. A
    caller must never be able to dress up a 2-window guess, or a
    confident-looking pooled average measured on a channel too noisy for
    the boundary to mean anything, as an answer.

    MEASURED, not MODEL: every number this function touches comes from the
    capture's own recovered samples, with no model involvement anywhere in
    the chain, and it fits nothing to an expected answer -- the boundary was
    calibrated on VALIDATION and only ever reported on TEST, the same
    discipline every other threshold in this project follows. It must never
    take a tier colour on screen; TEXT_DIM/INSTRUMENT styling only, exactly
    like everything else in src/measure.py.
    """
    from src.ui.plots import recover_symbols

    windows = list(windows)
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
            margin=float("nan"), accuracy=None, snr_db=None)

    mean_c42 = float(np.mean(per_window_c42))
    margin = abs(mean_c42 - C42_BOUNDARY)
    accuracy = _pooled_accuracy(n)
    snr_db = estimate_snr_db(np.concatenate(windows), noise_power)

    decision = None
    if n >= min_windows and snr_db >= min_snr_db:
        decision = "16QAM" if mean_c42 >= C42_BOUNDARY else "64QAM"

    return ConstellationOrderEstimate(
        decision=decision, mean_c42=mean_c42, n_windows=n, margin=margin,
        accuracy=accuracy, snr_db=snr_db)
