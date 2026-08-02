"""Windowing, normalization, SNR control, and training-time augmentation."""
import numpy as np
from scipy.signal import stft

from src.config import CFG


def add_awgn(signal, snr_db, rng=None, reference="active"):
    """Add complex AWGN to hit a target SNR (dB).

    reference="active" measures signal power over the samples that actually
    carry signal, ignoring silent gaps. This matters for pulsed emitters:
    averaging across a window that is 95% silence understates the power and so
    adds far too little noise. Measured before the fix, a radar labelled -10 dB
    was really sitting at +3 dB during its pulse, while FHSS and jamming (which
    are continuous) were labelled correctly. The model could then learn
    "clean signal at low labelled SNR => radar" — a shortcut, not a signal
    feature — and the accuracy-vs-SNR curve would be wrong for that class.

    reference="mean" restores the naive whole-array behaviour. For continuous
    signals the two are identical.
    """
    rng = rng or np.random.default_rng()
    mag_sq = np.abs(signal) ** 2

    if reference == "active":
        peak = mag_sq.max()
        active = mag_sq > 0.01 * peak       # -20 dB below peak counts as "on"
        sig_power = mag_sq[active].mean() if active.any() else mag_sq.mean()
    elif reference == "mean":
        sig_power = mag_sq.mean()
    else:
        raise ValueError(f"unknown reference {reference!r}")

    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return signal + noise


def preprocess_window(iq_complex, window_len=None):
    """Convert complex IQ to a normalized (2, window_len) real array."""
    window_len = window_len or CFG["signal"]["window_len"]
    iq = iq_complex[:window_len]
    if len(iq) < window_len:
        iq = np.pad(iq, (0, window_len - len(iq)))
    arr = np.stack([iq.real, iq.imag])
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.astype(np.float32)


def to_spectrogram(iq_complex, fs=None, nperseg=128):
    """Optional 2D representation for a 2D-CNN — an ALTERNATIVE to raw IQ, not
    an extra step. Nothing in the training pipeline calls this.

    WARNING: returns magnitude only, so phase is discarded. Radar chirps and
    FHSS hops stay clearly visible, but BPSK/QPSK/QAM are *defined* by their
    phase constellations and become very hard to separate without it. See
    docs/pipeline/05-preprocessing.md before reaching for this.
    """
    fs = fs or CFG["signal"]["fs"]
    _, _, Zxx = stft(iq_complex, fs=fs, nperseg=nperseg, return_onesided=False)
    return np.abs(Zxx).astype(np.float32)


def phase_rotate_batch(batch, theta):
    """Rotate a batch of (N, 2, L) IQ arrays by a fixed phase angle.

    A receiver's absolute phase is arbitrary, so rotating one carries no class
    information — the label is unchanged. That makes it usable at INFERENCE
    time: predict several rotations of the same window and average, and the
    per-rotation noise cancels while the signal does not. Free accuracy for
    compute only, since no new information is required.
    """
    c, s = np.cos(theta), np.sin(theta)
    i, q = batch[:, 0], batch[:, 1]
    return np.stack([c * i - s * q, s * i + c * q], axis=1)


def augment_iq(arr, rng=None):
    """Random phase rotation + time shift, applied per training batch.

    Both are label-preserving for these signal classes: a receiver's arbitrary
    phase offset and capture start time carry no class information.
    """
    rng = rng or np.random.default_rng()
    theta = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    arr = rot @ arr
    shift = rng.integers(-20, 20)
    return np.roll(arr, shift, axis=1)
