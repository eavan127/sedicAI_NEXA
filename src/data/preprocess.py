"""Windowing, normalization, SNR control, and training-time augmentation."""
import numpy as np
from scipy.signal import stft

from src.config import CFG


def add_awgn(signal, snr_db, rng=None):
    """Add complex AWGN to hit a target SNR (dB).

    Verified by tests/test_preprocess.py — the measured SNR of the output
    must match the requested value, or every SNR label in the dataset is a lie.
    """
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(signal) ** 2)
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
