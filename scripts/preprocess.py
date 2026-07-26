"""
Shared preprocessing: windowing, normalization, AWGN/SNR control, and the
optional spectrogram representation. Used by every generator + train.py.
"""
import numpy as np
from scipy.signal import stft


def add_awgn(signal, snr_db):
    """Add complex additive white Gaussian noise to hit a target SNR (dB)."""
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def preprocess_window(iq_complex, window_len=1024):
    """Convert complex IQ to a normalized (2, window_len) real array for the CNN."""
    iq = iq_complex[:window_len]
    if len(iq) < window_len:
        iq = np.pad(iq, (0, window_len - len(iq)))
    arr = np.stack([iq.real, iq.imag])
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr.astype(np.float32)


def to_spectrogram(iq_complex, fs, nperseg=128):
    """Optional path — recommended for radar/FHSS, whose time-frequency
    signature is visually obvious to a 2D-CNN."""
    _, _, Zxx = stft(iq_complex, fs=fs, nperseg=nperseg, return_onesided=False)
    return np.abs(Zxx).astype(np.float32)


def augment_iq(arr):
    """Random phase rotation + time shift, applied during training for extra diversity."""
    theta = np.random.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    arr = rot @ arr
    shift = np.random.randint(-20, 20)
    arr = np.roll(arr, shift, axis=1)
    return arr
