"""
Synthetic LFM (Linear Frequency Modulation) radar pulse generator.
Owner: Person A (per project timeline).

Produces complex-baseband IQ arrays representing radar chirps embedded
in a pulse train, ready to be windowed/labeled by preprocess.py.
"""
import numpy as np


def generate_lfm_chirp_iq(fs, duration, bandwidth, f_start=None):
    """Generate a single complex-baseband LFM radar pulse (IQ)."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    f_start = f_start if f_start is not None else -bandwidth / 2
    k = bandwidth / duration  # chirp rate (Hz/s)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.exp(1j * phase)


def embed_pulse_train(pulse, pri, fs, total_duration):
    """Embed repeating pulses at a Pulse Repetition Interval (PRI) into a zero-filled buffer."""
    total_samples = int(total_duration * fs)
    pri_samples = int(pri * fs)
    out = np.zeros(total_samples, dtype=complex)
    for start in range(0, total_samples - len(pulse), pri_samples):
        out[start:start + len(pulse)] += pulse
    return out


def random_radar_example(fs=1e6, total_duration=2e-3):
    """Generate one randomized radar example for dataset diversity.

    Randomizing pulse width / bandwidth / PRI / sweep direction prevents the
    model from overfitting to one specific radar signature instead of the
    general LFM concept.
    """
    pulse_width = np.random.uniform(10e-6, 100e-6)
    bandwidth = np.random.uniform(50e3, 1e6)
    pri = np.random.uniform(1e-3, 10e-3)
    f_start = -bandwidth / 2 if np.random.rand() > 0.5 else bandwidth / 2  # sweep direction

    pulse = generate_lfm_chirp_iq(fs, pulse_width, bandwidth, f_start)
    signal = embed_pulse_train(pulse, pri, fs, total_duration)
    return signal


if __name__ == "__main__":
    # Quick sanity check — plot a spectrogram for the self-QA step
    # (see docs/SEDIC2026_Track1_Documentation.md section 5.6)
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    fs = 1e6
    sig = random_radar_example(fs=fs)
    f, t, Zxx = stft(sig, fs=fs, nperseg=256, return_onesided=False)
    plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(np.abs(Zxx), axes=0), shading="gouraud")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("LFM Radar Chirp — Spectrogram (self-QA check)")
    plt.savefig("results/radar_qa_spectrogram.png")
    print("Saved results/radar_qa_spectrogram.png — compare against reference radar spectrograms.")
