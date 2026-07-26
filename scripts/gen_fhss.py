"""
Synthetic FHSS (Frequency Hopping Spread Spectrum) generator.
Owner: Person B (per project timeline).
"""
import numpy as np


def generate_fhss(fs, total_duration, hop_duration, hop_freqs):
    """Generate a frequency-hopping signal from a random hop sequence."""
    samples_per_hop = int(hop_duration * fs)
    n_hops = int(total_duration / hop_duration)
    t_hop = np.arange(samples_per_hop) / fs
    signal = np.zeros(n_hops * samples_per_hop, dtype=complex)
    for i in range(n_hops):
        f = np.random.choice(hop_freqs)
        signal[i * samples_per_hop:(i + 1) * samples_per_hop] = np.exp(2j * np.pi * f * t_hop)
    return signal


def random_fhss_example(fs=1e6, total_duration=2e-3):
    """Generate one randomized FHSS example.

    Hop rate 100-1000 hops/sec and 8-64 channels are literature-informed
    starting ranges — see docs appendix for references to validate against.
    """
    hop_rate = np.random.uniform(100, 1000)  # hops/sec
    hop_duration = 1 / hop_rate
    n_channels = np.random.randint(8, 64)
    channel_spacing = np.random.uniform(10e3, 50e3)
    hop_freqs = (np.arange(n_channels) - n_channels / 2) * channel_spacing
    return generate_fhss(fs, total_duration, hop_duration, hop_freqs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    fs = 1e6
    sig = random_fhss_example(fs=fs)
    f, t, Zxx = stft(sig, fs=fs, nperseg=128, return_onesided=False)
    plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(np.abs(Zxx), axes=0), shading="gouraud")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("FHSS — Spectrogram (self-QA check)")
    plt.savefig("results/fhss_qa_spectrogram.png")
    print("Saved results/fhss_qa_spectrogram.png — compare against reference FHSS spectrograms.")
