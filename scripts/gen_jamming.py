"""
Synthetic jamming generators: barrage (wideband noise), tone, and sweep.
Owner: Person C (per project timeline).
"""
import numpy as np
from gen_radar import generate_lfm_chirp_iq


def generate_barrage_jamming(n_samples):
    """Wideband noise jammer."""
    return np.random.randn(n_samples) + 1j * np.random.randn(n_samples)


def generate_tone_jamming(fs, n_samples, freqs):
    """Single or multi-tone continuous-wave jammer."""
    t = np.arange(n_samples) / fs
    return sum(np.exp(2j * np.pi * f * t) for f in freqs)


def generate_sweep_jamming(fs, duration, bandwidth):
    """Fast repeating sweep jammer (reuses the chirp generator with jamming-typical params)."""
    return generate_lfm_chirp_iq(fs, duration, bandwidth)


def apply_jamming(signal, jammer, jsr_db):
    """Overlay jammer onto a legitimate signal at a controlled Jammer-to-Signal Ratio."""
    sig_power = np.mean(np.abs(signal) ** 2)
    jam_power = np.mean(np.abs(jammer) ** 2)
    scale = np.sqrt((sig_power * 10 ** (jsr_db / 10)) / jam_power)
    return signal + scale * jammer[:len(signal)]


def random_jamming_example(fs=1e6, n_samples=2048):
    """Generate one randomized jamming example (kind chosen at random)."""
    kind = np.random.choice(["barrage", "tone", "sweep"])
    if kind == "barrage":
        return generate_barrage_jamming(n_samples)
    elif kind == "tone":
        n_tones = np.random.randint(1, 4)
        freqs = np.random.uniform(-fs / 4, fs / 4, n_tones)
        return generate_tone_jamming(fs, n_samples, freqs)
    else:
        return generate_sweep_jamming(fs, n_samples / fs, bandwidth=np.random.uniform(100e3, 500e3))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    fs = 1e6
    sig = random_jamming_example(fs=fs)
    f, t, Zxx = stft(sig, fs=fs, nperseg=128, return_onesided=False)
    plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(np.abs(Zxx), axes=0), shading="gouraud")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("Jamming — Spectrogram (self-QA check)")
    plt.savefig("results/jamming_qa_spectrogram.png")
    print("Saved results/jamming_qa_spectrogram.png — compare against reference jamming spectrograms.")
