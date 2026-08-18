"""
FHSS (Frequency Hopping Spread Spectrum) generator.
Owner: Person B.

Verified by tests/test_fhss.py, which checks each hop segment actually lands
on one of the declared channel frequencies.
"""
import numpy as np

from src.config import CFG, REPO_ROOT
from src.generators.radar import safe_freq_offset


def generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng=None):
    """Generate a frequency-hopping signal from a random hop sequence."""
    rng = rng or np.random.default_rng()
    samples_per_hop = max(int(hop_duration * fs), 1)
    n_hops = max(int(total_duration / hop_duration), 1)
    t_hop = np.arange(samples_per_hop) / fs

    signal = np.zeros(n_hops * samples_per_hop, dtype=complex)
    for i in range(n_hops):
        f = rng.choice(hop_freqs)
        signal[i * samples_per_hop:(i + 1) * samples_per_hop] = np.exp(2j * np.pi * f * t_hop)
    return signal


def random_fhss_example(fs=None, total_duration=None, rng=None):
    """One randomized FHSS example, parameters drawn from the config ranges."""
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    total_duration = total_duration or CFG["signal"]["total_duration"]

    hop_rate = rng.uniform(*CFG["fhss"]["hop_rate_hz"])
    hop_duration = 1 / hop_rate
    n_channels = rng.integers(*CFG["fhss"]["n_channels"])
    spacing = rng.uniform(*CFG["fhss"]["channel_spacing_hz"])
    # Carrier frequency offset -- simulates oscillator mismatch/Doppler.
    # Computed from the comb's own half-span so the outermost hop can never
    # be pushed past Nyquist, however wide this particular draw is. FHSS has
    # the least room of the three judged classes (comb can reach +/-1.536 MHz
    # against a 1.6 MHz Nyquist), so this offset is naturally the smallest.
    center_offset = safe_freq_offset(rng, (n_channels / 2) * spacing, fs)
    hop_freqs = (np.arange(n_channels) - n_channels / 2) * spacing + center_offset

    return generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng=rng)

def _qa_plot():
    """Generate one example, plot amplitude + spectrogram, and print the
    hop count so the picture can be checked against the configured hop rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import stft

    fs = CFG["signal"]["fs"]
    total_duration = CFG["signal"]["window_len"] / fs
    rng = np.random.default_rng(0)

    hop_rate = rng.uniform(*CFG["fhss"]["hop_rate_hz"])
    hop_duration = 1 / hop_rate
    n_channels = rng.integers(*CFG["fhss"]["n_channels"])
    spacing = rng.uniform(*CFG["fhss"]["channel_spacing_hz"])
    hop_freqs = (np.arange(n_channels) - n_channels / 2) * spacing

    sig = generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng=rng)
    expected_hops = total_duration / hop_duration

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    t = np.arange(len(sig)) / fs * 1e6
    axes[0].plot(t, sig.real, lw=0.8, label="I")
    axes[0].plot(t, sig.imag, lw=0.8, alpha=0.7, label="Q")
    axes[0].set_title(f"FHSS example  (hop_rate={hop_rate/1e3:.1f} kHz, "
                       f"n_channels={n_channels}, spacing={spacing/1e3:.1f} kHz)")
    axes[0].set_xlabel("time (us)")
    axes[0].legend(fontsize=8)

    f_, t_, Z = stft(sig, fs=fs, nperseg=32, return_onesided=False)
    axes[1].pcolormesh(t_ * 1e6, np.fft.fftshift(f_) / 1e6,
                        np.fft.fftshift(np.abs(Z), axes=0), shading="gouraud")
    axes[1].set_xlabel("time (us)")
    axes[1].set_ylabel("freq (MHz)")

    plt.tight_layout()
    out = REPO_ROOT / "results" / "fhss_qa.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")
    print(f"window = {total_duration*1e6:.1f} us, hop_duration = "
          f"{hop_duration*1e6:.2f} us -> expect ~{expected_hops:.1f} hops in this window")


if __name__ == "__main__":
    _qa_plot()
