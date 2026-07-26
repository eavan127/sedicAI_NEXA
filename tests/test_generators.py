"""
DSP correctness tests for the synthetic signal generators.

WHY THIS FILE MATTERS: the project's single biggest risk is shipping synthetic
training data whose physics is subtly wrong — the model then scores 95% on our
own data and fails the organizer's real Qualifier IQ Stream. Eyeballing a
spectrogram catches gross errors; these tests catch the quiet ones by asserting
the maths actually does what the class name claims.

These verify INTERNAL CONSISTENCY (the code implements the intended equations).
They cannot verify REALISM (whether those equations match a real-world emitter)
— that still needs a signal-processing expert or reference literature.

Run:  pytest tests/ -v
"""
import numpy as np
import pytest

from src.data.preprocess import add_awgn, augment_iq, preprocess_window
from src.generators.fhss import generate_fhss
from src.generators.jamming import apply_jamming, generate_barrage_jamming, generate_tone_jamming
from src.generators.radar import embed_pulse_train, generate_lfm_chirp_iq

FS = 1e6


def instantaneous_freq(sig, fs):
    """Recover instantaneous frequency from the phase derivative."""
    return np.diff(np.unwrap(np.angle(sig))) * fs / (2 * np.pi)


class TestRadar:
    def test_chirp_sweeps_linearly_at_requested_rate(self):
        """An LFM chirp's defining property: frequency rises linearly in time.

        If this fails, we are not generating radar — we are generating noise
        with a radar label, and the model learns the wrong thing.
        """
        duration, bandwidth = 100e-6, 200e3
        sig = generate_lfm_chirp_iq(FS, duration, bandwidth, f_start=-bandwidth / 2)

        f_inst = instantaneous_freq(sig, FS)
        t = np.arange(len(f_inst)) / FS
        slope, intercept = np.polyfit(t, f_inst, 1)

        expected_rate = bandwidth / duration
        assert slope == pytest.approx(expected_rate, rel=0.02)
        assert intercept == pytest.approx(-bandwidth / 2, abs=0.02 * bandwidth)

        # Residuals must be tiny — a *linear* sweep, not merely an increasing one
        residual = f_inst - (slope * t + intercept)
        assert np.std(residual) < 0.01 * bandwidth

    def test_chirp_spans_the_requested_bandwidth(self):
        duration, bandwidth = 100e-6, 200e3
        sig = generate_lfm_chirp_iq(FS, duration, bandwidth, f_start=-bandwidth / 2)
        f_inst = instantaneous_freq(sig, FS)
        assert (f_inst.max() - f_inst.min()) == pytest.approx(bandwidth, rel=0.05)

    def test_chirp_has_constant_envelope(self):
        """Phase modulation only — amplitude must not wobble."""
        sig = generate_lfm_chirp_iq(FS, 100e-6, 200e3)
        assert np.allclose(np.abs(sig), 1.0, atol=1e-9)

    def test_pulse_train_repeats_at_requested_pri(self):
        pulse = generate_lfm_chirp_iq(FS, 20e-6, 100e3)
        pri, total = 1e-3, 5e-3
        train = embed_pulse_train(pulse, pri, FS, total)

        assert len(train) == int(total * FS)
        # Energy should appear in bursts, with silent gaps between them
        active = np.abs(train) > 1e-9
        assert active.any() and not active.all(), "expected pulsed, not continuous"

        starts = np.flatnonzero(np.diff(active.astype(int)) == 1) + 1
        if len(starts) >= 2:
            assert np.diff(starts) == pytest.approx(int(pri * FS), rel=0.01)


class TestFHSS:
    def test_each_hop_lands_on_a_declared_channel(self):
        """Every hop segment's dominant tone must be one of the channels we
        declared — otherwise the 'hop sequence' is not what we labelled it."""
        hop_freqs = np.array([-200e3, -100e3, 0.0, 100e3, 200e3])
        hop_duration = 200e-6
        total = 2e-3
        rng = np.random.default_rng(0)

        sig = generate_fhss(FS, total, hop_duration, hop_freqs, rng=rng)
        n_per_hop = int(hop_duration * FS)
        freq_axis = np.fft.fftfreq(n_per_hop, d=1 / FS)

        for i in range(len(sig) // n_per_hop):
            seg = sig[i * n_per_hop:(i + 1) * n_per_hop]
            peak = freq_axis[np.argmax(np.abs(np.fft.fft(seg)))]
            assert np.min(np.abs(hop_freqs - peak)) < FS / n_per_hop, (
                f"hop {i} peaked at {peak:.0f} Hz, not on any declared channel"
            )

    def test_signal_actually_hops(self):
        """Guards against the degenerate case of a constant-frequency tone
        being mislabelled as frequency-hopping."""
        hop_freqs = np.array([-200e3, -100e3, 100e3, 200e3])
        hop_duration = 200e-6
        rng = np.random.default_rng(1)
        sig = generate_fhss(FS, 4e-3, hop_duration, hop_freqs, rng=rng)

        n_per_hop = int(hop_duration * FS)
        freq_axis = np.fft.fftfreq(n_per_hop, d=1 / FS)
        peaks = {
            round(float(freq_axis[np.argmax(np.abs(np.fft.fft(
                sig[i * n_per_hop:(i + 1) * n_per_hop])))]))
            for i in range(len(sig) // n_per_hop)
        }
        assert len(peaks) > 1, "signal never changed frequency"

    def test_length_matches_requested_duration(self):
        hop_duration = 100e-6
        total = 1e-3
        sig = generate_fhss(FS, total, hop_duration, np.array([0.0, 50e3]))
        assert len(sig) == int(total / hop_duration) * int(hop_duration * FS)


class TestJamming:
    def test_applied_jsr_matches_request(self):
        """If the achieved JSR drifts from the requested value, every JSR label
        in the dataset is wrong and the jamming class is mislabelled."""
        rng = np.random.default_rng(2)
        n = 8192
        signal = np.exp(2j * np.pi * 50e3 * np.arange(n) / FS)

        for jsr_db in (0, 5, 10, 20):
            jammer = generate_barrage_jamming(n, rng=rng)
            jammed = apply_jamming(signal, jammer, jsr_db)

            measured = 10 * np.log10(
                np.mean(np.abs(jammed - signal) ** 2) / np.mean(np.abs(signal) ** 2)
            )
            assert measured == pytest.approx(jsr_db, abs=0.5)

    def test_tone_jammer_sits_at_requested_frequencies(self):
        n = 4096
        freqs = [-120e3, 80e3]
        sig = generate_tone_jamming(FS, n, freqs)

        spectrum = np.abs(np.fft.fft(sig))
        freq_axis = np.fft.fftfreq(n, d=1 / FS)
        for f in freqs:
            bin_idx = np.argmin(np.abs(freq_axis - f))
            # Each requested tone should dominate its neighbourhood
            assert spectrum[bin_idx] > 0.5 * spectrum.max()

    def test_barrage_jammer_is_broadband(self):
        """Barrage jamming must spread energy widely, not concentrate in one bin."""
        rng = np.random.default_rng(3)
        spectrum = np.abs(np.fft.fft(generate_barrage_jamming(4096, rng=rng)))
        assert spectrum.max() < 20 * spectrum.mean()


class TestPreprocess:
    @pytest.mark.parametrize("snr_db", [-10, -5, 0, 5, 10, 15])
    def test_awgn_achieves_requested_snr(self, snr_db):
        """Every SNR label in the dataset depends on this being exact."""
        rng = np.random.default_rng(4)
        n = 16384
        clean = np.exp(2j * np.pi * 30e3 * np.arange(n) / FS)

        noisy = add_awgn(clean, snr_db, rng=rng)
        measured = 10 * np.log10(
            np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noisy - clean) ** 2)
        )
        assert measured == pytest.approx(snr_db, abs=0.3)

    def test_window_shape_and_normalisation(self):
        arr = preprocess_window(np.random.randn(5000) + 1j * np.random.randn(5000), 1024)
        assert arr.shape == (2, 1024)
        assert arr.dtype == np.float32
        assert arr.mean() == pytest.approx(0.0, abs=1e-5)
        assert arr.std() == pytest.approx(1.0, abs=1e-3)

    def test_short_input_is_zero_padded(self):
        arr = preprocess_window(np.ones(100, dtype=complex), 1024)
        assert arr.shape == (2, 1024)

    def test_augmentation_preserves_shape_and_energy(self):
        """Phase rotation and time shift must not change how much signal is
        present — only how it is presented."""
        rng = np.random.default_rng(5)
        arr = preprocess_window(rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
        out = augment_iq(arr.copy(), rng=rng)

        assert out.shape == arr.shape
        assert np.sum(out ** 2) == pytest.approx(np.sum(arr ** 2), rel=1e-6)
