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

from src.config import CFG
from src.data.composite import overlay_jamming
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

    def test_duty_cycle_never_exceeds_one_hundred_percent(self):
        """A pulse wider than its PRI means the next pulse starts before the
        previous one ends — impossible, and it produces overlapping garbage.

        This caught a real bug: after widening the PRI floor to match RadChar
        (17 us) while pulse width still reached 100 us, sampling the two
        independently allowed duty cycles up to 588%.
        """
        from src.config import CFG

        cfg = CFG["radar"]
        max_duty = cfg["max_duty_cycle"]
        assert 0 < max_duty <= 1.0

        # Worst case: widest pulse against the PRI it would be paired with
        widest = cfg["pulse_width_s"][1]
        pri_floor = max(cfg["pri_s"][0], widest / max_duty)
        assert widest / pri_floor <= max_duty + 1e-9

    def _count_pulses(self, sig):
        active = np.abs(sig) > 1e-9
        return int(np.sum(np.diff(active.astype(int)) == 1) + (1 if active[0] else 0))

    def test_n_pulses_caps_the_burst(self):
        """RadChar fires 2-6 pulses then falls silent; a scanning radar keeps
        transmitting. The cap is what lets us generate the first shape.

        Tested directly rather than statistically: with a long PRI a continuous
        train also yields one pulse per window, so counting silent tails cannot
        tell the two apart.
        """
        pulse = generate_lfm_chirp_iq(FS, 10e-6, 200e3)
        pri, total = 20e-6, 500e-6          # short PRI -> many pulses would fit

        uncapped = embed_pulse_train(pulse, pri, FS, total)
        assert self._count_pulses(uncapped) > 6, "expected a long continuous train"

        for n in (2, 4, 6):
            capped = embed_pulse_train(pulse, pri, FS, total, n_pulses=n)
            assert self._count_pulses(capped) == n

    def test_both_emission_patterns_are_generated(self):
        """Both code paths must actually be exercised by the sampler, or the
        training set only ever contains one shape."""
        from src.generators.radar import random_radar_example
        from src.config import CFG

        rng = np.random.default_rng(21)
        counts = set()
        for _ in range(200):
            sig = random_radar_example(rng=rng)
            counts.add(self._count_pulses(sig))

        # A capped burst yields at most 6; an uncapped short-PRI train yields far more
        assert any(c <= 6 for c in counts), "no burst-limited examples generated"
        assert any(c > 6 for c in counts), "no continuous-train examples generated"

    def test_generated_pulses_never_overlap(self):
        """Empirical check on the above: no sample should receive energy from
        two pulses at once, which would show up as amplitude above unity."""
        from src.generators.radar import random_radar_example

        rng = np.random.default_rng(12)
        for _ in range(20):
            sig = random_radar_example(rng=rng)
            assert np.abs(sig).max() < 1.5, "pulses are overlapping"

    def test_pulses_do_not_always_start_at_sample_zero(self):
        """RadChar randomises its pulse start (time_delay 1-10 us). If ours
        always begins at sample 0, the model gets a positional fingerprint
        separating synthetic from real instead of a signal feature."""
        from src.generators.radar import random_radar_example

        rng = np.random.default_rng(11)
        first_active = set()
        for _ in range(12):
            sig = random_radar_example(rng=rng)
            onset = int(np.argmax(np.abs(sig) > 1e-9))
            first_active.add(onset)

        assert first_active != {0}, "every radar example starts at sample 0"
        assert len(first_active) > 1, "pulse onset is not being randomised"

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
        """Each requested tone must appear as a distinct peak at its own bin.

        This test failed once in a full-suite run and passed on every rerun.
        The cause was never identified: the original assertion survives 5,000
        unseeded runs without a single failure, so the obvious theory (the
        per-tone amplitude draw of uniform(0.5, 1.0) landing near the 0.5
        factor the assertion used) does not hold.

        Rewritten to remove the two things that could plausibly carry
        nondeterminism, rather than to fix a diagnosed cause:

        1. The rng is now seeded, as every other test in this class already
           does. A seeded test cannot flake on a draw whatever the mechanism.
        2. The assertion now measures what its comment always claimed. It
           compared each tone's bin against the GLOBAL maximum, which is a
           statement about the two tones' relative amplitudes, not about
           whether a tone sits at the requested frequency. Comparing against
           the non-tone background tests the real property and is indifferent
           to how the amplitudes were drawn.

        Skirt width and threshold are measured, not guessed. These tones do
        not land on exact bin centres (-120 kHz / 781.25 Hz = -153.6), so
        leakage spreads well past a few bins; at +/-30 bins the worst observed
        peak-to-background ratio over 300 seeds is 38x, so 10x leaves ample
        headroom.
        """
        rng = np.random.default_rng(4)
        n = 4096
        freqs = [-120e3, 80e3]
        sig = generate_tone_jamming(FS, n, freqs, rng=rng)

        spectrum = np.abs(np.fft.fft(sig))
        freq_axis = np.fft.fftfreq(n, d=1 / FS)
        tone_bins = [int(np.argmin(np.abs(freq_axis - f))) for f in freqs]

        background = np.ones(n, dtype=bool)
        for b in tone_bins:
            background[max(b - 30, 0):b + 31] = False   # exclude leakage skirts
        background_peak = spectrum[background].max()

        for f, b in zip(freqs, tone_bins):
            assert spectrum[b] > 10 * background_peak, (
                f"tone at {f / 1e3:.0f} kHz is not a distinct peak: "
                f"bin={spectrum[b]:.1f} vs background={background_peak:.1f}"
            )

    def test_barrage_jammer_is_broadband(self):
        """Barrage jamming must spread energy widely, not concentrate in one bin."""
        rng = np.random.default_rng(3)
        spectrum = np.abs(np.fft.fft(generate_barrage_jamming(4096, rng=rng)))
        assert spectrum.max() < 20 * spectrum.mean()


class TestCompositeOverlay:
    """src/data/composite.py -- jammer overlaid on a real victim signal, the
    building block for the multi-label composite training examples."""

    def test_label_set_is_victim_plus_jamming(self):
        rng = np.random.default_rng(5)
        victim = np.exp(2j * np.pi * 50e3 * np.arange(2048) / FS)
        _, class_set = overlay_jamming(victim, "QPSK", rng=rng, fs=FS)
        assert class_set == {"QPSK", "JAMMING"}

    def test_output_length_matches_victim(self):
        rng = np.random.default_rng(6)
        victim = np.exp(2j * np.pi * 50e3 * np.arange(4096) / FS)
        jammed, _ = overlay_jamming(victim, "BPSK", rng=rng, fs=FS)
        assert len(jammed) == len(victim)

    def test_invalid_victim_class_rejected(self):
        """NOISE_FLOOR (nothing to jam) and JAMMING (already jamming, not a
        victim) must both be refused, not silently accepted -- a silent
        accept here would mean a composite example either jams silence or
        double-counts jamming as its own victim."""
        rng = np.random.default_rng(7)
        victim = np.zeros(1024, dtype=complex)
        with pytest.raises(ValueError):
            overlay_jamming(victim, "NOISE_FLOOR", rng=rng, fs=FS)
        with pytest.raises(ValueError):
            overlay_jamming(victim, "JAMMING", rng=rng, fs=FS)

    def test_achieved_jsr_stays_within_configured_range(self):
        """Same guarantee test_applied_jsr_matches_request makes for
        apply_jamming() directly. overlay_jamming draws jsr_db randomly from
        jamming.jsr_db instead of taking it as a parameter, so this checks
        the ACHIEVED ratio lands in the configured range across repeated
        draws, rather than requesting one exact value."""
        rng = np.random.default_rng(8)
        lo, hi = CFG["jamming"]["jsr_db"]
        victim = np.exp(2j * np.pi * 50e3 * np.arange(4096) / FS)
        for _ in range(10):
            jammed, _ = overlay_jamming(victim, "BPSK", rng=rng, fs=FS)
            measured = 10 * np.log10(
                np.mean(np.abs(jammed - victim) ** 2) / np.mean(np.abs(victim) ** 2)
            )
            assert lo - 0.5 <= measured <= hi + 0.5


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

    @pytest.mark.parametrize("duty", [0.05, 0.2, 0.5, 0.9])
    def test_pulsed_signals_get_the_snr_they_asked_for(self, duty):
        """SNR must be measured against the pulse, not diluted by silent gaps.

        This caught a real bug: averaging power across a window that is 95%
        silence added far too little noise, so a radar labelled -10 dB was
        really at +3 dB during its pulse — while continuous classes (FHSS,
        jamming) were labelled correctly. The model could then key on
        "clean at low labelled SNR => radar" instead of on the signal.
        """
        rng = np.random.default_rng(7)
        n = 20000
        k = int(n * duty)
        sig = np.zeros(n, dtype=complex)
        sig[:k] = np.exp(2j * np.pi * 30e3 * np.arange(k) / FS)

        noisy = add_awgn(sig, -10, rng=rng)
        during_pulse = 10 * np.log10(
            np.mean(np.abs(sig[:k]) ** 2) / np.mean(np.abs(noisy - sig) ** 2)
        )
        assert during_pulse == pytest.approx(-10, abs=0.5), (
            f"duty {duty:.0%}: asked for -10 dB, pulse actually sees {during_pulse:.1f} dB"
        )

    def test_snr_is_independent_of_duty_cycle(self):
        """Otherwise SNR-label error correlates with pulse width, and the
        accuracy-vs-SNR curve becomes meaningless for pulsed classes."""
        rng = np.random.default_rng(8)
        measured = []
        for duty in (0.05, 0.9):
            n, k = 20000, int(20000 * duty)
            sig = np.zeros(n, dtype=complex)
            sig[:k] = np.exp(2j * np.pi * 30e3 * np.arange(k) / FS)
            noisy = add_awgn(sig, 0, rng=rng)
            measured.append(10 * np.log10(
                np.mean(np.abs(sig[:k]) ** 2) / np.mean(np.abs(noisy - sig) ** 2)))
        assert abs(measured[0] - measured[1]) < 1.0

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
