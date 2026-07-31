"""
LFM (Linear Frequency Modulation) radar pulse generator.
Owner: Person A.

Verified by tests/test_radar.py, which checks the instantaneous frequency
actually sweeps linearly at the requested chirp rate.
"""
import numpy as np

from src.config import CFG


def generate_lfm_chirp_iq(fs, duration, bandwidth, f_start=None):
    """Generate a single complex-baseband LFM radar pulse.

    Phase is quadratic in time, so instantaneous frequency (its derivative)
    is linear — that linear sweep is the defining radar signature.
    """
    n = int(duration * fs)
    t = np.arange(n) / fs
    f_start = f_start if f_start is not None else -bandwidth / 2
    k = bandwidth / duration  # chirp rate (Hz/s)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return np.exp(1j * phase)


def embed_pulse_train(pulse, pri, fs, total_duration, time_delay=0.0, n_pulses=None):
    """Embed repeating pulses at a Pulse Repetition Interval (PRI).

    time_delay shifts where the first pulse begins. RadChar randomises this
    (1-10 us) and so must we: pulses that always start at sample 0 give the
    model a positional fingerprint rather than a signal feature.

    n_pulses caps how many pulses are emitted. RadChar sends a burst of 2-6 and
    then falls silent for the rest of the frame, whereas a continuously-scanning
    radar keeps transmitting. Both are real; None means fill the whole duration.
    """
    total_samples = int(total_duration * fs)
    pri_samples = max(int(pri * fs), 1)
    offset = int(time_delay * fs)

    out = np.zeros(total_samples, dtype=complex)
    emitted = 0
    for start in range(offset, max(total_samples - len(pulse), 1), pri_samples):
        if n_pulses is not None and emitted >= n_pulses:
            break
        out[start:start + len(pulse)] += pulse
        emitted += 1
    return out


def random_radar_example(fs=None, total_duration=None, rng=None):
    """One randomized radar example, parameters drawn from the config ranges.

    Randomizing per example prevents the model from memorizing one specific
    radar signature instead of learning the general LFM concept.
    """
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    total_duration = total_duration or CFG["signal"]["total_duration"]

    cfg = CFG["radar"]
    pulse_width = rng.uniform(*cfg["pulse_width_s"])
    bandwidth = rng.uniform(*cfg["bandwidth_hz"])
    time_delay = rng.uniform(*cfg["time_delay_s"])

    # PRI is drawn CONDITIONAL on pulse width, so pulse_width < PRI always.
    # Sampling them independently allows duty > 100%, i.e. a pulse starting
    # before the previous one ended — which is physically impossible and
    # produces overlapping garbage.
    pri_lo = max(cfg["pri_s"][0], pulse_width / cfg["max_duty_cycle"])
    pri_hi = max(pri_lo, cfg["pri_s"][1])
    pri = rng.uniform(pri_lo, pri_hi)
    # Randomize sweep direction (up-chirp vs down-chirp)
    f_start = -bandwidth / 2 if rng.random() > 0.5 else bandwidth / 2
    bandwidth = bandwidth if f_start < 0 else -bandwidth

    # Two emission patterns, both real. RadChar sends a short burst (2-6 pulses)
    # then falls silent; a continuously-scanning radar keeps transmitting. We
    # cover both, since the organisers' stream could resemble either.
    n_pulses = (int(rng.integers(*cfg["n_pulses"]))
                if rng.random() < cfg["burst_fraction"] else None)

    pulse = generate_lfm_chirp_iq(fs, pulse_width, bandwidth, f_start)
    return embed_pulse_train(pulse, pri, fs, total_duration, time_delay, n_pulses)
