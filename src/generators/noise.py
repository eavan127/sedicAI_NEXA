"""
NOISE_FLOOR: an empty channel -- nobody transmitting, receiver hearing only
thermal noise.

Why this is a class rather than an absence: softmax always assigns one of the
available labels, so without a "nothing here" option a blank window is forced
into whichever real class it least-poorly resembles. Barrage jamming is
band-limited noise by construction, so empty spectrum lands there -- the model
reports a hostile emitter on an empty channel, which is the most expensive
false alarm the system can make.

This class deliberately carries NO structure at all. It is generated as pure
complex AWGN and, unlike every other class, is NOT passed through add_awgn
downstream -- there is no underlying signal to set an SNR against, since SNR
of a signal that does not exist is undefined. See build_dataset.
"""
import numpy as np

from src.config import CFG


def random_noise_example(fs=None, total_duration=None, rng=None):
    """One noise-floor window: complex AWGN, unit average power.

    Amplitude is normalised away by preprocess_window (zero-mean/unit-std), so
    the absolute scale here is irrelevant -- what matters is that the result is
    structureless in both time and frequency, which is what distinguishes it
    from band-limited barrage jamming.
    """
    rng = rng or np.random.default_rng()
    fs = fs or CFG["signal"]["fs"]
    total_duration = total_duration or CFG["signal"]["total_duration"]
    n_samples = int(fs * total_duration)

    return (rng.standard_normal(n_samples)
            + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
