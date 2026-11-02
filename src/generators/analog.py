"""
Analog and ASK generators for ten of Eileen's week-1 classes.

  * AM (AM_DSB_WC, AM_DSB_SC, AM_SSB_WC, AM_SSB_SC): a random message moves the
    amplitude. WC keeps a carrier, SC suppresses it; DSB has both sidebands, SSB
    one, upper or lower at random.
  * ANGLE (WBFM, NBFM, PM): a random message moves the frequency (FM) or the
    phase directly (PM). The envelope stays constant.
  * ASK (OOK, 4ASK, 8ASK): random symbols on M equally spaced amplitudes,
    0 ... M-1, root-raised-cosine shaped as in digital.py. Levels start at zero
    because that is what keying a carrier does, so these keep a carrier and are
    not the bipolar PAM constellations.

The names are used loosely in the literature, so the definitions below are this
project's choice, written down so every teammate means the same thing:

  class      message bandwidth   strength (`index` in the params)
  AM_DSB_WC  8 - 60 kHz          depth 0.3 - 1.0
  AM_DSB_SC  8 - 60 kHz          none, the message is the signal
  AM_SSB_WC  8 - 60 kHz          sideband 0.3 - 1.0 of the carrier
  AM_SSB_SC  8 - 60 kHz          none, the message is the signal
  WBFM       15 - 53 kHz         peak deviation 50 - 75 kHz, as broadcast FM
  NBFM       8 - 60 kHz          peak deviation 0.2 - 0.8 x message bandwidth
  PM         8 - 60 kHz          peak phase 0.5 - 1.5 rad

Message bandwidth is the one place this departs from the real signals on
purpose. A window is 160 microseconds, one cycle of 6.25 kHz, so true voice
(3 kHz) would show less than half a cycle and every AM class would look like a
bare carrier. 8 - 60 kHz puts 1 - 10 message cycles in each window: the
modulation is visible, at the cost of being wider than real voice channels.
WBFM needs no such stretch, since broadcast FM really is this wide.

The message is band-limited Gaussian noise rather than recorded speech, so
nothing here depends on an audio dataset. Output follows the shard contract the
same way digital.py does, and impairments stay in the shared module.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn
from src.dsp import RRC_SPAN_SYMBOLS, rrc_taps
from src.generators.digital import LINEAR_SPS, RRC_BETA_RANGE

FS = CFG["signal"]["fs"]
WINDOW_LEN = CFG["signal"]["window_len"]

# The message is built in the frequency domain, over a span long enough that
# the lowest message frequency still lands on several FFT bins.
MESSAGE_LEN = 8 * WINDOW_LEN
MESSAGE_HZ = (8e3, 60e3)
WBFM_MESSAGE_HZ = (15e3, 53e3)
WBFM_DEVIATION_HZ = (50e3, 75e3)
NBFM_INDEX = (0.2, 0.8)
PM_PEAK_RAD = (0.5, 1.5)
AM_INDEX = (0.3, 1.0)
# Message band starts here as a share of its top, so the message has no DC to
# act as a carrier in the SC classes.
MESSAGE_LOW_EDGE = 0.05

AM_CLASSES = ("AM_DSB_WC", "AM_DSB_SC", "AM_SSB_WC", "AM_SSB_SC")
ANGLE_CLASSES = ("WBFM", "NBFM", "PM")
ASK_LEVELS = {"OOK": 2, "4ASK": 4, "8ASK": 8}
ASK_CLASSES = tuple(ASK_LEVELS)
CLASSES = AM_CLASSES + ANGLE_CLASSES + ASK_CLASSES


def _message(fm_hz, rng):
    """Band-limited Gaussian noise (real, peak 1) and its Hilbert transform,
    both up to fm_hz. Periodic over MESSAGE_LEN, so no filter edges to trim."""
    spec = np.zeros(MESSAGE_LEN // 2 + 1, dtype=complex)
    freqs = np.fft.rfftfreq(MESSAGE_LEN, 1 / FS)
    band = (freqs >= MESSAGE_LOW_EDGE * fm_hz) & (freqs <= fm_hz)
    spec[band] = rng.normal(size=band.sum()) + 1j * rng.normal(size=band.sum())
    m = np.fft.irfft(spec, MESSAGE_LEN)
    hilbert = np.fft.irfft(-1j * spec, MESSAGE_LEN)
    peak = np.max(np.abs(m))
    return m / peak, hilbert / peak


def _crop(sig, rng):
    start = rng.integers(0, len(sig) - WINDOW_LEN + 1)
    return sig[start:start + WINDOW_LEN]


def _am(cls, rng):
    fm = rng.uniform(*MESSAGE_HZ)
    m, hilbert = _message(fm, rng)
    carrier = cls.endswith("_WC")
    index = rng.uniform(*AM_INDEX) if carrier else np.nan
    if "_SSB_" in cls:
        sideband = rng.choice((-1, 1))  # lower or upper
        sig = m + 1j * sideband * hilbert
    else:
        sig = m.astype(complex)
    if carrier:
        sig = 1 + index * sig
    return _crop(sig, rng), {"fm_hz": fm, "index": index, "bandwidth_hz": 2 * fm}


def _angle(cls, rng):
    if cls == "WBFM":
        fm, deviation = rng.uniform(*WBFM_MESSAGE_HZ), rng.uniform(*WBFM_DEVIATION_HZ)
    else:
        fm = rng.uniform(*MESSAGE_HZ)
    m, _ = _message(fm, rng)
    if cls == "PM":
        index = rng.uniform(*PM_PEAK_RAD)
        phase = index * m
        # Same two-sided width Carson's rule gives, with index as the deviation ratio.
        bandwidth = 2 * fm * (1 + index)
    else:
        if cls == "NBFM":
            deviation = rng.uniform(*NBFM_INDEX) * fm
        index = deviation / fm
        phase = 2 * np.pi * np.cumsum(deviation * m) / FS
        bandwidth = 2 * (deviation + fm)
    return _crop(np.exp(1j * phase), rng), {"fm_hz": fm, "index": index, "bandwidth_hz": bandwidth}


def _ask(cls, rng):
    sps = int(rng.choice(LINEAR_SPS))
    beta = rng.uniform(*RRC_BETA_RANGE)
    taps = rrc_taps(sps, beta=beta)
    n_sym = WINDOW_LEN // sps + 2 * RRC_SPAN_SYMBOLS + 2
    up = np.zeros(n_sym * sps)
    up[::sps] = rng.integers(0, ASK_LEVELS[cls], n_sym)
    sig = np.convolve(up, taps, mode="same")
    # Drop the filter start-up and tail before choosing the window.
    edge = RRC_SPAN_SYMBOLS * sps
    sig = sig[edge:-edge]
    return _crop(sig, rng), {"sps": sps, "beta": beta, "bandwidth_hz": FS / sps * (1 + beta)}


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit average power) -- used by the
    tests to check the physics without noise in the way.
    """
    if cls in AM_CLASSES:
        sig, drawn = _am(cls, rng)
    elif cls in ANGLE_CLASSES:
        sig, drawn = _angle(cls, rng)
    elif cls in ASK_LEVELS:
        sig, drawn = _ask(cls, rng)
    else:
        raise ValueError(f"{cls!r} is not an analog.py class; choose from {CLASSES}")
    params = {"sps": np.nan, "beta": np.nan, "fm_hz": np.nan, "index": np.nan, **drawn}
    sig = sig * np.exp(1j * rng.uniform(0, 2 * np.pi))
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng, reference="mean")
    return sig.astype(np.complex64), params


def generate(cls, n, snr_db, rng):
    """The shared generator signature: `n` windows as (complex64 (n, WINDOW_LEN),
    meta) where meta holds one array per randomised parameter."""
    iq = np.empty((n, WINDOW_LEN), dtype=np.complex64)
    meta = {k: np.empty(n) for k in ("sps", "beta", "fm_hz", "index", "bandwidth_hz")}
    for i in range(n):
        iq[i], params = generate_window(cls, snr_db, rng)
        for k, v in params.items():
            meta[k][i] = v
    return iq, meta
