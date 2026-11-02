"""
Multicarrier generators for three of Eileen's week-1 classes: OFDM, LTE_LIKE
and NR_LIKE. All three are the same machine -- random subcarrier symbols, an
inverse FFT, a cyclic prefix -- and differ in spacing, prefix and pilots:

  class     subcarrier spacing      cyclic prefix        subcarrier data  pilots
  OFDM      12.5 / 25 / 50 kHz      1/4, 1/8, 1/16       QPSK - 64QAM     none
  LTE_LIKE  15 kHz                  4.7 or 16.7 us       QPSK - 64QAM     every 6th, DC empty
  NR_LIKE   30 / 60 kHz             2.3 / 1.2 us         QPSK - 256QAM    every 2nd, 1 symbol in 14

"-LIKE" is deliberate. These borrow the waveform shape of LTE and 5G NR (the
spacing, prefix length and pilot layout), not their frames, scrambling or
control channels. A 160 microsecond window is only 2 LTE symbols, so nothing
longer than that could show up anyway. Spacing and prefix are rounded to whole
samples at 3.2 MS/s: LTE's 15 kHz becomes 213 samples per symbol, 15.02 kHz.

OFDM stays away from 15 / 30 / 60 kHz so that spacing alone separates a generic
OFDM link from the two standards.

Built with NumPy rather than TorchSig so the generators need no extra
dependency and follow the shared signature. Output follows the shard contract
the same way digital.py does, and impairments stay in the shared module.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import add_awgn
from src.generators.digital import CONSTELLATIONS

FS = CFG["signal"]["fs"]
WINDOW_LEN = CFG["signal"]["window_len"]

OFDM_NFFT = (64, 128, 256)
OFDM_CP_FRACTIONS = (1 / 4, 1 / 8, 1 / 16)
OFDM_FILL = (0.5, 0.9)  # share of the FFT carrying subcarriers
OFDM_DATA = ("QPSK", "16QAM", "64QAM")

LTE_NFFT = 213
LTE_CP = (15, 53)        # normal, extended
LTE_SUBCARRIERS = (72, 180)  # the 1.4 MHz and 3 MHz channels
LTE_PILOT_STEP = 6
LTE_SLOT_SYMBOLS = 7
LTE_PILOT_SYMBOLS = (0, 4)
LTE_DATA = ("QPSK", "16QAM", "64QAM")

NR_NFFT_CP = ((107, 8), (53, 4))  # 30 kHz, 60 kHz
NR_RB_RANGE = (0.5, 0.9)          # share of the FFT carrying resource blocks
NR_RB_SUBCARRIERS = 12
NR_PILOT_STEP = 2
NR_SLOT_SYMBOLS = 14
NR_PILOT_SYMBOL = 2
NR_DATA = ("QPSK", "16QAM", "64QAM", "256QAM")

CLASSES = ("OFDM", "LTE_LIKE", "NR_LIKE")


def _ofdm_symbols(nfft, cp, bins, n_sym, data, pilot_bins, rng):
    """`n_sym` OFDM symbols with a cyclic prefix. `pilot_bins(i)` gives the
    positions inside `bins` that carry a QPSK pilot on symbol i."""
    points = CONSTELLATIONS[data]
    pilots = CONSTELLATIONS["QPSK"]
    out = []
    for i in range(n_sym):
        spec = np.zeros(nfft, dtype=complex)
        spec[bins] = points[rng.integers(0, len(points), len(bins))]
        pilot = pilot_bins(i)
        spec[bins[pilot]] = pilots[rng.integers(0, len(pilots), len(pilot))]
        body = np.fft.ifft(spec) * np.sqrt(nfft)
        out.append(np.concatenate([body[-cp:], body]))
    return np.concatenate(out)


def _centred_bins(n, nfft, skip_dc=False):
    """FFT indices of `n` subcarriers around DC, lowest frequency first."""
    if skip_dc:
        k = np.concatenate([np.arange(-n // 2, 0), np.arange(1, n // 2 + 1)])
    else:
        k = np.arange(n) - n // 2
    return k % nfft


def _n_sym(nfft, cp):
    return WINDOW_LEN // (nfft + cp) + 2


def _plain(rng):
    nfft = int(rng.choice(OFDM_NFFT))
    cp = int(round(nfft * rng.choice(OFDM_CP_FRACTIONS)))
    n = 2 * int(rng.uniform(*OFDM_FILL) * nfft / 2)
    data = str(rng.choice(OFDM_DATA))
    sig = _ofdm_symbols(nfft, cp, _centred_bins(n, nfft), _n_sym(nfft, cp), data,
                        lambda i: np.empty(0, dtype=int), rng)
    return sig, nfft, cp, n, data


def _lte(rng):
    cp = int(rng.choice(LTE_CP))
    n = int(rng.choice(LTE_SUBCARRIERS))
    data = str(rng.choice(LTE_DATA))
    bins = _centred_bins(n, LTE_NFFT, skip_dc=True)
    first = int(rng.integers(0, LTE_SLOT_SYMBOLS))  # where in the slot the window falls

    def pilot_bins(i):
        sym = (first + i) % LTE_SLOT_SYMBOLS
        if sym not in LTE_PILOT_SYMBOLS:
            return np.empty(0, dtype=int)
        # The two pilot symbols sit half a step apart, as LTE's reference signal does.
        shift = 0 if sym == LTE_PILOT_SYMBOLS[0] else LTE_PILOT_STEP // 2
        return np.arange(shift, len(bins), LTE_PILOT_STEP)

    sig = _ofdm_symbols(LTE_NFFT, cp, bins, _n_sym(LTE_NFFT, cp), data, pilot_bins, rng)
    return sig, LTE_NFFT, cp, n, data


def _nr(rng):
    nfft, cp = NR_NFFT_CP[rng.integers(0, len(NR_NFFT_CP))]
    n_rb = int(rng.uniform(*NR_RB_RANGE) * nfft / NR_RB_SUBCARRIERS)
    n = n_rb * NR_RB_SUBCARRIERS
    data = str(rng.choice(NR_DATA))
    bins = _centred_bins(n, nfft)
    first = int(rng.integers(0, NR_SLOT_SYMBOLS))

    def pilot_bins(i):
        if (first + i) % NR_SLOT_SYMBOLS != NR_PILOT_SYMBOL:
            return np.empty(0, dtype=int)
        return np.arange(0, len(bins), NR_PILOT_STEP)

    sig = _ofdm_symbols(nfft, cp, bins, _n_sym(nfft, cp), data, pilot_bins, rng)
    return sig, nfft, cp, n, data


def generate_window(cls, snr_db, rng):
    """One window of class `cls`: (complex64 (WINDOW_LEN,), params dict).

    snr_db=None returns the clean signal (unit average power) -- used by the
    tests to check the physics without noise in the way.
    """
    builders = {"OFDM": _plain, "LTE_LIKE": _lte, "NR_LIKE": _nr}
    if cls not in builders:
        raise ValueError(f"{cls!r} is not a multicarrier.py class; choose from {CLASSES}")
    sig, nfft, cp, n, data = builders[cls](rng)
    start = rng.integers(0, len(sig) - WINDOW_LEN + 1)
    sig = sig[start:start + WINDOW_LEN] * np.exp(1j * rng.uniform(0, 2 * np.pi))
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    if snr_db is not None:
        sig = add_awgn(sig, snr_db, rng=rng, reference="mean")
    params = {"nfft": nfft, "cp": cp, "n_subcarriers": n, "scs_hz": FS / nfft,
              "bits_per_symbol": int(np.log2(len(CONSTELLATIONS[data]))),
              "bandwidth_hz": n * FS / nfft}
    return sig.astype(np.complex64), params


def generate(cls, n, snr_db, rng):
    """The shared generator signature: `n` windows as (complex64 (n, WINDOW_LEN),
    meta) where meta holds one array per randomised parameter."""
    iq = np.empty((n, WINDOW_LEN), dtype=np.complex64)
    meta = {k: np.empty(n) for k in ("nfft", "cp", "n_subcarriers", "scs_hz",
                                     "bits_per_symbol", "bandwidth_hz")}
    for i in range(n):
        iq[i], params = generate_window(cls, snr_db, rng)
        for k, v in params.items():
            meta[k][i] = v
    return iq, meta
