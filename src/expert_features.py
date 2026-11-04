"""
Hand-made "expert" features for the week-1 60-class model: the numpy
reference. AMC_CNN's ExpertFeatures branch (src/models/amc_cnn.py) computes
the same numbers batched on the GPU; tests/test_expert_features.py keeps the
two in step.

Why these two (cheap probes, 2026-11-04, generated windows at 10-18 dB, a
small classifier on the feature alone, accuracy inside the confused group):

  amplitude histogram   APSK 0.50, ASK 0.73, QAM 0.28  (CNN seed 1, 60-way:
                        0.20, 0.61, 0.22). Amplitude does not change with a
                        carrier offset, so the rings of APSK / QAM and the
                        levels of ASK survive the random placement that
                        smears their phases.
  delay correlation     OFDM-like group 0.67, LTE_LIKE 0.67 (CNN: 0.03).
                        OFDM repeats the end of each symbol as its cyclic
                        prefix, so the signal correlates with itself at a
                        delay of one symbol; noise and noise jammers do not.
                        The delay where it peaks differs between LTE-like,
                        NR-like and plain OFDM.

Both are taken on a CHANNELIZED copy of the window: shift the strongest
occupied band to 0 Hz and drop everything outside it, so out-of-band noise
does not swamp the feature. Sampling at the symbol instants was tried and
did not help (APSK 0.43, ASK 0.51), so every sample is used.

Output layout (EXPERT_DIM = 104 values):
  [0:22]   soft histogram of |y| / rms(|y|), 22 bins over 0 .. 2.2
  [22]     std of that amplitude
  [23]     log(1 + kurtosis) of that amplitude
  [24]     occupied fraction of the band
  [25:104] |autocorrelation| at delays 8, 12, ..., 320 samples, / lag 0
The band's centre frequency is deliberately left out: placement is random,
so it carries no class information, only dataset quirks.
"""
import numpy as np

AMP_CENTERS = np.linspace(0.05, 2.15, 22)
AMP_WIDTH = 0.05
LAGS = np.arange(8, 321, 4)
SMOOTH = 7
FLOOR_QUANTILE = 0.3
FLOOR_FACTOR = 4.0
MIN_BINS = 6
KEEP = 0.55          # keep |bin| <= KEEP * occupied bins after centring
EXPERT_DIM = len(AMP_CENTERS) + 3 + len(LAGS)


def channelize(z):
    """complex (L,) -> (centred, band-limited complex (L,), occupied fraction)."""
    L = len(z)
    Z = np.fft.fft(z)
    P = np.abs(Z) ** 2
    h = SMOOTH // 2
    Ps = np.convolve(np.r_[P[-h:], P, P[:h]], np.ones(SMOOTH) / SMOOTH, mode="valid")
    floor = np.quantile(Ps, FLOOR_QUANTILE)
    mask = Ps > FLOOR_FACTOR * floor
    k = np.fft.fftfreq(L, 1 / L)
    w = np.clip(Ps - floor, 0, None) * mask
    fc = float(np.sum(k * w) / max(w.sum(), 1e-12))
    bins = max(int(mask.sum()), MIN_BINS)
    y = z * np.exp(-2j * np.pi * fc * np.arange(L) / L)
    Y = np.fft.fft(y)
    Y[np.abs(k) > KEEP * bins] = 0
    return np.fft.ifft(Y), bins / L


def expert_features(z):
    """complex (L,) window -> float (EXPERT_DIM,)."""
    y, occ = channelize(np.asarray(z, dtype=complex))
    a = np.abs(y) / (np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12)
    hist = np.exp(-0.5 * ((a[:, None] - AMP_CENTERS) / AMP_WIDTH) ** 2).mean(0)
    var = a.var()
    kurt = np.mean((a - a.mean()) ** 4) / (var ** 2 + 1e-12)
    yc = y - y.mean()
    R = np.fft.ifft(np.abs(np.fft.fft(yc, 2 * len(yc))) ** 2)
    corr = np.abs(R[LAGS]) / (np.abs(R[0]) + 1e-12)
    return np.r_[hist, np.sqrt(var), np.log1p(kurt), occ, corr].astype(np.float32)


def expert_features_from_input(x):
    """The model's (2, L) normalised input -> features, as the model computes them."""
    x = np.asarray(x, dtype=np.float64)
    return expert_features(x[0] + 1j * x[1])
