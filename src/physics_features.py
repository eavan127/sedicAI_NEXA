"""
Physics features for the 60-class model: the numpy reference. AMC_CNN's
PhysicsFeatures branch (src/models/amc_cnn.py) computes the same numbers
batched in torch; tests/test_physics_features.py keeps the two in step.

Why (docs/SEPARABILITY_ANALYSIS.md): the CNN confuses groups whose members
differ in small, well-defined physical ways. A specialist on these features
beat the CNN inside those groups (analog vs tones 58% vs 32%, APSK vs band
noise 46% vs 22%, PSK order 57% vs 49% at 15 dB), and fusing them with the
CNN's scores over all 60 classes gave 62.8% vs 56.6% for a control.

All measured blind on the channelized window (src/expert_features.channelize:
strongest band moved to 0 Hz, the rest dropped), except the spectral shape and
on/off timing, which use the raw window. PHYSICS_DIM = 56 values:

  [0:4]    power-law lines: peak / mean of |FFT(y^p)|, p = 1, 2, 4, 8, in dB.
           Squaring BPSK removes its data (a tone appears), the 4th power
           QPSK, the 8th 8PSK; noise, FSK and analog never make a line.
  [4:7]    symbol-rate line in |y|^2: peak / mean (dB), its position (share of
           the sample rate), peak / median (dB). Linear digital modulations have a
           symbol clock; noise jammers do not.
  [7:10]   instantaneous frequency (angle of y[n+1] conj(y[n])): spread,
           kurtosis, roughness (mean |step| / spread). Flat for a tone, wobbling
           for FM/PM, levels for FSK, ramps (smooth) for Gaussian FSK / GMSK.
  [10:26]  16-bin soft histogram of that frequency / (3 spread): FSK levels.
  [26:30]  how its own spectrum splits into 4 bands (slow ... fast).
  [30:33]  envelope spread, kurtosis, share below 0.2 (OQPSK / MSK never dip).
  [33:45]  12-bin soft histogram of the envelope (rings, levels).
  [45:48]  strongest autocorrelation at delays 16..320, its delay / 320, mean
           of the 3 strongest: the OFDM cyclic-prefix echo, chirp repetition.
  [48:52]  occupied fraction, spectral flatness inside the band, number of
           spectral peaks (multitone, max 20), symmetry about the centre (SSB).
  [52:56]  16-block power: max / min (dB), share of blocks on, a single
           switch-on (reactive jamming), number of on/off changes (pulsed).
"""
import numpy as np

from src import expert_features as ef

IF_CENTERS = np.linspace(-0.9375, 0.9375, 16)       # soft-histogram centres on [-1, 1]
IF_WIDTH = 0.0625
AMP_CENTERS = np.linspace(0.1, 2.3, 12)
AMP_WIDTH = 0.1
LAG_MIN, LAG_MAX = 16, 320
N_BLOCKS = 16
PHYSICS_DIM = 56


def _pa_db(a):
    return 10 * np.log10((a.max() + 1e-12) / (a.mean() + 1e-12))


def physics_features(z):
    """complex (L,) window -> float32 (PHYSICS_DIM,)."""
    z = np.asarray(z, dtype=complex)
    y, occ = ef.channelize(z)
    y = y / (np.sqrt(np.mean(np.abs(y) ** 2)) + 1e-12)
    n = len(y)
    han = np.hanning(n)
    f = []
    for p in (1, 2, 4, 8):
        f.append(_pa_db(np.abs(np.fft.fft(y ** p * han))))
    e = np.abs(y) ** 2
    E = np.abs(np.fft.rfft((e - e.mean()) * han))
    E[:3] = 0
    k = int(np.argmax(E))
    f += [_pa_db(E), k / n, 10 * np.log10((E[k] + 1e-12) / (np.median(E) + 1e-12))]
    d = y[1:] * np.conj(y[:-1])
    fi = np.arctan2(d.imag, d.real) / np.pi
    fi_c = fi - np.median(fi)
    sd = fi_c.std() + 1e-12
    f += [sd, np.mean(fi_c ** 4) / sd ** 4, np.mean(np.abs(np.diff(fi_c))) / sd]
    u = np.clip(fi_c / (3 * sd), -1, 1)
    f += list(np.exp(-0.5 * ((u[:, None] - IF_CENTERS) / IF_WIDTH) ** 2).mean(0))
    FI = np.abs(np.fft.rfft(fi_c * np.hanning(len(fi_c)), n))
    FI = FI / (FI.sum() + 1e-12)
    f += [FI[1:8].sum(), FI[8:32].sum(), FI[32:96].sum(), FI[96:].sum()]
    a = np.abs(y)
    f += [a.std(), np.mean((a - a.mean()) ** 4) / (a.var() ** 2 + 1e-12), np.mean(a < 0.2)]
    f += list(np.exp(-0.5 * ((a[:, None] - AMP_CENTERS) / AMP_WIDTH) ** 2).mean(0))
    yc = y - y.mean()
    R = np.abs(np.fft.ifft(np.abs(np.fft.fft(yc, 2 * n)) ** 2))[:n]
    R = R / (R[0] + 1e-12)
    seg = R[LAG_MIN:LAG_MAX + 1]
    j = int(np.argmax(seg))
    f += [seg[j], (LAG_MIN + j) / LAG_MAX, np.sort(seg)[-3:].mean()]
    P = np.abs(np.fft.fftshift(np.fft.fft(z * han))) ** 2
    Ps = np.convolve(np.r_[P[-2:], P, P[:2]], np.ones(5) / 5, mode="valid")
    floor = np.quantile(Ps, 0.3)
    band = Ps > 4 * floor
    if band.any():
        inb = Ps[band]
        flat = np.exp(np.mean(np.log(inb + 1e-20))) / (np.mean(inb) + 1e-20)
    else:
        flat = 1.0
    peaks = np.sum((Ps[1:-1] > Ps[:-2]) & (Ps[1:-1] > Ps[2:]) & (Ps[1:-1] > 30 * floor))
    Y = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    m = n // 2
    up, dn = Y[m + 1:m + 64], Y[m - 1:m - 64:-1]
    sym = np.corrcoef(up, dn)[0, 1] if up.std() > 0 and dn.std() > 0 else 0.0
    f += [occ, flat, min(peaks, 20), sym if np.isfinite(sym) else 0.0]
    bp = np.mean(np.abs(z.reshape(N_BLOCKS, -1)) ** 2, axis=1)
    on = bp > 0.5 * bp.max()
    onset = int(np.argmax(on))
    single = float(onset > 0 and on[onset:].all())
    f += [10 * np.log10(bp.max() / (bp.min() + 1e-12)), on.mean(), single, np.count_nonzero(on[1:] != on[:-1])]
    out = np.asarray(f, dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def physics_features_from_input(x):
    """The model's (2, L) input -> features, as the model computes them."""
    x = np.asarray(x, dtype=np.float64)
    return physics_features(x[0] + 1j * x[1])
