"""
Zoom in on one narrowband track and say which kind of voice/Morse radio it is,
with measurements instead of a trained model.

The 60-class classifier sees 0.16 ms at a time: too short to hear speech, so
NBFM / AM / SSB / a keyed carrier look much alike to it. A whole track is
hundreds of milliseconds long, and at a 50 kS/s zoom these kinds differ in
plain, measurable ways (piano version: is the note's loudness steady, does it
switch on and off, is there a strong held note in the middle, and are the
two sides of it mirror images):

  feature          what it measures                          tells apart
  occupied_hz      2-98% of the signal's power above noise   carrier/CW (< 400 Hz) vs voice
  keyed            on/off in 5 ms blocks, both states long   CW (Morse) vs a steady carrier
  env_var          loudness variation of the signal itself,  FM (constant, ~0) vs AM/SSB
                   with the noise's share removed            (voice makes it swing)
  env_ratio        that swing over the swing AM would have   FM (~0) vs AM (~1) vs SSB with
                   with the measured sidebands               carrier (~0.5) when a carrier
                                                             dominates the power
  carrier_frac     share of power in one line (+-150 Hz)      AM / SSB with carrier vs without
  asym_db          upper vs lower sideband around the        AM (both sides) vs SSB (one side)
                   carrier (or the band's middle)
  mirror           how alike the two sides are, bin by bin   DSB without carrier vs SSB
  tilt             where the power sits inside the band      USB vs LSB when there is no carrier
                                                             (speech is strongest low, so a USB
                                                             band is heavier at its lower edge)

kinds: CARRIER, CW, NBFM, WBFM, AM (DSB with carrier), AM_SC (DSB without),
USB, LSB, UNKNOWN (too weak, or not an analog voice/Morse signal).
"""
import numpy as np
from scipy.signal import resample_poly, welch

from src.config import CFG

FS = float(CFG["signal"]["fs"])
DECIM = 64
FA = FS / DECIM                 # 50 kS/s zoom
NARROW_HZ = 400.0
LINE_HZ = 150.0                 # the carrier line; speech sidebands start ~300 Hz away
SIDE_MIN_DB = -6.0              # sidebands this far below the noise of a 3 kHz band still count
FM_ENV_VAR = 0.08               # normalised envelope variance below this = constant envelope
FM_ENV_RATIO = 0.12             # with a carrier: loudness swing vs what AM with these sidebands would give
CARRIER_FRAC = 0.25
ASYM_DB = 6.0
MIRROR = 0.7
MIN_SNR_DB = 8.0               # below this a "voice" reading of noise is too likely
WBFM_HZ = 40e3


def zoom(x, decim=DECIM):
    """Centred complex at FS -> complex at FS/decim (anti-aliased)."""
    return resample_poly(np.asarray(x, dtype=np.complex128), 1, decim)


def _spectrum(y, fa):
    f, p = welch(y, fs=fa, nperseg=min(1024, len(y)), return_onesided=False, detrend=False)
    o = np.argsort(f)
    return f[o], p[o]


def _bandpass(y, fa, lo, hi):
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(len(y), 1 / fa)
    return np.fft.ifft(np.where((f >= lo) & (f <= hi), Y, 0))


def measure(y, fa=FA):
    """Features of a zoomed track y (complex, centred). Power numbers are
    measured in the signal's own band, not the whole 50 kHz zoom."""
    y = np.asarray(y, dtype=complex)
    f, p = _spectrum(y, fa)
    df = f[1] - f[0]
    floor = float(np.median(p))                         # noise density: most of the zoom is noise
    excess = np.clip(p - 1.5 * floor, 0, None)          # 1.5x: noise bins scatter above the median
    total = float(excess.sum())
    out = {"snr_db": -99.0}
    if total <= 0:
        return out
    k0 = int(np.argmax(excess))
    line = np.abs(f - f[k0]) <= LINE_HZ                # window leakage of a strong carrier stays inside
    out["carrier_hz"] = float(f[k0])
    out["carrier_frac"] = float(excess[line].sum() / total)
    # The modulation's own spread: everything except the strongest line,
    # when there is enough of it above the noise of a voice-wide band.
    resid = np.where(line, 0, excess)
    side_snr = 10 * np.log10(max(resid.sum() * df, 1e-20) / (floor * 3000.0))
    src = resid if side_snr > SIDE_MIN_DB else excess
    cum = np.cumsum(src) / src.sum()
    lo, hi = float(f[np.searchsorted(cum, 0.02)]), float(f[min(np.searchsorted(cum, 0.98), len(f) - 1)])
    lo, hi = min(lo, f[k0] - LINE_HZ), max(hi, f[k0] + LINE_HZ)
    out["band"] = (lo, hi)
    out["occupied_hz"] = hi - lo
    bw = max(hi - lo, 200.0)
    out["snr_db"] = float(10 * np.log10(total * df / (floor * bw)))
    # Is there modulation beside the line? (sideband power over the noise in a voice-wide band)
    out["sideband_snr_db"] = float(side_snr)

    # Envelope statistics inside the signal's band only.
    yb = _bandpass(y, fa, lo - 150, hi + 150)
    noise_pow = floor * (hi - lo + 300)
    blk = max(1, int(0.005 * fa))
    nb = len(yb) // blk
    bp = np.mean(np.abs(yb[:nb * blk].reshape(nb, blk)) ** 2, axis=1) if nb else np.ones(1)
    hi_p = np.percentile(bp, 95)
    on = bp > noise_pow + 0.3 * max(hi_p - noise_pow, 0)
    off_frac = 1 - on.mean()
    flips = int(np.count_nonzero(on[1:] != on[:-1]))
    out["keyed"] = bool(0.1 < off_frac < 0.9 and flips >= 4 and hi_p > 4 * noise_pow)
    out["on_frac"] = float(on.mean())
    mask = np.zeros(len(yb), bool)
    mask[:nb * blk] = np.repeat(on, blk)
    a2 = np.abs(yb[mask]) ** 2 if mask.sum() > blk else np.abs(yb) ** 2
    s = max(float(a2.mean()) - noise_pow, 1e-20)
    # |s + n|^2 with constant |s|^2 = S has variance N^2 + 2 S N: remove the noise's share.
    out["env_var"] = float((a2.var() - noise_pow ** 2 - 2 * s * noise_pow) / s ** 2)
    # With a carrier C and sideband power P, AM's loudness swings by ~4 P / C
    # (|C + m a|^2 = C^2 + 2 C m a + ...), SSB with carrier by ~2 P / C, and
    # FM by ~0 however strong its sidebands are. env_ratio = measured / AM's.
    p_line = float(excess[line].sum() * df)
    p_side = float(resid.sum() * df)
    out["env_ratio"] = float(out["env_var"] / (4 * p_side / p_line)) if p_line > 0 and p_side > 0 else float("nan")

    # Sidebands around the carrier (or the band's middle when there is no carrier)
    ref = out["carrier_hz"] if out["carrier_frac"] >= CARRIER_FRAC else (lo + hi) / 2
    up = (f > ref + 100) & (f < ref + 4000)
    dn = (f < ref - 100) & (f > ref - 4000)
    pu, pd = float(resid[up].sum()) + 1e-20, float(resid[dn].sum()) + 1e-20
    out["asym_db"] = float(10 * np.log10(pu / pd))
    k = np.arange(2, int(4000 / df))
    iu = np.searchsorted(f, ref + k * df)
    idn = np.searchsorted(f, ref - k * df)
    ok = (iu < len(f)) & (idn > 0)
    a, b = p[iu[ok]], p[idn[ok]]
    out["mirror"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and a.std() > 0 and b.std() > 0 else 0.0
    w = (f >= lo) & (f <= hi)
    centroid = float((f[w] * resid[w]).sum() / max(resid[w].sum(), 1e-20)) if w.any() else (lo + hi) / 2
    out["tilt"] = float((centroid - (lo + hi) / 2) / max(hi - lo, df))       # < 0: heavier at the lower edge
    return out


def classify(y, fa=FA, wide_hint=False):
    """Zoomed track -> (kind, features, reason)."""
    m = measure(y, fa)
    if m["snr_db"] < MIN_SNR_DB or "occupied_hz" not in m:
        return "UNKNOWN", m, "too weak to tell"
    occ = m["occupied_hz"]
    if m["keyed"] and m["carrier_frac"] > 0.5:
        return "CW", m, f"one line switching on and off ({100 * m['on_frac']:.0f}% on): Morse"
    if occ < NARROW_HZ:
        if m["keyed"]:
            return "CW", m, f"one narrow line ({occ:.0f} Hz) switching on and off: Morse"
        return "CARRIER", m, f"one steady narrow line ({occ:.0f} Hz), no modulation"
    has_carrier = m["carrier_frac"] >= CARRIER_FRAC
    steady = m["env_ratio"] < FM_ENV_RATIO if has_carrier and np.isfinite(m["env_ratio"])         else m["env_var"] < FM_ENV_VAR
    if steady:
        kind = "WBFM" if (wide_hint or occ > WBFM_HZ) else "NBFM"
        return kind, m, f"sidebands {occ / 1e3:.1f} kHz wide but steady loudness: FM"
    if m["carrier_frac"] >= CARRIER_FRAC:
        if abs(m["asym_db"]) < ASYM_DB:
            return "AM", m, f"strong carrier with sidebands on both sides ({m['asym_db']:+.1f} dB): AM"
        side = "USB" if m["asym_db"] > 0 else "LSB"
        return side, m, f"carrier with one sideband only ({m['asym_db']:+.1f} dB): {side}"
    if m["mirror"] >= MIRROR:
        return "AM_SC", m, f"no carrier, two mirror-image sidebands (mirror {m['mirror']:.2f}): DSB AM"
    side = "USB" if m["tilt"] < 0 else "LSB"
    return side, m, f"no carrier, one-sided band (mirror {m['mirror']:.2f}); speech weight sits at the " \
                    f"{'lower' if side == 'USB' else 'upper'} edge: {side}"
