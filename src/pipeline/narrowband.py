"""
The slow path's model: names a ZOOMED narrowband channel (50 kS/s, about 2 s)
as one of KINDS. The 60-class model looks at 0.16 ms, too short for signals
that change every 20-30 ms (RTTY, PSK31) or every few ms (voice); here the
look is long enough to see what makes them different:

  RTTY   two tones a fixed shift apart, switching at the baud rate (45-100 Bd)
  PSK31  one tone whose phase reverses at 31.25 Bd (a line in the squared signal,
         the envelope dipping at the symbol rate)
  AFSK   audio tones (1200/2200 Hz, or 1600/1800 Hz) inside FM or SSB
  USB/LSB  voice on one side of the (suppressed) carrier
  AM / NBFM  voice with a carrier / with a constant envelope
  CW     one line switching on and off;  CARRIER  one steady line;  NOISE  nothing

Trained on synthetic signals only (make()), with HF-style impairments: flat
fading, an echo, frequency offset and drift, SNR -5..30 dB in a 3 kHz band.
features() turns a channel into numbers a person can read; the classifier is a
gradient-boosted tree ensemble (scikit-learn), saved next to this file.

    python -m src.pipeline.narrowband --train          # ~2 min, writes narrowband_model.joblib
    python -m src.pipeline.narrowband --evaluate       # per-kind accuracy on fresh synthetic signals
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from src.intel import morse
from src.pipeline.analog_synth import speech_like

FA = 50e3
SECONDS = 2.0
KINDS = ("NOISE", "CARRIER", "CW", "RTTY", "PSK31", "AFSK", "AM", "NBFM", "USB", "LSB")
CLASS_NAME = {"NOISE": "NOISE_FLOOR", "CARRIER": "CARRIER", "CW": "CW", "RTTY": "2FSK", "PSK31": "BPSK",
              "AFSK": "AFSK", "AM": "AM_DSB_WC", "NBFM": "NBFM", "USB": "AM_SSB_SC", "LSB": "AM_SSB_SC"}
MODEL_PATH = Path(__file__).with_name("narrowband_model.joblib")
_MODEL = {}


# ---------- synthetic signals ----------

def _lp(x, hz, fs=FA, order=4):
    return sosfiltfilt(butter(order, hz, fs=fs, output="sos"), x)


def _cw(n, rng):
    wpm = rng.uniform(12, 35)
    dot = 1.2 / wpm
    letters = [c for c in morse.TO_CODE if len(c) == 1 and c.isalnum()]
    key = []
    while len(key) * dot * FA < n:
        for _ in range(rng.integers(2, 6)):
            for el in morse.TO_CODE[rng.choice(letters)]:
                key += [1] * (1 if el == "." else 3) + [0]
            key += [0] * 2
        key += [0] * 4
    lens = np.maximum(1, np.round(dot * FA * (1 + rng.normal(0, 0.08, len(key))))).astype(int)
    g = np.repeat(np.asarray(key, float), lens)[:n]
    g = np.pad(g, (0, n - len(g)))
    r = int(0.004 * FA)
    return np.convolve(g, np.ones(r) / r, mode="same").astype(complex)


def _fsk_audio(n, rng, f0, f1, baud):
    """Continuous-phase two-tone audio (real)."""
    bits = rng.integers(0, 2, int(n / FA * baud) + 2)
    f = np.repeat(np.where(bits, f1, f0), int(round(FA / baud)))[:n]
    f = np.pad(f, (0, n - len(f)), mode="edge")
    return np.cos(2 * np.pi * np.cumsum(f) / FA)


def _rtty(n, rng):
    shift = rng.choice([170.0, 425.0, 450.0, 850.0])
    baud = rng.choice([45.45, 50.0, 75.0, 100.0])
    bits = rng.integers(0, 2, int(n / FA * baud) + 2)
    f = np.repeat(np.where(bits, shift / 2, -shift / 2), int(round(FA / baud)))[:n]
    f = np.pad(f, (0, n - len(f)), mode="edge")
    return np.exp(2j * np.pi * np.cumsum(f) / FA)


def _psk(n, rng):
    baud = rng.choice([31.25, 31.25, 63.0, 125.0])
    m = int(round(FA / baud))
    k = n // m + 2
    # Varicode text has many zeros (= phase reversals); idle is all reversals.
    rev = rng.random(k) < rng.uniform(0.5, 1.0)
    ph = np.cumsum(rev) % 2
    sym = np.where(ph, -1.0, 1.0)
    # PSK31 shaping: the amplitude goes through zero (cosine) on a reversal.
    t = np.arange(m) / m
    out = np.empty(k * m)
    for i in range(k):
        a, b = sym[i - 1] if i else sym[0], sym[i]
        out[i * m:(i + 1) * m] = b if a == b else a * np.cos(np.pi * t)
    return out[:n].astype(complex)


def _afsk(n, rng):
    if rng.random() < 0.5:                                    # VHF packet: Bell 202 on FM
        a = _fsk_audio(n, rng, 1200.0, 2200.0, 1200.0)
        return np.exp(1j * 2 * np.pi * np.cumsum(3000 * a) / FA)
    a = _fsk_audio(n, rng, 1600.0, 1800.0, 300.0)             # HF packet: 300 Bd on USB
    return hilbert(_lp(a, 3000))


def _speech(n, rng):
    """Voice-like audio closer to real radio speech than speech_like alone:
    voiced sounds are a pitch (90-250 Hz, wandering) with its harmonics under
    formant-like peaks, mixed with noise-like consonants, in syllables and
    pauses; sometimes a roger beep at the end of the over. Real FM voice
    carries these steady tones, and without them the model mistook it for
    AFSK (2026-11-09)."""
    t = np.arange(n) / FA
    f0 = rng.uniform(90, 250) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t + rng.uniform(0, 6)))
    ph = 2 * np.pi * np.cumsum(f0) / FA
    voiced = np.zeros(n)
    formants = rng.uniform([500, 1200, 2300], [900, 1900, 3000])
    for k in range(1, int(3400 / f0.min())):
        fk = k * f0.mean()
        w = sum(np.exp(-((fk - F) / 250) ** 2) for F in formants) + 0.05
        voiced += w * np.sin(k * ph)
    voiced = _lp(voiced, 3400)
    a = 0.75 * voiced / (np.std(voiced) + 1e-12) + 0.5 * speech_like(n, FA, rng) / 0.3
    syll = np.clip(np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6)), 0, None) ** 1.5
    pause = (np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t + rng.uniform(0, 6)) > -0.6).astype(float)
    a = a * (0.15 + syll) * pause
    a = a / (np.abs(a).max() + 1e-12)
    if rng.random() < 0.5:                                    # roger beep, 0.1-0.25 s
        m = int(rng.uniform(0.1, 0.25) * FA)
        s0 = rng.integers(0, max(1, n - m))
        a[s0:s0 + m] = 0.6 * np.sin(2 * np.pi * rng.uniform(800, 2000) * np.arange(m) / FA)
    return a


def _voice(kind, n, rng):
    a = _speech(n, rng) if rng.random() < 0.75 else speech_like(n, FA, rng)
    if kind == "NBFM":
        if rng.random() < 0.35:                               # CTCSS sub-tone, as many walkie-talkies send
            a = 0.9 * a + 0.12 * np.sin(2 * np.pi * rng.uniform(67, 250) * np.arange(n) / FA)
        return np.exp(1j * 2 * np.pi * np.cumsum(rng.uniform(1500, 3000) * a) / FA)
    if kind == "AM":
        return (1 + rng.uniform(0.5, 0.9) * a).astype(complex)
    h = hilbert(a)
    return h if kind == "USB" else np.conj(h)


def make(kind, rng, seconds=SECONDS, snr_db=None):
    """One synthetic zoomed channel at FA: (complex64 iq, snr_db)."""
    n = int(seconds * FA)
    if kind == "NOISE":
        s = np.zeros(n, complex)
    elif kind == "CARRIER":
        s = np.ones(n, complex)
    elif kind == "CW":
        s = _cw(n, rng)
    elif kind == "RTTY":
        s = _rtty(n, rng)
    elif kind == "PSK31":
        s = _psk(n, rng)
    elif kind == "AFSK":
        s = _afsk(n, rng)
    else:
        s = _voice(kind, n, rng)
    t = np.arange(n) / FA
    if kind != "NOISE":
        if rng.random() < 0.5:                                # HF flat fading (slow, Rician-ish)
            fd = rng.uniform(0.05, 1.0)
            g = _lp(rng.standard_normal(n) + 1j * rng.standard_normal(n), max(fd, 0.2), order=2)
            g = g / (np.sqrt(np.mean(np.abs(g) ** 2)) + 1e-12)
            s = s * (0.6 + 0.8 * g)
        if rng.random() < 0.3:                                # one echo, 0.5-3 ms
            d = int(rng.uniform(0.5e-3, 3e-3) * FA)
            s = s + rng.uniform(0.2, 0.6) * np.concatenate([np.zeros(d, complex), s[:-d]])
        off = rng.uniform(-1500, 1500)                        # the zoom is rarely exactly centred
        drift = rng.uniform(-2, 2)
        s = s * np.exp(2j * np.pi * (off * t + 0.5 * drift * t ** 2))
    snr = rng.uniform(-5, 30) if snr_db is None else snr_db
    on = np.abs(s) > 1e-6
    p = np.mean(np.abs(s[on]) ** 2) if on.any() else 1.0
    noise_p = (p / 10 ** (snr / 10) * (FA / 3000)) if on.any() else 1.0
    y = s + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(noise_p / 2)
    return y.astype(np.complex64), snr


# ---------- features: what makes them different ----------

FEATURES = ("snr_db", "width20_hz", "width6_hz", "n_peaks", "gap12_hz", "balance12_db", "flatness",
            "carrier_frac", "side_db", "env_cv", "env_onoff", "env_on", "env_syllabic", "env_line",
            "env_line_hz", "if_std_hz", "if_bimodal", "if_switch_hz", "sq_line", "fm_tone_peaks", "fm_tone_db")


def _peaks(db, f, floor, min_gap=60.0, above=10.0, top=6):
    idx = np.argsort(db)[::-1]
    out = []
    for i in idx:
        if db[i] < floor + above or len(out) >= top:
            break
        if all(abs(f[i] - f[j]) > min_gap for j in out):
            out.append(i)
    return out


def features(y, fs=FA):
    y = np.asarray(y, np.complex64)
    y = y - y.mean() * 0                                    # keep DC: a carrier at 0 Hz is information
    nfft = 8192
    f, p = welch(y, fs=fs, nperseg=min(nfft, len(y)), return_onesided=False)
    order = np.argsort(f)
    f, p = f[order], p[order]
    db = 10 * np.log10(p + 1e-20)
    floor = np.median(db)
    top = db.max()
    snr = float(top - floor)
    w20 = db > max(floor + 6, top - 20)
    w6 = db > max(floor + 6, top - 6)
    df = f[1] - f[0]
    pk = _peaks(db, f, floor)
    gap = float(abs(f[pk[1]] - f[pk[0]])) if len(pk) > 1 else 0.0
    bal = float(db[pk[0]] - db[pk[1]]) if len(pk) > 1 else 40.0
    band = w20 if w20.any() else np.ones_like(w20)
    pb = p[band]
    flat = float(np.exp(np.mean(np.log(pb + 1e-20))) / (np.mean(pb) + 1e-20))
    carrier = float(p.max() / (pb.sum() + 1e-20))
    centre = f[int(np.argmax(db))]
    above = p[(f > centre + 150) & band].sum()
    below = p[(f < centre - 150) & band].sum()
    side = float(np.clip(10 * np.log10((above + 1e-20) / (below + 1e-20)), -40, 40))
    # Band-limit to the occupied band before time-domain measures.
    lo, hi = (f[band].min(), f[band].max()) if band.any() else (-1500, 1500)
    mid = (lo + hi) / 2
    yb = y * np.exp(-2j * np.pi * mid * np.arange(len(y)) / fs)
    half = max(150.0, (hi - lo) / 2 + 100)
    yb = _lp(yb, min(half, 0.45 * fs), fs)
    env = np.abs(yb)
    blk = int(0.002 * fs)
    e = env[:len(env) // blk * blk].reshape(-1, blk).mean(1)           # 2 ms envelope
    e = e / (np.percentile(e, 95) + 1e-12)
    on = e > 0.5
    cv = float(np.std(e) / (np.mean(e) + 1e-12))
    onoff = float(np.mean(on[1:] != on[:-1]))
    es = np.abs(np.fft.rfft(e - e.mean())) ** 2
    ef = np.fft.rfftfreq(len(e), 0.002)
    syl = float(es[(ef > 2) & (ef < 8)].sum() / (es[ef > 0.5].sum() + 1e-20))
    sel = (ef > 15) & (ef < 200)
    line = float(es[sel].max() / (np.median(es[sel]) + 1e-20)) if sel.any() else 0.0
    line_hz = float(ef[sel][np.argmax(es[sel])]) if sel.any() else 0.0
    # Instantaneous frequency inside the band.
    inst = np.angle(yb[1:] * np.conj(yb[:-1])) * fs / (2 * np.pi)
    strong = env[1:] > np.percentile(env, 50)
    ifr = inst[strong] if strong.sum() > 100 else inst
    ifr = np.clip(ifr, -half, half)
    if_std = float(np.std(ifr))
    m3 = float(np.mean((ifr - ifr.mean()) ** 3) / (if_std ** 3 + 1e-12))
    m4 = float(np.mean((ifr - ifr.mean()) ** 4) / (if_std ** 4 + 1e-12))
    bimodal = float((m3 ** 2 + 1) / (m4 + 1e-12))                       # Sarle: ~1 two clusters, ~0.33 one
    ifs = _lp(inst, 300, fs, order=2)[::50]                             # smoothed, 1 kS/s
    sw = np.sign(ifs - np.median(ifs))
    switch = float(np.mean(sw[1:] != sw[:-1]) * 1000)
    # Squared signal: a phase-reversal (BPSK) signal has a clean line at 2x carrier.
    sq = yb ** 2
    sqf = np.abs(np.fft.fft(sq[:1 << 16])) ** 2
    sq_line = float(sqf.max() / (np.median(sqf) + 1e-20))
    # Tones inside FM audio (AFSK on FM), measured on the FM-demodulated audio.
    fm = _lp(inst, 4000, fs, order=4)
    af, ap = welch(fm, fs=fs, nperseg=4096)
    aud = (af > 300) & (af < 3500)
    adb = 10 * np.log10(ap[aud] + 1e-20)
    afl = np.median(adb)
    tones = _peaks(adb, af[aud], afl, min_gap=150, above=8, top=4)
    return np.array([snr, float(w20.sum() * df), float(w6.sum() * df), len(pk), gap, bal, flat, carrier, side,
                     cv, onoff, float(on.mean()), syl, np.log10(line + 1), line_hz, if_std, bimodal, switch,
                     np.log10(sq_line + 1), len(tones), float(adb.max() - afl)], dtype=np.float32)


# ---------- model ----------

def dataset(n_per_kind, seed):
    rng = np.random.default_rng(seed)
    X, y, snr = [], [], []
    for k, kind in enumerate(KINDS):
        for _ in range(n_per_kind):
            iq, s = make(kind, rng)
            X.append(features(iq))
            y.append(k)
            snr.append(s)
    return np.array(X), np.array(y), np.array(snr)


def train(n_per_kind=500, seed=0, path=MODEL_PATH):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y, _ = dataset(n_per_kind, seed)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
    clf.fit(X, y)
    joblib.dump({"model": clf, "kinds": KINDS, "features": FEATURES, "n_per_kind": n_per_kind, "seed": seed}, path,
                compress=3)
    return clf


def load(path=MODEL_PATH):
    if path not in _MODEL:
        import joblib
        _MODEL[path] = joblib.load(path)
    return _MODEL[path]


def classify(y, fs=FA):
    """Zoomed channel (complex, at FA) -> (kind, probability, {kind: p}). The
    clip should be at least ~1 s; longer clips are cut to SECONDS pieces and
    their probabilities averaged."""
    m = load()
    n = int(SECONDS * fs)
    pieces = [y[i:i + n] for i in range(0, max(len(y) - n // 2, 1), n)] or [y]
    pieces = [p for p in pieces if len(p) >= int(0.5 * fs)][:10] or [y]
    P = np.mean(m["model"].predict_proba(np.array([features(p, fs) for p in pieces])), axis=0)
    i = int(np.argmax(P))
    return m["kinds"][i], float(P[i]), {k: float(v) for k, v in zip(m["kinds"], P)}


SLOW_ONLY = ("RTTY", "PSK31", "AFSK")      # kinds only this model can name
TAKE_P = 0.6                               # a slow-only kind replaces the zoom rules from this probability
FIX_P = 0.75                               # ... and overrides a rules answer the rules often get wrong


def refine(kind, why, y, fs=FA):
    """Second opinion on the zoom rules' answer (src/pipeline/analog.classify)
    for a zoomed channel y. -> (kind, why, demod_kind, info). demod_kind is the
    demodulator to use for listening (RTTY / PSK31 / AFSK are heard in SSB or FM).
    Leaves the answer alone when the model file is missing."""
    if not MODEL_PATH.exists() or len(y) < int(0.5 * fs):
        return kind, why, kind, None
    nk, p, probs = classify(y, fs)
    info = {"kind": nk, "p": round(p, 3), "probs": {k: round(v, 3) for k, v in probs.items()}}
    demod_of = {"RTTY": "USB", "PSK31": "USB", "AFSK": "NBFM" if kind == "NBFM" else "USB"}
    if nk in SLOW_ONLY and p >= TAKE_P:
        return nk, f"slow-path model ({p:.0%}): {DESCRIBE[nk]}; zoom rules had said {kind}", demod_of[nk], info
    if nk == "NOISE" and p >= FIX_P and kind in ("UNKNOWN", "CARRIER"):
        return "UNKNOWN", f"slow-path model ({p:.0%}): only noise here", "UNKNOWN", info
    # Where the rules are unsure (CW / carrier / unknown, e.g. SSB voice whose
    # roger beep looks like a keyed line) a confident model answer wins. The
    # rules' VOICE answers (AM / NBFM / USB / LSB) are kept: on realistic voice
    # (TTS and Common Voice through radio effects, the 91 military_radio_IQ
    # files) the rules were right 91/91, while the model -- trained on
    # simulated speech -- swapped NBFM for USB and AM for NBFM (2026-11-09).
    if nk not in ("AM", "CARRIER", "NOISE") and p >= FIX_P and nk != kind and kind in ("CW", "CARRIER", "UNKNOWN"):
        return nk, f"slow-path model ({p:.0%}): {PLAIN[nk]}; zoom rules had said {kind}", nk, info
    return kind, why, kind, info


PLAIN = {"CW": "one line switching on and off (Morse)", "NBFM": "constant-envelope voice (narrow FM)",
         "USB": "voice above a suppressed carrier (upper sideband)", "LSB": "voice below a suppressed carrier (lower sideband)"}


DESCRIBE = {"RTTY": "two tones a fixed shift apart, switching at the baud rate (RTTY / 2FSK)",
            "PSK31": "one tone with phase reversals at a slow symbol rate (PSK31-type BPSK)",
            "AFSK": "audio tones carried inside FM or SSB (AFSK, e.g. packet radio)"}


def evaluate(n_per_kind=200, seed=12345):
    m = load()
    X, y, snr = dataset(n_per_kind, seed)
    pred = m["model"].predict(X)
    print(f"{'kind':<9} {'all SNR':>8} {'>=10 dB':>8} {'<10 dB':>8}")
    for k, kind in enumerate(KINDS):
        s = y == k
        hi, lo = s & (snr >= 10), s & (snr < 10)
        print(f"{kind:<9} {np.mean(pred[s] == k):8.1%} {np.mean(pred[hi] == k) if hi.any() else np.nan:8.1%} "
              f"{np.mean(pred[lo] == k) if lo.any() else np.nan:8.1%}")
    print(f"{'overall':<9} {np.mean(pred == y):8.1%} {np.mean(pred[snr >= 10] == y[snr >= 10]):8.1%} "
          f"{np.mean(pred[snr < 10] == y[snr < 10]):8.1%}")
    return pred, y, snr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--n", type=int, default=500, help="training signals per kind")
    a = ap.parse_args()
    if a.train:
        train(a.n)
        print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    if a.evaluate:
        evaluate()
