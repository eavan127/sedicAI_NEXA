"""
RTTY (radioteletype) decoder: two-tone FSK -> 5-bit Baudot (ITA2) text.

    zoomed channel (complex, ~centred) -> 4 kS/s -> find the two tones (mark,
    space) and their shift -> per-tone energy -> mark-minus-space line ->
    try each standard speed and both polarities -> start / 5 data / stop
    frames -> ITA2 letters and figures.

Covers amateur RTTY (45.45 Bd, 170 Hz) and the common utility / military-style
settings (50 / 75 / 100 Bd, 425 / 450 / 850 Hz shift). The speed and polarity
are chosen by how many frames have a valid stop bit; on noise that share is
about 50 %, on real RTTY 90 %+, so `readable` says whether to trust the text.
Synthetic-tested only (tests/test_rtty.py) plus KiwiSDR clips for validation.
"""
import numpy as np
from scipy.signal import resample_poly, welch

FS2 = 4000.0                                  # working rate: fits the widest shift (850 Hz) with room
BAUDS = (45.45, 50.0, 75.0, 100.0)
MIN_SHIFT, MAX_SHIFT = 80.0, 1000.0
GOOD_STOP_SHARE = 0.8                         # frames with a valid stop bit, above this the text is real
LTRS_SHIFT, FIGS_SHIFT = 31, 27
# ITA2, indexed by the 5-bit code (first bit sent = least significant).
LTRS = ["", "E", "\n", "A", " ", "S", "I", "U", "\n", "D", "R", "J", "N", "F", "C", "K",
        "T", "Z", "L", "W", "H", "Y", "P", "Q", "O", "B", "G", "", "M", "X", "V", ""]
FIGS = ["", "3", "\n", "-", " ", "'", "8", "7", "\n", "$", "4", "'", ",", "!", ":", "(",
        "5", "+", ")", "2", "#", "6", "0", "1", "9", "?", "&", "", ".", "/", "=", ""]


def tones(x, fs=FS2):
    """The two FSK tones in x: -> (f_low, f_high, level_db above the floor) or None."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), 1024), return_onesided=False, detrend=False)
    o = np.argsort(f)
    f, p = f[o], p[o]
    db = 10 * np.log10(p + 1e-20)
    keep = np.abs(f) < 1500
    f, db = f[keep], db[keep]
    floor = np.median(db)
    i = int(np.argmax(db))
    away = (np.abs(f - f[i]) >= MIN_SHIFT) & (np.abs(f - f[i]) <= MAX_SHIFT)
    if not away.any():
        return None
    j = int(np.flatnonzero(away)[np.argmax(db[away])])
    if db[j] - floor < 6 or db[i] - floor < 8:
        return None
    lo, hi = sorted((float(f[i]), float(f[j])))
    return lo, hi, float(min(db[i], db[j]) - floor)


def _box(x, n):
    n = max(int(n), 1)
    return np.convolve(x, np.ones(n) / n, mode="same")


def discriminator(x, f_space, f_mark, fs=FS2):
    """+1 where the mark tone is on, -1 where the space tone is on."""
    t = np.arange(len(x)) / fs
    shift = abs(f_mark - f_space)
    n = round(fs / shift)                       # boxcar null sits exactly on the other tone
    m = np.abs(_box(x * np.exp(-2j * np.pi * f_mark * t), n)) ** 2
    s = np.abs(_box(x * np.exp(-2j * np.pi * f_space * t), n)) ** 2
    return (m - s) / (m + s + 1e-12), m + s


def estimate_baud(d, fs=FS2):
    """The speed whose bit length best explains how long each tone stays on
    (every run is a whole number of bits). -> (baud, misfit 0..0.5)"""
    bit = _box(d, fs / 200) > 0
    edges = np.flatnonzero(bit[1:] != bit[:-1])
    runs = np.diff(edges)
    best = (BAUDS[0], 0.5)
    for baud in BAUDS:
        k = runs / (fs / baud)
        k = k[(k > 0.6) & (k < 12)]
        if len(k) < 20:
            continue
        miss = float(np.median(np.abs(k - np.round(k))))
        if miss < best[1]:
            best = (baud, miss)
    return best


def squelch(power, fs=FS2):
    """True where a tone is on: the smoothed tone power sits well above the quiet
    level (noise between transmissions would otherwise print stray letters)."""
    p = _box(power, fs * 0.05)
    quiet, loud = np.percentile(p, 10), np.percentile(p, 90)
    if loud < 3 * quiet:                          # on all the time (or never): no gaps to skip
        return np.ones(len(p), bool)
    return p > quiet + 0.3 * (loud - quiet)


def frames(d, baud, fs=FS2, on=None):
    """Asynchronous start(0) / 5 data / stop(1) frames in the line d (>0 = mark),
    read only where `on` (squelch) is true.
    -> (codes with their start time in s, frames with a bad stop bit)"""
    sb = fs / baud
    d = _box(d, sb / 3)
    bit = d > 0
    on = np.ones(len(bit), bool) if on is None else on
    out, bad, i, n = [], 0, 1, len(bit)
    end = n - int(7.5 * sb)
    while i < end:
        if bit[i - 1] and not bit[i] and on[i] and on[int(i + 6.5 * sb)]:   # mark -> space: a start bit begins
            at = lambda k: bit[int(i + (k + 0.5) * sb)]   # noqa: E731 -- middle of bit k
            if not at(0):
                code = sum(int(at(1 + b)) << b for b in range(5))
                if at(6):
                    out.append((code, i / fs))
                    i += int(6.5 * sb)           # stop bit seen: wait for the next start
                    continue
                bad += 1
        i += 1
    return out, bad


def to_text(codes, unshift_on_space=True):
    figs, chars = False, []
    for code, t in codes:
        if code == LTRS_SHIFT:
            figs = False
        elif code == FIGS_SHIFT:
            figs = True
        else:
            c = (FIGS if figs else LTRS)[code]
            if c == " " and unshift_on_space:
                figs = False
            if c:
                chars.append((c, t))
    return chars


def segments(chars, gap_s=2.0):
    """Text split where the station paused, with times. -> [{start_s, end_s, text}]"""
    out, cur = [], []
    for c, t in chars:
        if cur and t - cur[-1][1] > gap_s:
            out.append(cur)
            cur = []
        cur.append((c, t))
    if cur:
        out.append(cur)
    segs = []
    for s in out:
        text = " ".join("".join(c for c, _ in s).split())       # one line, single spaces
        if text:
            segs.append({"start_s": round(s[0][1], 2), "end_s": round(s[-1][1], 2), "text": text})
    return segs


def decode(y, fs):
    """y: complex channel around the RTTY signal (any rate >= 4 kS/s).
    -> {text, segments, baud, shift_hz, mark_hz, space_hz, inverted, stop_ok_share, readable, why}"""
    from fractions import Fraction
    r = Fraction(FS2 / fs).limit_denominator(1000)
    x = resample_poly(np.asarray(y, np.complex64), r.numerator, r.denominator)
    found = tones(x)
    if found is None:
        return {"text": "", "raw_text": "", "segments": [], "readable": False, "why": "no two-tone FSK pair found"}
    lo, hi, level = found
    best = None
    for inverted in (False, True):              # mark is normally the higher tone (USB); some send it inverted
        mark, space = (lo, hi) if inverted else (hi, lo)
        d, power = discriminator(x, space, mark)
        on = squelch(power)
        guess, _ = estimate_baud(np.where(on, d, 1.0))
        for baud in (guess,):
            codes, bad = frames(d, baud, on=on)
            share = len(codes) / max(len(codes) + bad, 1)
            score = len(codes) * share ** 4      # many frames AND nearly all with a good stop bit
            if best is None or score > best[0]:
                best = (score, baud, inverted, codes, share, mark, space)
    _, baud, inverted, codes, share, mark, space = best
    chars = to_text(codes)
    segs = segments(chars)
    text = " / ".join(s["text"] for s in segs)
    readable = share >= GOOD_STOP_SHARE and len(codes) >= 10
    return {"text": text if readable else "", "raw_text": text, "segments": segs if readable else [],
            "baud": baud, "shift_hz": round(hi - lo, 1), "mark_hz": round(mark, 1), "space_hz": round(space, 1),
            "inverted": inverted, "level_db": round(level, 1), "frames": len(codes),
            "stop_ok_share": round(share, 3), "readable": readable,
            "why": (f"{baud:g} Bd, {hi - lo:.0f} Hz shift, {share:.0%} of {len(codes)} frames framed correctly"
                    + ("" if readable else ": too few clean frames, text not trusted"))}
