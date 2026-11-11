"""
PSK31 decoder (also PSK63 / PSK125): BPSK with Varicode -> text.

    zoomed channel (complex) -> 4 kS/s -> carrier from the squared signal
    (BPSK squared is a pure line at twice the carrier) -> mix down -> one
    symbol of averaging -> pick the symbol timing where the amplitude is
    highest -> differential decision (no phase reversal = 1, reversal = 0)
    -> Varicode (characters separated by "00").

Differential decoding needs no phase lock and does not care which sideband
the receiver used. `readable` is true when most codes are real Varicode
characters: on noise only short random codes come out.
Synthetic-tested only (tests/test_psk31.py).
"""
import numpy as np
from scipy.signal import resample_poly

FS2 = 4000.0
BAUDS = (31.25, 62.5, 125.0)
GOOD_SHARE = 0.85

# G3PLX Varicode. Every code starts and ends with 1 and never holds "00".
VARICODE = {
    " ": "1", "e": "11", "t": "101", "o": "111", "a": "1011", "i": "1101", "n": "1111",
    "r": "10101", "s": "10111", "l": "11011", "\n": "11101", "\r": "11111",
    "h": "101011", "d": "101101", "c": "101111", "-": "110101", "u": "110111", "m": "111011", "f": "111101",
    "p": "111111",
    "=": "1010101", ".": "1010111", "g": "1011011", "y": "1011101", "b": "1011111", "w": "1101011",
    "T": "1101101", "S": "1101111", ",": "1110101", "E": "1110111", "v": "1111011", "A": "1111101",
    "I": "1111111",
    "O": "10101011", "C": "10101101", "R": "10101111", "D": "10110101", "0": "10110111", "M": "10111011",
    "1": "10111101", "k": "10111111", "P": "11010101", "L": "11010111", "F": "11011011", "N": "11011101",
    "x": "11011111", "B": "11101011", "2": "11101101", ":": "11110101", ")": "11110111", "(": "11111011",
    "G": "11111101", "3": "11111111",
    "H": "101010101", "U": "101010111", "5": "101011011", "W": "101011101", '"': "101011111",
    "6": "101101011", "X": "101110101", "4": "101110111", "Y": "101111011", "K": "101111101",
    "'": "101111111", "8": "110101011", "7": "110101101", "/": "110101111", "V": "110110101",
    "9": "110110111", ";": "110111101", "q": "110111111", "z": "111010101", "Q": "111011101",
    "j": "111101011", "J": "111111101",
    "?": "1010101111", "Z": "1010101101",
}
CHAR_OF = {v: k for k, v in VARICODE.items()}


def _box(x, n):
    n = max(int(round(n)), 1)
    return np.convolve(x, np.ones(n) / n, mode="same")


def carrier(x, fs=FS2):
    """BPSK carrier from the line at twice its frequency in x**2. -> (hz, line strength in dB over the median)"""
    n = 1 << int(np.ceil(np.log2(max(len(x), 2))))
    x = np.asarray(x, np.complex128)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-20)        # level-free: any receiver gain
    X = np.abs(np.fft.fft(x * x, n)) ** 2
    f = np.fft.fftfreq(n, 1 / fs)
    keep = np.abs(f) < 1600
    i = np.flatnonzero(keep)[np.argmax(X[keep])]
    return float(f[i] / 2), float(10 * np.log10(X[i] / (np.median(X[keep]) + 1e-20)))


def varicode_text(bits):
    """bits (1 = no reversal) -> [(char or None for an unknown code, bit index)]"""
    out, buf, start = [], [], 0
    for i, b in enumerate(bits):
        buf.append(int(b))
        if len(buf) >= 2 and buf[-1] == 0 and buf[-2] == 0:
            code = "".join(map(str, buf[:-2])).lstrip("0")
            if code:
                out.append((CHAR_OF.get(code), start))
            buf, start = [], i + 1
        elif len(buf) > 14:                      # longer than any code: noise
            out.append((None, start))
            buf, start = [], i + 1
    return out


def _decode_at(x, fc, baud, fs=FS2):
    sps = fs / baud
    z = _box(x * np.exp(-2j * np.pi * fc * np.arange(len(x)) / fs), sps)
    a = np.abs(z)
    step = int(round(sps))
    off = int(np.argmax([a[o::step][:-1].mean() for o in range(step)]))
    s = z[off::step]
    d = s[1:] * np.conj(s[:-1])
    bits = (d.real > 0).astype(int)
    chars = varicode_text(bits)
    known = [(c, (k + off / sps) / baud) for c, k in chars if c is not None]
    share = len(known) / max(len(chars), 1)
    return known, share, len(chars)


def segments(chars, gap_s=2.0):
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
        text = " ".join("".join(c for c, _ in s).split())
        if text:
            segs.append({"start_s": round(s[0][1], 2), "end_s": round(s[-1][1], 2), "text": text})
    return segs


def decode(y, fs):
    """y: complex channel around the PSK signal. -> {text, segments, baud, carrier_hz, readable, why, ...}"""
    from fractions import Fraction
    r = Fraction(FS2 / fs).limit_denominator(1000)
    x = resample_poly(np.asarray(y, np.complex64), r.numerator, r.denominator)
    fc, line_db = carrier(x)
    if line_db < 15:
        return {"text": "", "raw_text": "", "segments": [], "readable": False,
                "why": f"no BPSK carrier (squared-signal line only {line_db:.0f} dB over the floor)"}
    best = None
    for baud in BAUDS:
        known, share, n = _decode_at(x, fc, baud)
        score = len(known) * share ** 4
        if best is None or score > best[0]:
            best = (score, baud, known, share, n)
    _, baud, known, share, n = best
    segs = segments(known)
    text = " / ".join(s["text"] for s in segs)
    readable = share >= GOOD_SHARE and len(known) >= 10
    name = {31.25: "PSK31", 62.5: "PSK63", 125.0: "PSK125"}[baud]
    return {"text": text if readable else "", "raw_text": text, "segments": segs if readable else [],
            "mode": name, "baud": baud, "carrier_hz": round(fc, 2), "line_db": round(line_db, 1),
            "codes": n, "known_share": round(share, 3), "readable": readable,
            "why": (f"{name}, carrier {fc:+.1f} Hz, {share:.0%} of {n} codes are real characters"
                    + ("" if readable else ": text not trusted"))}
