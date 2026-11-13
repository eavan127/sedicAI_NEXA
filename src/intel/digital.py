"""
Text decoders for digital narrowband modes, picked by the kind the zoom rules
and the slow-path model named (src/pipeline/narrowband.py):

    RTTY   -> src/intel/rtty.py   (two-tone FSK, Baudot)
    PSK31  -> src/intel/psk31.py  (BPSK, Varicode; also PSK63 / PSK125)

When the kind is unsure (CW / carrier / unknown / AFSK), both decoders are
tried and a clean decode wins: a high share of correctly framed characters
does not happen by chance, so it is stronger evidence than the classifier.
"""
from src.intel import psk31, rtty

UNSURE = ("CW", "CARRIER", "UNKNOWN", "AFSK", "USB", "LSB")   # SSB: a narrow data signal often looks like it


def _rtty(y, fs):
    return dict(rtty.decode(y, fs), mode="RTTY")


def _psk(y, fs):
    r = psk31.decode(y, fs)
    return dict(r, mode=r.get("mode") or "PSK31")


def read(kind, y, fs):
    """-> (kind, decode result or None). y: the zoomed complex channel."""
    if kind == "RTTY":
        return kind, _rtty(y, fs)
    if kind == "PSK31":
        return kind, _psk(y, fs)
    if kind in UNSURE:
        if kind == "CW" and keyed_off_share(y, fs) >= KEYED_OFF:
            # Morse switches its tone fully off between dots and dashes; RTTY and PSK31
            # keep a tone up the whole time they send. A signal that keeps going silent
            # is Morse, however tidy a data decode of it looks: on Morse both decoders
            # can frame almost every "character" and print gibberish (13 Nov 2026,
            # SDRconnect file: "SIERRA BASE WEATHER ..." came out as RTTY "OGVXB.96=9?").
            return kind, None
        peak = strongest_hz(y, fs)
        for name, fn in (("RTTY", _rtty), ("PSK31", _psk)):
            r = fn(y, fs)
            # the decode must be OF the signal: its tone(s) where the strongest line is,
            # not a stretch of noise elsewhere in the channel that happens to frame
            on_it = (min(abs(r.get("mark_hz", 1e9) - peak), abs(r.get("space_hz", 1e9) - peak)) <= ON_SIGNAL_HZ
                     if name == "RTTY" else abs(r.get("carrier_hz", 1e9) - peak) <= ON_SIGNAL_HZ)
            if r["readable"] and on_it:
                r["why"] = f"{r['why']} (the classifier had said {kind}; the clean decode wins)"
                return name, r
    return kind, None


ON_SIGNAL_HZ = 60.0       # a data decode that overrides the classifier must sit this close to the strongest line
KEYED_OFF = 0.15          # share of the transmission spent silent above which it is Morse, not data


def strongest_hz(y, fs):
    """Frequency (Hz, relative to the channel centre) of the strongest line in y."""
    import numpy as np
    from scipy.signal import welch
    f, p = welch(np.asarray(y), fs=fs, nperseg=min(len(y), int(fs)), return_onesided=False, detrend=False)
    return float(f[int(np.argmax(p))])


def keyed_off_share(y, fs, block_s=0.02):
    """Share of 20 ms blocks, between the first and last loud block, in which the
    signal is off (below a quarter of its typical 'on' level). Morse: 0.3-0.6.
    RTTY / PSK31 while sending: about 0 (a tone is always on; PSK31's phase
    reversals dip for a few ms only, inside one block)."""
    import numpy as np
    y = np.asarray(y)
    n = int(block_s * fs)
    if n < 1 or len(y) < 10 * n:
        return 0.0
    p = np.mean(np.abs(y[:len(y) // n * n].reshape(-1, n)) ** 2, axis=1)
    floor, loud = np.percentile(p, 10), np.percentile(p, 95)
    if loud < 4 * floor:                      # never clearly on: no keying to see
        return 0.0
    on = p > floor + 0.5 * (loud - floor)
    idx = np.flatnonzero(on)
    span = p[idx[0]:idx[-1] + 1]
    level = np.median(p[on])
    return float(np.mean(span < floor + 0.25 * (level - floor)))


def listen_fields(r):
    """The listen-dict keys for a decode (Comms Intel and the dashboard read these)."""
    return {"digital": {k: v for k, v in r.items() if k not in ("segments",)},
            "digital_text": r["text"], "digital_segments": r["segments"]}
