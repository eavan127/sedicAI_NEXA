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
        for name, fn in (("RTTY", _rtty), ("PSK31", _psk)):
            r = fn(y, fs)
            if r["readable"]:
                r["why"] = f"{r['why']} (the classifier had said {kind}; the clean decode wins)"
                return name, r
    return kind, None


def listen_fields(r):
    """The listen-dict keys for a decode (Comms Intel and the dashboard read these)."""
    return {"digital": {k: v for k, v in r.items() if k not in ("segments",)},
            "digital_text": r["text"], "digital_segments": r["segments"]}
