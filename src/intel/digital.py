"""
Text decoders for digital narrowband modes, picked by the kind the zoom rules
and the slow-path model named (src/pipeline/narrowband.py):

    RTTY   -> src/intel/rtty.py   (two-tone FSK, Baudot)
    PSK31  -> src/intel/psk31.py  (BPSK, Varicode; also PSK63 / PSK125)

When the kind is unsure (CW / carrier / unknown / AFSK), both decoders are
tried and a clean decode wins: a high share of correctly framed characters
does not happen by chance, so it is stronger evidence than the classifier.

Two exceptions, both seen on real captures (2026-11-13): an overloaded
walkie-talkie voice and a Morse file each "framed" as RTTY and lost their
transcript / Morse text. A regular signal (a held tone, a steady FM hum, keyed
Morse) can line up with RTTY's start and stop bits by chance, and then
decodes to the same two or three letters over and over ("FFDDFDDFDD"). So the
override is refused when
  * the slow-path model is clearly sure it is NOT a digital mode (voice or
    Morse ahead of RTTY / PSK31 / AFSK by OVERRIDE_MARGIN), or
  * the "text" is one or two letters repeating, not text (fewer than
    MIN_DISTINCT different characters; RTTY's own RYRY test pattern excepted).
"""
import re

from src.intel import psk31, rtty

UNSURE = ("CW", "CARRIER", "UNKNOWN", "AFSK", "USB", "LSB")   # SSB: a narrow data signal often looks like it
DIGITAL = ("RTTY", "PSK31", "AFSK")
OVERRIDE_MARGIN = 0.3        # the slow-path model's lead for "not digital" that the override may not overrule
MIN_DISTINCT = 6             # different characters a real decoded message has at least
RYRY = re.compile(r"^(?:RY|YR|\s)+$")


def model_says_not_digital(slow):
    """The slow-path model (narrowband.refine's info) is clearly sure this is
    voice, Morse or a carrier, not RTTY / PSK31 / AFSK."""
    probs = (slow or {}).get("probs") or {}
    if not probs:
        return False
    best_other = max((p for k, p in probs.items() if k not in DIGITAL and k != "NOISE"), default=0.0)
    best_digital = max((probs.get(k, 0.0) for k in DIGITAL), default=0.0)
    return best_other - best_digital >= OVERRIDE_MARGIN


def looks_like_text(text):
    """Real decoded traffic uses many different characters; a chance framing
    of a regular signal repeats two or three."""
    chars = set(re.sub(r"[\s/]", "", text or "").upper())
    return len(chars) >= MIN_DISTINCT or bool(RYRY.match((text or "").strip().upper()) and len(text.strip()) >= 6)


def _rtty(y, fs):
    return dict(rtty.decode(y, fs), mode="RTTY")


def _psk(y, fs):
    r = psk31.decode(y, fs)
    return dict(r, mode=r.get("mode") or "PSK31")


def read(kind, y, fs, slow=None):
    """-> (kind, decode result or None). y: the zoomed complex channel; slow:
    the slow-path model's opinion (narrowband.refine), when there is one."""
    if kind == "RTTY":
        return kind, _rtty(y, fs)
    if kind == "PSK31":
        return kind, _psk(y, fs)
    if kind in UNSURE and not model_says_not_digital(slow):
        for name, fn in (("RTTY", _rtty), ("PSK31", _psk)):
            r = fn(y, fs)
            if r["readable"] and looks_like_text(r.get("text")):
                r["why"] = f"{r['why']} (the classifier had said {kind}; the clean decode wins)"
                return name, r
    return kind, None


def listen_fields(r):
    """The listen-dict keys for a decode (Comms Intel and the dashboard read these)."""
    return {"digital": {k: v for k, v in r.items() if k not in ("segments",)},
            "digital_text": r["text"], "digital_segments": r["segments"]}
