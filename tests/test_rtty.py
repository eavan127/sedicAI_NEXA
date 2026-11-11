"""RTTY decoder (src/intel/rtty.py) on synthetic two-tone FSK: every common
speed / shift, both polarities, with noise; and noise alone gives no text."""
import numpy as np
import pytest

from src.intel import rtty

FS = 50000.0
MSG = "CQ CQ DE NEXA TEST RYRYRY THE QUICK BROWN FOX"


def make(text, baud=45.45, shift=170.0, snr_db=15.0, inverted=False, seed=1, off=120.0):
    rev = {c: i for i, c in enumerate(rtty.LTRS) if c and c != "\n"}
    bits = [1] * int(baud * 0.5)
    for ch in text:
        code = rev[ch]
        bits += [0] + [(code >> b) & 1 for b in range(5)] + [1, 1]
    bits += [1] * int(baud * 0.5)
    spb = FS / baud
    b = np.array(bits)[(np.arange(int(len(bits) * spb)) / spb).astype(int)]
    if inverted:
        b = 1 - b
    f = np.where(b == 1, off + shift / 2, off - shift / 2)
    s = np.exp(2j * np.pi * np.cumsum(f) / FS)
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s))) / np.sqrt(2)
    return s * np.sqrt(10 ** (snr_db / 10) * 3000 / FS) + noise


@pytest.mark.parametrize("baud,shift,inverted,snr", [(45.45, 170, False, 15), (50, 850, False, 15),
                                                     (75, 425, True, 15), (100, 450, False, 10),
                                                     (45.45, 170, True, 8)])
def test_decodes_common_settings(baud, shift, inverted, snr):
    r = rtty.decode(make(MSG, baud=baud, shift=shift, inverted=inverted, snr_db=snr), FS)
    assert r["readable"], r["why"]
    assert r["baud"] == baud and abs(r["shift_hz"] - shift) < 25
    assert r["inverted"] == inverted
    assert r["text"] == MSG


def test_figures_shift():
    rev = {c: i for i, c in enumerate(rtty.FIGS) if c}
    codes = [(rtty.FIGS_SHIFT, 0)] + [(rev[c], 0) for c in "599"] + [(rtty.LTRS_SHIFT, 0)] + [(3, 0)]
    assert "".join(c for c, _ in rtty.to_text(codes)) == "599A"


def test_noise_gives_no_text():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(200000) + 1j * rng.standard_normal(200000)
    assert rtty.decode(z, FS)["text"] == ""
