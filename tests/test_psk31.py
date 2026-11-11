"""PSK31 / PSK63 / PSK125 decoder (src/intel/psk31.py) on synthetic BPSK with
Varicode and cosine-shaped reversals; noise alone gives no text."""
import numpy as np
import pytest

from src.intel import psk31

FS = 50000.0
MSG = "CQ CQ de NEXA test 599 tnx"


def make(text, baud=31.25, snr_db=15.0, off=60.0, seed=2):
    bits = [0] * 32                                   # idle: reversals
    for ch in text:
        bits += [int(b) for b in psk31.VARICODE[ch]] + [0, 0]
    bits += [0] * 32
    sps = int(round(FS / baud))
    t = np.arange(sps) / sps
    sym, out = 1.0, []
    for b in bits:
        new = sym if b else -sym
        out.append(np.full(sps, new) if new == sym else sym * np.cos(np.pi * t))
        sym = new
    env = np.concatenate(out)
    n = np.arange(len(env))
    s = env * np.exp(2j * np.pi * off * n / FS + 0.7j)
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s))) / np.sqrt(2)
    return s * np.sqrt(10 ** (snr_db / 10) * 3000 / FS) + noise


@pytest.mark.parametrize("baud,snr", [(31.25, 15), (31.25, 5), (62.5, 15), (125.0, 15)])
def test_decodes(baud, snr):
    r = psk31.decode(make(MSG, baud=baud, snr_db=snr), FS)
    assert r["readable"], r["why"]
    assert r["baud"] == baud
    assert r["text"] == MSG
    assert abs(r["carrier_hz"] - 60.0) < 2


def test_noise_gives_no_text():
    rng = np.random.default_rng(5)
    z = rng.standard_normal(300000) + 1j * rng.standard_normal(300000)
    assert psk31.decode(z, FS)["text"] == ""


def test_varicode_is_a_valid_prefix_free_code():
    codes = list(psk31.VARICODE.values())
    assert len(codes) == len(set(codes))
    assert all("00" not in c and c[0] == "1" and c[-1] == "1" for c in codes)
