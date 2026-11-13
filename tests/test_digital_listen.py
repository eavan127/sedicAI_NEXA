"""RTTY and PSK31 through the live voice path (scripts/live_sdrconnect.py
voice_snapshot) into Comms Intel (src/intel/store.py), on a synthetic
wideband capture with one of each; speech-like SSB is not decoded as text."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
live = pytest.importorskip("live_sdrconnect")

from src.intel import digital, store  # noqa: E402
from tests.test_psk31 import make as make_psk  # noqa: E402
from tests.test_rtty import make as make_rtty  # noqa: E402

FS = 250e3


def at(sig, offset_hz, n):
    w = resample_poly(sig, 5, 1)                       # 50 kS/s -> 250 kS/s
    w = np.pad(w, (0, max(0, n - len(w))))[:n]
    return w * np.exp(2j * np.pi * offset_hz * np.arange(n) / FS)


def test_rtty_and_psk31_are_read_and_saved(tmp_path):
    rtty_sig = make_rtty("RYRYRY CQ CQ DE NEXA RTTY TEST K", snr_db=40, off=0.0)
    psk_sig = make_psk("CQ CQ de NEXA psk test k", snr_db=40, off=0.0)
    rtty_sig, psk_sig = (s / np.sqrt(np.mean(np.abs(s) ** 2)) for s in (rtty_sig, psk_sig))
    n = int(FS * max(len(rtty_sig), len(psk_sig)) / 50e3)
    rng = np.random.default_rng(1)
    amp = np.sqrt(10 ** (25 / 10) * 3000 / FS)
    iq = (amp * (at(rtty_sig, 30e3, n) + at(psk_sig, -40e3, n))
          + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
    tracks = live.voice_snapshot(iq, FS, tmp_path, "small", None, False, False, None)
    got = {t["listen"]["kind"]: t["listen"].get("digital_text") for t in tracks}
    assert got.get("RTTY") == "RYRYRY CQ CQ DE NEXA RTTY TEST K", tracks
    assert got.get("PSK31") == "CQ CQ de NEXA psk test k", tracks
    saved = store.save_decoded_tracks("20261111-120000-abcd", {"file": "test", "centre_hz": 7.04e6}, tracks,
                                      tmp_path, out=tmp_path / "intel")
    kinds = {json.loads((tmp_path / "intel" / f"{i}.json").read_text(encoding="utf-8"))["kind"] for i in saved}
    assert kinds == {"rtty", "psk31"}


def test_noise_and_voice_like_signals_stay_untouched():
    rng = np.random.default_rng(7)
    fs = 50e3
    t = np.arange(int(3 * fs)) / fs
    # SSB-like: a few wandering audio tones, the kind of thing speech looks like
    voice = sum(np.exp(2j * np.pi * (f0 + 80 * np.sin(2 * np.pi * 3 * t + k)) * t)
                for k, f0 in enumerate((300, 700, 1300)))
    for y, kind in ((voice + 0.3 * rng.standard_normal(len(t)), "USB"),
                    (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)), "UNKNOWN")):
        k, r = digital.read(kind, y, fs)
        assert k == kind and r is None


def test_voice_or_morse_is_not_taken_over_by_a_chance_rtty_framing():
    """2026-11-13: an overloaded walkie-talkie and a Morse file each framed
    as RTTY and lost their text. The model's clear 'not digital', or text that
    is two letters repeating, must keep the original kind."""
    assert digital.model_says_not_digital({"probs": {"NBFM": 0.545, "LSB": 0.179, "NOISE": 0.222, "RTTY": 0.0}})
    assert digital.model_says_not_digital({"probs": {"CW": 1.0, "RTTY": 0.0}})
    assert not digital.model_says_not_digital({"probs": {"AFSK": 0.6, "USB": 0.3}})
    assert not digital.model_says_not_digital(None)
    assert not digital.looks_like_text("FFDDFDDFDDFFDDFDDFDDFDDFDDFFF / FF / FSSS")
    assert not digital.looks_like_text("K / VYV / K / QV / K / QVV / K / Q")
    assert digital.looks_like_text("RYRYRY CQ CQ DE NEXA RTTY TEST K") and digital.looks_like_text("RYRYRYRY")
    fs = 50e3
    t = np.arange(int(3 * fs)) / fs
    tone = np.exp(2j * np.pi * 1000 * t)                     # a held carrier: framing it as RTTY is pure chance
    k, r = digital.read("USB", tone, fs, slow={"probs": {"NBFM": 0.6, "RTTY": 0.0}})
    assert k == "USB" and r is None
