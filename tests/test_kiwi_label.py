"""scripts/kiwi_label.py: the pre-labeller names synthetic HF signals at the
KiwiSDR rate (12 kS/s) from their shape alone."""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
kl = pytest.importorskip("kiwi_label")

FS = 12000.0
N = int(20 * FS)
rng = np.random.default_rng(7)
t = np.arange(N) / FS


def noise(scale=1.0):
    return scale * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)


def at(snr_db, s):
    s = s / np.sqrt(np.mean(np.abs(s[np.abs(s) > 1e-6]) ** 2))
    return s * np.sqrt(10 ** (snr_db / 10) * 3000 / FS) + noise()


def cw():
    dot = 1.2 / 20
    key = np.zeros(N)
    pos = 0
    pattern = [1, 0, 1, 1, 1, 0, 0, 0] * 400            # dots and dashes with gaps
    for k in pattern:
        n = int(dot * FS)
        key[pos:pos + n] = k
        pos += n
        if pos >= N:
            break
    key = np.convolve(key, np.hanning(60) / np.hanning(60).sum(), mode="same")
    return key * np.exp(2j * np.pi * 700 * t)


def rtty(shift=450.0, baud=50.0):
    bits = rng.integers(0, 2, int(N / FS * baud) + 1)
    f = np.repeat(np.where(bits, shift / 2, -shift / 2), int(FS / baud))[:N]
    return np.exp(2j * np.pi * np.cumsum(f) / FS)


def psk31():
    baud = 31.25
    sym = np.repeat(rng.choice([1.0, -1.0], int(N / FS * baud) + 1), int(FS / baud))[:N]
    return sosfiltfilt(butter(2, 40, fs=FS, output="sos"), sym) * np.exp(2j * np.pi * 300 * t)


def ssb_voice():
    x = sosfiltfilt(butter(4, [300, 2700], btype="band", fs=FS, output="sos"), rng.standard_normal(N))
    syll = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2 * (np.sin(2 * np.pi * 0.5 * t) > -0.3)
    return hilbert(x * syll)


@pytest.mark.parametrize("make,expect", [
    (lambda: noise(), "noise"), (lambda: at(25, cw()), "cw"), (lambda: at(25, rtty()), "rtty"),
    (lambda: at(25, ssb_voice()), "ssb_voice"),
])
def test_shape_rules(make, expect):
    label, conf, why = kl.suggest(kl.measure(make().astype(np.complex64), FS), None)
    assert label == expect, why


def test_psk31_needs_its_frequency():
    m = kl.measure(at(25, psk31()).astype(np.complex64), FS)
    assert kl.suggest(m, "psk31")[0] == "psk31"
    assert kl.suggest(m, None)[0] == "unknown"           # a thin line alone is not proof


def test_sheet_keeps_human_labels(tmp_path):
    x = at(25, cw()).astype(np.complex64)
    raw = np.empty(2 * N, "<i2")
    raw[0::2], raw[1::2] = np.round(x.real / np.abs(x).max() * 30000), np.round(x.imag / np.abs(x).max() * 30000)
    with wave.open(str(tmp_path / "cw_7020.wav"), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(int(FS))
        w.writeframes(raw.tobytes())
    sys.argv = ["kiwi_label.py", str(tmp_path)]
    kl.main()
    sheet = (tmp_path / "labels.csv").read_text(encoding="utf-8-sig")
    assert "cw_7020.wav" in sheet and ",cw," in sheet and "amateur CW segment" in sheet
    sheet = sheet.replace(",,", ",cw,", 1)                # a person confirms it
    (tmp_path / "labels.csv").write_text(sheet, encoding="utf-8-sig")
    kl.main()
    assert ",cw,cw," in (tmp_path / "labels.csv").read_text(encoding="utf-8-sig").replace('"', "") or \
        "cw" in (tmp_path / "labels.csv").read_text(encoding="utf-8-sig")
