"""Recording input, analog zoom rules, demodulation and the /api/pipeline routes."""
import json
import struct
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.pipeline import analog, demod, iqio, listen  # noqa: E402
from src.pipeline import analog_synth as S  # noqa: E402

FS = iqio.FS


def iq_wav(path, iq, fs, centre=None, bits=16):
    """A 2-channel IQ WAV like SDRconnect writes, with an optional auxi chunk."""
    x = np.empty(2 * len(iq), np.float32)
    x[0::2], x[1::2] = iq.real, iq.imag
    data = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    chunks = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 2, int(fs), int(fs) * 4, 4, bits)
    if centre:
        aux = bytes(32) + struct.pack("<I", int(centre)) + bytes(8)
        chunks += struct.pack("<4sI", b"auxi", len(aux)) + aux
    chunks += struct.pack("<4sI", b"data", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def test_reads_sdrconnect_style_wav_and_centre(tmp_path):
    t = np.arange(32000) / FS
    iq = (0.3 * np.exp(2j * np.pi * 100e3 * t)).astype(np.complex64)
    iq_wav(tmp_path / "rec.wav", iq, FS, centre=446_006_250)
    out, info = iqio.load(tmp_path / "rec.wav")
    assert info["centre_hz"] == 446_006_250 and not info["resampled"]
    assert np.abs(out - iq).max() < 1e-3


def test_resamples_narrower_recording_and_fills_its_edges(tmp_path):
    rng = np.random.default_rng(0)
    fs = 2e6
    n = 40000
    iq = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.05).astype(np.complex64)
    iq_wav(tmp_path / "rec2.wav", iq, fs)
    out, info = iqio.load(tmp_path / "rec2.wav")
    assert info["resampled"] and info["edge_filled"] and abs(len(out) - n * FS / fs) < 5
    P = np.abs(np.fft.fft(out)) ** 2
    f = np.fft.fftfreq(len(out), 1 / FS)
    inside, edge = np.median(P[np.abs(f) < 0.7e6]), np.median(P[np.abs(f) > 1.2e6])
    assert 0.5 < edge / inside < 2                    # the band edges hold noise at the floor, not nothing


def test_audio_wav_is_refused_with_a_hint(tmp_path):
    from src.intel import audio
    audio.write_wav(tmp_path / "audio.wav", np.zeros(800), 8000)
    with pytest.raises(ValueError, match="Recording type = IQ"):
        iqio.load(tmp_path / "audio.wav")


EXPECT = {"NBFM": "NBFM", "AM": "AM", "AM_SC": "AM_SC", "USB": "USB", "LSB": "LSB",
          "USB_WC": "USB", "LSB_WC": "LSB", "CW": "CW", "CARRIER": "CARRIER"}


@pytest.mark.parametrize("kind", list(EXPECT))
def test_analog_kind_at_a_strong_signal(kind):
    hits = sum(analog.classify(S.make(kind, 1.0, 30, np.random.default_rng(s * 7 + 1))[0])[0] == EXPECT[kind]
               for s in range(4))
    assert hits >= 3


def test_noise_alone_is_not_heard_as_voice():
    rng = np.random.default_rng(5)
    y = (rng.standard_normal(50000) + 1j * rng.standard_normal(50000))
    assert analog.classify(y)[0] in ("UNKNOWN", "CARRIER")


def _rhythm(x, fs, blk=0.02):
    b = int(blk * fs)
    n = len(x) // b
    return np.sqrt(np.mean(x[:n * b].reshape(n, b) ** 2, axis=1))


@pytest.mark.parametrize("kind", ["NBFM", "AM", "USB", "LSB"])
def test_demodulated_speech_follows_the_original(kind):
    y, a = S.make(kind, 2.0, 30, np.random.default_rng(3))
    k, m, _ = analog.classify(y)
    audio = demod.demodulate(y, S.FA, k, m)
    ra, ro = _rhythm(audio, demod.AUDIO_FS), _rhythm(a, S.FA)
    n = min(len(ra), len(ro))
    assert np.corrcoef(ra[:n], ro[:n])[0, 1] > 0.9


def test_tracks_on_one_frequency_are_listened_to_together():
    def ans(n, t0, lo, hi, cls="AM_SSB_WC"):
        return {"track": n, "t_start_ms": t0, "t_end_ms": t0 + 50, "f_lo_khz": lo, "f_hi_khz": hi,
                "answer": cls, "classes": [cls], "top5": [(cls, 0.3)]}
    answers = [ans(1, 0, 220, 312), ans(2, 260, 219, 313), ans(3, 10, 850, 937), ans(4, 5, -100, 900, "QPSK")]
    g = listen.groups(answers)
    assert sorted(sorted(a["track"] for a in grp) for grp in g) == [[1, 2], [3]]


# --- /api/pipeline -----------------------------------------------------------------

@pytest.fixture
def server(tmp_path, monkeypatch):
    import serve_local
    monkeypatch.setattr(serve_local, "PIPELINE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(serve_local, "CLASSIFIER_DIR", tmp_path / "none")
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0, demo=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    srv.server_close()


def call(url, method="GET", body=None):
    req = urllib.request.Request(url, data=body, method=method, headers={"X-NEXA-Client": "1"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_pipeline_lists_runs_and_reads_a_finished_one(server):
    base, tmp = server
    run = tmp / "jobs" / "20261106-101500-ab12"
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({"id": run.name, "source": "voice-demo"}))
    (run / "tracks.json").write_text(json.dumps({"tracks": [], "duration_ms": 1200}))
    status, data = call(base + "/api/pipeline")
    assert status == 200 and data["runs"][0]["status"] == "done" and data["classifiers"] == []
    status, one = call(base + f"/api/pipeline/{run.name}")
    assert status == 200 and one["result"]["duration_ms"] == 1200


@pytest.mark.parametrize("bad", ["../nexa", "20261106-101500-ab12/../../x", "nope"])
def test_pipeline_refuses_other_paths(server, bad):
    base, _ = server
    assert call(base + f"/api/pipeline/{bad}")[0] == 404


def test_pipeline_start_needs_a_classifier(server):
    base, _ = server
    status, data = call(base + "/api/pipeline?source=voice-demo", "POST", b"")
    assert status == 409 and "classifier" in data["error"]
