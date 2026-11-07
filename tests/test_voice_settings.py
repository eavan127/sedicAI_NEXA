"""The speech step's settings: model and language reach Whisper, a missing model
stops instead of downloading, and doubt marks reach the pipeline's result. Whisper
itself is faked, so these run without it installed."""
import sys
import threading
import urllib.error
import urllib.request
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.intel import audio, voice  # noqa: E402
from src.pipeline import analog_synth, extract, listen  # noqa: E402


class FakeWhisper:
    def __init__(self, segments=None):
        self.calls = []
        self.segments = segments if segments is not None else [
            {"start": 0.0, "end": 1.0, "text": " hari ini baik", "avg_logprob": -0.3, "no_speech_prob": 0.1},
            {"start": 1.0, "end": 2.0, "text": " tak pasti", "avg_logprob": -1.4, "no_speech_prob": 0.1},
        ]

    def transcribe(self, audio_in, **kw):
        self.calls.append(kw)
        return {"text": " ".join(s["text"].strip() for s in self.segments), "language": kw["language"] or "ms",
                "segments": self.segments}


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "x.wav"
    audio.write_wav(p, 0.3 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000), 16000)
    return audio.wav_info(p)


@pytest.fixture
def fake(monkeypatch):
    f = FakeWhisper()
    monkeypatch.setitem(voice._MODELS, "medium", f)
    return f


def test_default_is_medium():
    assert voice.DEFAULT_MODEL == "medium"


def test_a_missing_model_stops_and_never_imports_whisper(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delitem(voice._MODELS, "large-v3", raising=False)
    monkeypatch.setitem(sys.modules, "whisper", None)     # importing it would now raise ImportError instead
    with pytest.raises(FileNotFoundError, match="never downloads"):
        voice.load("large-v3")


def test_installed_models_reads_whisper_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert voice.installed_models() == []
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "medium.pt").write_bytes(b"x")
    (tmp_path / "whisper" / "base.pt").write_bytes(b"x")
    assert voice.installed_models() == ["base", "medium"]


@pytest.mark.parametrize("given,sent", [("ms", "ms"), ("en", "en"), ("auto", None), (None, None), ("", None)])
def test_language_reaches_whisper(clip, fake, given, sent):
    voice.transcribe(clip, model="medium", language=given)
    assert fake.calls[-1]["language"] == sent


def test_text_is_not_fed_forward_between_segments(clip, fake):
    voice.transcribe(clip, model="medium")
    assert fake.calls[-1]["condition_on_previous_text"] is False


def test_doubtful_segments_are_marked(clip, fake):
    tr = voice.transcribe(clip, model="medium", language="ms")
    assert [s["doubtful"] for s in tr["segments"]] == [False, True]
    assert tr["doubtful_share"] == pytest.approx(2 / 5, abs=0.001)
    assert tr["model"] == "medium"


def test_no_speech_probability_also_marks_doubt(clip, monkeypatch):
    f = FakeWhisper([{"start": 0, "end": 1, "text": " hello", "avg_logprob": -0.2, "no_speech_prob": 0.9}])
    monkeypatch.setitem(voice._MODELS, "medium", f)
    assert voice.transcribe(clip, model="medium")["segments"][0]["doubtful"] is True


def _nbfm_answer():
    rng = np.random.default_rng(1)
    iq, _ = analog_synth.wideband([("NBFM", -600e3, 30)], seconds=1.5, rng=rng)
    ans = {"track": 1, "t_start_ms": 0.0, "t_end_ms": 1500.0, "f_lo_khz": -620.0, "f_hi_khz": -580.0,
           "answer": "NBFM", "classes": ["NBFM"], "top5": [("NBFM", 0.5)]}
    return iq, ans, extract.noise_variance(np.asarray(iq[:32768])), rng


def test_listen_passes_the_settings_and_keeps_doubt_marks(fake, tmp_path, monkeypatch):
    iq, ans, nv, rng = _nbfm_answer()
    out = listen.listen_track(iq, ans, tmp_path, nv, rng, whisper_model="medium", language="ms", use_llm=False)
    assert out["kind"] == "NBFM", out
    assert fake.calls[-1]["language"] == "ms"
    assert out["whisper_model"] == "medium"
    assert [s["doubtful"] for s in out["segments"]] == [False, True]


def test_ai_reading_only_gets_what_whisper_was_sure_of(fake, tmp_path, monkeypatch):
    from src.intel import reason
    seen = {}

    def fake_read(name, segments, **kw):
        seen["segments"] = segments
        return {"summary": "ok"}
    monkeypatch.setattr(reason, "read_recording", fake_read)
    iq, ans, nv, rng = _nbfm_answer()
    listen.listen_track(iq, ans, tmp_path, nv, rng, whisper_model="medium", use_llm=True)
    assert [s["text"] for s in seen["segments"]] == ["hari ini baik"]


# --- the dashboard's /api/pipeline ---------------------------------------------------

@pytest.fixture
def server(tmp_path, monkeypatch):
    import serve_local
    monkeypatch.setattr(serve_local, "PIPELINE_DIR", tmp_path / "jobs")
    monkeypatch.setattr(serve_local, "CLASSIFIER_DIR", tmp_path / "classifiers")
    (tmp_path / "classifiers").mkdir()
    (tmp_path / "classifiers" / "seed4_best.pt").write_bytes(b"x")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    (tmp_path / "cache" / "whisper").mkdir(parents=True)
    (tmp_path / "cache" / "whisper" / "base.pt").write_bytes(b"x")
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


def test_dashboard_is_told_what_is_installed(server):
    base, _ = server
    status, data = call(base + "/api/pipeline")
    assert status == 200 and data["whisper_models"] == ["base"]
    assert data["whisper_default"] == "medium" and "ms" in data["languages"]


@pytest.mark.parametrize("query,expect", [("whisper_model=medium", "not on this machine"),
                                          ("whisper_model=../../evil", "not on this machine"),
                                          ("whisper_model=base&language=xx", "language must be")])
def test_bad_settings_are_refused_before_anything_starts(server, query, expect):
    base, tmp = server
    status, data = call(base + f"/api/pipeline?source=voice-demo&{query}", "POST", b"")
    assert status == 400 and expect in data["error"]
    assert not list((tmp / "jobs").glob("*"))             # no run folder was made
