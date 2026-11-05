"""Comms intel: Morse decoding, WAV reading, LLM grounding, /api/intel."""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.intel import audio, morse, reason  # noqa: E402

MSG = "CQ CQ DE YB1ABC K"


def wav(tmp_path, x, name="m.wav", bits=8):
    p = tmp_path / name
    audio.write_wav(p, x, 8000, bits=bits)
    return audio.wav_info(p)


@pytest.mark.parametrize("wpm,snr", [(12, 10), (20, 0), (30, 5)])
def test_decodes_synthetic_morse(tmp_path, wpm, snr):
    x = morse.synth(MSG, wpm=wpm, tone=700, snr_db=snr, rng=np.random.default_rng(wpm))
    r = morse.analyse(wav(tmp_path, x))
    assert [s["text"] for s in r["segments"]] == [MSG]
    assert abs(r["segments"][0]["wpm"] - wpm) / wpm < 0.15
    assert abs(r["segments"][0]["tone_hz"] - 700) < 10


def test_two_transmissions_at_different_pitch_and_speed(tmp_path):
    x = np.r_[morse.synth(MSG, wpm=15, tone=600), np.zeros(8000 * 20),
              morse.synth("QTH JAKARTA", wpm=25, tone=950)]
    segs = morse.analyse(wav(tmp_path, x))["segments"]
    assert [s["text"] for s in segs] == [MSG, "QTH JAKARTA"]
    assert segs[1]["start_s"] > segs[0]["end_s"] + 15


def test_pure_noise_gives_no_text(tmp_path):
    x = np.random.default_rng(1).normal(0, 0.2, 8000 * 60)
    r = morse.analyse(wav(tmp_path, x))
    assert r["segments"] == []


def test_wav_reader_handles_8_and_16_bit(tmp_path):
    x = 0.5 * np.sin(np.linspace(0, 100, 8000))
    for bits in (8, 16):
        info = wav(tmp_path, x, f"s{bits}.wav", bits)
        assert info.bits == bits and info.sample_rate == 8000 and info.n_frames == 8000
        y = np.concatenate([c for _, c in audio.iter_chunks(info, chunk_s=0.3)])
        assert np.max(np.abs(y - x)) < (0.01 if bits == 8 else 1e-4)


def test_ground_removes_what_the_text_does_not_say():
    src = "CQ CQ DE YB1ABC K\nQTH JAKARTA"
    out = reason.ground({"summary": "s", "category": "Calling / contact", "confidence": "high",
                         "call_signs": ["YB1ABC", "PK4XYZ", "VVV"], "places": ["Jakarta", "Bandung"],
                         "evidence": [{"quote": "de yb1abc", "why": "id"}, {"quote": "made up", "why": "x"}]}, src)
    assert out["call_signs"] == ["YB1ABC"] and out["places"] == ["Jakarta"]
    assert out["category"] == "calling / contact"
    assert [e["quote"] for e in out["evidence"]] == ["de yb1abc"]
    assert out["removed_ungrounded"] == 4 and out["confidence"] == "medium"


def test_ground_without_evidence_is_low_confidence():
    out = reason.ground({"category": "nonsense", "confidence": "high"}, "ABC")
    assert out["category"] == "other" and out["confidence"] == "low"


def test_facts_counts():
    f = reason.facts("CQ CQ DE YB1ABC K QTH JAKARTA 48213 99021")
    assert f["cq_calls"] == 2 and dict(f["call_sign_like"]) == {"YB1ABC": 1}
    assert dict(f["q_codes"]) == {"QTH": 1} and f["five_char_groups_share"] == pytest.approx(2 / 9, abs=1e-3)


def test_read_recording_without_ollama_reports_error():
    segs = [{"start_s": 0, "text": MSG, "wpm": 20}]
    r = reason.read_recording("x.wav", segs, url="http://127.0.0.1:9")     # nothing listens on port 9
    assert "error" in r and r["facts"]["cq_calls"] == 2


# --- /api/intel -------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    import serve_local
    intel = tmp_path / "intel"
    intel.mkdir()
    (intel / "index.json").write_text(json.dumps({"recordings": [{"id": "r1", "file": "r1.wav"}]}))
    (intel / "r1.json").write_text(json.dumps({"id": "r1", "segments": []}))
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0, demo=True, intel_dir=intel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def test_intel_api_lists_and_opens(server):
    status, data = get(server + "/api/intel")
    assert status == 200 and data["recordings"][0]["id"] == "r1"
    assert get(server + "/api/intel/r1") == (200, {"id": "r1", "segments": []})


@pytest.mark.parametrize("bad", ["index", "..%2Fnexa", "nope"])
def test_intel_api_refuses_other_files(server, bad):
    assert get(server + f"/api/intel/{bad}")[0] == 404
