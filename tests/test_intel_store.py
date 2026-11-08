"""src/intel/store.py: what Whisper analysed in a run becomes Comms Intel recordings."""
import json

import numpy as np

from src.intel import audio, store


def wav(run_dir, name="track_1.wav", seconds=30):
    audio.write_wav(run_dir / name, np.zeros(16000 * seconds, np.float32), 16000, bits=16)
    return name


def track(k, audio_name, text, **listen):
    return {"track": k, "f_lo_khz": 29.0, "f_hi_khz": 31.0,
            "listen": {"kind": "NBFM", "audio": audio_name, "transcript": text, "language": "en",
                       "speech_segments": [{"start_s": 2.0, "end_s": 6.0, "text": text}] if text else [], "snr_db": 25.0,
                       **listen}}


SOURCE = {"file": "LIVE · SDRconnect", "centre_hz": 145e6}


def test_transcript_saved_and_indexed(tmp_path):
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    name = wav(run)
    assert store.save_whisper_tracks("r1", SOURCE, [track(1, name, "alpha bravo")], run, out) == ["r1_track_1"]
    rec = json.loads((out / "r1_track_1.json").read_text(encoding="utf-8"))
    assert rec["kind"] == "voice" and rec["transcript"]["text"] == "alpha bravo"
    assert rec["segments"][0]["text"] == "alpha bravo" and rec["duration_s"] == 30
    assert "145.0300 MHz" in rec["file"]
    assert sum(rec["chunks"]["key_down"]) * rec["chunk_s"] == 4.0
    row = json.loads((out / "index.json").read_text(encoding="utf-8"))["recordings"][0]
    assert row["id"] == "r1_track_1" and row["summary"] == "alpha bravo" and row["transmissions"] == 1


def test_no_words_recorded_as_such_unless_slicing(tmp_path):
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    name = wav(run)
    assert store.save_whisper_tracks("r1", SOURCE, [track(1, name, "")], run, out, keep_empty=False) == []
    assert store.save_whisper_tracks("r2", SOURCE, [track(1, name, "")], run, out) == ["r2_track_1"]
    row = json.loads((out / "index.json").read_text(encoding="utf-8"))["recordings"][0]
    assert row["summary"] == store.NO_WORDS


def test_only_tracks_whisper_heard_once_each(tmp_path):
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    name = wav(run)
    joined = track(1, name, "same transmission")
    morse = {"track": 3, "f_lo_khz": 1.0, "f_hi_khz": 2.0, "listen": {"kind": "CW", "audio": name, "morse": "CQ"}}
    tracks = [joined, dict(joined, track=2), morse, {"track": 4, "f_lo_khz": 1, "f_hi_khz": 2}]
    assert store.save_whisper_tracks("r1", {"file": "a.wav"}, tracks, run, out) == ["r1_track_1"]
