"""src/intel/store.py: what Whisper or the Morse decoder read in a run becomes Comms Intel recordings."""
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
    assert store.save_decoded_tracks("r1", SOURCE, [track(1, name, "alpha bravo")], run, out) == ["r1_track_1"]
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
    assert store.save_decoded_tracks("r1", SOURCE, [track(1, name, "")], run, out, keep_empty=False) == []
    assert store.save_decoded_tracks("r2", SOURCE, [track(1, name, "")], run, out) == ["r2_track_1"]
    row = json.loads((out / "index.json").read_text(encoding="utf-8"))["recordings"][0]
    assert row["summary"] == store.NO_WORDS


def morse_track(k, audio_name, text, **listen):
    segments = [{"start_s": 1.0, "end_s": 9.0, "text": text, "wpm": 18.0, "snr_db": 21.0}] if text else []
    detail = {"chunk_s": 15.0, "chunks": {"key_down": [0.4], "snr_db": [21.0]}, "active_minutes": 0.2,
              "tone_purity": 0.8, "segments": segments, "dropped": 1}
    return {"track": k, "f_lo_khz": 1.0, "f_hi_khz": 1.4,
            "listen": {"kind": "CW", "audio": audio_name, "morse": text, "morse_detail": detail, **listen}}


def test_morse_saved_with_its_reasoning(tmp_path):
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    name = wav(run)
    reading = {"category": "amateur", "summary": "A station calling CQ", "confidence": "high", "call_signs": ["YB1AA"],
               "evidence": [{"quote": "CQ DE YB1AA", "why": "a general call"}]}
    assert store.save_decoded_tracks("r1", SOURCE, [morse_track(1, name, "CQ DE YB1AA", reading=reading)], run, out) == ["r1_track_1"]
    rec = json.loads((out / "r1_track_1.json").read_text(encoding="utf-8"))
    assert rec["kind"] == "morse" and "transcript" not in rec
    assert rec["segments"][0]["text"] == "CQ DE YB1AA" and rec["wpm_median"] == 18.0
    assert rec["reading"]["evidence"][0]["why"] == "a general call"
    assert rec["chunk_s"] == 15.0 and rec["dropped_noise_segments"] == 1
    row = json.loads((out / "index.json").read_text(encoding="utf-8"))["recordings"][0]
    assert row["kind"] == "morse" and row["summary"] == "A station calling CQ" and row["call_signs"] == ["YB1AA"]


def test_morse_activity_is_a_fine_timeline(tmp_path):
    from src.intel import morse
    keyed = np.r_[np.ones(100, bool), np.zeros(100, bool)]             # 0.5 s on, 0.5 s off at 200 values/s
    fine = morse.activity(keyed)
    assert fine["bin_s"] == 0.1 and fine["key_down"][:5] == [1.0] * 5 and fine["key_down"][5:] == [0.0] * 5
    assert morse.activity(np.ones(200 * 3600, bool))["bin_s"] == 12.0  # an hour stays at 300 bars
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    t = morse_track(1, wav(run), "CQ DE YB1AA")
    t["listen"]["morse_detail"]["activity"] = fine
    store.save_decoded_tracks("r1", SOURCE, [t], run, out)
    rec = json.loads((out / "r1_track_1.json").read_text(encoding="utf-8"))
    assert rec["chunk_s"] == 0.1 and rec["chunks"]["key_down"] == fine["key_down"]
    assert rec["chunks"]["snr_db"][:5] == [21.0] * 5 and rec["chunks"]["snr_db"][5:] == [0.0] * 5


def test_morse_without_ai_shows_its_text_and_unreadable_morse_is_marked(tmp_path):
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    name = wav(run)
    store.save_decoded_tracks("r1", SOURCE, [morse_track(1, name, "VVV DE TEST")], run, out)
    assert store.save_decoded_tracks("r2", SOURCE, [morse_track(1, name, "")], run, out, keep_empty=False) == []
    store.save_decoded_tracks("r3", SOURCE, [morse_track(1, name, "")], run, out)
    rows = {r["id"]: r for r in json.loads((out / "index.json").read_text(encoding="utf-8"))["recordings"]}
    assert rows["r1_track_1"]["summary"] == "VVV DE TEST"
    assert rows["r3_track_1"]["summary"] == store.NO_MORSE


def test_only_tracks_whisper_heard_once_each(tmp_path):
    run, out = tmp_path / "run", tmp_path / "intel"
    run.mkdir()
    name = wav(run)
    joined = track(1, name, "same transmission")
    morse = {"track": 3, "f_lo_khz": 1.0, "f_hi_khz": 2.0, "listen": {"kind": "CW", "audio": name, "morse": "CQ"}}
    tracks = [joined, dict(joined, track=2), morse, {"track": 4, "f_lo_khz": 1, "f_hi_khz": 2}]
    assert store.save_decoded_tracks("r1", {"file": "a.wav"}, tracks, run, out) == ["r1_track_1"]
