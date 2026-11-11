"""
The files behind the Comms Intel page, results/comms_intel/ (git-ignored): one
JSON per recording and index.json. Written by scripts/comms_intel.py for folders
of audio, and by save_whisper_tracks() for every track Whisper listens to in the
live bridge or in Signal Pipeline's "Analyse file", and for every track the
Morse decoder reads there. Each keeps its local AI reading when one was made.
"""
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import REPO_ROOT

from . import audio

INTEL_DIR = REPO_ROOT / "results" / "comms_intel"
VERSION = 1
NO_WORDS = "Whisper listened to this and recognised no words."
NO_MORSE = "The Morse decoder listened to this and could read no text."


def safe_id(stem):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "recording"


def summary_row(rec):
    r = rec.get("reading") or {}
    said = ((rec.get("transcript") or {}).get("text") or " / ".join(s["text"] for s in rec["segments"][:5]))[:200]
    return {k: rec.get(k) for k in ("id", "file", "recorded_at", "duration_s", "analysed_s", "kind",
                                    "active_minutes", "wpm_median", "processed_at")} | {
        "transmissions": len(rec["segments"]),
        "category": r.get("category"), "confidence": r.get("confidence"),
        "summary": r.get("summary") or said or rec.get("note"), "call_signs": r.get("call_signs", []),
        "error": r.get("error"), "model": r.get("model"),
    }


def _replace(tmp, target):
    """Windows refuses the swap for a moment while the dashboard is reading the file."""
    for attempt in range(20):
        try:
            return tmp.replace(target)
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def write_index(out):
    out = Path(out)
    rows = []
    for p in sorted(out.glob("*.json")):
        # Dot files are not recordings: the server's audit markers
        # (.audited-<db>.json, a list of logged ids) live here too.
        if p.name == "index.json" or p.name.startswith("."):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict):
                rows.append(summary_row(rec))
        except (json.JSONDecodeError, KeyError):
            continue
    rows.sort(key=lambda r: (r.get("recorded_at") or "", r["file"]))
    tmp = out / f"index.{os.getpid()}.tmp"                # the bridge and a file run can finish together
    tmp.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "recordings": rows}, indent=1), encoding="utf-8")
    _replace(tmp, out / "index.json")
    return rows


def _activity(segments, duration_s, level_db):
    """The page's "when was it sending" strip, from when Whisper heard speech. -> (chunk_s, chunks)"""
    chunk_s = max(1.0, duration_s / 200)
    speech = [0.0] * max(1, math.ceil(duration_s / chunk_s))
    for s in segments:
        i = int(s["start_s"] // chunk_s)
        while i < len(speech) and i * chunk_s < s["end_s"]:
            overlap = min(s["end_s"], (i + 1) * chunk_s) - max(s["start_s"], i * chunk_s)
            speech[i] = round(min(1.0, speech[i] + max(overlap, 0.0) / chunk_s), 3)
            i += 1
    return chunk_s, {"key_down": speech, "snr_db": [level_db if v > 0 else 0.0 for v in speech]}


def fine_activity(morse):
    """The decoder's short-moment key track, in the page's timeline form. -> (chunk_s, chunks)"""
    fine = morse["activity"]
    heard = [s for s, d in zip(morse["chunks"]["snr_db"], morse["chunks"]["key_down"]) if d > 0]
    level = float(statistics.median(heard)) if heard else 0.0
    return fine["bin_s"], {"key_down": fine["key_down"], "snr_db": [level if d > 0 else 0.0 for d in fine["key_down"]]}


def save_decoded_tracks(run_id, source, tracks, run_dir, out=INTEL_DIR, keep_empty=True):
    """Every track of this run that Whisper or the Morse decoder worked on
    becomes one Comms Intel recording. keep_empty=False leaves out tracks with
    no text (fixed slices of a live stream would otherwise add a row every few
    seconds). -> ids saved"""
    out, run_dir = Path(out), Path(run_dir)
    saved, seen = [], set()
    for t in tracks:
        L = t.get("listen") or {}
        morse = L.get("morse_detail")
        if (morse is None and "transcript" not in L) or not L.get("audio") or L["audio"] in seen:
            continue
        seen.add(L["audio"])                              # tracks joined on one frequency share one listen
        text = (L["morse"] if morse else L["transcript"]).strip()
        if not text and not keep_empty:
            continue
        info = audio.wav_info(run_dir / L["audio"])
        if morse:
            segments, chunk_s, chunks = morse["segments"], morse["chunk_s"], morse["chunks"]
            active_minutes, dropped = morse["active_minutes"], morse["dropped"]
            if morse.get("activity"):
                chunk_s, chunks = fine_activity(morse)
        else:
            level = L.get("snr_db", (L.get("features") or {}).get("snr_db")) or 0.0
            segments = [dict(s, snr_db=level) for s in L.get("speech_segments", [])]
            chunk_s, chunks = _activity(segments, info.duration_s, level)
            active_minutes, dropped = round(sum(s["end_s"] - s["start_s"] for s in segments) / 60, 2), 0
        wpm = [s["wpm"] for s in segments if s.get("wpm")]
        mid_khz = (t["f_lo_khz"] + t["f_hi_khz"]) / 2
        centre = source.get("centre_hz")
        where = f"{(centre + mid_khz * 1e3) / 1e6:.4f} MHz" if centre else f"{mid_khz:+.1f} kHz"
        rid = safe_id(f"{run_id}_track_{t['track']}")
        rec = {"version": VERSION, "id": rid, "file": f"{source.get('file', run_id)} · {where}",
               "recorded_at": datetime.now().isoformat(timespec="seconds"),
               "duration_s": round(info.duration_s, 1), "analysed_s": round(info.duration_s, 1),
               "sample_rate": info.sample_rate, "bits": info.bits, "kind": "morse" if morse else "voice",
               "active_minutes": active_minutes,
               "chunk_s": chunk_s, "chunks": chunks, "segments": segments, "dropped_noise_segments": dropped,
               "wpm_median": float(statistics.median(wpm)) if wpm else None, "pipeline_run": run_id,
               "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if morse:
            rec["tone_purity"] = morse["tone_purity"]
        else:
            rec["transcript"] = {"text": text, "language": L.get("language")}
        if L.get("reading"):
            rec["reading"] = L["reading"]
        if not text:
            rec["note"] = NO_MORSE if morse else NO_WORDS
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / f"{rid}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        _replace(tmp, out / f"{rid}.json")
        saved.append(rid)
    if saved:
        write_index(out)
    return saved
