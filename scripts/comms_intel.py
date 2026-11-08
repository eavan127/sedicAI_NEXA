"""
Comms intel over a folder of receiver AUDIO recordings (WAV): Morse decoding,
Whisper for voice, then a local Ollama reading of the text. Results go to
results/comms_intel/ (git-ignored), where the local server's Comms Intel
page (scripts/serve_local.py -> /api/intel) shows them.

Everything runs on this machine; no audio or text is sent anywhere.

    python scripts/comms_intel.py --in "F:/12234 indonesia cw"
    python scripts/comms_intel.py --in DIR --limit 3 --max-hours 1     # quick look
    python scripts/comms_intel.py --in DIR --no-llm                    # decode only
    python scripts/comms_intel.py --in DIR --relabel                   # redo only the LLM step
    python scripts/comms_intel.py --demo                               # made-up Morse, to try the page

A file already processed is skipped unless --force (or --relabel for the LLM
part only). Needs: Ollama running with the model pulled (`ollama pull
llama3.2:3b`); openai-whisper only if a recording turns out to be voice.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.intel import audio, morse, reason  # noqa: E402
from src.intel.store import VERSION, safe_id, write_index  # noqa: E402

OUT_DEFAULT = REPO_ROOT / "results" / "comms_intel"
MORSE_PURITY = 0.25        # share of audio-band power in the beep; speech spreads far wider


def recorded_at(stem):
    """'01Aug2018-091847' (the recorder's own naming) -> ISO time, else None."""
    try:
        return datetime.strptime(stem[:16], "%d%b%Y-%H%M%S").isoformat()
    except ValueError:
        return None


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


def sidecar(path):
    """The recorder's .log next to the WAV, if any (shown on the page, not sent to the model)."""
    log = path.with_suffix(".log")
    if log.is_file() and log.stat().st_size < 200_000:
        return log.read_bytes().decode("utf-8", errors="replace")[:4000]
    return None


def process(path, args):
    info = audio.wav_info(path)
    max_s = args.max_hours * 3600 if args.max_hours else None
    t0 = time.time()
    last = [0.0]

    def progress(start):
        if start - last[0] >= 3600:
            last[0] = start
            print(f"    {hms(start)} of {hms(info.duration_s)}  ({time.time() - t0:.0f}s)", flush=True)

    m = morse.analyse(info, max_s=max_s, progress=progress)
    kind = "morse" if m["tone_purity"] >= MORSE_PURITY or m["segments"] else "voice"
    if m["active_minutes"] == 0:
        kind = "silent"
    rec = {
        "version": VERSION, "id": safe_id(path.stem), "file": path.name,
        "recorded_at": recorded_at(path.stem), "duration_s": round(info.duration_s, 1),
        "analysed_s": round(min(info.duration_s, max_s or info.duration_s), 1),
        "sample_rate": info.sample_rate, "bits": info.bits, "kind": kind,
        "active_minutes": m["active_minutes"], "tone_purity": m["tone_purity"],
        "chunk_s": m["chunk_s"], "chunks": m["chunks"],
        "segments": m["segments"], "dropped_noise_segments": m["dropped"],
        "recorder_log": sidecar(path),
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if kind == "voice" and not args.no_whisper:
        from src.intel import voice
        print("    voice-like: Whisper on the loudest parts ...", flush=True)
        try:
            rec["transcript"] = voice.transcribe(info, model=args.whisper_model, max_minutes=args.whisper_minutes)
        except Exception as e:                       # noqa: BLE001 -- whisper missing or failed: keep the rest
            rec["transcript"] = {"error": str(e)}
    wpm = [s["wpm"] for s in rec["segments"]]
    rec["wpm_median"] = float(np.median(wpm)) if wpm else None
    print(f"    {kind}, {m['active_minutes']} min with signal, {len(rec['segments'])} transmissions"
          f" ({m['dropped']} noise-like dropped), {time.time() - t0:.0f}s", flush=True)
    return rec


def add_reading(rec, args):
    segs = rec["segments"]
    kind = rec["kind"]
    if kind == "voice" and rec.get("transcript", {}).get("segments"):
        segs = rec["transcript"]["segments"]
    if args.no_llm or kind == "silent":
        rec.pop("reading", None)
        return rec
    t0 = time.time()
    rec["reading"] = reason.read_recording(rec["file"], segs, kind="morse" if kind == "morse" else "voice",
                                           duration=hms(rec["analysed_s"]), active_min=rec["active_minutes"],
                                           model=args.model, url=args.ollama)
    r = rec["reading"]
    print(f"    LLM: {r.get('error') or (r.get('category'), r.get('confidence'))}  ({time.time() - t0:.0f}s)",
          flush=True)
    return rec


def demo_folder(out_dir):
    """A few made-up recordings, so the page can be tried without real data."""
    d = out_dir / "_demo_wavs"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    scenes = {
        "05Nov2026-090000": [("CQ CQ CQ DE YB1DEMO YB1DEMO K", 18, 700), ("YB1DEMO DE 9M2TST GM UR RST 579 QTH KUALA LUMPUR K", 22, 650)],
        "05Nov2026-101500": [("VVV VVV DE TEST BEACON =", 15, 800)] * 3,
        "05Nov2026-113000": [("QTC 3 = 48213 99021 55871 = AR", 20, 600)],
    }
    for stem, parts in scenes.items():
        x = [np.zeros(8000 * 2)]
        for text, wpm, tone in parts:
            x += [morse.synth(text, wpm=wpm, tone=tone, snr_db=float(rng.uniform(0, 12)), rng=rng), np.zeros(8000 * 5)]
        audio.write_wav(d / f"{stem}_DEMO.wav", 0.6 * np.concatenate(x), 8000, bits=8)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", help="folder of .wav files (or one .wav)")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    p.add_argument("--demo", action="store_true", help="make and process three made-up Morse recordings")
    p.add_argument("--limit", type=int, default=0, help="first N files only")
    p.add_argument("--max-hours", type=float, default=0, help="analyse at most this much of each file")
    p.add_argument("--force", action="store_true", help="redo files already processed")
    p.add_argument("--relabel", action="store_true", help="keep the decode, redo only the LLM reading")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--model", default=reason.DEFAULT_MODEL, help="Ollama model (e.g. llama3.2:3b, qwen2.5:7b)")
    p.add_argument("--ollama", default=reason.DEFAULT_URL)
    p.add_argument("--no-whisper", action="store_true")
    p.add_argument("--whisper-model", default="base")
    p.add_argument("--whisper-minutes", type=float, default=10.0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.demo:
        args.inp = str(demo_folder(out))
    if not args.inp:
        p.error("--in DIR (or --demo) is required")
    src = Path(args.inp)
    files = [src] if src.is_file() else sorted(src.glob("*.wav"), key=lambda f: f.name.lower())
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} recordings -> {out}")
    for i, f in enumerate(files, 1):
        target = out / f"{safe_id(f.stem)}.json"
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
        try:
            if target.exists() and not args.force:
                rec = json.loads(target.read_text(encoding="utf-8"))
                failed = "error" in (rec.get("reading") or {})      # e.g. Ollama timed out last time
                if not (args.relabel or (failed and not args.no_llm)):
                    print("    done before (--force to redo)")
                    continue
            else:
                rec = process(f, args)
            rec = add_reading(rec, args)
        except (ValueError, OSError) as e:
            print(f"    skipped: {e}")
            continue
        target.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        write_index(out)
    rows = write_index(out)
    print(f"index: {len(rows)} recordings in {out / 'index.json'}")


if __name__ == "__main__":
    main()
