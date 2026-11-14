"""
A clearer copy of the military voice IQ dataset (made by scripts/make_military_voice_dataset.py),
for demos: only the clips Whisper understands well, re-transmitted at a higher SNR so the
voice stands out on a waterfall, while staying a noisy radio signal.

  1. Whisper (default `small`) transcribes each clip's radio audio (radio_audio/<id>.wav,
     already band-limited and compressed like a handheld) and is scored against the known
     text: word error rate, or character error rate for Mandarin.
  2. Clips at or under --max-error are kept ("clear speakers").
  3. Each kept clip is modulated again with the SAME modulation and frequency offset as in
     the original set, at an SNR drawn from --snr (dB in 3 kHz, default 32-38 dB: the old
     set was 12-30 dB, which looks faint on a 250 kS/s waterfall).

No Common Voice download is needed: it reads the existing dataset folder.

    python scripts/make_clear_voice_iq.py                      # -> Downloads/military_radio_IQ_clear
    python scripts/make_clear_voice_iq.py --max-error 0.3 --snr 30 36
"""
import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mv", HERE / "make_military_voice_dataset.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

DL = Path.home() / "Downloads"
WHISPER_LANG = {"malay": "ms", "english": "en", "chinese": "zh"}


def norm_words(text):
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def error_rate(ref, hyp, chinese=False):
    """Character error rate with spaces and punctuation ignored, so a spacing difference
    ("dihadapan" / "di hadapan") is not counted as a mishearing."""
    r = [c for c in ref.lower() if re.match(r"\w", c)]
    h = [c for c in hyp.lower() if re.match(r"\w", c)]
    return edit_distance(r, h) / max(len(r), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(DL / "military_radio_IQ"))
    p.add_argument("--out", default=str(DL / "military_radio_IQ_clear"))
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--max-error", type=float, default=0.12, help="keep clips Whisper gets this right (character error, 0.12 = 12%%)")
    p.add_argument("--snr", type=float, nargs=2, default=[32.0, 38.0], help="SNR range, dB in 3 kHz")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--languages", nargs="+", default=["english", "malay"], choices=list(WHISPER_LANG),
                   help="voice folders to include (Malaysian English and Malay by default)")
    p.add_argument("--no-morse", action="store_true", help="do not copy the morse/ folder as it is")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # What Whisper heard, saved after every clip: a crash (Whisper can run out of memory on an
    # 8 GB laptop) resumes where it stopped instead of starting again.
    cache_f = out / "whisper_heard.json"
    heard = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.is_file() else {}
    model = None
    rows = [r for r in csv.DictReader(open(src / "manifest.csv", encoding="utf-8-sig")) if r["folder"] in args.languages]
    kept, scored = [], []
    t_all = time.time()
    for r in rows:
        audio_path = src / r["folder"] / "radio_audio" / f"{r['id']}.wav"
        x = mv.read_wav(audio_path).astype(np.float32)
        lang = WHISPER_LANG[r["folder"]]
        if r["id"] not in heard:
            if model is None:
                import whisper
                model = whisper.load_model(args.whisper_model)
            heard[r["id"]] = model.transcribe(x, language=lang, fp16=False, condition_on_previous_text=False)["text"].strip()
            cache_f.write_text(json.dumps(heard, indent=1, ensure_ascii=False), encoding="utf-8")
        hyp = heard[r["id"]]
        err = error_rate(r["text"], hyp, chinese=lang == "zh")
        scored.append({"id": r["id"], "folder": r["folder"], "error": round(err, 3), "whisper": hyp, "text": r["text"]})
        ok = err <= args.max_error
        print(f"{r['id']:<11} {'KEEP' if ok else 'drop'} error {err:5.2f}  {hyp[:70]}", flush=True)
        if not ok:
            continue
        snr = float(np.round(rng.uniform(*args.snr), 1))
        offset = float(r["offset_khz"]) * 1e3
        iq, t0, t1 = mv.to_iq(x, r["modulation"], offset, snr, rng)
        lang_dir = out / r["folder"]
        for sub in ("iq", "radio_audio", "clean_audio"):
            (lang_dir / sub).mkdir(parents=True, exist_ok=True)
        name = Path(r["iq_file"]).name
        mv.write_iq(lang_dir / "iq" / name, iq)
        mv.write_wav(lang_dir / "radio_audio" / f"{r['id']}.wav", x)
        clean = src / r["folder"] / "clean_audio" / f"{r['id']}.wav"
        if clean.is_file():
            (lang_dir / "clean_audio" / clean.name).write_bytes(clean.read_bytes())
        row = dict(r)
        row.update({"snr_db_3khz": snr, "tx_start_s": round(t0, 3), "tx_end_s": round(t1, 3),
                    "duration_s": round(len(iq) / mv.FS_OUT, 3), "whisper_error": round(err, 3)})
        kept.append(row)

    keys = list(kept[0].keys()) if kept else []

    def write_csv(path, rs):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rs)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest.csv", kept)
    for d in sorted({r["folder"] for r in kept}):
        write_csv(out / d / "manifest.csv", [r for r in kept if r["folder"] == d])
    (out / "manifest.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "whisper_scores.json").write_text(json.dumps(scored, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = {d: sum(r["folder"] == d for r in kept) for d in args.languages}
    if not args.no_morse and (src / "morse").is_dir():           # Morse is reused unchanged
        import shutil
        shutil.copytree(src / "morse", out / "morse", dirs_exist_ok=True)
        counts["morse (copied)"] = sum(1 for _ in (out / "morse" / "iq").glob("*.wav"))
    (out / "README.txt").write_text(
        f"Clearer demo copy of military_radio_IQ: {len(kept)} of {len(rows)} voice clips {counts}.\n"
        f"Kept: Whisper ({args.whisper_model}) error <= {args.max_error} on the radio audio (whisper_scores.json has all).\n"
        f"Re-transmitted at {args.snr[0]:g}-{args.snr[1]:g} dB SNR in 3 kHz (was 12-30 dB), same modulation and offset,\n"
        f"250 kS/s complex IQ WAV (2 channels: I, Q). Made by scripts/make_clear_voice_iq.py.\n", encoding="utf-8")
    print(f"\nkept {len(kept)} of {len(rows)} {counts} in {time.time() - t_all:.0f} s -> {out}")


if __name__ == "__main__":
    main()
