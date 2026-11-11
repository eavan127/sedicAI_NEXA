"""
Score the Signal Pipeline on the military_radio_IQ demo set (SED-44).

The set (scripts/make_military_voice_dataset.py, scripts/make_morse_dataset.py)
holds 57 voice transmissions (NBFM / AM / USB; English, Malay, Mandarin) and 27
Morse transmissions, each a 250 kS/s IQ WAV with its truth in a manifest. Every
file goes through scripts/run_pipeline.py exactly as the dashboard runs it
(--hybrid --listen), one process per file so memory is freed between files.

Per file it reports:
  found     a track covers the transmission's frequency
  kind      the listen step named the right analog kind (NBFM / AM / USB / CW)
  audio     an audio track was written
  text      Whisper's transcript vs the spoken text (word error rate; character
            error rate for Mandarin), or the Morse decoder's text vs the sent text
            (character error rate)
  junk      tracks that are not the transmission

    python scripts/eval_military_radio.py                         # all 84 (hours on a CPU)
    python scripts/eval_military_radio.py --limit 3               # a quick look
    python scripts/eval_military_radio.py --only cw_001 en_cv_001
    python scripts/eval_military_radio.py --rescore               # score existing runs again, no pipeline

Runs resume: a file whose run folder already holds tracks.json is not run again
(--rerun forces it). Output: <out>/scores.csv, <out>/summary.json, <out>/runs/<id>/.
Demo and validation only: the set is never used for training.
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.intel.voice_sim import wer  # noqa: E402

DEFAULT_CLASSIFIER = REPO_ROOT / "results" / "classifier" / "physics_aug_best.pt"
DEFAULT_OUT = REPO_ROOT / "results" / "eval_military_radio"
EXPECTED_KIND = {"NBFM": "NBFM", "AM": "AM", "USB": "USB", "CW": "CW"}
MATCH_KHZ = 10.0          # a track "covers" the transmission within this much of its frequency


def find_set(path=None):
    """The folder holding manifest.csv and morse/manifest.csv. A shared copy is
    often unzipped one level deeper (military_radio_IQ/military_radio_IQ)."""
    base = Path(path) if path else Path(os.path.expandvars(r"$USERPROFILE")) / "Downloads" / "military_radio_IQ"
    for d in (base, base / "military_radio_IQ"):
        if (d / "manifest.csv").exists():
            return d
    raise SystemExit(f"No manifest.csv in {base} (or {base / 'military_radio_IQ'}); pass --set")


def language_of(row):
    lang = (row.get("language") or "").split()[0].lower()
    return lang if lang in ("en", "ms", "zh") else "auto"


def items(root):
    """Every transmission: dict(id, iq, modulation, offset_khz, text, language)."""
    out = []
    with open(root / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out.append(dict(id=r["id"], iq=root / r["folder"] / r["iq_file"], modulation=r["modulation"],
                            offset_khz=float(r["offset_khz"]), snr_db=float(r["snr_db_3khz"]),
                            text=r["text"], language=language_of(r)))
    morse = root / "morse" / "manifest.csv"
    if morse.exists():
        with open(morse, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                out.append(dict(id=r["id"], iq=root / "morse" / r["iq_file"], modulation=r["modulation"],
                                offset_khz=float(r["offset_khz"]), snr_db=float(r["snr_db_3khz"]),
                                text=r["text"], language="morse"))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (x != y), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


def cer(truth, hyp):
    return edit_distance(truth, hyp) / max(len(truth), 1)


def morse_text(s):
    """Upper case letters, digits and single spaces: the decoder joins its
    segments with ' / ' and may space words differently."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())).strip()


def zh_chars(s):
    return [c for c in (s or "") if "一" <= c <= "鿿" or c.isalnum()]


def covers(track, it):
    """Does the track span the transmission's frequency? USB voice sits 0.3-3 kHz above its offset."""
    lo, hi = it["offset_khz"] - MATCH_KHZ, it["offset_khz"] + MATCH_KHZ
    return track["f_hi_khz"] >= lo and track["f_lo_khz"] <= hi


def score(it, doc, seconds):
    tracks = doc.get("tracks") or []
    hits = [t for t in tracks if covers(t, it)]
    heard = [t["listen"] for t in hits if t.get("listen")]
    want = EXPECTED_KIND[it["modulation"]]
    kinds = sorted({L.get("kind") for L in heard if L.get("kind")})
    row = dict(id=it["id"], modulation=it["modulation"], language=it["language"], snr_db=it["snr_db"],
               tracks=len(tracks), junk=len(tracks) - len(hits), found=bool(hits),
               listened=bool(heard), kind=" / ".join(kinds), kind_right=want in kinds,
               audio=any(L.get("audio") for L in heard), text_score="", text="", seconds=round(seconds, 1))
    if it["language"] == "morse":
        hyp = next((L.get("morse") for L in heard if L.get("morse")), "")
        row["text"] = hyp
        row["text_score"] = round(cer(morse_text(it["text"]), morse_text(hyp)), 3)
    else:
        hyp = next((L.get("transcript") for L in heard if L.get("transcript")), "")
        row["text"] = hyp
        if it["language"] == "zh":
            row["text_score"] = round(cer(zh_chars(it["text"]), zh_chars(hyp)), 3)
        else:
            row["text_score"] = round(wer(it["text"], hyp, it["language"])["wer"], 3)
    return row


def run_one(it, run_dir, args):
    cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_pipeline.py"), "--iq", str(it["iq"]),
           "--max-seconds", str(args.max_seconds), "--hybrid", "--listen", "--no-llm",
           "--classifier", str(args.classifier), "--out", str(run_dir),
           "--whisper-model", args.whisper_model,
           "--language", "auto" if it["language"] == "morse" else it["language"]]
    run_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT, timeout=args.timeout)
    return time.time() - t


def summarise(rows):
    def share(rs, k):
        return round(sum(bool(r[k]) for r in rs) / len(rs), 3) if rs else None

    def med(rs):
        v = [r["text_score"] for r in rs if r["text_score"] != ""]
        return round(statistics.median(v), 3) if v else None

    groups = {}
    for r in rows:
        groups.setdefault(r["modulation"], []).append(r)
        groups.setdefault("lang:" + r["language"], []).append(r)
    out = {"files": len(rows),
           "found": share(rows, "found"), "listened": share(rows, "listened"),
           "kind_right": share(rows, "kind_right"), "audio": share(rows, "audio"),
           "junk_tracks_per_file": round(statistics.mean(r["junk"] for r in rows), 2) if rows else None,
           "seconds_per_file": round(statistics.mean(r["seconds"] for r in rows), 1) if rows else None,
           "groups": {}}
    for g, rs in sorted(groups.items()):
        out["groups"][g] = {"files": len(rs), "found": share(rs, "found"), "kind_right": share(rs, "kind_right"),
                            "audio": share(rs, "audio"), "median_text_error": med(rs),
                            "junk_per_file": round(statistics.mean(r["junk"] for r in rs), 2)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default=None, help="the military_radio_IQ folder (default: Downloads)")
    p.add_argument("--only", nargs="+", default=None, help="ids (or the start of ids) to run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER))
    p.add_argument("--whisper-model", default="medium")
    p.add_argument("--max-seconds", type=float, default=13.0, help="the longest file is 12.7 s")
    p.add_argument("--timeout", type=float, default=3600, help="seconds allowed per file")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--rerun", action="store_true", help="run files again even if their run exists")
    p.add_argument("--rescore", action="store_true", help="only score runs that already exist")
    args = p.parse_args()

    root = find_set(args.set)
    todo = items(root)
    if args.only:
        todo = [it for it in todo if any(it["id"].startswith(o) for o in args.only)]
    if args.limit:
        todo = todo[:args.limit]
    out = Path(args.out)
    rows = []
    for k, it in enumerate(todo, 1):
        run_dir = out / "runs" / it["id"]
        done = (run_dir / "tracks.json").exists()
        seconds = 0.0
        if args.rescore and not done:
            continue
        if not args.rescore and (args.rerun or not done):
            print(f"[{k}/{len(todo)}] {it['id']} ({it['modulation']}, {it['language']}) ...", flush=True)
            try:
                seconds = run_one(it, run_dir, args)
            except subprocess.TimeoutExpired:
                print(f"    timed out after {args.timeout:.0f} s")
        if not (run_dir / "tracks.json").exists():
            print(f"    no tracks.json: see {run_dir / 'run.log'}")
            continue
        if not seconds and (run_dir / "score.json").exists():
            seconds = json.loads((run_dir / "score.json").read_text(encoding="utf-8")).get("seconds", 0.0)
        row = score(it, json.loads((run_dir / "tracks.json").read_text(encoding="utf-8")), seconds)
        (run_dir / "score.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append(row)
        print(f"    found={row['found']} kind={row['kind'] or '-'} ({'right' if row['kind_right'] else 'wrong'}) "
              f"audio={row['audio']} text_error={row['text_score']} junk={row['junk']} {row['seconds']} s", flush=True)
    if not rows:
        print("Nothing scored.")
        return
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows)
    summary["classifier"] = Path(args.classifier).name
    summary["whisper_model"] = args.whisper_model
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{summary['files']} files: found {summary['found']}, kind right {summary['kind_right']}, "
          f"audio {summary['audio']}, junk tracks per file {summary['junk_tracks_per_file']}")
    for g, s in summary["groups"].items():
        print(f"  {g:<10} n={s['files']:<3} found {s['found']}  kind {s['kind_right']}  audio {s['audio']}  "
              f"text error (median) {s['median_text_error']}  junk {s['junk_per_file']}")
    print(f"Wrote {out / 'scores.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
