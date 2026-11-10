"""
Real-world HF evaluation: score NEXA on KiwiSDR recordings against labels a
person confirmed by ear (data/kiwi/labels.csv, column human_label; see
docs/KIWI_LABELLING.md). Test data only, never training.

Each clip goes through the same analysis as a live SDRconnect recording
(scripts/live_sdrconnect.py voice path: channel finder -> zoom rules ->
slow-path model), and NEXA's answer for the strongest transmission is compared
with the human label. A clip with nothing above the noise is NEXA saying
"noise". Labels NEXA has no kind for yet (ft8) are reported as "not
supported", never counted as right or hidden.

    python scripts/eval_kiwisdr.py                       # human labels only (the real score)
    python scripts/eval_kiwisdr.py --use-suggested       # also the pre-labeller's guesses (NOT a real score)

Writes evals/kiwisdr/scorecard.json and evals/kiwisdr/clips.csv.
"""
import argparse
import csv
import datetime
import json
import sys
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import REPO_ROOT  # noqa: E402

# human label -> NEXA kinds that count as right
ACCEPT = {
    "noise": {"NOISE"},
    "cw": {"CW"},
    "rtty": {"RTTY"},
    "psk31": {"PSK31"},
    "ssb_voice": {"USB", "LSB"},
    "am": {"AM", "CARRIER"},          # a time station between ticks is a bare carrier
}
NOT_SUPPORTED = {"ft8"}               # 8 slow tones: no NEXA kind yet
OUT = REPO_ROOT / "evals" / "kiwisdr"


def read_iq(path):
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64), float(fs)


def nexa_answer(path):
    """-> (kind, why, n_channels) for the strongest transmission in the clip."""
    import live_sdrconnect as live
    iq, fs = read_iq(path)
    with tempfile.TemporaryDirectory() as d:
        tracks = live.voice_snapshot(iq, fs, Path(d), whisper_model=None, language=None,
                                     use_whisper=False, use_llm=False, llm_model=None)
    if not tracks:
        return "NOISE", "nothing above the noise", 0
    best = tracks[0]                                     # find_channels sorts by strength
    return best["listen"]["kind"], best["listen"].get("why", ""), len(tracks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", default=str(REPO_ROOT / "data" / "kiwi"))
    p.add_argument("--use-suggested", action="store_true",
                   help="also score clips without a human label against the pre-labeller's suggestion")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    clips = Path(a.clips)
    sheet = clips / "labels.csv"
    if not sheet.exists():
        sys.exit(f"{sheet} not found: run scripts/kiwi_collect.py (or kiwi_label.py) first")
    rows = list(csv.DictReader(open(sheet, encoding="utf-8-sig")))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, per = [], defaultdict(Counter)
    for r in rows:
        human = (r.get("human_label") or "").strip().lower()
        truth, source = (human, "human") if human else ((r.get("suggested") or "").strip().lower(), "suggested")
        if not human and not a.use_suggested:
            results.append({**{k: r.get(k) for k in ("file", "freq_khz", "signal_khz")}, "truth": "", "source": "unlabelled",
                            "nexa": "", "result": "not scored (no human_label)"})
            continue
        if truth in ("", "unknown"):
            results.append({**{k: r.get(k) for k in ("file", "freq_khz", "signal_khz")}, "truth": truth, "source": source,
                            "nexa": "", "result": "not scored (label unknown)"})
            continue
        kind, why, n = nexa_answer(clips / r["file"])
        if truth in NOT_SUPPORTED:
            res = "not supported"
        elif truth in ACCEPT:
            res = "right" if kind in ACCEPT[truth] else "wrong"
        else:
            res = f"unknown label '{truth}'"
        per[(truth, source)][res] += 1
        results.append({**{k: r.get(k) for k in ("file", "freq_khz", "signal_khz")}, "truth": truth, "source": source,
                        "nexa": kind, "channels": n, "result": res, "why": why[:160]})
        print(f"{r['file'][:44]:<44} {source:<9} {truth:<10} NEXA {kind:<8} {res}")
    summary = {}
    for (truth, source), c in sorted(per.items()):
        scored = c["right"] + c["wrong"]
        summary[f"{truth} ({source})"] = {"right": c["right"], "wrong": c["wrong"], "not_supported": c["not supported"],
                                          "accuracy": round(c["right"] / scored, 3) if scored else None}
    human = [x for x in results if x["source"] == "human" and x["result"] in ("right", "wrong")]
    doc = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "clips": len(rows),
           "human_scored": len(human),
           "human_accuracy": round(sum(x["result"] == "right" for x in human) / len(human), 3) if human else None,
           "per_label": summary,
           "note": "Only rows with source=human are a real score; 'suggested' rows are the pre-labeller's guesses."}
    (out / "scorecard.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    keys = ["file", "freq_khz", "signal_khz", "truth", "source", "nexa", "channels", "result", "why"]
    with open(out / "clips.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nhuman-labelled clips scored: {len(human)}"
          + (f", NEXA right on {doc['human_accuracy']:.0%}" if human else " (fill in human_label in labels.csv)"))
    print(f"wrote {out / 'scorecard.json'} and {out / 'clips.csv'}")


if __name__ == "__main__":
    main()
