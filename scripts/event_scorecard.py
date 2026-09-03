"""Event-level scorecard, scored against scenario GROUND TRUTH.

The submission metric is per-window -- src/infer.py writes one CSV row per
window, and the organiser's stream carries no event boundaries -- so this is
NOT the benchmark and must never be quoted as it. It answers a different and
operationally honest question: when the system can watch a sustained emission
rather than one 160 microsecond window, how well does it identify it?

Why it can exist at all: scenario captures carry ScenarioSegment ground truth,
so events are defined by the SCENARIO, not by the model's own detections.
Scoring against timeline.detections() would grade the system on boundaries it
drew itself, which is circular. That is also why this cannot run on the
qualifier stream, which has no truth.

Windows are matched to an event through radiating_spans rather than
start_s/end_s, per ScenarioSegment's own docstring: a pulsed radar is
"scheduled" across a span it transmits in under 2% of, so scoring the silent
gaps as misses measures the duty cycle instead of the model.

Two aggregation rules, reported separately because they disagree and the
disagreement matters:

  MEAN  average the class probability over the event's windows, then apply
        the per-class threshold. Cancels random per-window error.
  ANY   the class fires if ANY window in the event clears its threshold.

ANY looks better than it is. It reaches perfect recall, but its false-positive
rate is measured here too, and over an event spanning ~100 windows a single
stray window is enough to fire: absent classes are reported in up to every
capture. MEAN gives up a little recall and takes the false-positive rate to
zero, which is why both columns are printed side by side. Recall alone would
have picked the wrong rule.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT  # noqa: E402
from src.scenarios import CASES  # noqa: E402
from src.ui.app_models import EnsembleModel, _load_one, ensemble_paths  # noqa: E402
from src.ui.session import load_scenario  # noqa: E402

THRESHOLDS = CFG["multilabel_thresholds_per_class"]
FALLBACK = CFG["multilabel_threshold"]


def _threshold(cls):
    return THRESHOLDS.get(cls, FALLBACK)


def _window_overlaps(w_start_s, w_end_s, spans):
    """True if this window overlaps any span in which the emitter radiates."""
    return any(w_start_s < b and w_end_s > a for a, b in spans)


def score_capture(session, segments, window_len, fs):
    """One capture -> per-event outcomes, plus the classes that were absent.

    Returns (events, fps). events carries the truth class and both
    aggregation verdicts; fps carries, for each class with NO ground-truth
    segment in this capture, whether the pooled decision reported it anyway.
    """
    probs = session.result.probs
    starts = session.result.starts
    w_start = starts / fs
    w_end = (starts + window_len) / fs

    present = {s.class_name for s in segments}
    absent = [c for c in CLASSES if c not in present]

    events = []
    for seg in segments:
        spans = seg.radiating_spans or [(seg.start_s, seg.end_s)]
        idx = np.array([i for i in range(len(starts))
                        if _window_overlaps(w_start[i], w_end[i], spans)], dtype=int)
        if idx.size == 0:
            continue  # emitter radiates entirely between window centres
        col = CLASS_TO_IDX[seg.class_name]
        thr = _threshold(seg.class_name)
        events.append({
            "class": seg.class_name,
            "n_windows": int(idx.size),
            "mean_hit": bool(probs[idx, col].mean() > thr),
            "any_hit": bool((probs[idx, col] > thr).any()),
        })

    fps = {}
    for cls in absent:
        col = CLASS_TO_IDX[cls]
        thr = _threshold(cls)
        fps[cls] = {
            "mean_fires": bool(probs[:, col].mean() > thr),
            "any_fires": bool((probs[:, col] > thr).any()),
        }
    return events, fps


def main(seeds, snrs, duration, out_json, out_csv):
    model = EnsembleModel([_load_one(p) for p in ensemble_paths()])
    fs = CFG["signal"]["fs"]
    window_len = CFG["signal"]["window_len"]

    hits = defaultdict(lambda: {"mean": 0, "any": 0, "n": 0, "windows": 0})
    fp = defaultdict(lambda: {"mean": 0, "any": 0, "n": 0})

    # load_scenario rather than build_scenario directly: the civilian classes
    # come from a recorded library, not a generator, and load_scenario is what
    # supplies it (and caps the requested SNR at the library's own, since a
    # recording already carries its noise). Using it means this scores exactly
    # the captures the console builds.
    capped = 0
    for case in CASES:
        for snr_db in snrs:
            for seed in seeds:
                session = load_scenario(model, total_duration=duration,
                                        snr_db=snr_db, seed=seed, case=case)
                capped += bool(getattr(session, "snr_capped", False))
                segments = session.truth
                events, fps = score_capture(session, segments, window_len, fs)
                for e in events:
                    h = hits[e["class"]]
                    h["n"] += 1
                    h["windows"] += e["n_windows"]
                    h["mean"] += e["mean_hit"]
                    h["any"] += e["any_hit"]
                for cls, f in fps.items():
                    fp[cls]["n"] += 1
                    fp[cls]["mean"] += f["mean_fires"]
                    fp[cls]["any"] += f["any_fires"]

    result = {
        "note": ("Event-level, scored against scenario ground truth "
                 "(ScenarioSegment.radiating_spans). NOT the submission "
                 "benchmark, which is per-window."),
        "cases": list(CASES),
        "snr_bins_db": list(snrs),
        "seeds": list(seeds),
        "capture_duration_s": duration,
        "thresholds": {c: _threshold(c) for c in CLASSES},
        # Civilian recordings carry their own noise, so a requested SNR above
        # the library's cannot be achieved and load_scenario caps it. Recorded
        # here so a reader knows the civilian rows are not at the nominal SNR.
        "captures_with_snr_capped": capped,
        "per_class": {},
    }
    for cls in CLASSES:
        h, f = hits.get(cls), fp.get(cls)
        if not h or not h["n"]:
            continue
        result["per_class"][cls] = {
            "events": h["n"],
            "mean_windows_per_event": h["windows"] / h["n"],
            "recall_mean_rule": h["mean"] / h["n"],
            "recall_any_rule": h["any"] / h["n"],
            "absent_captures": f["n"] if f else 0,
            "false_positive_rate_mean_rule": (f["mean"] / f["n"]) if f and f["n"] else None,
            "false_positive_rate_any_rule": (f["any"] / f["n"]) if f and f["n"] else None,
        }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "events", "mean_windows_per_event", "recall_mean_rule",
                    "recall_any_rule", "false_positive_rate_mean_rule",
                    "false_positive_rate_any_rule", "is_judged_class"])
        for cls, m in result["per_class"].items():
            w.writerow([cls, m["events"], f"{m['mean_windows_per_event']:.1f}",
                        m["recall_mean_rule"], m["recall_any_rule"],
                        m["false_positive_rate_mean_rule"],
                        m["false_positive_rate_any_rule"],
                        cls in CFG["judged_classes"]])

    judged = CFG["judged_classes"]
    print(f"\nEvent-level, scenario ground truth  "
          f"({len(CASES)} cases x {len(snrs)} SNR x {len(seeds)} seeds)")
    print(f"{'class':<13}{'events':>7}{'win/ev':>8}{'recall MEAN':>13}"
          f"{'recall ANY':>12}{'FP MEAN':>9}{'FP ANY':>8}")
    print("-" * 71)
    for cls, m in result["per_class"].items():
        mark = " *" if cls in judged else ""
        fpm = ("n/a" if m["false_positive_rate_mean_rule"] is None
               else f"{m['false_positive_rate_mean_rule']:.3f}")
        fpa = ("n/a" if m["false_positive_rate_any_rule"] is None
               else f"{m['false_positive_rate_any_rule']:.3f}")
        print(f"{cls + mark:<13}{m['events']:>7}{m['mean_windows_per_event']:>8.1f}"
              f"{m['recall_mean_rule']:>13.3f}{m['recall_any_rule']:>12.3f}"
              f"{fpm:>9}{fpa:>8}")
    print("\n  * judged class.  MEAN averages the event's windows then thresholds;")
    print("    ANY fires if any single window clears. Read the two recall")
    print("    columns against the two FP columns: ANY wins on recall and")
    print("    loses badly on false positives, because over an event of this")
    print("    length one stray window is enough to fire.")
    print("\n  NOT the submission benchmark, which is per-window. Scenario")
    print("  ground truth only; the qualifier stream carries none.")
    print(f"\n  wrote {out_json}\n  wrote {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--snrs", type=float, nargs="+", default=CFG["snr_bins_db"])
    p.add_argument("--duration", type=float, default=0.05)
    p.add_argument("--out-json",
                   default=str(REPO_ROOT / CFG["paths"]["evals"] / "event_scorecard.json"))
    p.add_argument("--out-csv",
                   default=str(REPO_ROOT / CFG["paths"]["evals"] / "csv" / "event_scorecard.csv"))
    a = p.parse_args()
    main(a.seeds, a.snrs, a.duration, a.out_json, a.out_csv)
