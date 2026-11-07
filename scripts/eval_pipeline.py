"""
Score the wideband pipeline on synthetic captures with known emitters
(src/pipeline/capture.py): does tracking beat answering window by window?

For every true emitter, the track that overlaps it most in time x frequency
(IoU >= --match-iou) is its track. Then:

  found         the emitter got a track at all (detector + tracker)
  window top-1  share of that track's windows whose own top class is right
                (what the classifier alone would say, window by window)
  track         the track's final answer covers the truth (an undecided pair
                such as "RADAR_CW / JAM_TONE" counts when it contains it)
  track exact   the final answer is exactly the true class (pairs settled)

"Right" for a window accepts the classes one window cannot tell apart
(src/data/labels.py): RADAR_CW <-> JAM_TONE, LORA_CSS <-> JAM_SWEPT, and a
DRFM jammer named by the radar it copies.

    python scripts/eval_pipeline.py --classifier best.pt --n-captures 30
    python scripts/eval_pipeline.py --classifier best.pt --classes LORA_CSS JAM_SWEPT JAM_TONE RADAR_CW
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASS_TO_IDX, REPO_ROOT  # noqa: E402
from src.pipeline.capture import make_capture  # noqa: E402
from src.pipeline.classify import Classifier  # noqa: E402
from src.pipeline.run import EnergyDetector, HybridDetector, NetDetector, process_capture  # noqa: E402

FS = CFG["signal"]["fs"]
PARTNERS = {"RADAR_CW": {"JAM_TONE"}, "JAM_TONE": {"RADAR_CW"}, "LORA_CSS": {"JAM_SWEPT"},
            "JAM_SWEPT": {"LORA_CSS"}, "JAM_DRFM": {"RADAR_PULSE", "LFM_RADAR", "RADAR_BARKER"}}


def accepted(cls):
    return {cls} | PARTNERS.get(cls, set())


def iou_tf(a, b):
    """IoU of two (t0, t1, f0, f1) rectangles."""
    dt = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    df = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = dt * df
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", nargs="+", required=True)
    p.add_argument("--detector", default=str(REPO_ROOT / "results" / "detector" / "emitter_detector.pt"))
    p.add_argument("--energy", action="store_true")
    p.add_argument("--hybrid", action="store_true", help="YOLO + energy boxes for long narrow signals it missed")
    p.add_argument("--all-windows", action="store_true", help="vote with every window, also those where the emitter is off (old behaviour)")
    p.add_argument("--cut-margin", type=float, default=None, help="extra band kept around a box, as a share of its width (default: extract.CUT_MARGIN_FRAC)")
    p.add_argument("--score-thr", type=float, default=0.3, help="detector confidence threshold")
    p.add_argument("--n-captures", type=int, default=30)
    p.add_argument("--n-chunks", type=int, default=4)
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--snr-db", type=float, default=None, help="fix every emitter's SNR (default: -10..20 dB)")
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "pipeline" / "eval"))
    args = p.parse_args()

    detector = EnergyDetector() if args.energy else NetDetector(args.detector, score_thr=args.score_thr)
    if args.hybrid and not args.energy:
        detector = HybridDetector(detector)
    classifier = Classifier(args.classifier)
    if args.cut_margin is not None:
        from src.pipeline import extract
        extract.CUT_MARGIN_FRAC = args.cut_margin
    per = defaultdict(lambda: dict(n=0, found=0, win_hit=0, win_n=0, track=0, exact=0))
    matched = []                       # (track confidence, answer covers the truth, snr) per found emitter
    t0 = time.time()
    for k in range(args.n_captures):
        rng = np.random.default_rng(args.seed + k)
        iq, truth = make_capture(rng, n_chunks=args.n_chunks, classes=args.classes, snr_db=args.snr_db)
        answers, tracker = process_capture(iq, detector, classifier, rng=rng, active_only=not args.all_windows)
        by_id = {tr.id: tr for tr in tracker.tracks}
        for t in truth:
            s = per[t["cls"]]
            s["n"] += 1
            box = (t["t_start"] / FS, t["t_end"] / FS, t["f_lo"], t["f_hi"])
            best, best_iou = None, args.match_iou
            for a in answers:
                ov = iou_tf(box, (a["t_start_ms"] / 1e3, a["t_end_ms"] / 1e3, a["f_lo_khz"] * 1e3, a["f_hi_khz"] * 1e3))
                if ov >= best_iou:
                    best, best_iou = a, ov
            if best is None:
                continue
            s["found"] += 1
            ok = {DATASET_CLASS_TO_IDX[c] for c in accepted(t["cls"])}
            wins = by_id[best["track"]].window_top1()
            s["win_hit"] += int(np.isin(wins, list(ok)).sum())
            s["win_n"] += len(wins)
            right = int(t["cls"] in best["classes"] or bool(accepted(t["cls"]) & set(best["classes"])))
            s["track"] += right
            matched.append((best["confidence"], right, t.get("snr_db", float("nan"))))
            s["exact"] += int(best["answer"] == t["cls"])
        print(f"capture {k + 1}/{args.n_captures}  {time.time() - t0:.0f}s", flush=True)

    rows, tot = [], dict(n=0, found=0, win_hit=0, win_n=0, track=0, exact=0)
    for cls in sorted(per):
        s = per[cls]
        for key in tot:
            tot[key] += s[key]
        rows.append((cls, s))
    rows.append(("ALL", tot))
    print(f"\n{'class':>18} {'emitters':>8} {'found':>6} {'window top-1':>13} {'track':>6} {'track exact':>12}")
    table = {}
    for cls, s in rows:
        f = s["found"] or 1
        r = dict(emitters=s["n"], found=s["found"] / max(s["n"], 1), window_top1=s["win_hit"] / max(s["win_n"], 1),
                 track=s["track"] / f, track_exact=s["exact"] / f)
        table[cls] = r
        print(f"{cls:>18} {s['n']:>8} {r['found']:>6.0%} {r['window_top1']:>13.0%} {r['track']:>6.0%} {r['track_exact']:>12.0%}")
    # Operating points: answer only when the track's confidence clears a cut-off.
    # coverage = share of found emitters that get an answer; accuracy = right among those.
    conf = np.array([m[0] for m in matched]); right = np.array([m[1] for m in matched])
    ops = []
    print(f"\n{'cut-off':>8} {'coverage':>9} {'accuracy':>9}")
    for thr in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        keep = conf >= thr
        cov = float(keep.mean()) if len(conf) else 0.0
        acc = float(right[keep].mean()) if keep.any() else float("nan")
        ops.append(dict(cutoff=thr, coverage=cov, accuracy=acc, n=int(keep.sum())))
        print(f"{thr:>8.2f} {cov:>9.0%} {acc:>9.0%}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"args": vars(args), "results": table, "operating_points": ops,
                                               "matched": [list(map(float, m)) for m in matched]}, indent=1))
    print(f"\nWrote {out / 'eval.json'}")


if __name__ == "__main__":
    main()
