"""
Run the wideband pipeline on one capture: detector boxes -> cut-out IQ ->
60-class classifier -> tracker -> one answer per emitter (src/pipeline/).
Fully offline.

Writes <out>/tracks.json (one entry per emitter: answer, reason, time,
frequency, top-5 classes, behaviour features) and <out>/tracks.png (the
capture's spectrogram with every track boxed and named; on a synthetic
capture the true emitters are drawn dashed).

A recorded capture (complex64 .npy, or raw interleaved I/Q):
    python scripts/run_pipeline.py --iq capture.npy --classifier best.pt
    python scripts/run_pipeline.py --iq capture.bin --dtype int16 --classifier best.pt

A synthetic one with known emitters (4 chunks = 41 ms):
    python scripts/run_pipeline.py --synthetic 4 --seed 3 --classifier best.pt
    python scripts/run_pipeline.py --synthetic 4 --classes LORA_CSS JAM_SWEPT --classifier best.pt

--classifier takes one week-1 best.pt or several (averaged as an ensemble).
The detector defaults to results/detector/emitter_detector.pt; --energy uses
the classical energy detector instead.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, REPO_ROOT  # noqa: E402
from src.detect import scenes  # noqa: E402
from src.pipeline.capture import make_capture  # noqa: E402
from src.pipeline.classify import Classifier  # noqa: E402
from src.pipeline.run import EnergyDetector, NetDetector, process_capture  # noqa: E402

FS = CFG["signal"]["fs"]
DEFAULT_DETECTOR = REPO_ROOT / "results" / "detector" / "emitter_detector.pt"


def load_iq(path, dtype):
    if str(path).endswith(".npy"):
        return np.load(path).astype(np.complex64)
    raw = np.fromfile(path, dtype=dtype).astype(np.float32)
    raw = raw[:len(raw) // 2 * 2]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def plot(iq, answers, truth, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = scenes.spectrogram(iq)
    t_ms = len(iq) / FS * 1e3
    fig, ax = plt.subplots(figsize=(max(8, min(24, t_ms / 4)), 5.5))
    ax.imshow(img, aspect="auto", origin="lower", cmap="magma", extent=[0, t_ms, -FS / 2e6, FS / 2e6])
    for t in truth or []:
        ax.add_patch(plt.Rectangle((t["t_start"] / FS * 1e3, t["f_lo"] / 1e6),
                                   (t["t_end"] - t["t_start"]) / FS * 1e3, (t["f_hi"] - t["f_lo"]) / 1e6,
                                   fill=False, ec="white", ls="--", lw=0.8))
    for a in answers:
        x0, x1 = a["t_start_ms"], a["t_end_ms"]
        y0, y1 = a["f_lo_khz"] / 1e3, a["f_hi_khz"] / 1e3
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#4cd4ff", lw=1.4))
        ax.text(x0, y1 + 0.02, f"#{a['track']} {a['answer']} ({a['confidence']:.2f})", color="#4cd4ff",
                fontsize=7, va="bottom")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title("Tracks (solid) and, for synthetic captures, true emitters (dashed)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--iq", help="capture file: complex .npy, or raw interleaved I/Q (see --dtype)")
    src.add_argument("--synthetic", type=int, metavar="N_CHUNKS", help="make a synthetic capture this many 10.24 ms chunks long")
    p.add_argument("--dtype", default="float32", help="sample type of a raw --iq file (float32, int16, int8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classes", nargs="+", default=None, help="synthetic: only these classes")
    p.add_argument("--n-emitters", type=int, default=None)
    p.add_argument("--classifier", nargs="+", required=True, help="week-1 best.pt file(s)")
    p.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    p.add_argument("--energy", action="store_true", help="use the classical energy detector")
    p.add_argument("--score-thr", type=float, default=0.3, help="detector confidence threshold")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "pipeline" / "run"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    truth = None
    if args.iq:
        iq = load_iq(args.iq, args.dtype)
    else:
        iq, truth = make_capture(rng, n_chunks=args.synthetic, n_emitters=args.n_emitters, classes=args.classes)
    detector = EnergyDetector() if args.energy else NetDetector(args.detector, score_thr=args.score_thr)
    classifier = Classifier(args.classifier)
    answers, _ = process_capture(iq, detector, classifier, rng=rng)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracks.json").write_text(json.dumps({"n_samples": len(iq), "duration_ms": len(iq) / FS * 1e3,
                                                 "tracks": answers, "truth": truth}, indent=1, default=float))
    plot(iq, answers, truth, out / "tracks.png")
    print(f"{len(iq) / FS * 1e3:.1f} ms, {len(answers)} track(s)")
    for a in answers:
        print(f"  #{a['track']:<3} {a['t_start_ms']:7.2f}-{a['t_end_ms']:7.2f} ms  "
              f"{a['f_lo_khz']:8.0f}..{a['f_hi_khz']:6.0f} kHz  {a['answer']:<22} {a['confidence']:.2f}  {a['reason']}")
    if truth:
        print("true emitters:")
        for t in truth:
            print(f"      {t['t_start'] / FS * 1e3:7.2f}-{t['t_end'] / FS * 1e3:7.2f} ms  "
                  f"{t['f_lo'] / 1e3:8.0f}..{t['f_hi'] / 1e3:6.0f} kHz  {t['cls']:<22} {t['snr_db']:5.1f} dB")
    print(f"Wrote {out / 'tracks.json'} and {out / 'tracks.png'}")


if __name__ == "__main__":
    main()
