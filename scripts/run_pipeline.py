"""
Run the wideband pipeline on one capture: detector boxes -> cut-out IQ ->
60-class classifier -> tracker -> one answer per emitter (src/pipeline/).
Fully offline.

Writes <out>/tracks.json (one entry per emitter: answer, reason, time,
frequency, top-5 classes, behaviour features) and <out>/tracks.png (the
capture's spectrogram with every track boxed and named; on a synthetic
capture the true emitters are drawn dashed).

A recorded capture (SDRconnect / SDRuno IQ .wav, SigMF, complex .npy, or raw
interleaved I/Q); other sample rates are resampled to 3.2 MS/s:
    python scripts/run_pipeline.py --iq rec.wav --classifier best.pt --listen
    python scripts/run_pipeline.py --iq capture.bin --dtype int16 --fs 2e6 --classifier best.pt

--listen zooms into voice / Morse tracks (src/pipeline/listen.py): analog kind,
demodulated audio (track_<n>.wav), Morse text or Whisper transcript, and a
local Ollama reading. --voice-demo makes a capture with made-up walkie-talkie
speech (NBFM) and Morse (CW) in it, to try that path without a recording:
    python scripts/run_pipeline.py --voice-demo --classifier best.pt --listen
Choose the speech model and language (the model must already be on this machine):
    python scripts/run_pipeline.py --iq rec.wav --classifier best.pt --listen --whisper-model medium --language ms

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
from src.intel import voice  # noqa: E402
from src.pipeline import iqio, listen  # noqa: E402
from src.pipeline.capture import make_capture  # noqa: E402
from src.pipeline.classify import Classifier  # noqa: E402
from src.pipeline.run import EnergyDetector, HybridDetector, NetDetector, process_capture  # noqa: E402

FS = CFG["signal"]["fs"]
DEFAULT_DETECTOR = REPO_ROOT / "results" / "detector" / "emitter_detector.pt"


def load_iq(path, dtype):
    if str(path).endswith(".npy"):
        return np.load(path).astype(np.complex64)
    raw = np.fromfile(path, dtype=dtype).astype(np.float32)
    raw = raw[:len(raw) // 2 * 2]
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def finite(o):
    """NaN / inf -> None: browsers refuse them in JSON (a missing measurement
    is NaN in numpy, e.g. a power ratio without a carrier)."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


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
    src.add_argument("--voice-demo", action="store_true", help="synthetic NBFM speech + CW + AM capture (1.2 s)")
    p.add_argument("--fs", type=float, default=None, help="sample rate of the recording (taken from the file when it says)")
    p.add_argument("--max-seconds", type=float, default=1.0,
                   help="analyse at most this much of a recording (CPU: ~50 s of work per second of capture)")
    p.add_argument("--listen", action="store_true", help="zoom, demodulate and transcribe voice / Morse tracks")
    p.add_argument("--no-whisper", action="store_true")
    p.add_argument("--whisper-model", default=voice.DEFAULT_MODEL,
                   help="Whisper model already on this machine (it never downloads): base is fast but poor on Malay")
    p.add_argument("--language", default="auto", choices=voice.LANGUAGES,
                   help="spoken language; setting it is safer on short clips, a wrong one gives nonsense")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-model", default=None, help="Ollama model for the reading (default llama3.2:3b)")
    p.add_argument("--dtype", default="float32", help="sample type of a raw --iq file (float32, int16, int8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classes", nargs="+", default=None, help="synthetic: only these classes")
    p.add_argument("--n-emitters", type=int, default=None)
    p.add_argument("--classifier", nargs="+", required=True, help="week-1 best.pt file(s)")
    p.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    p.add_argument("--energy", action="store_true", help="use the classical energy detector")
    p.add_argument("--hybrid", action="store_true", help="YOLO + energy boxes for long narrow signals it missed")
    p.add_argument("--all-windows", action="store_true", help="vote with every window, also those where the emitter is off (old behaviour)")
    p.add_argument("--score-thr", type=float, default=0.3, help="detector confidence threshold")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "pipeline" / "run"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    truth = None
    source = None
    if args.iq:
        iq, source = iqio.load(args.iq, fs=args.fs, dtype=args.dtype, max_s=args.max_seconds, rng=rng)
        print(f"{source['file']}: {source['format']}, {source['fs_in'] / 1e6:.3f} MS/s"
              + (" (resampled to 3.2)" if source["resampled"] else "")
              + (f", centre {source['centre_hz'] / 1e6:.4f} MHz" if source["centre_hz"] else "")
              + f", {source['duration_s']:.2f} s analysed")
    elif args.voice_demo:
        from src.pipeline import analog_synth
        iq, voice_truth = analog_synth.wideband([("NBFM", -600e3, 30), ("CW", 250e3, 30), ("AM", 900e3, 35)],
                                                seconds=1.2, rng=rng)
        source = {"file": "voice-demo (synthetic)", "truth": voice_truth}
    else:
        iq, truth = make_capture(rng, n_chunks=args.synthetic, n_emitters=args.n_emitters, classes=args.classes)
    detector = EnergyDetector() if args.energy else NetDetector(args.detector, score_thr=args.score_thr)
    if args.hybrid and not args.energy:
        detector = HybridDetector(detector)
    classifier = Classifier(args.classifier)
    band_hz = source["fs_in"] if args.iq else None      # a narrower recording's edges are fill noise
    answers, _ = process_capture(iq, detector, classifier, rng=rng, active_only=not args.all_windows,
                                 band_hz=band_hz)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.listen:
        for old in out.glob("track_*.wav"):              # audio of an earlier run in this folder
            old.unlink()
        listen.listen_all(iq, answers, out, rng=rng, use_whisper=not args.no_whisper,
                          whisper_model=args.whisper_model, language=args.language,
                          use_llm=not args.no_llm, llm_model=args.llm_model)
    doc = finite({"n_samples": len(iq), "duration_ms": len(iq) / FS * 1e3,
                  "source": source, "tracks": answers, "truth": truth})
    (out / "tracks.json").write_text(json.dumps(doc, indent=1, default=float, allow_nan=False))
    if args.iq and args.listen:                           # real recordings only, not the synthetic demos
        from src.intel import store
        try:
            saved = store.save_decoded_tracks(out.name, doc["source"], doc["tracks"], out)
            if saved:
                print(f"Comms Intel: saved {len(saved)} decoded recording(s).")
        except Exception as e:                            # noqa: BLE001 -- the run itself is already safe
            print(f"Comms Intel: could not save this run: {type(e).__name__}: {e}")
    plot(iq, answers, truth, out / "tracks.png")
    print(f"{len(iq) / FS * 1e3:.1f} ms, {len(answers)} track(s)")
    for a in answers:
        print(f"  #{a['track']:<3} {a['t_start_ms']:7.2f}-{a['t_end_ms']:7.2f} ms  "
              f"{a['f_lo_khz']:8.0f}..{a['f_hi_khz']:6.0f} kHz  {a['answer']:<22} {a['confidence']:.2f}  {a['reason']}")
        if "listen" in a:
            L = a["listen"]
            said = L.get("morse") or L.get("transcript") or ""
            print(f"        zoom: {L['kind']:<8} {L.get('why', '')}"
                  + (f" | audio {L['audio']}" if L.get("audio") else "") + (f" | \"{said[:80]}\"" if said else ""))
    if truth:
        print("true emitters:")
        for t in truth:
            print(f"      {t['t_start'] / FS * 1e3:7.2f}-{t['t_end'] / FS * 1e3:7.2f} ms  "
                  f"{t['f_lo'] / 1e3:8.0f}..{t['f_hi'] / 1e3:6.0f} kHz  {t['cls']:<22} {t['snr_db']:5.1f} dB")
    print(f"Wrote {out / 'tracks.json'} and {out / 'tracks.png'}")


if __name__ == "__main__":
    main()
