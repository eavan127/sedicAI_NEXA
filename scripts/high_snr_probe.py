"""
Why does recall FALL as SNR rises?

Section 6.2 reports FHSS recall dropping 90.6% (+2 dB) -> 81.2% (+10 dB), with
the model's mean probability falling alongside it (0.736 -> 0.690), so it is
genuine uncertainty rather than a threshold artefact. LFM_RADAR shows the same
pattern mildly. That curve is measured over the whole test split, which mixes
standalone windows with overlays and mixtures -- so it cannot say WHICH of
those is losing recall.

This probe separates them. For each SNR bin it generates three kinds of window
with a known FHSS (or LFM_RADAR) component:

    standalone      the class on its own
    mixture         the class summed with the other military class at a random SIR
    overlay         the class with a jammer on top at a random JSR

and reports recall, mean probability, and what the model says INSTEAD when it
misses. If the decline appears only in composites, the cause is competition
inside the window (the other component gets cleaner as noise drops, and takes
over the representation). If it appears in standalone windows too, the cause is
the class's own shape at high SNR, and the parameter breakdown says which
configurations are lost.

Generation mirrors build_dataset.py exactly: mixtures use mix_components() then
one add_awgn() pass; overlays use overlay_jamming() then one add_awgn() pass.

Usage:
    python scripts/high_snr_probe.py --n 300 --ensemble --n-models 5
    python scripts/high_snr_probe.py --n 150 --class FHSS
    python scripts/high_snr_probe.py --n 300 --checkpoint results/experiment_rows.pt --stft-keep-rows
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
from src.data.composite import mix_components, overlay_jamming  # noqa: E402
from src.data.preprocess import add_awgn, preprocess_window  # noqa: E402
from src.generators.fhss import random_fhss_example  # noqa: E402
from src.generators.radar import random_radar_example  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LEN = CFG["signal"]["window_len"]
FS = CFG["signal"]["fs"]
GEN = {"FHSS": random_fhss_example, "LFM_RADAR": random_radar_example}
PARTNER = {"FHSS": "LFM_RADAR", "LFM_RADAR": "FHSS"}


def _load_models(ensemble, n_models, checkpoint=None):
    if checkpoint:
        paths = [Path(checkpoint)]
    else:
        paths = ([REPO_ROOT / "results" / f"ensemble_{i}.pt" for i in range(n_models)]
                 if ensemble else [REPO_ROOT / "results" / "best_model.pt"])
    models = []
    for p in paths:
        m = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def _probs(models, X):
    """Average sigmoid probabilities over the ensemble. X: (n, 2, window_len)."""
    t = torch.from_numpy(X).float().to(DEVICE)
    out = None
    for m in models:
        p = torch.sigmoid(m(t))
        out = p if out is None else out + p
    return (out / len(models)).cpu().numpy()


def _make_window(kind, cls, snr_db, rng):
    """One window containing `cls`, built the way build_dataset.py builds it."""
    iq = np.asarray(GEN[cls](rng=rng))
    if kind == "standalone":
        return add_awgn(iq, snr_db, rng=rng)
    if kind == "mixture":
        partner = np.asarray(GEN[PARTNER[cls]](rng=rng))
        combo = [(cls, iq), (PARTNER[cls], partner)] if cls == "LFM_RADAR" else \
                [(PARTNER[cls], partner), (cls, iq)]          # config order: LFM_RADAR first
        mixed, _ = mix_components(combo, rng=rng)
        return add_awgn(mixed, snr_db, rng=rng)
    if kind == "overlay":
        jammed, _ = overlay_jamming(iq, cls, rng=rng)
        return add_awgn(jammed, snr_db, rng=rng)
    raise ValueError(kind)


def main(n, cls, ensemble, n_models, seed, checkpoint=None, out=None):
    rng = np.random.default_rng(seed)
    models = _load_models(ensemble, n_models, checkpoint)
    thr = np.asarray(resolve_multilabel_thresholds())
    idx = CLASS_TO_IDX[cls]
    snr_bins = CFG["snr_bins_db"]
    kinds = ["standalone", "mixture", "overlay"]

    print(f"class={cls}  n={n} per cell  models={len(models)}  threshold={thr[idx]:.3f}")
    print(f"{'kind':11s} " + " ".join(f"{s:>+7.0f} dB" for s in snr_bins))

    table = {}
    outcomes = {}
    jam_idx = CLASS_TO_IDX["JAMMING"]
    misses = defaultdict(lambda: defaultdict(int))
    for kind in kinds:
        recalls, means = [], []
        for snr in snr_bins:
            X = np.stack([preprocess_window(_make_window(kind, cls, snr, rng)) for _ in range(n)])
            P = _probs(models, X)
            hit = P[:, idx] > thr[idx]
            if kind == "overlay":
                jhit = P[:, jam_idx] > thr[jam_idx]
                outcomes[snr] = ((hit & jhit).mean() * 100, (~hit & jhit).mean() * 100,
                                 (hit & ~jhit).mean() * 100, (~hit & ~jhit).mean() * 100)
            recalls.append(hit.mean() * 100)
            means.append(P[:, idx].mean())
            for row in P[~hit]:
                over = [c for c, p, t in zip(CLASSES, row, thr) if p > t]
                misses[(kind, snr)]["nothing at all" if not over else ", ".join(over)] += 1
        table[kind] = (recalls, means)
        print(f"{kind:11s} " + " ".join(f"{r:8.1f}%" for r in recalls))
    print()
    print("mean probability of the target class (same cells):")
    for kind in kinds:
        print(f"{kind:11s} " + " ".join(f"{m:9.3f}" for m in table[kind][1]))

    print()
    print(f"overlay windows truly contain BOTH {cls} and JAMMING. What is reported:")
    print(f"{'SNR':>7} {'both (correct)':>15} {'JAMMING only':>13} {cls + ' only':>13} {'neither':>8}")
    for snr in snr_bins:
        b, jo, co, ne = outcomes[snr]
        print(f"{snr:>+5.0f} dB {b:>14.1f}% {jo:>12.1f}% {co:>12.1f}% {ne:>7.1f}%")
    print("goal of any fix: move 'JAMMING only' into 'both' WITHOUT growing the '"
          + cls + " only' column (that would mean the jammer got missed).")

    print()
    print("what the model said INSTEAD, on misses at the two cleanest bins:")
    for kind in kinds:
        for snr in snr_bins[-2:]:
            top = sorted(misses[(kind, snr)].items(), key=lambda kv: -kv[1])[:3]
            if top:
                shown = "; ".join(f"{k} x{v}" for k, v in top)
                print(f"  {kind:11s} {snr:+3.0f} dB: {shown}")

    print()
    for kind in kinds:
        r = table[kind][0]
        peak_i = int(np.argmax(r))
        print(f"{kind:11s} peak {r[peak_i]:.1f}% at {snr_bins[peak_i]:+.0f} dB, "
              f"+10 dB {r[-1]:.1f}%  ->  drop of {r[peak_i] - r[-1]:.1f} points")

    if out:
        result = {
            "class": cls, "n": n, "seed": seed, "models": len(models),
            "checkpoint": str(checkpoint) if checkpoint else ("ensemble" if ensemble else "best_model"),
            "threshold": float(thr[idx]), "jamming_threshold": float(thr[jam_idx]),
            "snr_bins": [float(s_) for s_ in snr_bins],
            "recall": {k: [float(v) for v in table[k][0]] for k in kinds},
            "mean_probability": {k: [float(v) for v in table[k][1]] for k in kinds},
            "overlay_outcomes": {str(float(s_)): dict(zip(("both", "jamming_only", "class_only", "neither"),
                                                          (float(v) for v in outcomes[s_])))
                                 for s_ in snr_bins},
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--class", dest="cls", default="FHSS", choices=list(GEN))
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None,
                    help="score ONE explicit checkpoint (an experiment run) instead of results/")
    ap.add_argument("--stft-freq-summary", action="store_true",
                    help="build the model with model.stft_freq_summary on (needed for a checkpoint trained with it)")
    ap.add_argument("--stft-keep-rows", action="store_true",
                    help="build the model with model.stft_keep_rows on (variant C2)")
    ap.add_argument("--thresholds", default=None,
                    help="JSON {class: threshold} to use instead of configs/default.yaml. A retrained "
                         "model needs its OWN thresholds (scripts/evaluate_experiment.py writes them).")
    ap.add_argument("--out", default=None, help="write the tables as JSON here")
    a = ap.parse_args()
    if a.thresholds:
        CFG.setdefault("multilabel_thresholds_per_class", {}).update(json.loads(Path(a.thresholds).read_text()))
    if a.stft_freq_summary:
        CFG.setdefault("model", {})["stft_freq_summary"] = True
    if a.stft_keep_rows:
        CFG.setdefault("model", {})["stft_keep_rows"] = True
    main(a.n, a.cls, a.ensemble, a.n_models, a.seed, a.checkpoint, a.out)
