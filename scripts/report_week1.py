"""
Detailed report for one trained week-1 member (scripts/train_week1.py):
where it is right, where it is wrong, and what it confuses.

Runs <ckpt-dir>/best.pt on the test split and writes <ckpt-dir>/report/:
  report.json         everything below as numbers
  per_class.csv       one row per class: windows, top-1, top-1 at >= 10 dB,
                      recall / precision at 0.5, and the class it is most
                      often mistaken for
  snr_curve.png       top-1 against SNR (all windows, single-signal, mixtures)
  class_by_snr.png    top-1 for every class at every SNR (60 x 16 heatmap)
  confusion_all.png   true class -> predicted class, single-signal windows
  confusion_high.png  the same, only windows at >= --high-snr dB, where the
                      noise is weak -- mistakes left here are real confusions,
                      not just "too noisy to tell"

"Top-1" is the trainer's: the highest-scoring class is one of the true ones.
The confusion matrices use single-signal windows only, where "the true class"
is one class.

Usage:
    python scripts/report_week1.py --ckpt-dir results/2026-11-04_nfft64_ep15_seed1 --data data/combined
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASSES, REPO_ROOT  # noqa: E402
from src.data import labels  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_week1 import DEVICE, EVAL_BATCH, load_split  # noqa: E402

C = len(DATASET_CLASSES)


def predict(model, X):
    model.eval()
    out = np.empty((len(X), C), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), EVAL_BATCH):
            xb = torch.from_numpy(X[s:s + EVAL_BATCH]).to(DEVICE).float()
            out[s:s + EVAL_BATCH] = model(xb).cpu().numpy()
    return out


def rate(hit, mask):
    n = int(mask.sum())
    return (float(hit[mask].mean()) if n else float("nan")), n


def confusion(true, pred):
    """Row-normalised: row = true class, each row sums to 1 (NaN if no windows)."""
    m = np.zeros((C, C))
    np.add.at(m, (true, pred), 1)
    rows = m.sum(axis=1, keepdims=True)
    return np.where(rows > 0, m / np.maximum(rows, 1), np.nan), m


def plot_matrix(m, path, title, xlabels, ylabels, xlabel, ylabel, figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.nan_to_num(m, nan=0), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_snr_curve(bins, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals in curves.items():
        ax.plot(bins, vals, marker="o", label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("top-1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Test top-1 by SNR")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="folder holding best.pt (the training run's folder)")
    p.add_argument("--data", default=str(REPO_ROOT / CFG["week1"]["combined_data"]))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--high-snr", type=float, default=10, help="threshold for confusion_high.png and top1_high")
    p.add_argument("--limit", type=int, default=0, help="quick check: first N windows")
    args = p.parse_args()

    ckpt = torch.load(Path(args.ckpt_dir) / "best.pt", map_location=DEVICE, weights_only=False)
    meta = ckpt["meta"]
    if meta["classes"] != DATASET_CLASSES:
        sys.exit("best.pt was trained on a different class list than src/config.py has now")
    model = AMC_CNN(num_classes=C, input_len=meta["input_len"], stft_n_fft=meta["stft_n_fft"],
                    **meta["model_flags"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    X, y, snr = load_split(args.data, args.split, args.limit)
    print(f"{args.split}: {len(X)} windows, best.pt from epoch {meta['epoch']} (seed {meta['seed']})")
    logits = predict(model, X)

    yt = y.astype(bool)
    top = logits.argmax(axis=1)
    hit = yt[np.arange(len(yt)), top]
    # One emitter per window: one label, a DRFM pair or an ambiguous group
    # (src/data/labels.py). Each is scored under its own class; an ambiguous
    # group's windows count under every member, as one merged result.
    rows, true = labels.standalone_pairs(yt)
    single = np.zeros(len(yt), dtype=bool)
    single[rows] = True
    mixed = ~single
    pred05 = logits > 0  # sigmoid > 0.5
    high = snr >= args.high_snr
    p_hit, p_top, p_snr, p_high = hit[rows], labels.scored_prediction(true, top[rows], yt[rows]), snr[rows], high[rows]

    bins = sorted(set(CFG["dataset_snr_bins_db"]) | set(np.unique(snr).tolist()))
    by_snr = []
    for b in bins:
        at = snr == b
        if not at.any():
            continue
        by_snr.append({"snr_db": b,
                       "all": rate(hit, at)[0], "n": int(at.sum()),
                       "single": rate(hit, at & single)[0], "mixtures": rate(hit, at & mixed)[0]})

    conf_all, counts_all = confusion(true, p_top)
    conf_high, _ = confusion(true[p_high], p_top[p_high])

    per_class = []
    for i, c in enumerate(DATASET_CLASSES):
        mine = true == i
        row = counts_all[i].copy()
        row[i] = 0
        j = int(row.argmax())
        tp = int((pred05[:, i] & yt[:, i]).sum())
        per_class.append({
            "class": c,
            "n_single": int(mine.sum()),
            "top1": rate(p_hit, mine)[0],
            "top1_high": rate(p_hit, mine & p_high)[0],
            "recall_05": tp / max(int(yt[:, i].sum()), 1),
            "precision_05": tp / max(int(pred05[:, i].sum()), 1),
            "most_confused_with": DATASET_CLASSES[j] if row[j] else "",
            "confused_share": float(row[j] / max(counts_all[i].sum(), 1)),
            "top1_by_snr": {str(b): rate(p_hit, mine & (p_snr == b))[0] for b in CFG["dataset_snr_bins_db"]},
        })

    pairs = [(DATASET_CLASSES[i], DATASET_CLASSES[j], float(conf_all[i, j]))
             for i in range(C) for j in range(C) if i != j and np.isfinite(conf_all[i, j])]
    pairs.sort(key=lambda t: -t[2])

    out = Path(args.ckpt_dir) / "report"
    out.mkdir(exist_ok=True)
    summary = {"top1": float(hit.mean()), "top1_single": rate(hit, single)[0],
               "top1_mixtures": rate(hit, mixed)[0],
               f"top1_at_or_above_{args.high_snr:g}dB": rate(hit, high)[0]}
    (out / "report.json").write_text(json.dumps({
        "meta": {k: v for k, v in meta.items() if k != "classes"}, "split": args.split, "n": len(X),
        "high_snr_db": args.high_snr, "summary": summary, "by_snr": by_snr,
        "top_confusions": pairs[:30], "per_class": per_class,
        "confusion_single": conf_all.tolist(), "classes": DATASET_CLASSES}, indent=1))
    with open(out / "per_class.csv", "w", newline="") as f:
        cols = ["class", "n_single", "top1", "top1_high", "recall_05", "precision_05",
                "most_confused_with", "confused_share"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in per_class:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    snr_bins = CFG["dataset_snr_bins_db"]
    plot_snr_curve([r["snr_db"] for r in by_snr],
                   {"all windows": [r["all"] for r in by_snr],
                    "single signal": [r["single"] for r in by_snr],
                    "mixtures": [r["mixtures"] for r in by_snr]}, out / "snr_curve.png")
    grid = np.array([[r["top1_by_snr"][str(b)] for b in snr_bins] for r in per_class])
    plot_matrix(grid, out / "class_by_snr.png", "Top-1 per class and SNR (single-signal windows)",
                [str(b) for b in snr_bins], DATASET_CLASSES, "SNR (dB)", "true class", (8, 14))
    plot_matrix(conf_all, out / "confusion_all.png", "Confusion, all SNRs (row = true, share of its windows)",
                DATASET_CLASSES, DATASET_CLASSES, "predicted (top-1)", "true", (15, 14))
    plot_matrix(conf_high, out / "confusion_high.png",
                f"Confusion, SNR >= {args.high_snr:g} dB (row = true, share of its windows)",
                DATASET_CLASSES, DATASET_CLASSES, "predicted (top-1)", "true", (15, 14))

    print("\n".join(f"{k:>28}: {v:.3f}" for k, v in summary.items()))
    print("\nSNR (dB)   all   single  mixtures   windows")
    for r in by_snr:
        print(f"{r['snr_db']:>7g}  {r['all']:.3f}  {r['single']:.3f}   {r['mixtures']:.3f}   {r['n']:>7}")
    print("\nMost common mistakes (true -> predicted, share of the true class's windows):")
    for a, b, s in pairs[:15]:
        print(f"  {a:>14} -> {b:<14} {s:.2f}")
    print(f"\nWeakest 15 classes (top-1 all SNR | at >= {args.high_snr:g} dB | mistaken for):")
    for r in sorted(per_class, key=lambda r: np.nan_to_num(r["top1"], nan=9))[:15]:
        print(f"  {r['class']:>14}  {r['top1']:.2f} | {r['top1_high']:.2f} | "
              f"{r['most_confused_with']} ({r['confused_share']:.2f})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
