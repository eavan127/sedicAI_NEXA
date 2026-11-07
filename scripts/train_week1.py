"""
Train ONE member of the week-1 60-class ensemble on data/combined/
(scripts/merge_shards.py). Each teammate runs their own fixed seed
(WEEK1_PLAN "Training rules"); Eavan packs the five into nexa_week1.pt.

Separate from src/train.py and scripts/train_ensemble.py on purpose: those
train the deployed 8-class model from data/processed/ and must keep working
until week 1 replaces it.

What it does differently:
  * reads the fixed train / val / test splits from <data>/<split>/ instead of
    re-splitting one pile, so every seed is validated and tested on the same
    windows and nothing leaks across splits;
  * loads X as float16 (train is 2 GB) and casts each batch to float32 on the
    GPU, so it fits in free Colab RAM;
  * builds AMC_CNN with 60 outputs and week1.stft_n_fft (64), plus the
    expert-feature branch when week1.expert_features or --expert-features;
  * saves <ckpt-dir>/last.pt after EVERY epoch and resumes from it when run
    again with the same arguments -- a dropped Colab session costs at most one
    epoch;
  * best.pt (lowest val loss) carries its own metadata: classes, n_fft, seed,
    combined version, git commit -- what the day-7 packer needs.

Usage:
    python scripts/train_week1.py --seed 1
    python scripts/train_week1.py --seed 1 --ckpt-dir "/content/drive/MyDrive/.../week1_seed1" \\
        --member-out "/content/drive/MyDrive/.../members/seed1.pt" --eval-test
Quick check (a few hundred windows, 1 epoch):
    python scripts/train_week1.py --seed 1 --limit 2000 --epochs 1 --ckpt-dir data/interim/week1_try
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASSES, REPO_ROOT  # noqa: E402
from src.data import labels  # noqa: E402
from src.data.shards import _git_commit  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import compute_class_weights, compute_snr_weights, set_seed  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = len(DATASET_CLASSES)
NOISE_IDX = DATASET_CLASSES.index("NOISE_FLOOR")
EVAL_BATCH = 1024


def load_split(data_dir, split, limit=0):
    d = Path(data_dir) / split
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(limit, len(X)) if limit else len(X)
    # Rows are already shuffled by the merge, so a prefix is a fair sample.
    X = np.ascontiguousarray(X[:n])
    y = np.load(d / "y.npy", mmap_mode="r")[:n].astype(np.uint8)
    snr = np.load(d / "snr_labels.npy")[:n].astype(np.float32)
    if np.isnan(snr).any():
        raise ValueError(f"{split}: NaN in snr_labels.npy -- rebuild with scripts/merge_shards.py")
    if y.shape[1] != C:
        raise ValueError(f"{split}: y has {y.shape[1]} columns, the class list has {C}")
    return X, y, snr


def batch_to_device(X, y, idx):
    xb = torch.from_numpy(X[idx]).to(DEVICE, non_blocking=True).float()
    yb = torch.from_numpy(y[idx]).to(DEVICE, non_blocking=True).float()
    return xb, yb


def evaluate(model, criterion, X, y):
    """Loss, plus the two numbers the team has been comparing runs on:
    top-1 (argmax lands on one of the true classes) and macro recall at 0.5,
    with per-class values for the scorecard."""
    model.eval()
    loss, hits = 0.0, 0
    tp, pos, top1_hit, top1_n = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
    with torch.no_grad():
        for s in range(0, len(X), EVAL_BATCH):
            idx = np.arange(s, min(s + EVAL_BATCH, len(X)))
            xb, yb = batch_to_device(X, y, idx)
            out = model(xb)
            loss += criterion(out, yb).item() * len(idx)
            yt = y[idx].astype(bool)
            pred = (torch.sigmoid(out) > 0.5).cpu().numpy()
            top = out.argmax(dim=1).cpu().numpy()
            hit = yt[np.arange(len(idx)), top]
            hits += int(hit.sum())
            tp += (pred & yt).sum(axis=0)
            pos += yt.sum(axis=0)
            # One emitter per window: one label, a DRFM pair or an ambiguous
            # group (src/data/labels.py), each scored under its own class.
            rows, cls = labels.standalone_pairs(yt)
            np.add.at(top1_n, cls, 1)
            np.add.at(top1_hit, cls, hit[rows])
    recall = np.where(pos > 0, tp / np.maximum(pos, 1), np.nan)
    top1 = np.where(top1_n > 0, top1_hit / np.maximum(top1_n, 1), np.nan)
    return {"loss": loss / len(X), "top1": hits / len(X),
            "macro_recall_05": float(np.nanmean(recall)),
            "per_class": {c: {"recall_05": float(recall[i]), "top1_standalone": float(top1[i])}
                          for i, c in enumerate(DATASET_CLASSES)}}


def save_atomic(obj, path):
    """Write next to the target, then rename: a session killed mid-save (or
    a Drive sync catching it half-written) never leaves a broken last.pt."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_when_visible(path, tries=30, wait_s=2.0):
    """torch.load, waiting for the file to appear. On the Colab Drive mount a
    file renamed into place (save_atomic) can be missing for a few seconds;
    the 2026-11-07 resume finished all 20 epochs, then died here on best.pt."""
    for k in range(tries):
        try:
            return torch.load(path, map_location=DEVICE, weights_only=False)
        except FileNotFoundError:
            if k == tries - 1:
                raise
            time.sleep(wait_s)


def main():
    t = CFG["training"]
    w = CFG["week1"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True, help="your fixed ensemble seed (1-5)")
    p.add_argument("--data", default=str(REPO_ROOT / w["combined_data"]), help="folder with train/ val/ test/")
    p.add_argument("--ckpt-dir", default=None, help="default results/week1_seed<k>; put it on Drive in Colab")
    p.add_argument("--member-out", default=None, help="copy best.pt here when done, e.g. .../members/seed1.pt")
    p.add_argument("--epochs", type=int, default=t["epochs"])
    p.add_argument("--batch-size", type=int, default=t["batch_size"])
    p.add_argument("--eval-test", action="store_true", help="score the test split at the end -> scorecard.json")
    p.add_argument("--limit", type=int, default=0, help="quick check: first N train windows (N/8 val/test)")
    p.add_argument("--fresh", action="store_true", help="ignore an existing last.pt and start over")
    p.add_argument("--expert-features", action="store_true",
                   help="add the amplitude-histogram + delay-correlation branch (overrides week1.expert_features)")
    p.add_argument("--full-spectrum", action="store_true",
                   help="add the whole-window spectrum branch (SED-39 experiment)")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir or REPO_ROOT / CFG["paths"]["checkpoints"] / f"week1_seed{args.seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    info_path = Path(args.data) / "combined.json"
    combined_version = json.loads(info_path.read_text())["combined_version"] if info_path.exists() else "unknown"
    commit, dirty = _git_commit()
    print(f"device {DEVICE}  seed {args.seed}  combined {combined_version}  commit {commit[:10]}"
          f"{' (dirty)' if dirty else ''}  n_fft {w['stft_n_fft']}")

    small = args.limit // 8 if args.limit else 0
    Xtr, ytr, snr_tr = load_split(args.data, "train", args.limit)
    Xva, yva, _ = load_split(args.data, "val", small)
    print(f"train {len(Xtr)}  val {len(Xva)}")

    set_seed(args.seed)
    model_flags = {k: CFG.get("model", {}).get(k, False)
                   for k in ("stft_freq_summary", "stft_keep_rows", "cumulant_features")}
    model_flags["expert_features"] = bool(args.expert_features or w.get("expert_features", False))
    # Only recorded when on, so runs saved before this flag existed still resume.
    if args.full_spectrum:
        model_flags["full_spectrum"] = True
    model = AMC_CNN(num_classes=C, input_len=Xtr.shape[-1], stft_n_fft=w["stft_n_fft"], **model_flags).to(DEVICE)
    dampen = {DATASET_CLASSES.index(c): m for c, m in w.get("class_weight_multipliers", {}).items()}
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(ytr, C, dampen=dampen).to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=t["scheduler_patience"])
    sample_w = compute_snr_weights(snr_tr, ytr, [NOISE_IDX]).double()

    meta = {"format": "nexa-week1-member-v1", "classes": DATASET_CLASSES, "seed": args.seed,
            "stft_n_fft": w["stft_n_fft"], "input_len": int(Xtr.shape[-1]),
            "model_flags": model_flags,
            "combined_version": combined_version, "git_commit": commit, "git_dirty": dirty,
            "batch_size": args.batch_size, "limit": args.limit}

    start, best_loss, best_epoch, history = 0, float("inf"), None, []
    last = ckpt_dir / "last.pt"
    if last.exists() and not args.fresh:
        state = torch.load(last, map_location=DEVICE, weights_only=False)
        for key in ("seed", "stft_n_fft", "combined_version", "batch_size", "limit", "model_flags"):
            if state["meta"].get(key) != meta[key]:
                sys.exit(f"{last} was saved with {key}={state['meta'][key]!r}, this run has {meta[key]!r}. "
                         "Use the same arguments to resume, or --fresh to start over.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, best_loss, best_epoch, history = (state["epoch"], state["best_loss"],
                                                 state["best_epoch"], state["history"])
        print(f"resuming after epoch {start} (best so far: epoch {best_epoch}, val_loss {best_loss:.4f})")

    for epoch in range(start, args.epochs):
        t0 = time.time()
        model.train()
        # Seeded per (seed, epoch) so a resumed epoch draws the same batches.
        g = torch.Generator().manual_seed(args.seed * 100_000 + epoch)
        order = torch.multinomial(sample_w, len(Xtr), replacement=True, generator=g).numpy()
        tloss = 0.0
        for s in range(0, len(order), args.batch_size):
            xb, yb = batch_to_device(Xtr, ytr, order[s:s + args.batch_size])
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * len(xb)
        tloss /= len(order)

        val = evaluate(model, criterion, Xva, yva)
        sched.step(val["loss"])
        improved = val["loss"] < best_loss
        if improved:
            best_loss, best_epoch = val["loss"], epoch + 1
            save_atomic({"state_dict": model.state_dict(),
                         "meta": {**meta, "epoch": epoch + 1, "val": {k: v for k, v in val.items()
                                                                     if k != "per_class"}}},
                        ckpt_dir / "best.pt")
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "train_loss": tloss, "val_loss": val["loss"],
                        "val_top1": val["top1"], "val_macro_recall_05": val["macro_recall_05"],
                        "lr": lr, "seconds": round(time.time() - t0, 1), "improved": improved})
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "epoch": epoch + 1, "best_loss": best_loss, "best_epoch": best_epoch,
                     "history": history, "meta": meta}, last)
        (ckpt_dir / "history.json").write_text(json.dumps({"meta": meta, "best_epoch": best_epoch,
                                                           "epochs": history}, indent=2))
        left = (args.epochs - epoch - 1) * (time.time() - t0) / 60
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={tloss:.4f}  val_loss={val['loss']:.4f}  "
              f"top1={val['top1']:.3f}  recall@0.5={val['macro_recall_05']:.3f}  lr={lr:g}  "
              f"{time.time() - t0:.0f}s  (~{left:.0f} min left){'  <- best' if improved else ''}", flush=True)

    best = load_when_visible(ckpt_dir / "best.pt")
    print(f"Done. Best epoch {best['meta']['epoch']}: {ckpt_dir / 'best.pt'}")

    if args.eval_test:
        model.load_state_dict(best["state_dict"])
        Xte, yte, _ = load_split(args.data, "test", small)
        test = evaluate(model, criterion, Xte, yte)
        (ckpt_dir / "scorecard.json").write_text(json.dumps({"meta": best["meta"], "test": test}, indent=2))
        zero = [c for c, v in test["per_class"].items() if v["top1_standalone"] == 0]
        print(f"test: top1={test['top1']:.3f}  macro recall@0.5={test['macro_recall_05']:.3f}  "
              f"classes at 0 top-1: {len(zero)} {zero}")

    if args.member_out:
        Path(args.member_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt_dir / "best.pt", args.member_out)
        print(f"member copied to {args.member_out}")


if __name__ == "__main__":
    main()
