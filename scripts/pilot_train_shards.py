"""
Pilot: train one AMC_CNN directly on a few shard groups, before combined/ exists.

This is a look at how the chosen classes separate on their own. It is NOT an
ensemble member: it has fewer classes and no mixtures, so its scores run higher
than the merged model's and must not be reported as a result. Nothing here
writes to results/ or members/.

It keeps the real training choices (model, BCE loss, optimiser, low-SNR
sampling, batch size) so per-class recall can be set against the merged model
later. The one difference is the split: shards already come as train / val /
test from different seeds, so those are used rather than stratified_split.

Output goes to --out:
    pilot_last.pt      resume point, rewritten every epoch
    pilot_best.pt      best validation loss
    history.json       loss per epoch
    scorecard.json     per-class recall and false alarm, recall by SNR, top confusions

Usage (each group is a folder of .npz shards under --shards):
    python scripts/pilot_train_shards.py --shards /content/shards --out /content/pilot
A few seconds on a laptop, to check the code runs:
    python scripts/pilot_train_shards.py --shards data/interim/pilot --manifests data/interim/pilot_m --out data/interim/pilot_out --limit 800 --epochs 1

--wide-iq swaps in WideIQBranch to test whether a longer view helps. It lives
here rather than in src/models/ because the shared model is not changed yet.
--fraction keeps a share of every shard as it loads, so a run over all 60
classes fits in Colab's memory.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASSES  # noqa: E402
from src.data.shards import MANIFEST_DIR, SPLITS, load_shard  # noqa: E402
from src.models.amc_cnn import AMC_CNN, IQBranch  # noqa: E402
from src.train import compute_class_weights, compute_snr_weights, set_seed  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUPS = ("analog", "multicarrier", "apsk")
THRESHOLD = 0.5
TOP_CONFUSIONS = 5


class WideIQBranch(IQBranch):
    """IQBranch plus two more dilated layers: each output position sees about
    520 samples instead of 43, enough to span the whole window."""

    def __init__(self, out_channels=128):
        super().__init__(out_channels)
        self.dilated3 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=48, dilation=16)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.dilated4 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=192, dilation=64)
        self.bn5 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = super().forward(x)
        x = self.relu(self.bn4(self.dilated3(x)))
        return self.relu(self.bn5(self.dilated4(x)))


def normalise(iq):
    """preprocess_window for a whole batch: (N, 512) complex to (N, 2, 512)
    float32, zero mean and unit std per window."""
    arr = np.stack([iq.real, iq.imag], axis=1)
    mean = arr.mean(axis=(1, 2), keepdims=True)
    std = arr.std(axis=(1, 2), keepdims=True)
    return ((arr - mean) / (std + 1e-8)).astype(np.float32)


def load_split(shards, groups, split, limit=None, manifest_dir=MANIFEST_DIR, fraction=1.0):
    """All windows of one split as (X, y over DATASET_CLASSES, snr_db). Each
    shard is checked against its manifest when one is present, then cut to
    `fraction` of its windows, the same ones on every run."""
    parts = []
    rng = np.random.default_rng(0)
    paths = [p for g in groups for p in sorted((Path(shards) / g).glob(f"{g}_{split}_seed*.npz"))]
    if not paths:
        raise FileNotFoundError(f"no {split} shards for {groups} under {shards}")
    for path in paths:
        manifest_path = Path(manifest_dir) / f"{path.stem}.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        shard = load_shard(path, manifest=manifest)
        if fraction < 1:
            keep = rng.choice(len(shard["iq"]), int(len(shard["iq"]) * fraction), replace=False)
            shard = {k: shard[k][keep] for k in ("iq", "y", "snr_db")}
        parts.append(shard)
        print(f"  {path.name}: {len(parts[-1]['iq'])} windows"
              f"{'' if manifest else ' (no manifest, not checked)'}")
    iq = np.concatenate([p["iq"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    snr = np.concatenate([p["snr_db"] for p in parts])
    if limit and limit < len(iq):
        keep = np.random.default_rng(0).choice(len(iq), limit, replace=False)
        iq, y, snr = iq[keep], y[keep], snr[keep]
    return normalise(iq), y, snr


def present_classes(y):
    """Column indices of the classes these shards contain, and their names."""
    cols = np.flatnonzero(y.sum(axis=0))
    return cols, [DATASET_CLASSES[c] for c in cols]


def predict(model, X, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(DEVICE)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def scorecard(probs, y, snr, names):
    """Per-class recall and false alarm at the threshold, recall per SNR, and
    each class's commonest wrong answers (highest score that isn't the truth)."""
    preds = probs > THRESHOLD
    truth = y.astype(bool)
    top = probs.argmax(axis=1)
    per_class = {}
    for i, name in enumerate(names):
        pos, neg = truth[:, i], ~truth[:, i]
        confused = {names[j]: int(np.sum(top[pos] == j)) for j in range(len(names)) if j != i}
        per_class[name] = {
            "recall": float(preds[pos, i].mean()),
            "false_alarm": float(preds[neg, i].mean()),
            "top1_recall": float((top[pos] == i).mean()),
            "recall_by_snr": {str(int(s)): float(preds[pos & (snr == s), i].mean())
                              for s in np.unique(snr[pos])},
            "confused_with": dict(sorted(((k, v) for k, v in confused.items() if v),
                                         key=lambda kv: -kv[1])[:TOP_CONFUSIONS]),
        }
    return {"threshold": THRESHOLD, "n_windows": len(y),
            "macro_recall": float(np.mean([c["recall"] for c in per_class.values()])),
            "macro_top1": float(np.mean([c["top1_recall"] for c in per_class.values()])),
            "per_class": per_class}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="folder holding one sub-folder per group")
    p.add_argument("--out", required=True, help="folder for checkpoints and the scorecard")
    p.add_argument("--groups", nargs="+", default=list(GROUPS))
    p.add_argument("--manifests", default=str(MANIFEST_DIR), help="folder of manifest JSONs")
    p.add_argument("--epochs", type=int, default=CFG["training"]["epochs"])
    p.add_argument("--seed", type=int, default=99, help="not 1-5: those are the ensemble's")
    p.add_argument("--limit", type=int, default=None, help="max windows per split (quick check)")
    p.add_argument("--fraction", type=float, default=1.0, help="share of each shard to keep")
    p.add_argument("--wide-iq", action="store_true", help="use WideIQBranch")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = CFG["training"]
    set_seed(args.seed)

    data = {s: load_split(args.shards, args.groups, s, args.limit, args.manifests, args.fraction)
            for s in SPLITS}
    cols, names = present_classes(data["train"][1])
    data = {s: (X, y[:, cols], snr) for s, (X, y, snr) in data.items()}
    print(f"{len(names)} classes on {DEVICE}: {names}")

    X_tr, y_tr, snr_tr = data["train"]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=t["batch_size"],
        sampler=WeightedRandomSampler(compute_snr_weights(snr_tr), len(X_tr), replacement=True))
    X_va, y_va, _ = data["val"]
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.tensor(y_va, dtype=torch.float32)),
        batch_size=t["batch_size"])

    model = AMC_CNN(num_classes=len(names), input_len=X_tr.shape[-1])
    if args.wide_iq:
        model.iq_branch = WideIQBranch()
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_class_weights(y_tr, len(names)).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=t["scheduler_patience"])

    # A Colab session can die mid-run, so every epoch can be picked up again.
    start, best, history = 0, float("inf"), []
    resume = out / "pilot_last.pt"
    if resume.exists():
        state = torch.load(resume, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, history = state["epoch"] + 1, state["best"], state["history"]
        print(f"Resuming after epoch {start}")

    for epoch in range(start, args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_tr)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(X_va)
        scheduler.step(val_loss)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "classes": names}, out / "pilot_best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best,
                    "history": history}, resume)
        (out / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(out / "pilot_best.pt", map_location=DEVICE)["model"])
    X_te, y_te, snr_te = data["test"]
    card = scorecard(predict(model, X_te, t["batch_size"]), y_te, snr_te, names)
    card.update(groups=args.groups, seed=args.seed, epochs=len(history), wide_iq=args.wide_iq)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2))

    print(f"\nTest set, {card['n_windows']} windows, macro recall {card['macro_recall']:.3f}")
    print(f"{'class':<12}{'recall':>8}{'false alarm':>13}  most confused with")
    for name, c in sorted(card["per_class"].items(), key=lambda kv: kv[1]["recall"]):
        print(f"{name:<12}{c['recall']:>8.3f}{c['false_alarm']:>13.3f}  "
              f"{', '.join(list(c['confused_with'])[:2])}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
