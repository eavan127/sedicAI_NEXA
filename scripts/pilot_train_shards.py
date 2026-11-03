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
--features (amplitude and instantaneous frequency as extra IQ inputs) and
--stft-nfft (finer STFT bins) are tests of the same kind.
--fraction keeps a share of every shard as it loads, so a run over all 60
classes fits in Colab's memory.

The scorecard also gives each class its own threshold, picked on the
validation set, alongside the fixed 0.5. Rerunning a finished --out with the
same options trains nothing and just rescores the saved model.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASSES  # noqa: E402
from src.data.shards import MANIFEST_DIR, SPLITS, load_shard  # noqa: E402
from src.models.amc_cnn import AMC_CNN, IQBranch, STFTBranch  # noqa: E402
from src.train import compute_class_weights, compute_snr_weights, set_seed  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUPS = ("analog", "multicarrier", "apsk")
THRESHOLD = 0.5
CALIBRATION_GRID = np.linspace(0.02, 0.98, 49)
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


def polar_features(x):
    """(batch, 2, time) I/Q to (batch, 2, time): amplitude, and the phase step
    between neighbouring samples over pi (instantaneous frequency as a share of
    fs/2). The first sample has no step, so its frequency is 0."""
    z = torch.complex(x[:, 0], x[:, 1])
    freq = torch.angle(z[:, 1:] * z[:, :-1].conj()) / math.pi
    return torch.stack([z.abs(), F.pad(freq, (1, 0))], dim=1)


class FeatureIQBranch(IQBranch):
    """IQBranch fed amplitude and instantaneous frequency alongside I and Q.
    AM moves the first, FM and PM the second, and a constant tone leaves both
    flat; from raw I/Q the network has to work these out itself."""

    def __init__(self, out_channels=128):
        super().__init__(out_channels)
        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, padding=3)

    def forward(self, x):
        return super().forward(torch.cat([x, polar_features(x)], dim=1))


def build_model(num_classes, input_len, wide_iq=False, features=False, stft_nfft=16):
    """AMC_CNN with this script's test options swapped in. stft_nfft other than
    16 rebuilds the STFT branch with finer frequency bins (fs / stft_nfft each),
    keeping its other settings."""
    if wide_iq and features:
        raise ValueError("--wide-iq and --features are separate tests; pick one")
    model = AMC_CNN(num_classes=num_classes, input_len=input_len)
    if wide_iq:
        model.iq_branch = WideIQBranch()
    if features:
        model.iq_branch = FeatureIQBranch()
    if stft_nfft != model.stft_branch.n_fft:
        old = model.stft_branch
        model.stft_branch = STFTBranch(n_fft=stft_nfft, hop_length=old.hop_length,
                                       out_channels=old.conv2.out_channels,
                                       freq_summary=old.freq_summary, keep_rows=old.keep_rows)
    return model


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


def sampling_weights(snr, y, names):
    """compute_snr_weights for shards that hold NOISE_FLOOR. Its windows are
    stored with NaN SNR, which would turn every weight into NaN, so they count
    as 0 dB here and are then given weight 1 as neutral examples."""
    neutral = [names.index("NOISE_FLOOR")] if "NOISE_FLOOR" in names else []
    return compute_snr_weights(np.nan_to_num(snr, nan=0.0), y, neutral)


def predict(model, X, batch_size):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(DEVICE)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def calibrate(probs, y):
    """One threshold per class, the one on CALIBRATION_GRID with the best F1 on
    these (validation) windows. A class no threshold finds keeps THRESHOLD."""
    truth = y.astype(bool)[:, :, None]
    preds = probs[:, :, None] > CALIBRATION_GRID                 # (N, classes, grid)
    tp = (preds & truth).sum(axis=0)
    f1 = 2 * tp / np.maximum(preds.sum(axis=0) + truth.sum(axis=0), 1)
    best = CALIBRATION_GRID[f1.argmax(axis=1)]
    return np.where(f1.max(axis=1) > 0, best, THRESHOLD)


def _at_threshold(preds, pos, neg, snr):
    return {"recall": float(preds[pos].mean()),
            "false_alarm": float(preds[neg].mean()),
            "recall_by_snr": {str(int(s)): float(preds[pos & (snr == s)].mean())
                              for s in np.unique(snr[pos & np.isfinite(snr)])}}


def scorecard(probs, y, snr, names, thresholds=None):
    """Per-class recall and false alarm at the threshold, recall per SNR, and
    each class's commonest wrong answers (highest score that isn't the truth).
    With `thresholds` (from calibrate), each class also gets the same counts
    at its own threshold under "calibrated"."""
    truth = y.astype(bool)
    top = probs.argmax(axis=1)
    per_class = {}
    for i, name in enumerate(names):
        pos, neg = truth[:, i], ~truth[:, i]
        confused = {names[j]: int(np.sum(top[pos] == j)) for j in range(len(names)) if j != i}
        per_class[name] = {
            **_at_threshold(probs[:, i] > THRESHOLD, pos, neg, snr),
            "top1_recall": float((top[pos] == i).mean()),
            "confused_with": dict(sorted(((k, v) for k, v in confused.items() if v),
                                         key=lambda kv: -kv[1])[:TOP_CONFUSIONS]),
        }
        if thresholds is not None:
            per_class[name]["calibrated"] = {
                "threshold": float(thresholds[i]),
                **_at_threshold(probs[:, i] > thresholds[i], pos, neg, snr)}
    macro = lambda key, sub=None: float(np.mean(
        [(c[sub] if sub else c)[key] for c in per_class.values()]))
    card = {"threshold": THRESHOLD, "n_windows": len(y),
            "macro_recall": macro("recall"), "macro_false_alarm": macro("false_alarm"),
            "macro_top1": macro("top1_recall"), "per_class": per_class}
    if thresholds is not None:
        card.update(calibrated_macro_recall=macro("recall", "calibrated"),
                    calibrated_macro_false_alarm=macro("false_alarm", "calibrated"))
    return card


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
    p.add_argument("--features", action="store_true", help="use FeatureIQBranch")
    p.add_argument("--stft-nfft", type=int, default=16, help="STFT length; 64 gives 50 kHz bins")
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
        sampler=WeightedRandomSampler(sampling_weights(snr_tr, y_tr, names), len(X_tr), replacement=True))
    X_va, y_va, _ = data["val"]
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.tensor(y_va, dtype=torch.float32)),
        batch_size=t["batch_size"])

    model = build_model(len(names), X_tr.shape[-1], args.wide_iq, args.features,
                        args.stft_nfft).to(DEVICE)
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
    # Thresholds come from validation, so the test set stays unseen.
    thresholds = calibrate(predict(model, X_va, t["batch_size"]), y_va)
    X_te, y_te, snr_te = data["test"]
    card = scorecard(predict(model, X_te, t["batch_size"]), y_te, snr_te, names, thresholds)
    card.update(groups=args.groups, seed=args.seed, epochs=len(history), wide_iq=args.wide_iq,
                features=args.features, stft_nfft=args.stft_nfft)
    (out / "scorecard.json").write_text(json.dumps(card, indent=2))

    print(f"\nTest set, {card['n_windows']} windows. Macro top-1 {card['macro_top1']:.3f}, "
          f"recall at {THRESHOLD} {card['macro_recall']:.3f}, "
          f"recall at calibrated thresholds {card['calibrated_macro_recall']:.3f} "
          f"(false alarm {card['calibrated_macro_false_alarm']:.4f})")
    print(f"{'class':<18}{'top-1':>7}{'recall':>8}{'calibrated':>12}{'threshold':>11}  most confused with")
    for name, c in sorted(card["per_class"].items(), key=lambda kv: kv[1]["calibrated"]["recall"]):
        print(f"{name:<18}{c['top1_recall']:>7.3f}{c['recall']:>8.3f}{c['calibrated']['recall']:>12.3f}"
              f"{c['calibrated']['threshold']:>11.2f}  {', '.join(list(c['confused_with'])[:2])}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
