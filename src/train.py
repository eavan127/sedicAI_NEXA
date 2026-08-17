"""
Training loop for AMC_CNN.

Usage:
    python -m src.train
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(y, snr_labels, val_frac, test_frac, seed):
    """Split stratified by (class, SNR bin) jointly.

    Stratifying by class alone would let an SNR bin land entirely in one split,
    making the accuracy-vs-SNR curve meaningless for that bin.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(y):
        for snr in np.unique(snr_labels):
            group = np.flatnonzero((y == cls) & (snr_labels == snr))
            if group.size == 0:
                continue
            rng.shuffle(group)
            n_test = int(round(group.size * test_frac))
            n_val = int(round(group.size * val_frac))
            test_idx.append(group[:n_test])
            val_idx.append(group[n_test:n_test + n_val])
            train_idx.append(group[n_test + n_val:])

    return (np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx))


def compute_class_weights(y, num_classes):
    """Inverse-frequency weights, so the rarer judged classes aren't drowned out."""
    counts = np.maximum(np.bincount(y, minlength=num_classes), 1)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def compute_snr_weights(snr_labels):
    """Per-example sampling weight favouring low-SNR (harder) examples.

    Every SNR bin has equal example counts, but errors do not distribute
    equally across them -- on the judged classes, nearly all misclassification
    is concentrated in the two lowest bins (-10dB, -6dB), where a faded signal
    of any class starts to resemble noise-like classes. Equal sampling gives
    the model equal exposure to bins it already solves well and bins it
    doesn't, wasting gradient steps on the easy majority.

    Weighted by sqrt(linear noise-to-signal ratio) = 10^(-SNR_db/20), so the
    lowest bin (-10dB) gets roughly 10x the sampling weight of the highest
    (+10dB) -- steep enough to shift real attention toward the hard bins,
    without the ~100x an undamped linear-power weighting would give, which
    risked starving the easy bins the model already handles well.
    """
    ratio = 10 ** (-np.asarray(snr_labels, dtype=np.float64) / 20)
    return torch.tensor(ratio / ratio.mean(), dtype=torch.float32)


def load_data():
    data_dir = REPO_ROOT / CFG["paths"]["processed_data"]
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    snr_labels = np.load(data_dir / "snr_labels.npy")
    return X, y, snr_labels


def set_seed(seed):
    """Make a training run reproducible.

    Without this, weight initialisation and batch shuffling differ every run, so
    two runs of the SAME config give different numbers — and a config change
    cannot be told apart from random variation. Measure the spread with
    scripts/measure_variance.py before attributing any result to a change.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(seed=None):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    t = CFG["training"]

    set_seed(d["seed"] if seed is None else seed)

    train_idx, val_idx, _ = stratified_split(
        y, snr_labels, d["val_frac"], d["test_frac"], d["seed"]
    )

    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)
    train_sampler = WeightedRandomSampler(
        compute_snr_weights(snr_labels[train_idx]),
        num_samples=len(train_idx), replacement=True,
    )
    train_loader = DataLoader(
        TensorDataset(X_t[train_idx], y_t[train_idx]),
        batch_size=t["batch_size"], sampler=train_sampler,
    )
    val_loader = DataLoader(
        TensorDataset(X_t[val_idx], y_t[val_idx]), batch_size=t["batch_size"]
    )

    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y, len(CLASSES)).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=t["scheduler_patience"]
    )

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(t["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"epoch {epoch+1}/{t['epochs']}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    print(f"Done. Best checkpoint: {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    train()
