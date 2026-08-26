"""
Training loop for AMC_CNN.

Usage:
    python -m src.train
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import (CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT,
                         resolve_class_weight_multipliers, resolve_multilabel_thresholds)
from src.models.amc_cnn import AMC_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(y, snr_labels, val_frac, test_frac, seed):
    """Split stratified by (label-combination, SNR bin) jointly.

    Stratifying by class alone would let an SNR bin land entirely in one split,
    making the accuracy-vs-SNR curve meaningless for that bin. `y` is now
    multi-hot (N, num_classes) -- grouping by the FULL combination (not just
    one "primary" class) keeps composite examples (e.g. BPSK+JAMMING)
    balanced across splits the same way standalone examples are, instead of
    collapsing a two-class row into whichever single class got picked as
    representative.
    """
    rng = np.random.default_rng(seed)
    combo_lookup = defaultdict(list)
    for i, row in enumerate(y):
        combo_lookup[tuple(row.astype(int))].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for combo, rows in combo_lookup.items():
        rows = np.array(rows)
        for snr in np.unique(snr_labels[rows]):
            group = rows[snr_labels[rows] == snr]
            if group.size == 0:
                continue
            rng.shuffle(group)
            n_test = int(round(group.size * test_frac))
            n_val = int(round(group.size * val_frac))
            test_idx.append(group[:n_test])
            val_idx.append(group[n_test:n_test + n_val])
            train_idx.append(group[n_test + n_val:])

    return (np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx))


def compute_class_weights(y, num_classes, dampen=None):
    """Inverse-frequency weight PER CLASS, used as BCEWithLogitsLoss's
    `pos_weight` -- so the rarer judged classes aren't drowned out.

    `y` is multi-hot (N, num_classes); a class's count is how many windows
    have that bit set, whether standalone or as part of a composite example.

    `dampen`: optional {class_idx: multiplier} applied after the
    inverse-frequency calculation. NOISE_FLOOR is the current use case: with
    equal class counts it would get the same weight (~1.0) as every other
    class, but predicting it is disproportionately cheap for the loss --
    it's a structureless class, so "say NOISE_FLOOR" is a low-risk guess
whenever a real class looks ambiguous. Measured effect: it became the
single largest source of errors for BPSK/QPSK/16QAM/64QAM/LFM_RADAR (see
evals/confusion_matrix.png), not just the low-SNR judged classes it was
added to protect. Halving its weight makes that guess less rewarding
without touching the other seven classes' balance.
    """
    counts = np.maximum(y.sum(axis=0), 1)
    weights = counts.sum() / (num_classes * counts)
    if dampen:
        for idx, mult in dampen.items():
            weights[idx] *= mult
    return torch.tensor(weights, dtype=torch.float32)


def compute_snr_weights(snr_labels, y=None, neutral_classes=()):
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

    `neutral_classes` (class indices, e.g. NOISE_FLOOR) are exempted and
    always get weight 1.0 regardless of their labelled SNR bin. Their SNR
    label is a balancing bookkeeping value, not a measure of real difficulty
    -- there is no signal fading, so there is nothing that gets harder at low
    SNR. Weighting those examples the same as a genuinely fading signal at
    that bin meant the model saw far more undifferentiated noise during
    training than intended, on top of already having a "say NOISE_FLOOR when
    unsure" option newly available -- measurably reinforcing that shortcut.

    `y`, if given, is multi-hot (N, num_classes); an example is neutral when
    ANY of its neutral-class bits is set (NOISE_FLOOR never co-occurs with
    anything else by construction, so in practice this is just "is this a
    NOISE_FLOOR example").
    """
    ratio = 10 ** (-np.asarray(snr_labels, dtype=np.float64) / 20)
    weights = ratio / ratio.mean()
    if y is not None and len(neutral_classes):
        is_neutral = np.zeros(len(weights), dtype=bool)
        for idx in neutral_classes:
            is_neutral |= (y[:, idx] == 1)
        weights = np.where(is_neutral, 1.0, weights)
    return torch.tensor(weights, dtype=torch.float32)


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


def train_model(X, y, snr_labels, train_idx, val_idx, seed, verbose=False):
    """Train one model to its best-validation-loss state and return it.

    SHARED on purpose -- train() (the single submitted checkpoint),
    scripts/train_ensemble.py (N seeds averaged) and scripts/measure_variance.py
    (N seeds compared) used to each hand-copy this loop. That divergence is how
    compute_class_weights ended up called on the FULL dataset (train+val+test)
    in two of the three copies instead of `y[train_idx]` -- fixed here, once,
    instead of three places that can silently drift apart again.

    Returns (model, best_val_loss); does not save anything to disk -- callers
    decide whether/where a checkpoint belongs.
    """
    set_seed(seed)
    t = CFG["training"]

    # NOISE_FLOOR needs different treatment from every other class in the
    # sampler -- see compute_snr_weights. The loss's per-class multipliers
    # (NOISE_FLOOR's dampen plus any boost, e.g. LFM_RADAR) live in config,
    # see resolve_class_weight_multipliers.
    noise_floor_idx = CLASS_TO_IDX.get("NOISE_FLOOR")
    neutral_classes = [noise_floor_idx] if noise_floor_idx is not None else []
    dampen = resolve_class_weight_multipliers()

    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.float32)  # multi-hot -> BCEWithLogitsLoss wants float targets
    train_sampler = WeightedRandomSampler(
        compute_snr_weights(snr_labels[train_idx], y[train_idx], neutral_classes),
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
    # Multi-label: each class is an independent yes/no, so BCEWithLogitsLoss
    # (not CrossEntropyLoss, which assumes exactly one correct class per
    # example) -- pos_weight reuses the same inverse-frequency idea per class.
    # TRAIN split only -- held-out val/test label statistics must never
    # influence what the loss rewards during training.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_class_weights(y[train_idx], len(CLASSES), dampen=dampen).to(DEVICE)
    )
    # torch tensor, not the bare numpy array resolve_multilabel_thresholds()
    # returns -- compared directly against `out`, which stays on DEVICE for
    # the whole epoch loop below rather than round-tripping through numpy.
    # Only needed for the verbose per-epoch val_bit_acc printout.
    threshold = torch.tensor(resolve_multilabel_thresholds(), device=DEVICE) if verbose else None
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=t["scheduler_patience"]
    )

    best_val_loss, best_state = float("inf"), None

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
        val_loss, correct_bits, total_bits = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                if verbose:
                    # Per-class-bit accuracy (Hamming accuracy), not argmax
                    # match -- argmax doesn't apply once more than one class
                    # can be true at once. This counts each of the 8
                    # independent yes/no calls per window, not "did every
                    # bit in the window match".
                    preds_bin = (torch.sigmoid(out) > threshold).float()
                    correct_bits += (preds_bin == yb).sum().item()
                    total_bits += yb.numel()
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if verbose:
            val_acc = correct_bits / total_bits
            print(f"epoch {epoch+1}/{t['epochs']}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_bit_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def train(seed=None):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]

    train_idx, val_idx, _ = stratified_split(
        y, snr_labels, d["val_frac"], d["test_frac"], d["seed"]
    )

    model, best_val_loss = train_model(
        X, y, snr_labels, train_idx, val_idx,
        seed=d["seed"] if seed is None else seed, verbose=True,
    )

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
    print(f"Done. Best checkpoint: {ckpt_dir / 'best_model.pt'} (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    train()
