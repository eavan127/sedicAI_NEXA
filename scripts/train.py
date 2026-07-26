"""
Training loop for AMC_CNN on the dataset built by build_dataset.py.
Owner: Person D (pipeline), with input from A/B/C once data is validated.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import AMC_CNN
from build_dataset import CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_split(X, y, snr_labels, val_frac=0.15, test_frac=0.15, seed=42):
    """Simple stratified split by class (extend to also stratify by SNR bin
    once real SNR labels are populated for every example)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def make_loaders(X, y, batch_size=64):
    train_idx, val_idx, test_idx = stratified_split(X, y, None)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y, dtype=torch.long)

    train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])
    test_ds = TensorDataset(X_t[test_idx], y_t[test_idx])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train(epochs=30, batch_size=64, lr=1e-3):
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    train_loader, val_loader, test_loader = make_loaders(X, y, batch_size)

    model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    class_weights = compute_class_weights(y, len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model.pt")

    print("Training complete. Best model saved to results/best_model.pt")
    return model, test_loader


if __name__ == "__main__":
    train()
