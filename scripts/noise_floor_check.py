"""What does the model answer when a test window is really NOISE_FLOOR only?

Sorts every missed NOISE_FLOOR window into: (a) another class fired,
(b) nothing fired at all, (c) NOISE_FLOOR fired but also something else.
Usage: python scripts/noise_floor_check.py --n-models 5
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, CLASSES, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
CFG["model"]["stft_keep_rows"] = False  # shipped ensemble_*.pt were trained with it off
from src.evaluate import _predict_probs, DEVICE  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-models", type=int, default=5)
a = ap.parse_args()

X, y, snr = load_data()
d = CFG["dataset"]
_, _, ti = stratified_split(y, snr, d["val_frac"], d["test_frac"], d["seed"])
X, y, snr = X[ti], y[ti].astype(int), snr[ti]
nf = CLASSES.index("NOISE_FLOOR")
sel = (y[:, nf] == 1) & (y.sum(1) == 1)
X, snr = X[sel], snr[sel]
print(f"pure NOISE_FLOOR test windows: {len(X)}")

models = []
for i in range(a.n_models):
    m = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    m.load_state_dict(torch.load(REPO_ROOT / CFG["paths"]["checkpoints"] / f"ensemble_{i}.pt", map_location=DEVICE))
    m.eval()
    models.append(m)
th = np.asarray(resolve_multilabel_thresholds())
p = _predict_probs(models, X)
pred = p > th
print("thresholds:", dict(zip(CLASSES, np.round(th, 3))))
print(f"NOISE_FLOOR recall: {pred[:, nf].mean():.3f}")
print("mean prob NOISE_FLOOR on these windows:", round(float(p[:, nf].mean()), 3))
miss = ~pred[:, nf]
print(f"missed: {miss.sum()}")
nothing = miss & ~pred.any(1)
other = miss & pred.any(1)
print(f"  (b) nothing fired at all : {nothing.sum()}  -> cause 2 (bar too high)")
print(f"  (a) other class fired    : {other.sum()}  -> cause 1/3 (sees structure)")
c = Counter()
for r in pred[other]:
    c[",".join(CLASSES[j] for j in np.where(r)[0])] += 1
for k, v in c.most_common(8):
    print(f"      {k}: {v}")
print("also-fired while NOISE_FLOOR fired:", int((pred[:, nf] & (pred.sum(1) > 1)).sum()))
print("NOISE_FLOOR recall by SNR label:")
for s in sorted(set(snr)):
    print(f"  {s:+.0f} dB: {pred[snr == s, nf].mean():.3f}  (n={int((snr == s).sum())})")
# where the miss-probability sits: near-miss means the bar is the issue
print("prob quantiles on missed (10/50/90%):", np.round(np.quantile(p[miss, nf], [.1, .5, .9]), 3) if miss.any() else "none")
