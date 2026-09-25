"""Test the rule: if no class fires, answer NOISE_FLOOR. Whole test split, 5-model ensemble."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, CLASSES, REPO_ROOT, resolve_multilabel_thresholds
CFG["model"]["stft_keep_rows"] = False  # shipped ensemble_*.pt were trained with it off
from src.evaluate import _predict_probs, DEVICE
from src.models.amc_cnn import AMC_CNN
from src.train import load_data, stratified_split

X, y, snr = load_data(); d = CFG["dataset"]
_, _, ti = stratified_split(y, snr, d["val_frac"], d["test_frac"], d["seed"])
X, y = X[ti], y[ti].astype(int)
models = []
for i in range(5):
    m = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
    m.load_state_dict(torch.load(REPO_ROOT/CFG["paths"]["checkpoints"]/f"ensemble_{i}.pt", map_location=DEVICE)); m.eval(); models.append(m)
th = np.asarray(resolve_multilabel_thresholds(), dtype=float)
p = _predict_probs(models, X); nf = CLASSES.index("NOISE_FLOOR")

def report(name, pred):
    print(f"\n== {name}")
    for c in CLASSES:
        j = CLASSES.index(c); tp = (pred[:, j] & (y[:, j] == 1)).sum()
        fp = (pred[:, j] & (y[:, j] == 0)).sum(); fn = ((~pred[:, j]) & (y[:, j] == 1)).sum()
        print(f"  {c:12s} recall {tp/max(tp+fn,1):.3f}  precision {tp/max(tp+fp,1):.3f}  (FP {fp})")

base = p > th; report("baseline (current thresholds)", base)
rule = base.copy(); rule[~base.any(1), nf] = True; report("RULE: nothing fired -> NOISE_FLOOR", rule)
print("\n== NOISE_FLOOR threshold sweep (no rule)")
for t in [0.265, 0.2, 0.15, 0.1, 0.07, 0.05]:
    pr = base.copy(); pr[:, nf] = p[:, nf] > t
    j = nf; tp = (pr[:, j] & (y[:, j] == 1)).sum(); fp = (pr[:, j] & (y[:, j] == 0)).sum(); fn = ((~pr[:, j]) & (y[:, j] == 1)).sum()
    print(f"  t={t:.3f} recall {tp/(tp+fn):.3f} precision {tp/max(tp+fp,1):.3f} FP {fp}")
print("\nwindows where nothing fired:", int((~base.any(1)).sum()), "of", len(y))
print("  of which truly NOISE_FLOOR-only:", int((~base.any(1) & (y[:, nf] == 1) & (y.sum(1) == 1)).sum()),
      "| truly holding a signal:", int((~base.any(1) & (y[:, :nf].sum(1) > 0)).sum()))
