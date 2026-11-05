"""
Score windows with the week-1 60-class model: one best.pt, or several
(an ensemble, averaged).

Windows are normalised exactly as the merged training set was
(src/data/merge.normalize_batch: zero mean, unit spread per window), so a cut
from a capture reaches the model in the same form as a training window.
"""
from pathlib import Path

import numpy as np
import torch

from src.config import DATASET_CLASSES
from src.data.merge import normalize_batch
from src.models.amc_cnn import AMC_CNN

BATCH = 1024


class Classifier:
    """paths: one or more week-1 best.pt files. Calling it with complex
    windows (k, L) returns sigmoid probabilities (k, 60), averaged over the
    models."""

    def __init__(self, paths, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
        if not paths:
            raise ValueError("give at least one classifier best.pt")
        self.models, self.meta = [], []
        for p in paths:
            ck = torch.load(p, map_location=self.device, weights_only=False)
            meta = ck["meta"]
            if list(meta["classes"]) != list(DATASET_CLASSES):
                raise ValueError(f"{p} was trained on a different class list than src/config.py has now")
            m = AMC_CNN(num_classes=len(DATASET_CLASSES), input_len=meta["input_len"],
                        stft_n_fft=meta["stft_n_fft"], **meta.get("model_flags", {})).to(self.device)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models.append(m)
            self.meta.append({k: meta.get(k) for k in ("seed", "combined_version", "model_flags", "epoch")})
        self.input_len = ck["meta"]["input_len"]

    @torch.no_grad()
    def __call__(self, windows):
        x = normalize_batch(np.asarray(windows))
        out = np.zeros((len(x), len(DATASET_CLASSES)), dtype=np.float32)
        for s in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[s:s + BATCH]).to(self.device)
            out[s:s + BATCH] = np.mean([torch.sigmoid(m(xb)).cpu().numpy() for m in self.models], axis=0)
        return out
