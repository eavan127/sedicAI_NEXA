"""Checkpoint loading for the console — single model or 5-model ensemble.

Lives apart from app.py so pages can import it without importing app.py,
which imports the pages. That cycle is why this is its own module.
"""
import gradio as gr
import torch
import torch.nn as nn

from src.config import CFG, CLASSES, REPO_ROOT
from src.models.amc_cnn import AMC_CNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"

CKPT_DIR = REPO_ROOT / CFG["paths"]["checkpoints"]
N_ENSEMBLE = 5


class EnsembleModel(nn.Module):
    """Averages sigmoid probabilities across ensemble members.

    Averages PROBABILITIES, not logits -- matching _predict_probs in
    src/evaluate.py and _predict in train_ensemble.py, so the console shows
    the same numbers the scorecard does. Averaging logits instead would give
    a different (and unsubmitted) answer.

    forward() returns the average back in LOGIT space, because every caller
    does sigmoid(model(x)). sigmoid(logit(p)) == p exactly, so the interface
    is preserved without any call site needing to know whether it holds one
    model or five.

    attn_pool is deliberately member 0's, not an average: attention weights
    are a per-model internal, and averaging five models' attention would
    produce a curve no model actually computed. The UI labels it as member 0's.
    """

    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)
        self.attn_pool = models[0].attn_pool

    def forward(self, x):
        probs = torch.stack([torch.sigmoid(m(x)) for m in self.members]).mean(0)
        probs = probs.clamp(1e-6, 1 - 1e-6)
        return torch.log(probs / (1 - probs))


def ensemble_paths():
    return [CKPT_DIR / f"ensemble_{i}.pt" for i in range(N_ENSEMBLE)]


def ensemble_available():
    return all(p.exists() for p in ensemble_paths())


def _load_one(path):
    model = AMC_CNN(num_classes=len(CLASSES),
                     input_len=CFG["signal"]["window_len"]).to(DEVICE)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except RuntimeError as exc:
        raise gr.Error(
            f"{path.name} does not match the current model architecture. "
            f"Train a fresh one with the current code. Details: {exc}")
    model.eval()
    return model


def load_model(which="auto"):
    """Read from disk on every call -- no caching, so dropping a freshly
    trained checkpoint into results/ takes effect without a restart.

    "auto" prefers the ensemble when all five members are present, because
    that is what the team submits; the single checkpoint is what a lone
    training run produces.
    """
    if which == "auto":
        which = "ensemble" if ensemble_available() else "single"

    if which == "ensemble":
        missing = [p.name for p in ensemble_paths() if not p.exists()]
        if missing:
            raise gr.Error(f"Missing ensemble checkpoints: {missing}")
        return EnsembleModel([_load_one(p) for p in ensemble_paths()]).to(DEVICE)

    if not CKPT_PATH.exists():
        raise gr.Error(f"No checkpoint at {CKPT_PATH}. Train one first.")
    return _load_one(CKPT_PATH)


def model_label(which="auto"):
    if which == "auto":
        which = "ensemble" if ensemble_available() else "single"
    return (f"{N_ENSEMBLE}-model ensemble average" if which == "ensemble"
            else "single checkpoint — best_model.pt")


THEME = gr.themes.Base(primary_hue="teal", neutral_hue="slate")


