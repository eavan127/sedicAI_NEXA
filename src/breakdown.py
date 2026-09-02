"""Single-signal vs multi-signal recall across the SNR sweep.

The headline scorecard reports one recall per judged class over the whole test
split, which averages two quite different regimes together: windows carrying
one emitter, and windows where emitters overlap. Those behave differently
enough that the average hides the interesting part -- radar and FHSS are far
easier alone than overlapped, while jamming is slightly EASIER overlapped,
because a jammer sits 0-20 dB above its victim by construction.

This is additive analysis, not a second scorecard. It uses the same test
split, the same per-class thresholds and the same model as src/evaluate.py, so
its per-class totals reconcile with the scorecard rather than competing with
it. Nothing here is smoothed, gated or held -- those are display-layer rules
that never touch measurement.
"""
from dataclasses import dataclass

import numpy as np
import torch

from src.config import CFG, CLASSES, resolve_multilabel_thresholds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BreakdownResult:
    """recall[group][class][snr] plus per-class totals and support counts.

    `group` is "single" (exactly one class true in the window) or "multi"
    (more than one).
    """
    snr_bins: list
    classes: list
    recall: dict            # {group: {cls: {snr: pct or None}}}
    totals: dict            # {group: {cls: pct or None}}
    support: dict           # {group: {cls: {snr: int}}}
    n_windows: dict         # {group: int}


def predict_probs(model, X, batch_size=1024):
    """Sigmoid probabilities for every row of X, batched."""
    # The model's own device, not the module-level DEVICE -- see the same
    # note in src/timeline.py:classify_capture. `model` is a parameter, so
    # this cannot assume the caller moved it to DEVICE first.
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(np.array(X[i:i + batch_size])).to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def single_vs_multi(model, X, y, snr_labels, classes=None, thresholds=None):
    """Recall per class per SNR bin, split by single- vs multi-signal windows.

    `X`, `y`, `snr_labels` should already be the TEST split -- this function
    does no splitting of its own, so the caller controls that and cannot
    accidentally measure on training data.
    """
    classes = classes or list(CFG["judged_classes"])
    if thresholds is None:
        thresholds = np.array(resolve_multilabel_thresholds())

    probs = predict_probs(model, X)
    pred = probs > thresholds
    truth = y > 0.5
    n_labels = truth.sum(axis=1)
    groups = {"single": n_labels == 1, "multi": n_labels > 1}
    snr_bins = list(CFG["snr_bins_db"])

    recall, totals, support = {}, {}, {}
    for gname, gmask in groups.items():
        recall[gname], totals[gname], support[gname] = {}, {}, {}
        for cls in classes:
            j = CLASSES.index(cls)
            per_snr, per_snr_n = {}, {}
            for sb in snr_bins:
                sel = gmask & (snr_labels == sb) & truth[:, j]
                n = int(sel.sum())
                per_snr_n[sb] = n
                per_snr[sb] = float(pred[sel][:, j].mean() * 100) if n else None
            allsel = gmask & truth[:, j]
            recall[gname][cls] = per_snr
            support[gname][cls] = per_snr_n
            totals[gname][cls] = (float(pred[allsel][:, j].mean() * 100)
                                   if allsel.sum() else None)

    return BreakdownResult(
        snr_bins=snr_bins, classes=classes, recall=recall, totals=totals,
        support=support,
        n_windows={g: int(m.sum()) for g, m in groups.items()},
    )
