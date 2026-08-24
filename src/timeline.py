"""Sliding-window inference over a continuous IQ capture.

Everything in this module is MODEL-derived -- windows fed to the classifier,
the probabilities it returns, its attention weights, and the events grouped
from them. Nothing here measures the signal itself; that lives in
src/measure.py. Keeping the two apart is what makes the UI's provenance rule
(see docs/superpowers/specs/2026-08-24-omni-ui-design.md) structural rather
than a naming convention.
"""
import numpy as np

from src.config import CFG
from src.data.preprocess import preprocess_window


def sliding_windows(iq, window_len=None, hop=None):
    """Cut a continuous complex IQ capture into overlapping model inputs.

    Returns (windows, starts): windows is (n, 2, window_len) float32, already
    normalized per window by preprocess_window -- which is exactly how every
    training example was normalized, so the model sees the same distribution
    it was trained on. `starts` holds each window's first sample index, which
    is what maps a window back to a position in the capture.

    The capture itself is NEVER normalized as a whole -- see the spec's
    normalization rule. Raw amplitude has to survive for the waterfall and the
    noise-floor estimate to mean anything.
    """
    window_len = window_len or CFG["signal"]["window_len"]
    # `hop or window_len` would silently turn an invalid hop=0 into a valid
    # default, so the guard below could never fire. Check for None explicitly.
    hop = window_len if hop is None else hop
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")

    iq = np.asarray(iq)
    if len(iq) < window_len:
        iq = np.pad(iq, (0, window_len - len(iq)))

    n = 1 + (len(iq) - window_len) // hop
    starts = np.arange(n) * hop
    out = np.empty((n, 2, window_len), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = preprocess_window(iq[s:s + window_len], window_len)
    return out, starts
