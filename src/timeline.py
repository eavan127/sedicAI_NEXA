"""Sliding-window inference over a continuous IQ capture.

Everything in this module is MODEL-derived -- windows fed to the classifier,
the probabilities it returns, its attention weights, and the events grouped
from them. Nothing here measures the signal itself; that lives in
src/measure.py. Keeping the two apart is what makes the UI's provenance rule
(see docs/superpowers/specs/2026-08-24-omni-ui-design.md) structural rather
than a naming convention.
"""
from dataclasses import dataclass, replace

import numpy as np
import torch

from src.config import CFG
from src.data.preprocess import preprocess_window

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@dataclass
class TimelineResult:
    """One sliding-window pass over a capture.

    probs: (n_windows, num_classes) INDEPENDENT sigmoid probabilities -- rows
           do not sum to 1, and must never be normalized so that they do.
    attn:  (n_windows, window_len) attention weights, each row a softmax over
           time. Rows are separate distributions: a tall spike in one window
           says nothing about magnitude relative to another window.
    """
    probs: np.ndarray
    starts: np.ndarray
    attn: np.ndarray
    hop: int
    window_len: int
    fs: float

    @property
    def n_windows(self):
        return len(self.starts)

    @property
    def times_us(self):
        """Start time of each window, in microseconds from capture start."""
        return self.starts / self.fs * 1e6

    @property
    def window_duration_us(self):
        return self.window_len / self.fs * 1e6


def classify_capture(iq, model, hop=None, window_len=None, fs=None,
                     batch_size=256):
    """Run the model over every sliding window of a capture.

    Batched deliberately: looping one window at a time through a model this
    small is dominated by per-call overhead, and a 0.1 s capture at hop 256 is
    ~1,250 windows.

    Attention is captured with a forward hook on attn_pool.score, the same
    approach scripts/inspect_attention.py uses. The hook sees the RAW scores,
    before the softmax the model applies internally, so this function applies
    that softmax itself.
    """
    window_len = window_len or CFG["signal"]["window_len"]
    hop = window_len if hop is None else hop
    fs = fs or CFG["signal"]["fs"]

    windows, starts = sliding_windows(iq, window_len, hop)

    captured = {}

    def _hook(module, inputs, output):
        captured["scores"] = output.detach()

    handle = model.attn_pool.score.register_forward_hook(_hook)
    probs_batches, attn_batches = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                xb = torch.tensor(windows[i:i + batch_size]).to(DEVICE)
                logits = model(xb)
                probs_batches.append(torch.sigmoid(logits).cpu().numpy())
                scores = captured["scores"]           # (batch, 1, time)
                weights = torch.softmax(scores, dim=2)[:, 0, :]
                attn_batches.append(weights.cpu().numpy())
    finally:
        handle.remove()

    return TimelineResult(
        probs=np.concatenate(probs_batches).astype(np.float32),
        starts=starts,
        attn=np.concatenate(attn_batches).astype(np.float32),
        hop=hop, window_len=window_len, fs=fs,
    )
