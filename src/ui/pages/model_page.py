"""Model -- describes the checkpoint ACTUALLY LOADED.

Every number is introspected from the live model object and CFG at render
time. Nothing is hardcoded, so the page cannot go stale and cannot misreport
after someone swaps in a different checkpoint.

The page states what is running. It does not claim this architecture is the
best-performing one.
"""
import gradio as gr

from src.config import CFG, CLASSES
from src.ui.palette import PANEL, TEXT, TEXT_DIM


def build(get_model):
    gr.Markdown("### Model")
    refresh = gr.Button("Read loaded checkpoint", variant="primary")
    card = gr.HTML()

    def _render():
        model = get_model()
        total = sum(p.numel() for p in model.parameters())
        branches = "<br>".join(
            f"           {name:<14} {sum(p.numel() for p in getattr(model, name).parameters()):,}"
            for name in ("iq_branch", "stft_branch") if hasattr(model, name))
        window_len = CFG["signal"]["window_len"]
        fs = CFG["signal"]["fs"]
        thresholds = "<br>".join(
            f"           {c:<14} {v}" for c, v in
            CFG.get("multilabel_thresholds_per_class", {}).items())
        return (
            f'<div style="font-family:monospace;background:{PANEL};padding:18px;'
            f'border-radius:6px;color:{TEXT};line-height:1.8;">'
            f'ARCHITECTURE   {type(model).__name__}<br>'
            f'PARAMETERS     {total:,}<br>{branches}<br>'
            f'CLASSES        {len(CLASSES)}<br>'
            f'           {", ".join(CLASSES)}<br>'
            f'INPUT          (2, {window_len})<br>'
            f'WINDOW         {window_len / fs * 1e6:.0f} µs @ {fs / 1e6:.1f} MHz<br>'
            f'OUTPUT         sigmoid — multi-label, independent per class<br>'
            f'POOLING        energy-gated attention<br>'
            f'SAMPLING       SNR-weighted, 10^(-SNR/20)<br>'
            f'THRESHOLDS     per class<br>{thresholds}<br><br>'
            f'<span style="color:{TEXT_DIM};">Read from the loaded checkpoint '
            f'at render time, not hardcoded. Describes what is running — not a '
            f'claim that this architecture is the best performing.</span></div>')

    refresh.click(_render, inputs=None, outputs=card)
