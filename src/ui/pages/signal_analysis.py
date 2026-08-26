"""Signal Analysis -- inspect one 512-sample window.

This is where the window mechanism is stated plainly: the console looks
continuous, but the classifier decides one 160 us window at a time.
"""
import gradio as gr

from src.config import CFG, CLASSES
from src.measure import estimate_snr_db
from src.ui import plots
from src.timeline import tier_of_classes
from src.ui.palette import (BG, GRID, MONO_STACK, PANEL, TEXT, TEXT_DIM,
                             tier_color)


def _probability_html(session, window_index):
    """All 8 classes as bars, each marked against its OWN threshold.

    Bars, not a ranked list: the length carries the magnitude at a glance and
    makes it visible that the values do NOT sum to 100%. This model is
    multi-label sigmoid, so QPSK + JAMMING together is a legitimate answer,
    and a softmax-style ranked list would quietly imply otherwise.

    NOISE_FLOOR is separated below the rule as a channel state rather than an
    eighth threat class.
    """
    probs = session.result.probs[window_index]

    def row(cls):
        i = CLASSES.index(cls)
        p = float(probs[i])
        hit = p > session.thresholds[cls]
        colour = tier_color(tier_of_classes((cls,))) if hit else GRID
        text = TEXT if hit else TEXT_DIM
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
            f'<span style="width:14px;color:{text};font-weight:700;">'
            f'{"&#10003;" if hit else "&#9675;"}</span>'
            f'<span style="width:96px;font-family:{MONO_STACK};font-size:12px;'
            f'color:{text};">{cls}</span>'
            f'<span style="flex:1;background:{BG};height:13px;border-radius:2px;'
            f'overflow:hidden;">'
            f'<span style="display:block;width:{p * 100:.1f}%;height:100%;'
            f'background:{colour};"></span></span>'
            f'<span style="width:42px;text-align:right;font-family:{MONO_STACK};'
            f'font-size:12px;color:{text};">{p:.2f}</span></div>')

    bars = "".join(row(c) for c in CLASSES if c != "NOISE_FLOOR")

    noise_p = float(probs[CLASSES.index("NOISE_FLOOR")])
    quiet = noise_p > session.thresholds["NOISE_FLOOR"]
    noise_block = (
        f'<div style="margin-top:12px;padding-top:10px;'
        f'border-top:1px solid {GRID};">{row("NOISE_FLOOR")}'
        f'<div style="color:{TEXT_DIM};font-size:11px;margin-left:22px;">'
        f'Signal state: {"QUIET / NO SIGNAL" if quiet else "ACTIVE"}</div></div>')

    return (f'<div style="background:{PANEL};padding:16px;border-radius:6px;">'
            f'<div style="color:{TEXT_DIM};font-size:11px;margin-bottom:10px;">'
            f'independent probabilities &middot; multi-label &mdash; these do '
            f'not sum to 100%</div>{bars}{noise_block}</div>')


def _metadata_html(session, window_index):
    result = session.result
    start = int(result.starts[window_index])
    if session.snr_known:
        snr = (f'{session.true_snr_db:.1f} dB '
               f'<span style="color:{TEXT_DIM};">KNOWN</span>')
    else:
        window = session.iq[start:start + result.window_len]
        snr = f"est. {estimate_snr_db(window, session.noise_power):.1f} dB"
    return (
        f'<div style="font-family:{MONO_STACK};color:{TEXT};background:{PANEL};'
        f'padding:14px;border-radius:6px;margin-top:10px;">'
        f'WINDOW   #{window_index + 1} / {result.n_windows}<br>'
        f'OFFSET   {start / CFG["signal"]["fs"] * 1000:.3f} ms<br>'
        f'SAMPLES  {result.window_len}<br>'
        f'DURATION {result.window_duration_us:.0f} µs<br>'
        f'SNR      {snr}</div>')


def build(state):
    gr.Markdown("### Signal Analysis")
    gr.Markdown(
        "One 512-sample window — 160 µs — exactly as the classifier sees it. "
        "Load a capture on RF Replay first, then pick a window."
    )
    with gr.Row():
        index = gr.Slider(1, 1000, value=1, step=1, label="Window")
        show = gr.Button("Inspect", variant="primary")
    with gr.Row():
        with gr.Column():
            probs_out = gr.HTML()
            meta_out = gr.HTML()
        with gr.Column(scale=2):
            attn_out = gr.Plot(label="Amplitude (measured) + attention (model)")

    def _render(session, i):
        if session is None:
            return "Load a capture on RF Replay first.", "", None
        idx = max(0, min(int(i) - 1, session.result.n_windows - 1))
        return (_probability_html(session, idx), _metadata_html(session, idx),
                plots.attention_figure(session, idx))

    show.click(_render, inputs=[state, index],
                outputs=[probs_out, meta_out, attn_out])
    index.change(_render, inputs=[state, index],
                  outputs=[probs_out, meta_out, attn_out])
