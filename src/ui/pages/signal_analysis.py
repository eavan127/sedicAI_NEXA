"""Signal Analysis -- inspect one 512-sample window.

This is where the window mechanism is stated plainly: the console looks
continuous, but the classifier decides one 160 us window at a time.
"""
import gradio as gr

from src.config import CFG, CLASSES
from src.measure import estimate_snr_db
from src.ui import plots
from src.ui.palette import GRID, MONO_STACK, PANEL, TEXT, TEXT_DIM


def _probability_html(session, window_index):
    """All 8 classes, each marked against its OWN threshold.

    Not a ranked single-winner list: the model is multi-label sigmoid, so
    QPSK + JAMMING is a legitimate answer and the column does not sum to 100%.
    NOISE_FLOOR is presented separately as a channel state, not as an eighth
    threat class.
    """
    probs = session.result.probs[window_index]
    rows = []
    for i, cls in enumerate(CLASSES):
        if cls == "NOISE_FLOOR":
            continue
        hit = probs[i] > session.thresholds[cls]
        mark, color = ("✓", TEXT) if hit else ("○", TEXT_DIM)
        rows.append(
            f'<div style="font-family:{MONO_STACK};color:{color};">'
            f'{mark} {cls:<11} {probs[i]:.2f} '
            f'<span style="color:{TEXT_DIM};font-size:11px;">'
            f'thr {session.thresholds[cls]:.2f}</span></div>')

    noise_p = probs[CLASSES.index("NOISE_FLOOR")]
    quiet = noise_p > session.thresholds["NOISE_FLOOR"]
    noise_block = (
        f'<div style="margin-top:12px;padding-top:8px;border-top:1px solid {GRID};'
        f'font-family:{MONO_STACK};color:{TEXT};">'
        f'{"✓" if quiet else "○"} NOISE_FLOOR {noise_p:.2f}<br>'
        f'<span style="color:{TEXT_DIM};">Signal state: '
        f'{"QUIET / NO SIGNAL" if quiet else "ACTIVE"}</span></div>')

    return (f'<div style="background:{PANEL};padding:14px;border-radius:6px;">'
            f'<div style="color:{TEXT_DIM};font-size:11px;margin-bottom:8px;">'
            f'independent probabilities · multi-label — these do not sum to 100%'
            f'</div>{"".join(rows)}{noise_block}</div>')


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
