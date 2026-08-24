"""Overview -- the at-a-glance page.

Status fields have fixed provenance. `Occupancy` is MEASURED (from the STFT);
`Detections` is MODEL (grouped events). `Channel Load` is deliberately NOT
used: in an RF console that name reads as an energy measurement, but the
obvious implementation here would be model output wearing a measurement's
name.
"""
import gradio as gr

from src.measure import occupancy
from src.timeline import tier_of_classes
from src.ui import plots
from src.ui.palette import PANEL, TEXT, TEXT_DIM, tier_color


def build(state):
    gr.Markdown("### Overview")
    gr.Markdown("Load a capture on RF Replay, then refresh here.")
    refresh = gr.Button("Refresh from loaded capture", variant="primary")
    status = gr.HTML()
    with gr.Row():
        mini = gr.Plot(label="Waterfall")
        latest = gr.HTML()

    def _render(session):
        if session is None:
            return "No capture loaded.", None, ""

        occ = occupancy(session.iq)
        events = session.emitter_events()
        tier_counts = {}
        for e in events:
            t = tier_of_classes(e.classes)
            tier_counts[t] = tier_counts.get(t, 0) + 1

        status_html = (
            f'<div style="font-family:monospace;background:{PANEL};padding:14px;'
            f'border-radius:6px;color:{TEXT};line-height:1.8;">'
            f'Occupancy   {occ * 100:5.1f}%   '
            f'<span style="color:{TEXT_DIM};">measured — fraction of the '
            f'spectrogram above the noise floor</span><br>'
            f'Detections  {len(events):5d}   '
            f'<span style="color:{TEXT_DIM};">model — grouped events, '
            f'not windows</span><br>'
            f'Windows     {session.result.n_windows:5d}   '
            f'<span style="color:{TEXT_DIM};">hop {session.result.hop} · '
            f'{session.duration_ms:.1f} ms capture</span></div>')

        chips = "".join(
            f'<span style="display:inline-block;margin:4px 8px 0 0;padding:2px 10px;'
            f'border-radius:9px;font-size:11px;font-weight:600;'
            f'background:{tier_color(t)}22;color:{tier_color(t)};">'
            f'{t} {n}</span>'
            for t, n in sorted(tier_counts.items()))

        if events:
            e = events[-1]
            color = tier_color(tier_of_classes(e.classes))
            latest_html = (
                f'<div style="background:{PANEL};padding:16px;border-radius:6px;'
                f'color:{TEXT};">'
                f'<div style="color:{TEXT_DIM};font-size:11px;">LATEST DETECTION</div>'
                f'<div style="font-size:20px;font-weight:600;color:{color};'
                f'margin:6px 0;">{e.label}</div>'
                f'<div style="color:{TEXT_DIM};font-family:monospace;font-size:12px;">'
                f'{e.start_us / 1000:.2f} ms · {e.duration_us / 1000:.2f} ms long<br>'
                + " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)
                + f'</div><div>{chips}</div></div>')
        else:
            latest_html = (
                f'<div style="background:{PANEL};padding:16px;border-radius:6px;'
                f'color:{TEXT_DIM};">No emitter detected in this capture.</div>')

        return status_html, plots.waterfall_figure(session), latest_html

    refresh.click(_render, inputs=state, outputs=[status, mini, latest])
