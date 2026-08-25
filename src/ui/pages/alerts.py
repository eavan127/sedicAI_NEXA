"""Alerts -- judged classes only.

NOISE_FLOOR can never raise an alert. It denotes the ABSENCE of an emitter;
"alert: nothing is transmitting" would invert the purpose of both this page
and the class. session.judged_events() enforces that.
"""
import gradio as gr

from src.config import CFG
from src.timeline import tier_of_classes
from src.ui import plots


def build(state):
    gr.Markdown("### Alerts")
    gr.Markdown(
        f"Events involving a judged class: **{', '.join(CFG['judged_classes'])}**. "
        "NOISE_FLOOR never raises an alert — it is the absence of an emitter."
    )
    refresh = gr.Button("Refresh from loaded capture", variant="primary")
    timeline = gr.Plot(label="Alert timeline")
    table = gr.Dataframe(
        headers=["Tier", "Start (ms)", "Duration (ms)", "Detected", "Peak"],
        label="Alerts", interactive=False)

    def _render(session):
        if session is None:
            return None, []
        rows = [
            [tier_of_classes(e.classes), f"{e.start_us / 1000:.2f}",
             f"{e.duration_us / 1000:.2f}", e.label,
             " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)]
            for e in session.judged_events()
        ]
        return plots.alerts_timeline_figure(session), rows

    refresh.click(_render, inputs=state, outputs=[timeline, table])
