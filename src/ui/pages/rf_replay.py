"""RF Replay -- the replay deck.

Named Replay, not Live: there is no SDR. The page name states that rather than
relying on a badge to walk back a misleading title. OmniSIG itself supports
recorded-file playback, so this is the same mode, not a lesser one.
"""
import gradio as gr

from src.ui import plots
from src.ui.session import load_scenario, load_upload

HOP_CHOICES = [("no overlap — 512", 512), ("50% — 256", 256),
                ("75% — 128", 128), ("87.5% — 64", 64)]


def _rows(session, smoothed):
    return [
        [i + 1, f"{e.start_us / 1000:.2f}", f"{e.duration_us / 1000:.2f}",
         e.label, " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)]
        for i, e in enumerate(session.emitter_events(smoothed=smoothed))
    ]


def _render(session, smoothing_choice):
    smoothed = smoothing_choice == "Smoothed"
    rows = _rows(session, smoothed)
    head = (
        f"**● REPLAY** &nbsp; source `{session.source}` &nbsp;·&nbsp; "
        f"{session.duration_ms:.1f} ms &nbsp;·&nbsp; "
        f"{session.result.n_windows} windows @ hop {session.result.hop} "
        f"&nbsp;·&nbsp; **{len(rows)} events**"
    )
    return (session, head,
            plots.spectrum_figure(session),
            plots.waterfall_figure(session, smoothed=smoothed),
            plots.ribbon_figure(session, smoothed=smoothed),
            rows)


def build(state, get_model):
    gr.Markdown("### RF Replay")
    gr.Markdown(
        "Recorded or synthesized capture — baseband, fs 3.2 MHz. "
        "There is no live SDR ingest."
    )

    with gr.Row():
        with gr.Column(scale=1):
            scenario_btn = gr.Button("Synthesize scenario", variant="primary")
            file_in = gr.File(label="…or upload raw IQ (interleaved float32)")
            upload_btn = gr.Button("Analyze upload")
            hop = gr.Dropdown(choices=HOP_CHOICES, value=256,
                               label="Window hop (overlap)")
            smoothing = gr.Radio(choices=["Smoothed", "Raw"], value="Smoothed",
                                  label="Display")
            gr.Markdown(
                "<small>Smoothing, the NOISE_FLOOR gate and event hold are "
                "display-only. The Performance page is always per-window, "
                "ungated and unsmoothed.</small>"
            )
        with gr.Column(scale=3):
            header = gr.Markdown()
            spectrum = gr.Plot(label="Power spectrum — measured")
            with gr.Row():
                waterfall = gr.Plot(label="Waterfall (measured) + detections (model)")
                ribbon = gr.Plot(label="Tier")
            events = gr.Dataframe(
                headers=["#", "Start (ms)", "Duration (ms)", "Detected", "Peak"],
                label="Detection events", interactive=False)

    outputs = [state, header, spectrum, waterfall, ribbon, events]

    scenario_btn.click(
        lambda h, sm: _render(load_scenario(get_model(), total_duration=0.05,
                                             hop=h), sm),
        inputs=[hop, smoothing], outputs=outputs)

    upload_btn.click(
        lambda f, h, sm: _render(
            load_upload(f.name if hasattr(f, "name") else f, get_model(), hop=h),
            sm),
        inputs=[file_in, hop, smoothing], outputs=outputs)

    smoothing.change(
        lambda s, sm: _render(s, sm) if s is not None
        else (s, "Load a capture first.", None, None, None, []),
        inputs=[state, smoothing], outputs=outputs)
