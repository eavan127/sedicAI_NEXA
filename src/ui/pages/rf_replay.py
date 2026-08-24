"""RF Replay -- the replay deck.

Named Replay, not Live: there is no SDR. The page name states that rather than
relying on a badge to walk back a misleading title. OmniSIG itself supports
recorded-file playback, so this is the same mode, not a lesser one.

Layout note: controls sit in one compact bar across the top, not in a side
column. A side column is only as useful as its tallest control and leaves a
tall empty gutter beside the waterfall -- which pushed the detections table
into a narrow strip that had to be zoomed to read. Full width below the
controls gives the waterfall and the table the space they actually need.
"""
import gradio as gr

from src.ui import plots
from src.ui.session import load_scenario, load_upload

HOP_CHOICES = [("no overlap — 512", 512), ("50% — 256", 256),
                ("75% — 128", 128), ("87.5% — 64", 64)]


def _rows(session, smoothed):
    """One row per event.

    Class names and their peak confidences share a single column. Splitting
    them across "Detected" and "Peak" printed every class name twice, and a
    window carrying five simultaneous classes then needed ~1560px of table --
    more than the page has, so it scrolled sideways and had to be zoomed to
    read. One column says the same thing in roughly half the width.
    """
    return [
        [i + 1, f"{e.start_us / 1000:.2f}", f"{e.duration_us / 1000:.2f}",
         " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)]
        for i, e in enumerate(session.emitter_events(smoothed=smoothed))
    ]


def _render(session, smoothing_choice):
    smoothed = smoothing_choice == "Smoothed"
    rows = _rows(session, smoothed)
    head = (
        f"**● REPLAY** &nbsp; source `{session.source}` &nbsp;·&nbsp; "
        f"BASEBAND · fs 3.2 MHz &nbsp;·&nbsp; "
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

    # --- compact control bar, one row, above everything -------------------
    with gr.Row(equal_height=True):
        scenario_btn = gr.Button("Synthesize scenario", variant="primary",
                                  scale=2, min_width=170)
        file_in = gr.File(label="Upload raw IQ (interleaved float32)",
                           file_count="single", height=78, scale=3,
                           min_width=200)
        upload_btn = gr.Button("Analyze upload", scale=2, min_width=140)
        hop = gr.Dropdown(choices=HOP_CHOICES, value=256, scale=2,
                           min_width=150, label="Window hop")
        smoothing = gr.Radio(choices=["Smoothed", "Raw"], value="Smoothed",
                              scale=2, min_width=150, label="Display")

    gr.Markdown(
        "<div style='font-size:12px;color:#5F6B72;margin:-6px 0 4px 0;'>"
        "Smoothing, the NOISE_FLOOR gate and event hold are display-only. "
        "The Performance page is always per-window, ungated and unsmoothed."
        "</div>")

    header = gr.Markdown()
    spectrum = gr.Plot(label="Power spectrum — measured")

    with gr.Row(equal_height=True):
        waterfall = gr.Plot(label="Waterfall (measured) + detections (model)",
                             scale=9)
        ribbon = gr.Plot(label="Tier", scale=1, min_width=110)

    # Kept plain: fixed column widths fought the content. A window with five
    # simultaneous classes puts "BPSK + QPSK + 16QAM + LFM_RADAR + FHSS +
    # JAMMING" in one cell and a five-way peak breakdown in the next, so the
    # honest fix was giving the page more width (see app.py) rather than
    # squeezing columns. wrap lets long cells break; max_height keeps a long
    # event list scrollable in place instead of pushing the page down.
    # Explicit widths are required for wrap to do anything: without them the
    # component sizes each column to its content, so a six-class cell just
    # makes the table wider than the page and scrolls sideways. Constraining
    # the last column is what forces the long cell to wrap instead.
    #
    # (An earlier pass removed these on the belief they stopped the table
    # rendering. That was a bad diagnosis -- the data was rendering fine and
    # the DOM query used to check it was wrong.)
    events = gr.Dataframe(
        headers=["#", "Start (ms)", "Duration (ms)", "Detected (peak confidence)"],
        label="Detection events", interactive=False, wrap=True,
        max_height=520, column_widths=["5%", "12%", "14%", "69%"])

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
