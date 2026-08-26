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
from src.ui.app_models import ensemble_available, load_model, model_label
from src.config import CFG
from src.scenarios import CASES
from src.ui.session import load_scenario, load_upload, reanalyze

# SNR choices are the training bins, not arbitrary round numbers -- asking the
# model about an SNR it never saw in training conflates two questions.
SNR_CHOICES = [(f"{int(s):+d} dB", int(s)) for s in CFG["snr_bins_db"]]

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


def _render(session, smoothing_choice, model_choice="auto", case_note=""):
    smoothed = smoothing_choice == "Smoothed"
    rows = _rows(session, smoothed)
    snr_note = (f"SNR {session.true_snr_db:.1f} dB KNOWN &nbsp;·&nbsp; "
                if session.snr_known and session.true_snr_db is not None
                else "")
    head = (
        f"**● REPLAY** &nbsp; source `{session.source}` &nbsp;·&nbsp; "
        f"BASEBAND · fs 3.2 MHz &nbsp;·&nbsp; {snr_note}"
        f"{model_choice} &nbsp;·&nbsp; "
        + (f"{case_note} &nbsp;·&nbsp; " if case_note else "") +
        f"{session.duration_ms:.1f} ms &nbsp;·&nbsp; "
        f"{session.result.n_windows} windows @ hop {session.result.hop} "
        f"&nbsp;·&nbsp; **{len(rows)} events**"
    )
    return (session, head, plots.console_figure(session, smoothed=smoothed),
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
        model_sel = gr.Dropdown(
            choices=[("Ensemble (5 models)", "ensemble"),
                      ("Single — best_model.pt", "single")],
            value="ensemble" if ensemble_available() else "single",
            scale=2, min_width=170, label="Model")

    with gr.Row(equal_height=True):
        case_sel = gr.Dropdown(choices=list(CASES), value="All three",
                                scale=3, min_width=190, label="Scenario case")
        snr_sel = gr.Dropdown(choices=SNR_CHOICES, value=0 if 0 in [v for _, v in SNR_CHOICES] else SNR_CHOICES[len(SNR_CHOICES)//2][1],
                               scale=2, min_width=140, label="SNR (per emitter)")
        gr.Markdown(
            "<div style='font-size:12px;color:#5F6B72;padding-top:22px;'>"
            "Single emitter through fully contested band. SNR is per emitter, "
            "so the same value means the same thing in every case."
            "</div>")

    gr.Markdown(
        "<div style='font-size:12px;color:#5F6B72;margin:-6px 0 4px 0;'>"
        "Smoothing, the NOISE_FLOOR gate and event hold are display-only. "
        "The Performance page is always per-window, ungated and unsmoothed."
        "</div>")

    header = gr.Markdown()
    # One figure, not three. Spectrum, waterfall, detection lanes and the tier
    # ribbon share a single time axis, so a detection can be read straight down
    # against the signal that produced it and against ground truth. As
    # separate plots they were three independent axes that only looked
    # adjacent.
    console = gr.Plot(label="Spectrum · waterfall · detections · tier — shared time axis")

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

    outputs = [state, header, console, events]

    scenario_btn.click(
        lambda h, sm, mw, cs, sn: _render(
            load_scenario(load_model(mw), total_duration=0.05, hop=h,
                           snr_db=sn, case=cs),
            sm, model_label(mw), f"case `{cs}`"),
        inputs=[hop, smoothing, model_sel, case_sel, snr_sel], outputs=outputs)

    upload_btn.click(
        lambda f, h, sm, mw: _render(
            load_upload(f.name if hasattr(f, "name") else f, load_model(mw),
                         hop=h),
            sm, model_label(mw)),
        inputs=[file_in, hop, smoothing, model_sel], outputs=outputs)

    model_sel.change(
        lambda s, sm, mw: _render(reanalyze(s, load_model(mw)), sm,
                                   model_label(mw)) if s is not None
        else (s, "Load a capture first.", None, []),
        inputs=[state, smoothing, model_sel], outputs=outputs)

    smoothing.change(
        lambda s, sm, mw: _render(s, sm, model_label(mw)) if s is not None
        else (s, "Load a capture first.", None, []),
        inputs=[state, smoothing, model_sel], outputs=outputs)
