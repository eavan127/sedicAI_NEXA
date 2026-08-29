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
from src.measure import occupancy
from src.timeline import TIER_PRIORITY, tier_of_classes
from src.ui.palette import PANEL, TEXT, TEXT_DIM, BRAND_OLIVE, GRID, tier_color

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


def _channel_state(session, smoothed):
    """How much of the capture the model reports as empty.

    Without this the console has no way to SAY "nothing is transmitting". The
    detections table lists emitters, and an empty channel has none -- so a
    capture the model correctly reads as 118/119 windows of empty spectrum
    displayed as "1 events: LFM_RADAR 28%", showing only the single false
    positive and hiding the right answer entirely.

    MODEL provenance: this is derived from classifications, not measured from
    the samples. Deliberately not called occupancy -- Overview already uses
    that name for the MEASURED figure, and the two must not be confused.
    """
    tiers = session.tiers(smoothed=smoothed)
    if not tiers:
        return ""
    empty = tiers.count("Empty") / len(tiers) * 100
    if empty >= 90:
        return (f"<span style='color:#627143;font-weight:700;'>CHANNEL EMPTY "
                f"— {empty:.0f}% of windows</span>")
    return f"channel {empty:.0f}% empty"


# A detection only headlines if the class that SETS its tier is this confident.
#
# Measured on the verification pack: on pure civilian files the military
# classes fire on roughly one window in five -- LFM_RADAR 11/60 on 16qam.f32,
# FHSS 10/60 on 64qam.f32 -- at mean probability 0.37-0.47 against thresholds
# of 0.26-0.27. Just over the line, exactly as the scorecard's LFM_RADAR
# precision of 0.631 predicts. Real detections in the same files sit at
# 93-100%. So confidence separates the two populations cleanly where duration
# does not: the jammer burst at 0.16 ms in mixed_sequence.f32 is genuine at
# 97%, and a duration floor would have thrown it away.
MIN_HEADLINE_CONFIDENCE = 0.5


def _tier_confidence(event):
    """How strongly the event supports the tier it is claiming.

    The worst class present sets the tier, so that class's confidence is what
    the claim rests on -- not the event's highest confidence overall. An event
    reading "16QAM 27% - FHSS 26%" is claiming Military on 26%, and reporting
    its 27% civilian figure instead would dress up a weak military claim in a
    number that has nothing to do with it.
    """
    tier = tier_of_classes(event.classes)
    return max(event.peak[c] for c in event.classes
               if tier_of_classes((c,)) == tier)


def _priority_key(event):
    """Sort key: worst tier first, then longest, then most confident.

    Worst tier first uses the same TIER_PRIORITY the ribbon and the detection
    boxes already use, so this panel cannot disagree with the colours beside
    it. Within a tier the longest event wins -- a sustained emitter is the
    finding, a 0.16 ms blip beside it is not, and duration says that more
    honestly than peak confidence, which one lucky window can spike.
    Confidence only breaks a duration tie.
    """
    return (TIER_PRIORITY.index(tier_of_classes(event.classes)),
            -event.duration_us,
            -max(event.peak[c] for c in event.classes))


def _events_by_priority(events):
    """Every detection in the capture, worst first."""
    return sorted(events, key=_priority_key)


def _headline_event(events):
    """The event worth putting at the top of the panel, or None.

    NOT the last event, which is what this used to show. "Latest" is a
    live-monitoring idea: on a stream the newest detection is the urgent one.
    On an uploaded capture every event is equally historical and the last one
    is simply whatever the recording happened to end on.

    And not simply the worst tier either, which is what it did next. Tier as
    an absolute gate let a 0.24 ms military false positive at 27% outrank a
    3.12 ms civilian detection at 99% -- so a pure 64QAM capture headlined
    "LFM_RADAR + FHSS". Weak, isolated detections do not get to headline; they
    are still listed below, where the operator can see them in context.

    Falls back to the strongest available event when nothing clears the bar,
    because a capture with detections must not show an empty panel. The caller
    can tell the two apart with _headline_is_confident.
    """
    if not events:
        return None
    confident = [e for e in events
                 if _tier_confidence(e) >= MIN_HEADLINE_CONFIDENCE]
    return min(confident or events, key=_priority_key)


def _headline_is_confident(events):
    """Did anything clear MIN_HEADLINE_CONFIDENCE? Drives the caveat line."""
    return any(_tier_confidence(e) >= MIN_HEADLINE_CONFIDENCE for e in events)


MAX_LISTED_DETECTIONS = 8


def _detection_list_html(events):
    """All detections, worst tier first, as a compact ordered list.

    Capped, because a 50 ms scenario at hop 256 can produce dozens of events
    and an unbounded list would push the console figure off the screen. The
    cap keeps the ones that matter -- the ordering already put them first --
    and says plainly how many were not shown rather than truncating silently.
    """
    ordered = _events_by_priority(events)
    rows = []
    for e in ordered[:MAX_LISTED_DETECTIONS]:
        tier = tier_of_classes(e.classes)
        conf = " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)
        rows.append(
            f'<div style="display:flex;align-items:baseline;gap:8px;'
            f'padding:3px 0;border-bottom:1px solid {GRID};">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:{tier_color(tier)};display:inline-block;'
            f'flex:0 0 auto;"></span>'
            f'<span style="color:{tier_color(tier)};font-weight:600;'
            f'font-size:12px;min-width:120px;">{conf}</span>'
            f'<span style="color:{TEXT_DIM};font-family:monospace;'
            f'font-size:11px;">{e.start_us / 1000:.2f} ms · '
            f'{e.duration_us / 1000:.2f} ms</span></div>')
    hidden = len(ordered) - len(rows)
    if hidden > 0:
        rows.append(
            f'<div style="color:{TEXT_DIM};font-size:11px;padding-top:6px;">'
            f'+{hidden} more, lower priority</div>')
    return (f'<div style="margin-top:12px;padding-top:10px;'
            f'border-top:1px solid {GRID};">'
            f'<div style="color:{TEXT_DIM};font-size:11px;margin-bottom:4px;">'
            f'ALL DETECTIONS · worst tier first</div>'
            + "".join(rows) + '</div>')


def _render(session, smoothing_choice, model_choice="auto", case_note=""):
    smoothed = smoothing_choice == "Smoothed"
    # Record the choice on the session so Overview and Alerts show the same
    # view. They read the same capture; without this they silently stayed on
    # smoothed while this page showed raw.
    session.display_smoothed = smoothed
    rows = _rows(session, smoothed)
    # "capped from X dB" says what was requested AND what was delivered --
    # "capped by library" only gave the achieved figure and named an
    # implementation detail ("library") that appears nowhere else in the UI.
    snr_note = (
        f"SNR {session.true_snr_db:.1f} dB KNOWN"
        + (f" (capped from {session.requested_snr_db:.0f} dB)"
           if session.snr_capped else "")
        + " &nbsp;·&nbsp; "
        if session.snr_known and session.true_snr_db is not None
        else "")
    head = (
        f"**● REPLAY** &nbsp; source `{session.source}` &nbsp;·&nbsp; "
        f"BASEBAND · fs 3.2 MHz &nbsp;·&nbsp; {snr_note}"
        f"{model_choice} &nbsp;·&nbsp; "
        + (f"{case_note} &nbsp;·&nbsp; " if case_note else "") +
        f"{session.duration_ms:.1f} ms &nbsp;·&nbsp; "
        f"{session.result.n_windows} windows @ hop {session.result.hop} "
        f"&nbsp;·&nbsp; **{len(rows)} emitter events** &nbsp;·&nbsp; "
        + _channel_state(session, smoothed)
    )
    # The constellation is a separate component, not another panel inside the
    # console figure: that figure's whole premise is one shared time axis, and
    # a constellation has no time axis at all. Hidden outright when the
    # capture has no civilian window, so military-only cases look exactly as
    # they did before this panel existed.
    constellation = plots.constellation_figure(session, smoothed=smoothed)
    constellation_update = (gr.update(value=constellation, visible=True)
                             if constellation is not None
                             else gr.update(visible=False))
    # --- Overview Metrics ---
    occ = occupancy(session.iq)
    events = session.emitter_events(smoothed=smoothed)
    tiers_list = session.tiers(smoothed=smoothed)
    empty_pct_val = tiers_list.count("Empty") / max(len(tiers_list), 1) * 100
    channel_empty = empty_pct_val >= 90
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
        f'{session.duration_ms:.1f} ms capture</span><br>'
        f'Channel     {empty_pct_val:5.0f}%   '
        f'<span style="color:{TEXT_DIM};">model — windows reported as '
        f'empty spectrum</span></div>')

    chips = "".join(
        f'<span style="display:inline-block;margin:4px 8px 0 0;padding:2px 10px;'
        f'border-radius:9px;font-size:11px;font-weight:600;'
        f'background:{tier_color(t)}22;color:{tier_color(t)};">'
        f'{t} {n}</span>'
        for t, n in sorted(tier_counts.items()))

    if channel_empty:
        extra = ""
        headline = _headline_event(events)
        if headline is not None:
            e = headline
            extra = (
                f'<div style="margin-top:12px;padding-top:10px;'
                f'border-top:1px solid {GRID};color:{TEXT_DIM};'
                f'font-size:12px;">Isolated detection, not sustained: '
                f'<span style="color:{tier_color(tier_of_classes(e.classes))};'
                f'font-weight:600;">{e.label}</span> at '
                f'{e.start_us / 1000:.2f} ms for {e.duration_us / 1000:.2f} ms'
                f'</div>')
        latest_html = (
            f'<div style="background:{PANEL};padding:16px;border-radius:6px;'
            f'color:{TEXT};">'
            f'<div style="color:{TEXT_DIM};font-size:11px;">CHANNEL STATE</div>'
            f'<div style="font-size:22px;font-weight:700;color:{BRAND_OLIVE};'
            f'margin:6px 0;">EMPTY</div>'
            f'<div style="color:{TEXT_DIM};font-family:monospace;font-size:12px;">'
            f'{empty_pct_val:.0f}% of windows report no emitter</div>'
            f'{extra}</div>')
    elif events:
        e = _headline_event(events)
        color = tier_color(tier_of_classes(e.classes))
        latest_html = (
            f'<div style="background:{PANEL};padding:16px;border-radius:6px;'
            f'color:{TEXT};">'
            f'<div style="color:{TEXT_DIM};font-size:11px;">PRIMARY DETECTION</div>'
            f'<div style="font-size:20px;font-weight:600;color:{color};'
            f'margin:6px 0;">{e.label}</div>'
            f'<div style="color:{TEXT_DIM};font-family:monospace;font-size:12px;">'
            f'{e.start_us / 1000:.2f} ms · {e.duration_us / 1000:.2f} ms long<br>'
            + " · ".join(f"{c} {e.peak[c] * 100:.0f}%" for c in e.classes)
            + f'</div>'
            + ('' if _headline_is_confident(events) else
               f'<div style="color:{TEXT_DIM};font-size:11px;margin-top:6px;">'
               f'nothing in this capture cleared '
               f'{MIN_HEADLINE_CONFIDENCE:.0%} on the class setting its tier — '
               f'showing the strongest available</div>')
            + f'<div>{chips}</div>'
            + _detection_list_html(events)
            + '</div>')
    else:
        latest_html = (
            f'<div style="background:{PANEL};padding:16px;border-radius:6px;'
            f'color:{TEXT_DIM};">No emitter detected in this capture.</div>')

    return (session, head, status_html, latest_html, plots.console_figure(session, smoothed=smoothed),
            rows, constellation_update)


def build(state, get_model):
    gr.Markdown("### RF Replay")

    # --- Global Settings ---
    with gr.Row(equal_height=True):
        hop = gr.Dropdown(choices=HOP_CHOICES, value=256, scale=2,
                           min_width=150, label="Window hop")
        smoothing = gr.Radio(choices=["Smoothed", "Raw"], value="Smoothed",
                              scale=2, min_width=150, label="Display")
        model_sel = gr.Dropdown(
            choices=[("Ensemble (5 models)", "ensemble"),
                      ("Single — best_model.pt", "single")],
            value="ensemble" if ensemble_available() else "single",
            scale=2, min_width=170, label="Model")

    # --- Source Input ---
    with gr.Row(equal_height=True):
        file_in = gr.UploadButton("📁 Select & Analyze Source File", variant="primary", scale=1, min_width=300)

    # --- Synthesize Scenario ---
    with gr.Row(equal_height=True):
        case_sel = gr.Dropdown(choices=list(CASES), value="All three",
                                scale=3, min_width=190, label="Scenario case")
        snr_sel = gr.Dropdown(choices=SNR_CHOICES, value=0 if 0 in [v for _, v in SNR_CHOICES] else SNR_CHOICES[len(SNR_CHOICES)//2][1],
                               scale=2, min_width=140, label="SNR (per emitter)")
        scenario_btn = gr.Button("Synthesize Scenario", variant="secondary", scale=2, min_width=170)

    gr.Markdown(
        "<div style='font-size:12px;color:#5F6B72;margin:0px 0 4px 0;'>"
        "Smoothing, the NOISE_FLOOR gate and event hold are display-only. "
        "The Performance page is always per-window, ungated and unsmoothed."
        "</div>")

    header = gr.Markdown()
    with gr.Row():
        status_box = gr.HTML()
        latest_box = gr.HTML()
    # One figure, not three. Spectrum, waterfall, detection lanes and the tier
    # ribbon share a single time axis, so a detection can be read straight down
    # against the signal that produced it and against ground truth. As
    # separate plots they were three independent axes that only looked
    # adjacent.
    console = gr.Plot(label="Spectrum · waterfall · detections · tier — shared time axis")

    # Starts hidden: most cases are military-only, and an empty panel below
    # the console would read as a broken plot rather than as "not applicable".
    constellation = gr.Plot(
        label="Civilian constellation — raw I/Q vs recovered symbols",
        visible=False)

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

    outputs = [state, header, status_box, latest_box, console, events, constellation]

    scenario_btn.click(
        lambda h, sm, mw, cs, sn: _render(
            load_scenario(load_model(mw), total_duration=0.05, hop=h,
                           snr_db=sn, case=cs),
            sm, model_label(mw), f"case `{cs}`"),
        inputs=[hop, smoothing, model_sel, case_sel, snr_sel], outputs=outputs)

    file_in.upload(
        lambda f, h, sm, mw: _render(
            load_upload(f.name if hasattr(f, "name") else f, load_model(mw),
                         hop=h),
            sm, model_label(mw)),
        inputs=[file_in, hop, smoothing, model_sel], outputs=outputs)

    model_sel.change(
        lambda s, sm, mw: _render(reanalyze(s, load_model(mw)), sm,
                                   model_label(mw)) if s is not None
        else (s, "Load a capture first.", "", "", None, [], gr.update(visible=False)),
        inputs=[state, smoothing, model_sel], outputs=outputs)

    smoothing.change(
        lambda s, sm, mw: _render(s, sm, model_label(mw)) if s is not None
        else (s, "Load a capture first.", "", "", None, [], gr.update(visible=False)),
        inputs=[state, smoothing, model_sel], outputs=outputs)
