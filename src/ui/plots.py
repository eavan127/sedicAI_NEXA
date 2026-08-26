"""Figures for the OMNI console.

Colors come from src/ui/palette.py, never inline, so provenance is checkable
at each call site: tier_color() marks MODEL, INSTRUMENT marks MEASURED,
TRUTH_STYLE marks TRUTH.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from src.config import CFG, CLASSES
from src.measure import estimate_snr_db, power_spectrum_db
from src.timeline import tier_of_classes
from src.ui.palette import (BG, GRID, INSTRUMENT, MPL_FONT, PANEL, TEXT_DIM,
                             TRUTH_STYLE, WATERFALL_CMAP, style_axes,
                             tier_color)

# Sans-serif across every figure this package owns, matching the SEDIC brand.
# Set once at import: matplotlib resolves the first family actually installed.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = MPL_FONT

# RadioML 2018.01A is stored at 8 samples per symbol. Named rather than
# inlined because it is the one constant a capture at another rate would
# invalidate: the decimation below would then sample the pulse shape instead
# of the symbol instants, and the constellation would be wrong without looking
# wrong.
SAMPLES_PER_SYMBOL = 8

# RadioML's transmitters pulse-shape with a root-raised cosine, so a receiver
# matched to it is the standard front end -- and the one this panel was
# missing. Without it a sample taken at the symbol instant carries the noise of
# the whole 8x-oversampled band while the signal occupies only the symbol
# bandwidth, which is most of why a +10 dB capture drew a cloud instead of four
# clusters (measured: 0.62 -> 0.79 4th-power phase concentration at +10 dB,
# 0.20 -> 0.61 at +2 dB).
#
# The roll-off is a guess at RadioML's, and deliberately a safe one: 0.20 and
# 0.35 score the same on those measurements, so being wrong about it costs
# nothing visible.
RRC_ROLLOFF = 0.35
RRC_SPAN_SYMBOLS = 8


def rrc_taps(sps, beta=RRC_ROLLOFF, span=RRC_SPAN_SYMBOLS):
    """Root-raised-cosine taps, unit energy, odd length so they add no delay.

    The two singular points -- t = 0 and t = 1/(4*beta) -- are written out
    separately because the general expression divides by zero at exactly those
    samples. Both branches are the limit of the closed form.

    `span * sps` must be even -- an odd `span` and an odd `sps` together
    (this project's own SAMPLES_PER_SYMBOL/RRC_SPAN_SYMBOLS are 8 and 8, so it
    never happens here, but both are parameters a caller could still pass
    oddly) would produce an even-length filter with no centre tap: the t = 0
    branch above would never fire, and the "adds no delay" claim in this
    docstring would silently stop being true. Raising here turns that into a
    loud failure instead of a filter that quietly lies about its own delay.
    """
    if (span * sps) % 2:
        raise ValueError(
            f"rrc_taps needs an odd tap count for a centre tap (no delay): "
            f"span ({span}) and sps ({sps}) cannot both be odd")
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-8:
            taps[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            taps[i] = ((np.sin(np.pi * ti * (1 - beta))
                         + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
                        / (np.pi * ti * (1 - (4 * beta * ti) ** 2)))
    return taps / np.sqrt(np.sum(taps ** 2))


def carrier_offset(window, order=4):
    """Blind estimate of residual carrier offset, in cycles per sample.

    Raising the signal to the 4th power collapses a QPSK or QAM constellation
    onto a single tone at 4x the offset, which then shows as an FFT peak. The
    4th power is used for BPSK too: it locks there as well, at the cost of a
    90-degree phase ambiguity, which is harmless here because the panel only
    de-rotates and never labels an axis with an absolute phase.

    MEASURED, not MODEL -- this reads the capture's own samples and fits
    nothing to an expected constellation, so it cannot manufacture clusters
    the samples do not contain.
    """
    z = np.asarray(window)
    # An FFT over fewer than a couple of cycles' worth of samples has no
    # meaningful peak to find, so the honest answer is no estimate rather
    # than a spurious one. `order * 2` happens to equal SAMPLES_PER_SYMBOL
    # when order=4, but that is a coincidence, not coupling between the two.
    if len(z) < order * 2:
        return 0.0
    spectrum = np.abs(np.fft.fft(z ** order))
    k = int(np.argmax(spectrum))
    if k >= len(z) / 2:          # negative frequencies live in the upper half
        k -= len(z)
    return k / len(z) / order


def recover_symbols(window, sps=SAMPLES_PER_SYMBOL):
    """Symbol points from one raw IQ window.

    Returns (points, offset_estimate, timing_phase).

    Four operations, none of them model-derived: unit-power scaling, matched
    filtering with the RRC receive filter (rrc_taps), de-rotation by the
    estimated carrier offset, and decimation to one sample per symbol at the
    timing phase whose points have the tightest amplitude spread.

    The matched filter runs with mode="same", which convolves the samples
    near each edge of the window against implicit zero padding rather than
    real signal -- measured, the first recovered symbol comes out about 46%
    low and the second about 12% low, with the rest of the window
    undistorted. Rather than let a couple of points quietly pull toward the
    origin and misrepresent the constellation, any symbol whose filter
    support (half the filter length either side) extends past the window
    edge is dropped from the returned points. On a 512-sample window at
    SAMPLES_PER_SYMBOL=8 with the default RRC taps (65 taps, so a 32-sample
    margin each side) this drops 4 symbols per edge, leaving 56 of the 64.

    Degenerate windows -- shorter than one symbol, or carrying no power --
    come back unchanged rather than raising. This feeds a display; a capture
    with a silent stretch in it must render, not crash the page.
    """
    z = np.asarray(window).astype(complex)
    power = float(np.mean(np.abs(z) ** 2)) if len(z) else 0.0
    if len(z) < sps or power <= 0:
        return z, 0.0, 0

    z = z / np.sqrt(power)
    # Matched filter first: the carrier estimate is a 4th-power FFT peak, and
    # it finds that peak more reliably once the out-of-band noise is gone.
    taps = rrc_taps(sps)
    z = np.convolve(z, taps, mode="same")
    offset = carrier_offset(z)
    z = z * np.exp(-2j * np.pi * offset * np.arange(len(z)))

    # Samples whose filter support ran off the edge of `window` and into
    # mode="same"'s implicit zero padding -- see the docstring above.
    margin = len(taps) // 2
    lo, hi = margin, len(z) - 1 - margin

    best_phase, best_score, best_points = 0, -np.inf, np.array([], dtype=complex)
    for phase in range(sps):
        idx = np.arange(phase, len(z), sps)
        idx = idx[(idx >= lo) & (idx <= hi)]
        points = z[idx]
        if not len(points):
            continue
        # Power over amplitude spread. At the symbol instant the amplitudes
        # take the constellation's own discrete levels; between symbols they
        # smear across the pulse shape, which widens the spread.
        score = float(np.mean(np.abs(points) ** 2) /
                       (np.var(np.abs(points)) + 1e-9))
        if score > best_score:
            best_phase, best_score, best_points = phase, score, points
    return best_points, offset, best_phase


def _symbols_per_window(window_len, sps=SAMPLES_PER_SYMBOL):
    """How many decimated symbol points recover_symbols actually returns for
    a non-degenerate window of this length.

    Not window_len // sps -- recover_symbols now drops edge symbols whose
    matched-filter support runs past the window edge (see its docstring), so
    the true count is smaller. The formula matches recover_symbols' own
    index selection at phase 0; it comes out the same for every phase (see
    the reasoning in recover_symbols), so phase never needs to be picked
    here. Used by constellation_figure's caption so the printed symbol count
    stays true after the edge trim, rather than quoting the un-trimmed
    figure.
    """
    margin = len(rrc_taps(sps)) // 2
    idx = np.arange(0, window_len, sps)
    idx = idx[(idx >= margin) & (idx <= window_len - 1 - margin)]
    return len(idx)


def constellation_figure(session, smoothed=None, count=4):
    """IQ constellations for `count` civilian windows spread across the span,
    or None.

    Why this panel exists: the waterfall cannot tell civilian modulations
    apart. BPSK, QPSK, 16QAM and 64QAM are the same flat wideband smear on it
    at every SNR. Cluster count IS the modulation order -- 2, 4, 16, 64 -- so
    this is the one display that carries the distinction.

    `count` columns, 2 rows each, all MEASURED. Top row is the exact (2, 512)
    array the model is fed for that window; bottom row is the SAME samples
    through recover_symbols: unit-power scaling, matched filter, de-rotation,
    decimation. Neither row is model output, so neither wears a tier colour.
    The one MODEL element is each column's class-probability text.

    Four SEPARATE windows, not one pooled scatter. The 4th-power carrier
    estimate leaves a 90-degree ambiguity per window, so pooling would render
    a BPSK capture as four clusters instead of two -- asserting the wrong
    modulation order. Keeping each window in its own axes avoids that without
    giving up the spread session.civilian_windows() exists to show.

    Returns None when civilian_windows() returns no windows; the page hides
    the component rather than drawing an empty panel.
    """
    picks = session.civilian_windows(count=count, smoothed=smoothed)
    if not picks:
        return None
    class_name = picks[0][1]           # one class for the whole figure

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    ax_top, ax_bot = axes[0], axes[1]

    for col, (index, cls, prob) in enumerate(picks):
        start = int(session.result.starts[index])
        window = session.iq[start:start + session.result.window_len]
        points, offset, phase = recover_symbols(window)
        raw = window / np.sqrt(np.mean(np.abs(window) ** 2) + 1e-20)

        # recover_symbols hands a degenerate window back UNCHANGED -- no
        # scaling, no de-rotation, no decimation -- rather than raising,
        # because a silent stretch in a capture must still render.
        # Decimation is the one operation that always shrinks the array, so
        # "points is shorter than window" is the honest test for whether
        # recovery actually ran. A window can still reach here with no
        # power: the model classifies windows independently of this panel
        # and can call a near-silent window civilian above threshold. This
        # display exists to prove clusters came from real recovery, so a
        # column must never dress up 512 raw samples as symbol points --
        # that would be the one lie this console cannot afford.
        recovered = len(points) < len(window)

        ax_r, ax_s = ax_top[col], ax_bot[col]
        ax_r.scatter(raw.real, raw.imag, s=4, alpha=0.45, linewidths=0,
                      color=INSTRUMENT["color"])
        ax_s.scatter(points.real, points.imag, s=20, alpha=0.85, linewidths=0,
                      color=INSTRUMENT["color"])

        t_ms = start / CFG["signal"]["fs"] * 1e3
        # Window index and time are MEASURED -- TEXT_DIM. The class
        # probability sits above it as a separate text so it alone can carry
        # the tier colour; set_title only takes one colour for the whole
        # string, which cannot express that split. The two are placed at
        # axes-fraction y = 1.02 and 1.16 -- far enough apart at this font
        # size that the probability's bounding box clears the title's rather
        # than sitting on top of it, which is what happened when the
        # probability was drawn 0.02 above a title carrying its own `pad`
        # (the pad added vertical space matplotlib does not report back to
        # a sibling `ax.text` call, so the two crept into the same band).
        ax_r.set_title(f"win {index} @ {t_ms:.2f} ms", fontsize=7,
                        color=TEXT_DIM, y=1.02)
        ax_r.text(0.5, 1.16, f"{cls} {prob * 100:.0f}%",
                   transform=ax_r.transAxes, ha="center", va="bottom",
                   fontsize=7, fontweight="bold", color=tier_color("Civilian"))

        if recovered:
            ax_s.set_title(f"{len(points)} symbol points", fontsize=7,
                            color=TEXT_DIM)
        else:
            ax_s.set_title("no power in this window", fontsize=7,
                            color=TEXT_DIM)

        for ax in (ax_r, ax_s):
            # Equal aspect, or a QPSK square renders as a rectangle and the
            # eye reads a constellation that is not there.
            ax.set_aspect("equal")

    # Only the leftmost column carries a Y label and only the bottom row
    # carries X labels -- repeating "I (measured)" 2 * count times is noise,
    # not information, once every column shares the same units.
    ax_top[0].set_ylabel("Q (measured)")
    ax_bot[0].set_ylabel("Q (measured)")
    for ax in ax_bot:
        ax.set_xlabel("I (measured)")

    symbols_per_window = _symbols_per_window(session.result.window_len)
    chain_text = (f"{class_name} — unit-power scale → matched filter → "
                   f"de-rotate → decimate 1-in-{SAMPLES_PER_SYMBOL}")
    caveat_text = (f"cluster count is the modulation order — "
                    f"{symbols_per_window} symbols separates 2 clusters "
                    f"from 4, not enough to resolve 64QAM")
    selection_text = (
        "four windows spaced evenly across the civilian span, not chosen "
        "for how they look — a synthesized scene splices independent "
        "recordings, so some windows straddle a seam and will not cluster")
    fig.text(0.01, 0.075, chain_text, color=TEXT_DIM, fontsize=7)
    fig.text(0.01, 0.045, caveat_text, color=TEXT_DIM, fontsize=7)
    fig.text(0.01, 0.015, selection_text, color=TEXT_DIM, fontsize=7)

    style_axes(fig, list(fig.axes))
    # Top of the rect sits just above the highest per-column text (the
    # class-probability line at axes y=1.16) rather than at the figure's own
    # edge -- tight_layout was otherwise reserving a full blank band above
    # that text for a suptitle this figure does not have.
    fig.tight_layout(rect=[0, 0.12, 1, 0.97])
    return fig


def waterfall_figure(session, smoothed=True, nperseg=256):
    """Waterfall with x = frequency (MHz) and y = time, matching the classic
    RF-console convention, plus MODEL detection overlays.

    Overlays span the FULL width deliberately. The classifier has no frequency
    axis -- STFTBranch collapses it via f.mean(dim=2) -- so a box bounded in
    frequency would assert something the model never computed. A full-width
    box bounded in time is exactly the claim the model can support.
    """
    fs = CFG["signal"]["fs"]
    f, t, Z = stft(session.iq, fs=fs, nperseg=nperseg, return_onesided=False)
    freqs_mhz = np.fft.fftshift(f) / 1e6
    power_db = 10 * np.log10(np.abs(np.fft.fftshift(Z, axes=0)) ** 2 + 1e-20)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Clip the colour scale to percentiles of the actual data. Letting
    # pcolormesh autoscale maps the noise floor into mid-turbo, so the whole
    # display comes out uniformly orange and emitters do not stand out at all.
    # Anchoring the low end near the noise floor puts noise in dark blue and
    # gives the emitters the top of the ramp.
    vmin, vmax = np.percentile(power_db, [60, 99.5])
    ax.pcolormesh(freqs_mhz, t * 1e3, power_db.T, shading="auto",
                   cmap=WATERFALL_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xlabel("frequency (MHz) — BASEBAND")
    ax.set_ylabel("time (ms)")
    ax.invert_yaxis()                      # time runs downward, waterfall style

    span = freqs_mhz[-1] - freqs_mhz[0]

    # MODEL: full-width, time-bounded detection boxes.
    #
    # Labels get an opaque backing box and are only drawn for events long
    # enough to carry one. Without that they collide into unreadable mush on a
    # busy capture -- a 0.16 ms event and a 7 ms event sitting 0.1 ms apart
    # both want the same pixel row.
    events = session.emitter_events(smoothed=smoothed)
    min_labelled_ms = max(session.duration_ms * 0.04, 0.0)
    for event in events:
        color = tier_color(tier_of_classes(event.classes))
        ax.add_patch(mpatches.Rectangle(
            (freqs_mhz[0], event.start_us / 1000.0), span,
            event.duration_us / 1000.0,
            fill=False, edgecolor=color, linewidth=2.0, zorder=5))
        if event.duration_us / 1000.0 < min_labelled_ms:
            continue

        # Class + confidence, one line per class, the way an RF console
        # labels a detection. MODEL provenance -- tier colour.
        classes_text = "\n".join(
            f"{c}  {event.peak[c] * 100:.0f}%" for c in event.classes)
        y_mid = event.start_us / 1000.0 + event.duration_us / 2000.0
        snr_text = _event_snr_text(session, event)

        # Class block sits just above the event's midline, signal level just
        # below it, so the two stack without overlapping regardless of how
        # many classes the event carries.
        ax.text(freqs_mhz[0] + span * 0.012, y_mid, classes_text,
                 color=color, fontsize=8,
                 va="bottom" if snr_text else "center", ha="left",
                 fontweight="bold", linespacing=1.4, zorder=6,
                 bbox=dict(facecolor=PANEL, edgecolor=color, alpha=0.93,
                            boxstyle="round,pad=0.35", linewidth=0.9))

        # Signal level, drawn SEPARATELY in instrument styling. It shares the
        # label visually but not provenance: the classifier does not produce
        # SNR, so it must not wear the tier colour that marks model output.
        if snr_text:
            ax.text(freqs_mhz[0] + span * 0.012, y_mid, snr_text,
                     color=INSTRUMENT["color"], fontsize=7,
                     va="top", ha="left", zorder=6,
                     bbox=dict(facecolor=PANEL, edgecolor=GRID, alpha=0.9,
                                boxstyle="round,pad=0.2", linewidth=0.6))

    # TRUTH: scenario only, dashed outline, never filled.
    if session.truth:
        for seg in session.truth:
            ax.add_patch(mpatches.Rectangle(
                (freqs_mhz[0], seg.start_s * 1000.0), span,
                (seg.end_s - seg.start_s) * 1000.0,
                fill=TRUTH_STYLE["fill"], edgecolor=TRUTH_STYLE["color"],
                linestyle=TRUTH_STYLE["linestyle"],
                linewidth=TRUTH_STYLE["linewidth"]))
            ax.text(freqs_mhz[-1] - span * 0.01,
                     (seg.start_s + seg.end_s) * 500.0,
                     f"TRUTH {seg.class_name}", color=TRUTH_STYLE["color"],
                     fontsize=7, ha="right", va="center", zorder=6,
                     bbox=dict(facecolor=BG, edgecolor=TRUTH_STYLE["color"],
                                alpha=0.85, boxstyle="round,pad=0.25",
                                linewidth=0.7, linestyle="dashed"))

    style_axes(fig, ax)
    ax.grid(False)      # a grid drawn over a waterfall obscures the data
    plt.tight_layout()
    return fig


def console_figure(session, smoothed=True, nperseg=256):
    """The main RF Replay view: every time-indexed panel on one shared X axis.

    Layout, top to bottom, all sharing time on X:

        spectrum | WATERFALL      x = time, y = frequency
                 | DETECTIONS     one lane per class, model output
                 | TIER
                 | TRUTH          scenario captures only

    Time on X because everything worth comparing here -- waterfall, detections,
    tier, truth -- is a function of time. The spectrum is the only panel that
    is a function of frequency, so it rotates to the left and shares the
    waterfall's Y instead. That is also how a real spectrum analyser is laid
    out, so nothing is being contorted to fit.

    Detections get their own lanes rather than boxes drawn on the waterfall.
    With time on X a 2 ms event in a 50 ms capture is 4% of the width -- far
    too narrow to carry a multi-line label -- and stacking lanes puts each
    class directly above its truth segment, which is the comparison that
    matters.
    """
    fs = CFG["signal"]["fs"]
    f, t, Z = stft(session.iq, fs=fs, nperseg=nperseg, return_onesided=False)
    freqs_mhz = np.fft.fftshift(f) / 1e6
    power_db = 10 * np.log10(np.abs(np.fft.fftshift(Z, axes=0)) ** 2 + 1e-20)
    t_ms = t * 1e3
    duration_ms = session.duration_ms

    events = session.emitter_events(smoothed=smoothed)
    truth_classes = {seg.class_name for seg in (session.truth or [])}
    lanes = [c for c in CLASSES
             if c != "NOISE_FLOOR"
             and (any(c in e.classes for e in events) or c in truth_classes)]
    has_truth = bool(session.truth)

    heights = [6.0, max(len(lanes) * 0.52, 0.6), 0.5]
    fig = plt.figure(figsize=(13, sum(heights) + 1.2))
    gs = fig.add_gridspec(len(heights), 2, height_ratios=heights,
                           width_ratios=[1, 9], hspace=0.12, wspace=0.02)

    ax_spec = fig.add_subplot(gs[0, 0])
    ax_wf = fig.add_subplot(gs[0, 1], sharey=ax_spec)
    ax_det = fig.add_subplot(gs[1, 1], sharex=ax_wf)
    ax_tier = fig.add_subplot(gs[2, 1], sharex=ax_wf)

    # --- spectrum, rotated: MEASURED -------------------------------------
    freqs_hz, spectrum = power_spectrum_db(session.iq)
    ax_spec.plot(spectrum, freqs_hz / 1e6, color=INSTRUMENT["color"],
                  linewidth=INSTRUMENT["linewidth"])
    ax_spec.fill_betweenx(freqs_hz / 1e6, spectrum, spectrum.min(),
                           color=INSTRUMENT["color"], alpha=0.30)
    ax_spec.tick_params(labelbottom=False)
    ax_spec.set_ylabel("frequency (MHz) — BASEBAND")
    ax_spec.invert_xaxis()

    # --- waterfall: MEASURED ---------------------------------------------
    vmin, vmax = np.percentile(power_db, [60, 99.5])
    ax_wf.pcolormesh(t_ms, freqs_mhz, power_db, shading="auto",
                      cmap=WATERFALL_CMAP, vmin=vmin, vmax=vmax)
    ax_wf.tick_params(labelleft=False, labelbottom=False)
    ax_wf.set_xlim(0, duration_ms)

    # MODEL overlays on the waterfall itself: FULL frequency height, bounded
    # only in time. The classifier has no frequency axis -- STFTBranch
    # collapses it via f.mean(dim=2) -- so a band that stopped partway up the
    # display would assert a frequency the model never computed. Unlabelled
    # here on purpose; the labels live in the lanes below, where a narrow
    # event still has room for them.
    for e in events:
        ax_wf.add_patch(mpatches.Rectangle(
            (e.start_us / 1000.0, freqs_mhz[0]), e.duration_us / 1000.0,
            freqs_mhz[-1] - freqs_mhz[0], fill=False,
            edgecolor=tier_color(tier_of_classes(e.classes)),
            linewidth=1.6, alpha=0.9, zorder=5))

    # --- detection lanes: MODEL ------------------------------------------
    for i, cls in enumerate(lanes):
        color = tier_color(tier_of_classes((cls,)))
        for e in events:
            if cls not in e.classes:
                continue
            ax_det.add_patch(mpatches.Rectangle(
                (e.start_us / 1000.0, i + 0.12), e.duration_us / 1000.0, 0.76,
                facecolor=color, edgecolor=color, alpha=0.85))
            if e.duration_us / 1000.0 > duration_ms * 0.06:
                ax_det.text(e.start_us / 1000.0 + e.duration_us / 2000.0,
                             i + 0.5, f"{e.peak[cls] * 100:.0f}%",
                             color="#ffffff", fontsize=7, ha="center",
                             va="center", fontweight="bold")
    # TRUTH drawn into the SAME lane as its class, as a dashed outline over
    # the filled detection bar. A separate strip put all three classes on one
    # row where their spans overlapped and became unreadable; here each class
    # shows detected-vs-actual on one line, which is the comparison wanted.
    for i, cls in enumerate(lanes):
        for seg in (session.truth or []):
            if seg.class_name != cls:
                continue
            ax_det.add_patch(mpatches.Rectangle(
                (seg.start_s * 1e3, i + 0.04),
                (seg.end_s - seg.start_s) * 1e3, 0.92,
                fill=False, edgecolor=TRUTH_STYLE["color"],
                linestyle=TRUTH_STYLE["linestyle"],
                linewidth=TRUTH_STYLE["linewidth"], zorder=5))

    ax_det.set_ylim(0, max(len(lanes), 1))
    ax_det.set_yticks([i + 0.5 for i in range(len(lanes))])
    ax_det.set_yticklabels(lanes, fontsize=7)
    ax_det.tick_params(labelbottom=False)
    ax_det.set_ylabel("detections\n(model)", fontsize=8)

    # --- tier ribbon: MODEL ----------------------------------------------
    tiers = session.tiers(smoothed=smoothed)
    starts_ms = session.result.starts / fs * 1e3
    step = session.result.hop / fs * 1e3
    for tm, tier in zip(starts_ms, tiers):
        ax_tier.add_patch(mpatches.Rectangle((tm, 0), step, 1,
                                              color=tier_color(tier), lw=0))
    ax_tier.set_ylim(0, 1)
    ax_tier.set_yticks([])
    ax_tier.set_ylabel("tier", fontsize=8, rotation=0, ha="right", va="center")
    ax_tier.set_xlabel("time (ms)")

    axes = [ax_spec, ax_wf, ax_det, ax_tier]
    style_axes(fig, axes)
    ax_wf.grid(False)
    ax_det.grid(axis="y", visible=False)
    return fig


def alerts_timeline_figure(session, smoothed=True):
    """Gantt timeline of judged-class events, for the Alerts page.

    A table alone gives times as numbers, which makes overlap and sequencing
    hard to read -- "did the jammer start before or during the FHSS burst?" is
    a glance on a timeline and arithmetic in a table. This is the same event
    data the table lists, drawn against time.

    Judged classes only, so NOISE_FLOOR can never appear: it denotes the
    ABSENCE of an emitter, and an alert on it would invert the page's purpose.
    """
    judged = [c for c in CFG["judged_classes"]]
    events = session.judged_events(smoothed=smoothed)
    lanes = [c for c in judged if any(c in e.classes for e in events)] or judged

    fig, ax = plt.subplots(figsize=(12, max(len(lanes) * 0.6, 1.2)))
    for i, cls in enumerate(lanes):
        colour = tier_color(tier_of_classes((cls,)))
        for e in events:
            if cls not in e.classes:
                continue
            ax.add_patch(mpatches.Rectangle(
                (e.start_us / 1000.0, i + 0.15), e.duration_us / 1000.0, 0.7,
                facecolor=colour, edgecolor=colour, alpha=0.85))
            if e.duration_us / 1000.0 > session.duration_ms * 0.05:
                ax.text(e.start_us / 1000.0 + e.duration_us / 2000.0, i + 0.5,
                         f"{e.peak[cls] * 100:.0f}%", color="#ffffff",
                         fontsize=7, ha="center", va="center",
                         fontweight="bold")
    ax.set_xlim(0, session.duration_ms)
    ax.set_ylim(0, max(len(lanes), 1))
    ax.set_yticks([i + 0.5 for i in range(len(lanes))])
    ax.set_yticklabels(lanes, fontsize=8)
    ax.set_xlabel("time (ms)")
    style_axes(fig, ax)
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    return fig


def _event_snr_text(session, event):
    """Signal level for one event.

    Always MEASURED and per-event, estimated from the capture's own noise
    floor over that event's samples, and always prefixed `est.` so it can
    never be read as a calibrated measurement.

    Deliberately NOT the scenario's known SNR: that value describes the whole
    capture, so printing it on every box repeated one number down the page and
    told an operator nothing about the individual detection. The known
    capture-level figure belongs in the status line, and is shown there.

    Never MODEL -- the classifier does not estimate SNR.
    """
    result = session.result
    lo = int(result.starts[event.start_window])
    hi = int(result.starts[event.end_window]) + result.window_len
    segment = session.iq[lo:min(hi, len(session.iq))]
    if len(segment) < 8:
        return None
    return f"est. {estimate_snr_db(segment, session.noise_power):.1f} dB"


def spectrum_figure(session):
    """MEASURED average power spectrum. Instrument styling only."""
    freqs, spectrum = power_spectrum_db(session.iq)
    fig, ax = plt.subplots(figsize=(11, 1.5))
    ax.fill_between(freqs / 1e6, spectrum, spectrum.min(),
                     color=INSTRUMENT["color"], alpha=0.30)
    ax.plot(freqs / 1e6, spectrum, color=INSTRUMENT["color"],
             linewidth=INSTRUMENT["linewidth"])
    ax.set_ylabel("dB")
    ax.set_xlim(freqs.min() / 1e6, freqs.max() / 1e6)
    style_axes(fig, ax)
    plt.tight_layout()
    return fig


def ribbon_figure(session, smoothed=True):
    """MODEL tier ribbon, one cell per window."""
    tiers = session.tiers(smoothed=smoothed)
    # Height matches waterfall_figure so the two sit flush in a Row rather
    # than leaving a band of empty panel beneath the shorter one.
    fig, ax = plt.subplots(figsize=(1.15, 6.5))
    for i, tier in enumerate(tiers):
        ax.add_patch(mpatches.Rectangle((0, i), 1, 1, color=tier_color(tier)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(len(tiers), 1))
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_ylabel("window")
    style_axes(fig, ax)
    plt.tight_layout()
    return fig


def attention_figure(session, window_index):
    """MODEL attention over the raw amplitude of one window.

    Attention is a per-window softmax, so the curve is that window's own
    distribution -- heights are NOT comparable across windows. The axis label
    says so, because the plot itself cannot.
    """
    result = session.result
    start = int(result.starts[window_index])
    window = session.iq[start:start + result.window_len]
    t_us = np.arange(len(window)) / CFG["signal"]["fs"] * 1e6

    fig, ax = plt.subplots(figsize=(9, 3.2))
    # I and Q separately, not |IQ|. The model's input is a (2, 512) real array
    # of exactly these two traces, so showing the magnitude would display
    # something the classifier never sees -- and magnitude discards the phase
    # that distinguishes BPSK from QPSK from QAM.
    ax.plot(t_us, window.real, color=INSTRUMENT["color"],
             linewidth=INSTRUMENT["linewidth"], label="I")
    ax.plot(t_us, window.imag, color=tier_color("Civilian"),
             linewidth=INSTRUMENT["linewidth"], alpha=0.85, label="Q")
    ax.set_xlabel("time within window (µs)")
    ax.set_ylabel("I / Q (measured)")
    leg = ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    leg.get_frame().set_edgecolor(GRID)

    twin = ax.twinx()
    twin.fill_between(t_us, result.attn[window_index], 0,
                       color=tier_color("Military"), alpha=0.35)
    # Kept short: the long form ran off the right edge of the figure. The
    # "relative to this window" caveat still has to be stated -- attention is a
    # per-window softmax, so heights are not comparable between windows.
    # Short enough to fit the axis. The caveat it carries is not optional:
    # attention is a per-window softmax, so heights are not comparable between
    # windows.
    twin.set_ylabel("attention · sums to 1 per window")
    twin.tick_params(colors=TEXT_DIM, labelsize=8)
    twin.yaxis.label.set_color(TEXT_DIM)
    for spine in twin.spines.values():
        spine.set_color(GRID)

    style_axes(fig, ax)
    plt.tight_layout()
    return fig
