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

from src.config import CFG
from src.measure import power_spectrum_db
from src.timeline import tier_of_classes
from src.ui.palette import (BG, INSTRUMENT, PANEL, TEXT_DIM, TRUTH_STYLE,
                             WATERFALL_CMAP, style_axes, tier_color)


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
        if event.duration_us / 1000.0 >= min_labelled_ms:
            ax.text(freqs_mhz[0] + span * 0.01,
                     event.start_us / 1000.0 + event.duration_us / 2000.0,
                     event.label, color=color, fontsize=8, va="center",
                     fontweight="bold", zorder=6,
                     bbox=dict(facecolor=PANEL, edgecolor=color, alpha=0.92,
                                boxstyle="round,pad=0.3", linewidth=0.8))

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


def spectrum_figure(session):
    """MEASURED average power spectrum. Instrument styling only."""
    freqs, spectrum = power_spectrum_db(session.iq)
    fig, ax = plt.subplots(figsize=(8, 1.7))
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
    fig, ax = plt.subplots(figsize=(1.1, 6))
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
    ax.plot(t_us, np.abs(window), color=INSTRUMENT["color"],
             linewidth=INSTRUMENT["linewidth"])
    ax.set_xlabel("time within window (µs)")
    ax.set_ylabel("|IQ| (measured)")

    twin = ax.twinx()
    twin.fill_between(t_us, result.attn[window_index], 0,
                       color=tier_color("Military"), alpha=0.35)
    # Kept short: the long form ran off the right edge of the figure. The
    # "relative to this window" caveat still has to be stated -- attention is a
    # per-window softmax, so heights are not comparable between windows.
    twin.set_ylabel("attention (relative, this window)")
    twin.tick_params(colors=TEXT_DIM, labelsize=8)
    twin.yaxis.label.set_color(TEXT_DIM)
    for spine in twin.spines.values():
        spine.set_color("#1f2933")

    style_axes(fig, ax)
    plt.tight_layout()
    return fig
