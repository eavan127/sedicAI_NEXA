"""Light SEDIC-branded palette, and the mechanism that enforces the
provenance rule.

The spec requires every on-screen element to be identifiably MODEL, MEASURED
or TRUTH. That is enforced here rather than by convention: plotting code takes
colors from tier_color() (MODEL), INSTRUMENT (MEASURED) or TRUTH_STYLE
(TRUTH), and never writes a hex value inline. A reviewer can then check
provenance by looking at which constant a call site used.

Two colour systems, kept separate on purpose:

  BRAND_*  -- SEDIC 26 identity. Header, buttons, rules, accents.
  TIER_*   -- functional/semantic. What an operator reads to judge threat.

Folding the tier colours into the brand olive would make Military and the page
chrome the same colour, so an operator could no longer tell a detection from
decoration. Brand is not semantics.
"""
import numpy as np

# --- SEDIC 26 brand -------------------------------------------------------
# Sampled directly out of assets/sedic_logo.png rather than eyeballed -- the
# olive wordmark and the dark slate tagline, as the file actually stores them.
BRAND_OLIVE = "#627143"
BRAND_OLIVE_DARK = "#4A552F"
BRAND_OLIVE_TINT = "#EFF1EA"
BRAND_SLATE = "#121C27"

# --- light ground ---------------------------------------------------------
BG = "#F7F8F5"
PANEL = "#FFFFFF"
GRID = "#DFE3D9"
TEXT = BRAND_SLATE
TEXT_DIM = "#5F6B72"

# Sans-serif throughout, per the SEDIC brand.
FONT_STACK = ('"Inter", "Segoe UI", "Helvetica Neue", Arial, '
              '"Noto Sans", sans-serif')
# No italics anywhere in the UI. Emphasis is carried by weight and colour
# instead -- see the font-style rule in app.py's CSS, which enforces it
# globally rather than relying on every call site to remember.
MONO_STACK = ('"JetBrains Mono", "Cascadia Mono", Consolas, '
              '"DejaVu Sans Mono", monospace')

# MODEL provenance. Darkened for a light ground -- the dark-theme hues
# (#4fd1c5 etc.) have far too little contrast against white. Same hue family,
# same civilian/military/hostile/empty language, readable on paper.
TIER_COLOR = {
    "Civilian": "#0F766E",
    "Military": "#B45309",
    "Hostile": "#C1121F",
    "Empty": "#6B7280",
}

# PER-CLASS hues, for charts that draw every class at once.
#
# TIER_COLOR above is right when the tier IS the message -- four categories,
# four colours. It is wrong when eight classes are plotted together: four
# civilian classes all render in the same teal, and the reader cannot tell
# which line is which. Ordered within each tier family so the tier is still
# readable at a glance (cool = civilian, warm = military, red = hostile,
# grey = empty) while every class stays individually identifiable.
CLASS_COLOR = {
    "BPSK": "#1F6FB2",        # civilian -- blue
    "QPSK": "#2AA3A3",        # civilian -- teal
    "16QAM": "#7A5BC0",       # civilian -- violet
    "64QAM": "#C2569A",       # civilian -- magenta
    "LFM_RADAR": "#627143",   # military -- brand olive
    "FHSS": "#B07D2B",        # military -- ochre
    "JAMMING": "#C1121F",     # hostile  -- red
    "NOISE_FLOOR": "#6B7280", # empty    -- grey
}


def lighten(hex_color, amount=0.45):
    """Mix a hex colour toward white. Used to separate a secondary series from
    its primary without spending a second hue on it -- the hue keeps carrying
    class identity, lightness carries the series.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c + (255 - c) * amount))
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


# MEASURED provenance. Deliberately outside the tier hues AND outside the
# brand olive, so a waterfall, spectrum trace or SNR readout can never be
# mistaken for a detection or for page chrome.
INSTRUMENT = {"color": "#42505C", "linewidth": 1.0}

# TRUTH provenance. Outline only, dashed, never filled -- ground truth must be
# impossible to confuse with something the model produced.
TRUTH_STYLE = {"color": BRAND_SLATE, "linestyle": "dashed", "fill": False,
                "linewidth": 1.3}

# Perceptually better than jet, reads almost identically on a waterfall.
WATERFALL_CMAP = "turbo"

# Sans-serif for every figure this package owns.
MPL_FONT = ["Inter", "Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]


def tier_color(tier):
    """Color for a MODEL element of the given tier.

    Raises on an unknown tier rather than falling back to grey: a fallback
    would render a typo as a plausible-looking Empty cell and hide the bug.
    """
    return TIER_COLOR[tier]


def style_axes(fig, axes):
    """Apply the light SEDIC palette to a figure this package owns."""
    fig.patch.set_facecolor(PANEL)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT)
        ax.grid(color=GRID, alpha=0.8, linewidth=0.6)
