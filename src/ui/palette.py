"""Dark console palette, and the mechanism that enforces the provenance rule.

The spec requires every on-screen element to be identifiably MODEL, MEASURED
or TRUTH. That is enforced here rather than by convention: plotting code takes
colors from tier_color() (MODEL), INSTRUMENT (MEASURED) or TRUTH_STYLE
(TRUTH), and never writes a hex value inline. A reviewer can then check
provenance by looking at which constant a call site used.
"""
import numpy as np

BG = "#0b0f14"
PANEL = "#121820"
GRID = "#1f2933"
TEXT = "#e6edf3"
TEXT_DIM = "#8b98a5"

# MODEL provenance. Brighter values than the light-ground UI used, matching
# the Overwatch waterfall artifact's tier hues on a dark ground.
TIER_COLOR = {
    "Civilian": "#4fd1c5",
    "Military": "#f6ad55",
    "Hostile": "#fc8181",
    "Empty": "#4a5568",
}

# MEASURED provenance. Deliberately outside the tier hues so a waterfall,
# spectrum trace or SNR readout can never be mistaken for a detection.
INSTRUMENT = {"color": "#9aa5b1", "linewidth": 1.0}

# TRUTH provenance. Outline only, dashed, never filled -- ground truth must be
# impossible to confuse with something the model produced.
TRUTH_STYLE = {"color": "#cbd5e0", "linestyle": "dashed", "fill": False,
                "linewidth": 1.2}

# Perceptually better than jet, reads almost identically on a waterfall.
WATERFALL_CMAP = "turbo"


def tier_color(tier):
    """Color for a MODEL element of the given tier.

    Raises on an unknown tier rather than falling back to grey: a fallback
    would render a typo as a plausible-looking Empty cell and hide the bug.
    """
    return TIER_COLOR[tier]


def style_axes(fig, axes):
    """Apply the dark console palette to a figure this package owns."""
    fig.patch.set_facecolor(BG)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.label.set_color(TEXT_DIM)
        ax.yaxis.label.set_color(TEXT_DIM)
        ax.title.set_color(TEXT)
        ax.grid(color=GRID, alpha=0.5, linewidth=0.6)
