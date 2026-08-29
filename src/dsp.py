"""Shared, model-agnostic DSP building blocks.

Lives outside src/ui/plots.py (which pulls in matplotlib) and outside
src/models/ (which must stay import-light for training/inference) so both
sides -- the constellation panel's recover_symbols and the model's
expert-feature branch (src/models/amc_cnn.py, behind
`model.cumulant_features`) -- can share exactly ONE definition of the RRC
matched-filter taps instead of drifting apart with two.
"""
import numpy as np

# RadioML 2018.01A is stored at 8 samples per symbol. Named rather than
# inlined because it is the one constant a capture at another rate would
# invalidate: decimation/filtering built on it would then operate on the
# pulse shape instead of the symbol instants.
SAMPLES_PER_SYMBOL = 8

# RadioML's transmitters pulse-shape with a root-raised cosine, so a receiver
# matched to it is the standard front end -- and the one the constellation
# panel was originally missing. Without it a sample taken at the symbol
# instant carries the noise of the whole 8x-oversampled band while the
# signal occupies only the symbol bandwidth, which is most of why a +10 dB
# capture drew a cloud instead of four clusters (measured: 0.62 -> 0.79
# 4th-power phase concentration at +10 dB, 0.20 -> 0.61 at +2 dB).
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

    THE ONE DEFINITION: src/ui/plots.py's recover_symbols (numpy, mode="same"
    convolution) and src/models/amc_cnn.py's CumulantFeatures (torch, a fixed
    depthwise Conv1d, behind `model.cumulant_features`) both call this
    function rather than each keeping its own copy of the formula above.
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
