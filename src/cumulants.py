"""Normalised fourth- and sixth-order cumulant magnitudes.

Higher-order cumulants are the standard AMC-literature tool for separating
constellations of the same modulation family (e.g. 16QAM vs 64QAM) that
amplitude/phase/energy features -- and, per this project's own measurements,
a CNN trained on raw IQ -- cannot: 51.4% single-window accuracy picking the
larger class probability, 49.7% even after averaging 64 windows (biased, not
noisy, so pooling the model's own output cannot fix it), and 47.0% on true
16QAM specifically -- worse than chance. See src/measure.py's
constellation_order for the full story of the pooled, calibrated |C42|
decision rule built on top of normalized_c42, and
src/models/amc_cnn.py's CumulantFeatures (behind `model.cumulant_features`)
for the same math folded directly into the classifier as an expert feature.

All three functions below share one normalisation, applied to a 1-D array of
complex points: zero-mean is assumed already true of the input (recovered
QAM symbols and a matched-filtered IQ window both centre on the origin by
construction, so nothing here re-centres them), then scaled to unit average
power so absolute amplitude -- an AGC/scaling artefact, not information about
the constellation -- cannot move the result. The formulas themselves are the
reduced forms used throughout the cumulant-based AMC literature (Swami &
Sadler and follow-ons), which drop the E[x^2]-dependent cross terms on the
assumption of (approximate) circular symmetry -- the same simplification
normalized_c42 already made before this module existed; normalized_c40 and
normalized_c63 just extend it to the sixth order.

ONE DEFINITION: src/measure.py imports normalized_c42 from here (aliased to
its historical private name, _normalized_c42, so existing imports/tests keep
working) rather than keeping its own copy, and
src/models/amc_cnn.py's torch implementation is checked against these numpy
functions directly in tests/test_amc_cnn.py
(test_torch_cumulants_match_numpy_cumulants) rather than trusted to agree by
construction -- two implementations of the same formula (numpy here, batched
torch there, for autograd/GPU compatibility) will drift apart silently
otherwise.
"""
import numpy as np


def normalized_c42(points):
    """|C42| = |E[|x|^4] - 2*E[|x|^2]^2| of a set of complex points,
    normalised to unit average power. Returns None for an empty set or a
    set with zero power -- both mean there is nothing here to measure, not
    a C42 of 0.

    C42_THEORY in src/measure.py pins this against the ideal, noiseless
    16QAM/64QAM constellations: 0.680 and 0.619 respectively -- a real but
    modest gap that channel noise erodes further (see C42_BOUNDARY's
    comment), which is why constellation_order pools many windows rather
    than trusting one.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return None
    m2 = float(np.mean(np.abs(points) ** 2))
    if m2 <= 0:
        return None
    p = points / np.sqrt(m2)
    m2n = float(np.mean(np.abs(p) ** 2))
    m4n = float(np.mean(np.abs(p) ** 4))
    return abs(m4n - 2.0 * m2n ** 2)


def normalized_c40(points):
    """|C40| = |E[x^4] - 3*E[x^2]^2| of a set of complex points, normalised
    to unit average power. Unlike C42, this one is NOT conjugate-symmetric
    in its moments -- it is sensitive to the constellation's actual phase
    structure (a square QAM grid has 4-fold, not full, rotational symmetry,
    so E[x^4] does not vanish the way it would for a truly circularly
    symmetric source), which is what makes it a useful complement to C42
    rather than a redundant copy of it. Returns None for an empty set or a
    set with zero power.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return None
    m2 = float(np.mean(np.abs(points) ** 2))
    if m2 <= 0:
        return None
    p = points / np.sqrt(m2)
    m20 = np.mean(p ** 2)
    m40 = np.mean(p ** 4)
    return abs(m40 - 3.0 * m20 ** 2)


def normalized_c63(points):
    """|C63| = |E[|x|^6] - 9*E[|x|^2]*E[|x|^4] + 12*E[|x|^2]^3| of a set of
    complex points, normalised to unit average power -- the sixth-order
    analogue of C42, and the reduced form used throughout the cumulant-AMC
    literature under the same approximate-circular-symmetry assumption
    normalized_c42 already makes. Returns None for an empty set or a set
    with zero power.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return None
    m2 = float(np.mean(np.abs(points) ** 2))
    if m2 <= 0:
        return None
    p = points / np.sqrt(m2)
    m21 = float(np.mean(np.abs(p) ** 2))
    m42 = float(np.mean(np.abs(p) ** 4))
    m63 = float(np.mean(np.abs(p) ** 6))
    return abs(m63 - 9.0 * m21 * m42 + 12.0 * m21 ** 3)
