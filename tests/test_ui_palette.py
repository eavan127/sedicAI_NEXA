import pytest

from src.config import TIERS
from src.ui.palette import INSTRUMENT, TIER_COLOR, TRUTH_STYLE, tier_color


def test_every_tier_has_a_color():
    for tier in TIERS:
        assert tier in TIER_COLOR


def test_tier_color_rejects_unknown_tier():
    """Silently returning grey for a typo'd tier would hide the bug on screen."""
    with pytest.raises(KeyError):
        tier_color("Nonexistent")


def test_instrument_color_is_not_a_tier_color():
    """MEASURED elements must never be styled as detections."""
    assert INSTRUMENT["color"] not in TIER_COLOR.values()


def test_truth_style_is_not_a_tier_color():
    assert TRUTH_STYLE["color"] not in TIER_COLOR.values()


def test_truth_style_is_visually_distinct_from_solid_detections():
    assert TRUTH_STYLE["linestyle"] != "solid"
    assert TRUTH_STYLE["fill"] is False


def test_tier_colors_are_all_distinct():
    assert len(set(TIER_COLOR.values())) == len(TIER_COLOR)
