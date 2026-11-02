"""Checks the RadDet converter (scripts/convert_raddet.py): jet colours decode
back to their values, YOLO boxes land on our flipped / stretched image, and
(when RadDet is downloaded) a real frame converts into our format."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_raddet as rd  # noqa: E402

from src.detect import scenes  # noqa: E402

RADDET = Path(__file__).resolve().parents[1] / "data" / "external" / "raddet" / "RadDet40k128HW009Tv2"


def test_jet_colours_decode_to_their_values():
    values = np.linspace(0, 1, 256).reshape(16, 16)
    rgb = rd.jet_lut()[np.round(values * 255).astype(int)].round().astype(np.uint8)
    # jet has 252 distinct 8-bit colours for 256 values: a few neighbours share
    # a colour, so decoding can be a few steps off there.
    assert np.abs(rd.decode_jet(rgb) - values).max() <= 3 / 255


def test_yolo_box_is_flipped_and_stretched():
    # A box in RadDet's top-left quarter (y down = high frequency) is, on our
    # image, at the left in time and the top quarter of rows (high frequency).
    x0, y0, x1, y1 = rd.yolo_to_pixels(0.25, 0.25, 0.5, 0.5)
    assert (x0, x1) == (0, scenes.IMG_W / 2)
    assert (y0, y1) == (scenes.IMG_H / 2, scenes.IMG_H)


def test_noise_is_set_to_our_scale():
    rng = np.random.default_rng(0)
    v = 0.8 + 0.01 * rng.standard_normal((128, 128))
    img, med, spread = rd.to_our_scale(v)
    assert 0 <= img.min() and img.max() <= 1
    assert np.median(img[img > 0]) / 0.6745 == pytest.approx(rd.OUR_NOISE_SPREAD, rel=0.1)


@pytest.mark.skipif(not (RADDET / "images").is_dir(), reason="RadDet not downloaded")
def test_real_frame_converts_into_our_format():
    for stem in ("000000000000", "000000000002", "000000000004"):
        img, boxes, meta = rd.convert_frame(RADDET, "train", stem)
        assert img.shape == (scenes.IMG_H, scenes.IMG_W) and 0 <= img.min() and img.max() <= 1
        assert len(boxes) <= rd.RADDET_MAX and all(len(v) == len(boxes) for v in meta.values())
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert set(meta["cls"]) <= set(rd.CLASS_NAMES)
