"""Checks the TorchSig scene converter (scripts/make_torchsig_scenes.py)
gives scenes in our format, reproducible from their seed, with every box
inside the band and around its emitter. Skipped when torchsig is not
installed (pip install torchsig==2.2.0)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torchsig")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_torchsig_scenes as ts  # noqa: E402

from src.detect import scenes  # noqa: E402


def test_scene_matches_our_format():
    for seed in range(5):
        iq, img, boxes, meta = ts.make_scene(seed)
        assert iq.shape == (scenes.SCENE_SAMPLES,) and iq.dtype == np.complex64
        assert img.shape == (scenes.IMG_H, scenes.IMG_W) and 0 <= img.min() and img.max() <= 1
        assert boxes.shape[1] == 4 and len(boxes) <= scenes.MAX_EMITTERS
        assert all(len(v) == len(boxes) for v in meta.values())


def test_same_seed_same_scene():
    a, b = ts.make_scene(11), ts.make_scene(11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_boxes_stay_inside_the_band():
    """Wide emitters near the band edge are slid inside, so no box is cut
    off at the top or bottom of the image."""
    for seed in range(40):
        _, _, boxes, _ = ts.make_scene(seed)
        assert ((boxes[:, 1] > 0) & (boxes[:, 3] < scenes.IMG_H)).all()


def test_box_holds_the_emitter_energy(monkeypatch):
    """A single strong emitter, no receiver effects: nearly all the image's
    excess energy lies inside its box (same check as tests/test_detect.py)."""
    monkeypatch.setitem(ts.METADATA, "num_signals_max", 1)
    monkeypatch.setitem(ts.METADATA, "snr_db_min", 20.0)
    monkeypatch.setitem(ts.METADATA, "snr_db_max", 20.0)
    checked = 0
    for seed in range(30):
        _, img, boxes, meta = ts.make_scene(seed, impair=False)
        if not len(boxes):
            continue
        x0, y0, x1, y1 = boxes[0]
        excess = np.clip(img - 0.25, 0, None)             # > 10 dB above the floor
        inside = excess[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))].sum()
        assert inside >= 0.9 * excess.sum(), meta["cls"]
        checked += 1
    assert checked >= 20
