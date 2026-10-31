"""
Tests for the emitter detector: the burst generators it draws scenes from,
the scene generator's boxes, the YOLO-style target encoding / decoding,
scoring, and the classical energy detector it is compared against.
"""
import numpy as np
import pytest
import torch

from src.detect import baseline, model as det, scenes
from src.generators import digital, radar, spread


# --- bursts --------------------------------------------------------------------

@pytest.mark.parametrize("gen", [radar, spread, digital])
def test_burst_at_window_length_is_the_clean_window(gen):
    for cls in gen.CLASSES[:4]:
        a, _ = gen.generate_window(cls, None, np.random.default_rng(3))
        b, _ = gen.generate_burst(cls, gen.WINDOW_LEN, np.random.default_rng(3))
        assert np.array_equal(a, b.astype(np.complex64))


@pytest.mark.parametrize("gen", [radar, spread, digital])
def test_long_bursts_have_the_asked_length_and_unit_power(gen):
    """Radar is unit power while on; spread and comms are unit average power,
    the same conventions as their generate_window."""
    for cls in gen.CLASSES:
        sig, _ = gen.generate_burst(cls, 20_000, np.random.default_rng(1))
        assert sig.shape == (20_000,) and np.isfinite(sig).all()
        mag_sq = np.abs(sig) ** 2
        if gen is radar:
            mag_sq = mag_sq[mag_sq > 0.01 * mag_sq.max()]
        assert mag_sq.mean() == pytest.approx(1.0, rel=0.02)


# --- scenes ----------------------------------------------------------------------

def test_scene_shapes_and_ranges():
    iq, img, boxes, meta = scenes.make_scene(np.random.default_rng(0))
    assert iq.shape == (scenes.SCENE_SAMPLES,) and iq.dtype == np.complex64
    assert img.shape == (scenes.IMG_H, scenes.IMG_W) and 0 <= img.min() and img.max() <= 1
    assert boxes.shape[1] == 4 and len(boxes) <= scenes.MAX_EMITTERS
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
    assert (boxes >= 0).all() and (boxes[:, 2] <= scenes.IMG_W).all() and (boxes[:, 3] <= scenes.IMG_H).all()
    assert all(len(v) == len(boxes) for v in meta.values())


def test_same_seed_same_scene():
    a = scenes.make_scene(np.random.default_rng(5))
    b = scenes.make_scene(np.random.default_rng(5))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_empty_scene_has_no_boxes():
    _, img, boxes, _ = scenes.make_scene(np.random.default_rng(1), n_emitters=0)
    assert len(boxes) == 0 and img.mean() < 0.1


@pytest.mark.parametrize("group", scenes.GROUP_NAMES)
def test_box_holds_the_emitter_energy(group):
    """A single strong emitter, no receiver effects: nearly all the image's
    excess energy lies inside its box."""
    for seed in range(5):
        _, img, boxes, meta = scenes.make_scene(np.random.default_rng(seed), n_emitters=1,
                                                snr_db=20, impair=False, groups=(group,))
        x0, y0, x1, y1 = boxes[0]
        excess = np.clip(img - 0.25, 0, None)             # > 10 dB above the floor
        inside = excess[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))].sum()
        assert inside >= 0.9 * excess.sum(), meta["cls"]


def test_pixel_conversion_round_trip():
    b = scenes.to_pixels(1280, 2559, -400e3, 200e3)
    assert b[0] == pytest.approx(10) and b[2] == pytest.approx(20)
    assert b[1] == pytest.approx(48 - scenes.FREQ_PAD_PX) and b[3] == pytest.approx(72 + scenes.FREQ_PAD_PX)


def test_too_faint_is_flagged():
    _, _, _, meta = scenes.make_scene(np.random.default_rng(2), n_emitters=1, snr_db=-60)
    assert meta["ignore"] == [True]
    _, _, _, meta = scenes.make_scene(np.random.default_rng(2), n_emitters=1, snr_db=20)
    assert meta["ignore"] == [False]


# --- detector: encode / decode / scoring ------------------------------------------

def test_iou_and_nms():
    a = np.array([[0, 0, 10, 10]])
    assert det.iou(a, [[0, 0, 10, 10]])[0, 0] == pytest.approx(1)
    assert det.iou(a, [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)
    assert det.iou(a, [[20, 20, 30, 30]])[0, 0] == 0
    keep = det.nms(np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 0, 60, 10]]), [0.9, 0.8, 0.7])
    assert list(keep) == [0, 2]


def _perfect_logits(boxes, gh=16, gw=32):
    obj, box, _ = det.encode([boxes], [np.zeros(len(boxes), bool)], gh, gw)
    logit = lambda p: torch.log(p.clamp(1e-6, 1 - 1e-6) / (1 - p.clamp(1e-6, 1 - 1e-6)))
    pred = torch.zeros(1, 5, gh, gw)
    pred[0, 0] = torch.where(obj[0] > 0, 10.0, -10.0)
    pred[0, 1], pred[0, 2] = logit(box[0, 0]), logit(box[0, 1])
    pred[0, 3], pred[0, 4] = box[0, 2], box[0, 3]
    return pred


def test_encode_then_decode_recovers_the_boxes():
    boxes = np.array([[10.0, 20.0, 200.0, 30.0], [100.0, 60.0, 130.0, 120.0], [3.5, 1.0, 9.0, 4.0]])
    (got, scores), = det.decode(_perfect_logits(boxes), score_thr=0.5)
    order = np.argsort(got[:, 0])
    assert np.allclose(got[order], boxes[np.argsort(boxes[:, 0])], atol=1e-3)


def test_ignored_boxes_switch_off_their_cells():
    obj, _, weight = det.encode([np.array([[0, 0, 64, 32]])], [np.array([True])], 16, 32)
    assert obj.sum() == 0 and weight[0, :4, :8].sum() == 0 and weight[0, 8:, 16:].min() == 1


def test_match_counts_hits_misses_and_ignores():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 0, 110, 10]])
    ign = np.array([False, False, True])
    pred = np.array([[0, 0, 10, 10], [200, 0, 210, 10], [100, 0, 110, 10]])
    tp, sc, found = det.match(pred, [0.9, 0.8, 0.7], gt, ign)
    assert list(tp) == [True, False]           # the hit and the false alarm; the ignored one is dropped
    assert list(found) == [True, False, False]


def test_average_precision_of_perfect_and_useless_detectors():
    assert det.average_precision([True, True], [0.9, 0.8], 2) == pytest.approx(1.0)
    assert det.average_precision([False, False], [0.9, 0.8], 2) == pytest.approx(0.0)


def test_model_output_shape():
    net = det.EmitterDetector(width=8)
    assert net(torch.zeros(2, 1, scenes.IMG_H, scenes.IMG_W)).shape == (2, 5, 16, 32)


def test_model_learns_a_tiny_batch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    imgs, boxes = [], []
    for _ in range(4):
        _, img, b, _ = scenes.make_scene(rng, n_emitters=2, snr_db=15)
        imgs.append(torch.from_numpy(img)[None])
        boxes.append(b)
    x = torch.stack(imgs)
    obj, box, weight = det.encode(boxes, [np.zeros(len(b), bool) for b in boxes], 16, 32)
    net = det.EmitterDetector(width=16)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss, _ = det.loss(net(x), obj, box, weight)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.3 * first


# --- energy detector ----------------------------------------------------------------

def test_energy_detector_boxes_a_bright_patch():
    img = np.zeros((128, 256), dtype=np.float32)
    img[40:60, 100:180] = 0.6
    boxes, scores = baseline.detect(img)
    assert len(boxes) == 1 and np.allclose(boxes[0], [100, 40, 180, 60])
    assert baseline.detect(np.zeros((128, 256), dtype=np.float32))[0].shape == (0, 4)
