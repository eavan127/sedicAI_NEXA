"""
The whole chain over one capture, one detector chunk (10.24 ms) at a time:

    chunk -> spectrogram -> detector boxes -> cut each box from the capture
          -> classifier scores per window -> tracker -> resolve -> answers

Detectors are callables image (IMG_H, IMG_W) -> (boxes (k, 4) pixels, scores):
NetDetector wraps the trained YOLO-style model (results/detector/
emitter_detector.pt), EnergyDetector the classical threshold baseline.
"""
import numpy as np
import torch

from src.config import CFG
from src.detect import baseline
from src.detect import model as det
from src.detect import scenes
from src.pipeline import extract
from src.pipeline.resolve import resolve
from src.pipeline.tracker import Detection, Tracker, top_classes

FS = CFG["signal"]["fs"]
CHUNK = scenes.SCENE_SAMPLES


class NetDetector:
    def __init__(self, path, score_thr=0.3, iou_thr=0.5, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = det.EmitterDetector(width=ck["width"]).to(device).eval()
        self.net.load_state_dict(ck["state_dict"])
        self.score_thr, self.iou_thr, self.device = score_thr, iou_thr, device

    @torch.no_grad()
    def __call__(self, image):
        x = torch.from_numpy(np.asarray(image, dtype=np.float32))[None, None].to(self.device)
        boxes, scores = det.decode(self.net(x), self.score_thr, self.iou_thr,
                                   img_w=image.shape[1], img_h=image.shape[0])[0]
        return boxes, scores


class EnergyDetector:
    def __init__(self, threshold_db=8.0):
        self.threshold_db = threshold_db

    def __call__(self, image):
        return baseline.detect(image, threshold_db=self.threshold_db)


def box_to_physical(box, chunk_start):
    """[x0, y0, x1, y1] pixels of a chunk -> (t0, t1) samples of the capture,
    (f_lo, f_hi) Hz."""
    x0, y0, x1, y1 = (float(v) for v in box)
    t0 = chunk_start + int(np.floor(x0 * scenes.N_FFT))
    t1 = chunk_start + int(np.ceil(x1 * scenes.N_FFT))
    f_lo = y0 / scenes.IMG_H * FS - FS / 2
    f_hi = y1 / scenes.IMG_H * FS - FS / 2
    return t0, t1, f_lo, f_hi


def process_capture(iq, detector, classifier, rng=None, tracker=None, cw_min_s=None):
    """Run the chain over complex `iq`. Returns (answers, tracker): one dict
    per track, in order of first appearance."""
    rng = rng or np.random.default_rng(0)
    tracker = tracker or Tracker()
    iq = np.asarray(iq)
    for k, s in enumerate(range(0, len(iq), CHUNK)):
        chunk = iq[s:s + CHUNK]
        nv = extract.noise_variance(chunk) if len(chunk) >= scenes.N_FFT else 1.0
        if len(chunk) < CHUNK:      # pad the last chunk with noise so the image keeps its size
            pad = (rng.standard_normal(CHUNK - len(chunk)) + 1j * rng.standard_normal(CHUNK - len(chunk)))
            chunk = np.concatenate([chunk, pad * np.sqrt(nv / 2)])
        boxes, scores = detector(scenes.spectrogram(chunk))
        dets = []
        for box, score in zip(boxes, scores):
            t0, t1, f_lo, f_hi = box_to_physical(box, s)
            t1 = min(t1, len(iq))
            if t1 - t0 <= 0:
                continue
            c = extract.cut(iq, t0, t1, f_lo, f_hi, nv, rng)
            probs = classifier(extract.windows(c))
            dets.append(Detection(c.t0, c.t1, c.f_lo, c.f_hi, float(score), probs, c.features, chunk=k))
        tracker.update(dets)
    tracks = tracker.finish()
    kw = {} if cw_min_s is None else {"cw_min_s": cw_min_s}
    answers = []
    for tr in sorted(tracks, key=lambda t: t.t_first):
        a = resolve(tr, tracks, **kw)
        probs = tr.probs()
        answers.append(dict(
            track=tr.id, answer=a.label, classes=list(a.classes), confidence=round(a.confidence, 3),
            reason=a.reason,
            t_start_ms=round(tr.t_first / FS * 1e3, 3), t_end_ms=round(tr.t_last / FS * 1e3, 3),
            f_lo_khz=round(tr.f_lo / 1e3, 1), f_hi_khz=round(tr.f_hi / 1e3, 1),
            n_boxes=len(tr.dets), n_windows=tr.n_windows,
            top5=[(c, round(p, 3)) for c, p in top_classes(probs)],
            features={k: (round(v, 4) if isinstance(v, float) else v) for k, v in tr.features().items()},
        ))
    return answers, tracker
