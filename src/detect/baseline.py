"""
Classical energy detector: the yardstick the learned detector has to beat.

No training. Threshold the spectrogram image a fixed number of dB above the
noise floor, close small gaps (so the pulses of one radar and the hops of one
hopper join up), drop specks, and box each connected region. Score = the
region's brightest pixel.

Its known weaknesses are exactly what the learned detector is for: emitters
that touch or overlap merge into one box, faint emitters fall under the
threshold, and noise peaks above it become false alarms.
"""
import numpy as np
from scipy import ndimage

from src.detect.scenes import RANGE_DB


def detect(image, threshold_db=8.0, close=(3, 7), min_pixels=6):
    """image (H, W) in 0..1 -> (boxes (k, 4) [x0, y0, x1, y1] pixels, scores (k,))."""
    mask = image >= threshold_db / RANGE_DB
    mask = ndimage.binary_closing(mask, structure=np.ones(close, dtype=bool))
    labels, n = ndimage.label(mask)
    boxes, scores = [], []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[sl] == k
        if region.sum() < min_pixels:
            continue
        rows, cols = sl
        boxes.append([cols.start, rows.start, cols.stop, rows.stop])
        scores.append(float(image[sl][region].max()))
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(scores, dtype=np.float32)
