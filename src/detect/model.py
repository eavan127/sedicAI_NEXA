"""
YOLO-style emitter detector: one pass over a spectrogram image, one box per
emitter. Plain PyTorch, no detection library, so it trains and runs fully
offline on a CPU.

How it works (the YOLO idea, single class):
  * A small CNN turns the (1, IMG_H, IMG_W) image into a grid of cells, one
    per STRIDE x STRIDE pixels (16 x 32 cells for a 128 x 256 image).
  * Each cell predicts 5 numbers: objectness ("an emitter's centre is in this
    cell"), the centre's position inside the cell (tx, ty), and the box's
    width and height (tw, th, in log units of STRIDE).
  * Training: the cell holding a box's centre is the positive for that box;
    every other cell should say "nothing here". Cells under an `ignore` box
    (too faint to see, src/detect/scenes.py) are left out of the objectness
    loss.
  * Inference: keep cells whose objectness clears a threshold, turn their
    numbers into boxes, and drop duplicates with non-maximum suppression.

Boxes are [x0, y0, x1, y1] in image pixels everywhere in this module.

Receptive field: emitter boxes can span the whole 256-column image, so a
centre cell must see far along time. Dilated convolutions at the coarsest
scale give each cell a view of roughly +/-120 columns without pooling away the
resolution the box edges need.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 8


def _conv(c_in, c_out, stride=1, dilation=1):
    pad = dilation
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(c_out), nn.SiLU())


class EmitterDetector(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _conv(1, w // 2), _conv(w // 2, w, stride=2),            # 1/2
            _conv(w, w), _conv(w, 2 * w, stride=2),                  # 1/4
            _conv(2 * w, 2 * w), _conv(2 * w, 4 * w, stride=2),      # 1/8
            _conv(4 * w, 4 * w, dilation=2), _conv(4 * w, 4 * w, dilation=4),
            _conv(4 * w, 4 * w, dilation=8), _conv(4 * w, 4 * w, dilation=1),
        )
        self.head = nn.Conv2d(4 * w, 5, 1)
        nn.init.constant_(self.head.bias[0], -4.0)     # start out predicting "nothing"

    def forward(self, x):
        """(B, 1, H, W) -> (B, 5, H / STRIDE, W / STRIDE): obj, tx, ty, tw, th."""
        return self.head(self.features(x))


# --- targets ------------------------------------------------------------------

def encode(boxes_per_image, ignore_per_image, grid_h, grid_w):
    """Training targets for a batch.

    Returns obj (B, gh, gw) 0/1, box (B, 4, gh, gw) [tx, ty, tw, th], and
    obj_weight (B, gh, gw): 1 everywhere except cells under an ignored box.
    When two boxes put their centre in the same cell, the larger one wins.
    """
    b = len(boxes_per_image)
    obj = torch.zeros(b, grid_h, grid_w)
    box = torch.zeros(b, 4, grid_h, grid_w)
    weight = torch.ones(b, grid_h, grid_w)
    for i, (boxes, ignore) in enumerate(zip(boxes_per_image, ignore_per_image)):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        ignore = np.asarray(ignore, dtype=bool).reshape(-1)
        area_at = {}
        for (x0, y0, x1, y1), ign in zip(boxes, ignore):
            if ign:
                c0, c1 = int(x0 // STRIDE), int(np.ceil(x1 / STRIDE))
                r0, r1 = int(y0 // STRIDE), int(np.ceil(y1 / STRIDE))
                weight[i, r0:r1, c0:c1] = 0
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            col = min(int(cx // STRIDE), grid_w - 1)
            row = min(int(cy // STRIDE), grid_h - 1)
            area = (x1 - x0) * (y1 - y0)
            if area_at.get((row, col), -1) >= area:
                continue
            area_at[(row, col)] = area
            obj[i, row, col] = 1
            box[i, :, row, col] = torch.tensor([cx / STRIDE - col, cy / STRIDE - row,
                                                np.log(max(x1 - x0, 1e-3) / STRIDE),
                                                np.log(max(y1 - y0, 1e-3) / STRIDE)])
        weight[i][obj[i] > 0] = 1                 # a real centre always counts
    return obj, box, weight


def loss(pred, obj, box, weight, pos_weight=10.0, box_gain=2.0):
    """Objectness BCE over all non-ignored cells + L1 on the box numbers of
    positive cells. Returns (total, parts dict)."""
    obj_logit = pred[:, 0]
    l_obj = F.binary_cross_entropy_with_logits(
        obj_logit, obj, weight=weight,
        pos_weight=torch.tensor(pos_weight, device=pred.device), reduction="sum") / max(weight.sum().item(), 1)
    pos = obj > 0
    if pos.any():
        p = torch.stack([torch.sigmoid(pred[:, 1]), torch.sigmoid(pred[:, 2]), pred[:, 3], pred[:, 4]], 1)
        l_box = (p - box).abs().sum(1)[pos].mean()
    else:
        l_box = pred.sum() * 0
    return l_obj + box_gain * l_box, {"obj": float(l_obj.detach()), "box": float(l_box.detach())}


# --- decoding -----------------------------------------------------------------

def iou(a, b):
    """IoU between boxes a (n, 4) and b (m, 4) -> (n, m)."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes, scores, iou_thr=0.5):
    """Indices kept by greedy non-maximum suppression, best score first."""
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        overlap = iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap < iou_thr]
    return np.array(keep, dtype=int)


def decode(pred, score_thr=0.3, iou_thr=0.5, img_w=None, img_h=None):
    """Network output (B, 5, gh, gw) -> list of (boxes (k, 4), scores (k,))."""
    pred = pred.detach().cpu()
    _, _, gh, gw = pred.shape
    img_w, img_h = img_w or gw * STRIDE, img_h or gh * STRIDE
    rows, cols = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    out = []
    for p in pred:
        score = torch.sigmoid(p[0])
        cx = (cols + torch.sigmoid(p[1])) * STRIDE
        cy = (rows + torch.sigmoid(p[2])) * STRIDE
        w = torch.exp(p[3].clamp(max=8)) * STRIDE
        h = torch.exp(p[4].clamp(max=8)) * STRIDE
        keep = score >= score_thr
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)[keep].numpy()
        boxes = np.clip(boxes, 0, [img_w, img_h, img_w, img_h])
        s = score[keep].numpy()
        k = nms(boxes, s, iou_thr) if len(s) else np.array([], dtype=int)
        out.append((boxes[k].reshape(-1, 4), s[k]))
    return out


# --- scoring --------------------------------------------------------------------

def match(pred_boxes, scores, gt_boxes, gt_ignore, iou_thr=0.5):
    """Greedy match, best score first. Returns (tp flags for the counted
    predictions, their scores, found flags per ground-truth box). Predictions
    that land on an ignored box are dropped, not counted as false alarms."""
    pred_boxes, scores = np.asarray(pred_boxes).reshape(-1, 4), np.asarray(scores).reshape(-1)
    gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
    gt_ignore = np.asarray(gt_ignore, dtype=bool).reshape(-1)
    found = np.zeros(len(gt_boxes), dtype=bool)
    tp, kept_scores = [], []
    ov = iou(pred_boxes, gt_boxes) if len(gt_boxes) and len(pred_boxes) else np.zeros((len(pred_boxes), 0))
    for i in np.argsort(-scores):
        cand = [j for j in np.argsort(-ov[i]) if ov[i, j] >= iou_thr and not found[j]] if ov.shape[1] else []
        if cand:
            j = cand[0]
            if gt_ignore[j]:
                continue
            found[j] = True
            tp.append(True)
        elif ov.shape[1] and any(gt_ignore[j] and ov[i, j] >= iou_thr for j in range(ov.shape[1])):
            continue
        else:
            tp.append(False)
        kept_scores.append(scores[i])
    return np.array(tp, dtype=bool), np.array(kept_scores), found


def average_precision(tp, scores, n_gt):
    """Area under the precision-recall curve (all-points interpolation)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp, dtype=float)[order]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum(np.diff(recall) * precision[1:]))
