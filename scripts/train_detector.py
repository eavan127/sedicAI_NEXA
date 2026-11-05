"""
Train the YOLO-style emitter detector (src/detect/model.py) on the scenes
from scripts/make_scenes.py. CPU only, fully offline.

    python scripts/train_detector.py --epochs 30

Augmentation: random time reversal and frequency mirroring. Both keep every
box an emitter box (a reversed chirp is still a chirp), so they double the
variety of shapes for free.

Writes --out/emitter_detector.pt (the best epoch by validation AP@0.5, with
its settings) and --out/history.json.

Uses a CUDA GPU when there is one (Colab / Kaggle), else every CPU core;
--device cpu forces the CPU. The saved weights are CPU tensors either way, so
the file loads on any machine.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.detect import model as det  # noqa: E402


def load(path):
    d = np.load(path)
    imgs = torch.from_numpy(d["images"]).float().div_(255).unsqueeze(1)
    boxes = [d["boxes"][i, :d["n_boxes"][i]] for i in range(len(imgs))]
    ignore = [d["ignore"][i, :d["n_boxes"][i]] for i in range(len(imgs))]
    return imgs, boxes, ignore


def augment(img, boxes, rng):
    h, w = img.shape[-2:]
    boxes = boxes.copy()
    if rng.random() < 0.5:                                   # reverse time
        img = img.flip(-1)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if rng.random() < 0.5:                                   # mirror frequency
        img = img.flip(-2)
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    return img, boxes


def evaluate(net, imgs, boxes, ignore, score_thr=0.05, batch=64, device="cpu"):
    net.eval()
    tps, scores, n_gt = [], [], 0
    with torch.no_grad():
        for s in range(0, len(imgs), batch):
            preds = det.decode(net(imgs[s:s + batch].to(device)).cpu(), score_thr=score_thr)
            for (pb, ps), gb, gi in zip(preds, boxes[s:s + batch], ignore[s:s + batch]):
                tp, sc, _ = det.match(pb, ps, gb, gi)
                tps.append(tp)
                scores.append(sc)
                n_gt += int((~np.asarray(gi, bool)).sum())
    return det.average_precision(np.concatenate(tps), np.concatenate(scores), n_gt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "detect"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "detector"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="use only this many training scenes (smoke runs)")
    p.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count())
    rng = np.random.default_rng(args.seed)
    tr_imgs, tr_boxes, tr_ign = load(Path(args.data) / "train.npz")
    va_imgs, va_boxes, va_ign = load(Path(args.data) / "val.npz")
    if args.limit:
        tr_imgs, tr_boxes, tr_ign = tr_imgs[:args.limit], tr_boxes[:args.limit], tr_ign[:args.limit]
    _, _, img_h, img_w = tr_imgs.shape
    gh, gw = img_h // det.STRIDE, img_w // det.STRIDE

    net = det.EmitterDetector(args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tr_imgs) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tr_imgs)} train / {len(va_imgs)} val scenes, "
          f"{sum(x.numel() for x in net.parameters()):,} parameters, "
          + (f"GPU {torch.cuda.get_device_name(device)}" if device.type == "cuda"
             else f"CPU threads {torch.get_num_threads()}"))

    history, best = [], -1.0
    for epoch in range(args.epochs):
        net.train()
        t0, total, parts = time.time(), 0.0, {"obj": 0.0, "box": 0.0}
        order = rng.permutation(len(tr_imgs))
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            batch_imgs, batch_boxes = [], []
            for i in idx:
                img, bx = augment(tr_imgs[i], tr_boxes[i], rng)
                batch_imgs.append(img)
                batch_boxes.append(bx)
            obj, box, weight = det.encode(batch_boxes, [tr_ign[i] for i in idx], gh, gw)
            loss, lp = det.loss(net(torch.stack(batch_imgs).to(device)),
                                obj.to(device), box.to(device), weight.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
            for k in parts:
                parts[k] += lp[k] * len(idx)
        ap = evaluate(net, va_imgs, va_boxes, va_ign, device=device)
        n = len(order)
        history.append({"epoch": epoch + 1, "loss": total / n, "obj": parts["obj"] / n,
                        "box": parts["box"] / n, "val_ap50": ap, "seconds": time.time() - t0})
        flag = ""
        if ap > best:
            best = ap
            torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "width": args.width, "stride": det.STRIDE,
                        "img_h": img_h, "img_w": img_w, "epoch": epoch + 1, "val_ap50": ap},
                       out / "emitter_detector.pt")
            flag = "  <- saved"
        print(f"epoch {epoch + 1:>2}/{args.epochs}  loss {total / n:.4f}  "
              f"(obj {parts['obj'] / n:.4f}, box {parts['box'] / n:.4f})  "
              f"val AP@0.5 {ap:.3f}  {time.time() - t0:.0f}s{flag}", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val AP@0.5 {best:.3f} -> {out / 'emitter_detector.pt'}")


if __name__ == "__main__":
    main()
