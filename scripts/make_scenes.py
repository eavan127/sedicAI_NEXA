"""
Make the detector's scene dataset (src/detect/scenes.py): one .npz per split
with the spectrogram images and their boxes. Fully offline, uses every CPU core.

    python scripts/make_scenes.py --n-train 4000 --n-val 500 --n-test 500

Saved per split in --out (default data/detect/, gitignored):
  images      uint8 (N, IMG_H, IMG_W)      spectrogram, 0-255 = 0-40 dB above the floor
  boxes       float32 (N, MAX, 4)          [x0, y0, x1, y1] pixels, unused rows zero
  n_boxes     int16 (N,)
  ignore      bool (N, MAX)                too faint to see (scenes.MIN_VISIBILITY)
  snr_db, visibility   float32 (N, MAX)
  group, cls  str (N, MAX)
  seed        int64 (N,)                    scenes.make_scene(default_rng(seed)) rebuilds it

Seeds: --seed is the train seed, val and test use +1 / +2. The default 2200
keeps the detector apart from radar (2000) and spread (2100) inside Jessy's block.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.detect import scenes  # noqa: E402

M = scenes.MAX_EMITTERS


def scene_seed(split_seed, i):
    return int(np.random.SeedSequence([split_seed, i]).generate_state(1, dtype=np.uint64)[0] >> 1)


def one(seed):
    _, img, boxes, meta = scenes.make_scene(np.random.default_rng(seed))
    k = len(boxes)
    row = {
        "images": np.round(img * 255).astype(np.uint8),
        "boxes": np.zeros((M, 4), np.float32), "n_boxes": k,
        "ignore": np.zeros(M, bool), "snr_db": np.zeros(M, np.float32),
        "visibility": np.zeros(M, np.float32),
        "group": np.full(M, "", "<U8"), "cls": np.full(M, "", "<U20"), "seed": seed,
    }
    row["boxes"][:k] = boxes
    for key in ("ignore", "snr_db", "visibility", "group", "cls"):
        row[key][:k] = meta[key]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "detect"))
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--seed", type=int, default=2200)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, (split, n) in enumerate((("train", args.n_train), ("val", args.n_val), ("test", args.n_test))):
        if n <= 0:
            continue
        t0 = time.time()
        seeds = [scene_seed(args.seed + k, i) for i in range(n)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one, seeds, chunksize=16))
        data = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
        np.savez_compressed(out / f"{split}.npz", **data)
        kept = data["n_boxes"].sum()
        print(f"{split}: {n} scenes, {kept} emitters ({data['ignore'].sum()} too faint), "
              f"{time.time() - t0:.0f}s -> {out / (split + '.npz')}")


if __name__ == "__main__":
    main()
