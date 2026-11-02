"""
Make detector scenes from TorchSig (github.com/TorchDSP/torchsig, v2.2), in
the same format as scripts/make_scenes.py, so the detector can train and be
scored on emitters our own generators did not draw.

    python scripts/make_torchsig_scenes.py --n-train 10000 --n-val 1000 --n-test 1000

Why a second source: our scenes come only from our own generators, so the
detector could learn their habits instead of "find any emitter". TorchSig is
another team's simulator (53 comms classes: PSK, QAM, FSK, OFDM, AM/FM, LoRa,
ADS-B, Bluetooth ...), so it is an independent check. The detector is single
class, so TorchSig's class names do not need to match ours.

Each scene is made to match ours exactly (src/detect/scenes.py):
  * TorchSig runs at our receiver: 3.2 MS/s, 32,768 samples (10.24 ms),
    1-6 emitters (10% of scenes empty), bursts of at least MIN_BURST samples,
    bandwidth up to 1.4 MHz.
  * Rescaled to unit noise power (TorchSig's noise sits near -13 dB), then our
    receiver impairments (impairments.apply with scenes.SCENE_SKIP), then our
    spectrogram() -- so the image is built by the same code as ours.
  * Boxes: TorchSig's own time and frequency labels, converted with
    scenes.to_pixels. Measured on 150 single-emitter scenes at 20 dB they hold
    99.4% of the emitter's visible energy (our 99%-power rule held 98.4% and
    fails on chirps, whose power spreads over the whole band).
  * visibility / ignore: the same rule as ours, from each emitter on its own.

TorchSig's snr_db is measured against noise in the emitter's own bandwidth,
ours against noise in the whole band, so the two snr_db columns are not on
the same scale. visibility is computed the same way for both.

Saved per split in --out (default data/detect_torchsig/, gitignored), keys as
in make_scenes.py; group is "torchsig" and cls is TorchSig's class name.
Seeds: --seed is the train seed, val and test use +1 / +2. The default 1300
keeps these apart from our scenes (1200-1202) inside Chua's block.
"""
import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.data import impairments  # noqa: E402
from src.detect import scenes  # noqa: E402

M = scenes.MAX_EMITTERS
N = scenes.SCENE_SAMPLES
FS = scenes.FS
EDGE_HZ = scenes.BAND_FRACTION * FS / 2

METADATA = {
    "num_iq_samples_dataset": N,
    "num_signals_min": 1,
    "num_signals_max": M,
    "fft_size": scenes.N_FFT,
    "fft_stride": scenes.N_FFT,
    "sample_rate": FS,
    "noise_power_db": 0.0,
    "snr_db_min": scenes.EMITTER_SNR_DB[0],
    "snr_db_max": scenes.EMITTER_SNR_DB[1],
    "cochannel_overlap_probability": 0.2,
    "signal_duration_in_samples_min": scenes.MIN_BURST,
    "signal_duration_in_samples_max": N,
    "bandwidth_min": 20_000,
    "bandwidth_max": 1_400_000,
    "signal_center_freq_min": -EDGE_HZ,
    "signal_center_freq_max": EDGE_HZ,
    "frequency_min": -FS / 2,
    "frequency_max": FS / 2 - 1,
}


def scene_seed(split_seed, i):
    return int(np.random.SeedSequence([split_seed, i]).generate_state(1, dtype=np.uint64)[0] >> 1)


def make_scene(seed, impair=True):
    """One TorchSig scene in our format: (iq, image, boxes (k, 4), meta)."""
    from torchsig.datasets.datasets import TorchSigIterableDataset

    rng = np.random.default_rng(seed)
    empty = rng.random() < scenes.EMPTY_FRACTION
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = next(TorchSigIterableDataset(metadata=METADATA, seed=seed))
    comps = [] if empty else list(sig.component_signals)

    placed = np.zeros(N, dtype=complex)
    for c in sig.component_signals:
        a = int(c["start_in_samples"])
        placed[a:a + len(c.data)] += c.data[:N - a]
    noise = np.asarray(sig.data) - placed
    scale = 1 / np.sqrt(np.mean(np.abs(noise) ** 2))

    scene = noise * scale
    boxes = []
    meta = {"group": [], "cls": [], "snr_db": [], "visibility": [], "ignore": []}
    for c in comps:
        a = int(c["start_in_samples"])
        alone = np.zeros(N, dtype=complex)
        alone[a:a + len(c.data)] = c.data[:N - a] * scale
        # TorchSig can centre a wide emitter so close to the band edge that it
        # wraps round to the other side of the image, outside its box. Slide
        # it back inside; a shift of sampled data is circular in frequency, so
        # this also rejoins the part that had wrapped.
        lo, hi = c.lower_freq, c.upper_freq
        shift = 0.0
        if hi - lo <= 2 * EDGE_HZ:
            if lo < -EDGE_HZ:
                shift = -EDGE_HZ - lo
            elif hi > EDGE_HZ:
                shift = EDGE_HZ - hi
        if shift:
            alone = impairments.frequency_shift(alone, shift)
        scene += alone
        boxes.append(scenes.to_pixels(a, min(a + len(c.data), N) - 1, lo + shift, hi + shift))
        vis = scenes.visibility(scenes.pixel_power(alone))
        for k, v in zip(meta, ("torchsig", c["class_name"], c["snr_db"], vis, vis < scenes.MIN_VISIBILITY)):
            meta[k].append(v)

    if impair:
        scene, _ = impairments.apply(scene, None, rng, bandwidth_hz=FS, skip=scenes.SCENE_SKIP)
    iq = np.asarray(scene, dtype=np.complex64)
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return iq, scenes.spectrogram(iq), boxes, meta


def one(seed):
    _, img, boxes, meta = make_scene(seed)
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
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "detect_torchsig"))
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1300)
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
