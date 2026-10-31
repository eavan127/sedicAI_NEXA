"""
Make the week-1 spread-spectrum shards: FHSS, DSSS and LORA_CSS, every SNR in
dataset_snr_bins_db, train / val / test, written in the shard format with a
manifest per shard (README "Shared data workflow"). Same layout as
scripts/make_radar_shards.py.

All windows come from src/generators/spread.py through the impairment module
(src/data/impairments.py). No external dataset is used yet: the open FHSS and
DSSS recordings do not show their class inside one 160 us window (slow hoppers
dwell longer than a window; GPS sits below the noise), and the EPFL LoRaIQ set
is not downloaded. The 30% external cap is a maximum, so 0% is within the
contract.

Seeds: --seed is the train seed; val and test use --seed + 1 and + 2. The
default 2100 keeps spread apart from radar's 2000-2002 inside Jessy's block.
Every window stores its own seed and can be regenerated alone with
    impairments.generate_impaired(spread, cls, snr, default_rng(seed)).

Quick check first (a few windows, separate output folders):
    python scripts/make_spread_shards.py --n-per-bin 10 --out data/interim/try_spread --manifests data/interim/try_spread_manifests
Full run:
    python scripts/make_spread_shards.py --out "G:/My Drive/NEXA_data/shards/spread"
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASS_TO_IDX  # noqa: E402
from src.data import impairments  # noqa: E402
from src.data.shards import MANIFEST_DIR, SPLITS, ShardWriter  # noqa: E402
from src.generators import spread  # noqa: E402

GROUP = "spread"
CLASSES = list(spread.CLASSES)
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED_BLOCK = (2000, 2999)  # Jessy's block, README "Step by step"
DEFAULT_SEED = 2100        # radar uses 2000-2002


def window_seed(split_seed, cls, snr_db, i):
    """Deterministic per-window seed, non-negative so it fits int64."""
    ss = np.random.SeedSequence([split_seed, DATASET_CLASS_TO_IDX[cls], int(snr_db) + 100, i])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> 1)


def split_counts(n_per_bin):
    counts = {s: int(round(n_per_bin * f)) for s, f in SPLIT_FRACTIONS.items()}
    counts["train"] = n_per_bin - counts["val"] - counts["test"]
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="folder the .npz shards go into")
    p.add_argument("--manifests", default=str(MANIFEST_DIR), help="folder the manifest JSONs go into")
    p.add_argument("--n-per-bin", type=int, default=1000, help="windows per (class, SNR), all splits together")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="train seed; val/test use +1/+2")
    p.add_argument("--classes", nargs="+", default=CLASSES)
    p.add_argument("--no-impair", action="store_true", help="skip the impairment module (debug only)")
    args = p.parse_args()

    if not (SEED_BLOCK[0] <= args.seed and args.seed + 2 <= SEED_BLOCK[1]):
        sys.exit(f"--seed must leave seed..seed+2 inside Jessy's block {SEED_BLOCK}")
    unknown = set(args.classes) - set(CLASSES)
    if unknown:
        sys.exit(f"not a spread class: {sorted(unknown)}")

    snr_bins = CFG["dataset_snr_bins_db"]
    counts = split_counts(args.n_per_bin)
    split_seed = {s: args.seed + k for k, s in enumerate(SPLITS)}
    print(f"{len(args.classes)} classes x {len(snr_bins)} SNRs; per bin {counts}; "
          f"impairments {'OFF' if args.no_impair else 'on'}")

    notes = {"script": "scripts/make_spread_shards.py", "n_per_bin": args.n_per_bin,
             "ext_frac": 0.0, "impairments": not args.no_impair}
    for split in SPLITS:
        t0 = time.time()
        with ShardWriter(args.out, GROUP, split, split_seed[split], manifest_dir=args.manifests,
                         notes=notes) as w:
            for cls in args.classes:
                for snr in snr_bins:
                    for i in range(counts[split]):
                        s = window_seed(split_seed[split], cls, snr, i)
                        rng = np.random.default_rng(s)
                        if args.no_impair:
                            window, _ = spread.generate_window(cls, snr, rng)
                        else:
                            window, _ = impairments.generate_impaired(spread, cls, snr, rng)
                        w.add(window, {cls}, snr, "gen", s)
        total = sum(m["n_windows"] for m in w.written)
        print(f"{split}: {total} windows in {len(w.written)} shard(s), {time.time() - t0:.0f}s")
    print(f"Shards in {args.out}\nManifests in {args.manifests}")


if __name__ == "__main__":
    main()
