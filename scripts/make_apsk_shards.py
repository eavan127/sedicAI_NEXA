"""
Make Eileen's APSK shards: the 4 APSK classes, every SNR
in dataset_snr_bins_db, train / val / test, written in the shard format with a
manifest per shard (README "Shared data workflow"). Same layout as
scripts/make_analog_shards.py.

Per (class, SNR, split) the windows are:
  * up to --ext-frac (default 30%) from RadioML 2018, which has all four, and
  * the rest from src/generators/apsk.py, impaired by src/data/impairments.py.

Impairments: generated windows get the full chain. RadioML windows are stored
as they are, because RadioML already carries its own noise and channel effects,
and the plan applies the module to generators. --no-impair turns the
generated side off.

Seeds: --seed is the train seed; val and test use --seed + 1 and + 2, so the
three splits never share a generated window. Every generated window stores
its own seed, derived from (split seed, class, SNR, index), and can be
regenerated alone with
    impairments.generate_impaired(apsk, cls, snr, default_rng(seed)).
External windows store their row in the source file instead.

Quick check first (a few windows, separate output folders):
    python scripts/make_apsk_shards.py --n-per-bin 10 --out data/interim/try --manifests data/interim/try_manifests
Full run into the shared Drive folder:
    python scripts/make_apsk_shards.py --out "G:/My Drive/NEXA_data/shards/apsk"
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASS_TO_IDX  # noqa: E402
from src.data import external, impairments  # noqa: E402
from src.data.shards import MANIFEST_DIR, SPLITS, ShardWriter  # noqa: E402
from src.generators import apsk  # noqa: E402

GROUP = "apsk"
CLASSES = list(apsk.CLASSES)
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED_BLOCK = (3200, 3299)  # Eileen's block; analog 3000s, multicarrier 3100s


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
    p.add_argument("--ext-frac", type=float, default=0.3, help="max share from external datasets")
    p.add_argument("--seed", type=int, default=SEED_BLOCK[0], help="train seed; val/test use +1/+2")
    p.add_argument("--classes", nargs="+", default=CLASSES)
    p.add_argument("--no-radioml", action="store_true")
    p.add_argument("--no-impair", action="store_true", help="skip the impairment module (debug only)")
    args = p.parse_args()

    if not (SEED_BLOCK[0] <= args.seed and args.seed + 2 <= SEED_BLOCK[1]):
        sys.exit(f"--seed must leave seed..seed+2 inside {SEED_BLOCK}")
    unknown = set(args.classes) - set(CLASSES)
    if unknown:
        sys.exit(f"not an apsk class: {sorted(unknown)}")

    snr_bins = CFG["dataset_snr_bins_db"]
    counts = split_counts(args.n_per_bin)
    ext_counts = {s: int(args.ext_frac * n) for s, n in counts.items()}
    split_seed = {s: args.seed + k for k, s in enumerate(SPLITS)}
    print(f"{len(args.classes)} classes x {len(snr_bins)} SNRs; per bin {counts}, "
          f"external at most {ext_counts}; impairments {'OFF' if args.no_impair else 'on'}")

    ext = {}
    t0 = time.time()
    if not args.no_radioml:
        if external.RADIOML_PATH.exists():
            print("Reading RadioML 2018 ...")
            ext.update({k: (v, "radioml2018") for k, v in external.radioml_windows(
                args.classes, snr_bins, ext_counts, seed=args.seed).items()})
        else:
            print(f"  ! RadioML not found at {external.RADIOML_PATH}; generator only")
    print(f"External windows ready in {time.time() - t0:.0f}s")

    notes = {"script": "scripts/make_apsk_shards.py", "n_per_bin": args.n_per_bin,
             "ext_frac": args.ext_frac, "impairments": not args.no_impair}
    for split in SPLITS:
        t0 = time.time()
        n_ext_total = 0
        with ShardWriter(args.out, GROUP, split, split_seed[split], manifest_dir=args.manifests,
                         notes=notes) as w:
            for cls in args.classes:
                for snr in snr_bins:
                    n = counts[split]
                    got = ext.get((cls, snr, split))
                    n_ext = 0
                    if got is not None:
                        (iq, rows), source = got
                        for window, row in zip(iq[:n], rows[:n]):
                            w.add(window, {cls}, snr, source, row)
                        n_ext = min(n, len(iq))
                        n_ext_total += n_ext
                    for i in range(n - n_ext):
                        s = window_seed(split_seed[split], cls, snr, i)
                        rng = np.random.default_rng(s)
                        if args.no_impair:
                            window, _ = apsk.generate_window(cls, snr, rng)
                        else:
                            window, _ = impairments.generate_impaired(apsk, cls, snr, rng)
                        w.add(window, {cls}, snr, "gen", s)
        total = sum(m["n_windows"] for m in w.written)
        print(f"{split}: {total} windows ({n_ext_total} external) in {len(w.written)} shard(s), "
              f"{time.time() - t0:.0f}s")
    print(f"Shards in {args.out}\nManifests in {args.manifests}")


if __name__ == "__main__":
    main()
