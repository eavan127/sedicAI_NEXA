"""
Merge every teammate's shards into the combined train / val / test set
(README "Shared data workflow", step 2; WEEK1_PLAN "combined/").

For each split it:
  1. reads every manifest in --manifests (except the mix/ ones this script
     writes itself) and loads its shard from --shards/<group>/, checking the
     class-list version and SHA-256 -- a half-synced or edited shard stops
     the merge before anything is trained on it;
  2. fills the NaN SNR of NOISE_FLOOR windows (see merge.fill_noise_floor_snr);
  3. builds the mixtures -- signal + jammer, signal + signal, 3-signal scenes
     -- from that split's own high-SNR windows only, so no window is shared
     across splits, and writes them as mix/ shards with manifests like any
     other group;
  4. writes <out>/<split>/X.npy (float16 (N, 2, 512) by default, normalised
     like preprocess_window), y.npy (uint8 (N, 60) multi-hot) and
     snr_labels.npy (float32 (N,)), rows shuffled. The trainer casts X and y
     to float32 per batch.

<out>/combined.json records the combined version (fingerprint of every input
and mix shard's SHA-256), the git commit and per-class counts per split.
Record that version in every training run log.

Quick check first (a few windows per shard, separate output folders):
    python scripts/merge_shards.py --limit-per-shard 200 --out data/interim/combined_try --mix-out data/interim/mix_try --mix-manifests data/interim/mix_try_manifests
Full run:
    python scripts/merge_shards.py
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, DATASET_CLASSES, REPO_ROOT  # noqa: E402
from src.data import merge  # noqa: E402
from src.data.shards import (MANIFEST_DIR, SPLITS, ShardWriter, _git_commit,  # noqa: E402
                             class_list_version, load_shard)

MIX_GROUP = "mix"
SEED_BLOCK = (4000, 4999)  # Eavan's block; fskpsk uses 4000-4002
CHUNK = 10_000
NOISE_IDX = DATASET_CLASSES.index("NOISE_FLOOR")


def window_seed(split_seed, i):
    """Deterministic per-mixture seed, non-negative so it fits int64."""
    return int(np.random.SeedSequence([split_seed, i]).generate_state(1, dtype=np.uint64)[0] >> 1)


def read_manifests(folder):
    manifests = [json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))]
    return [m for m in manifests if m["group"] != MIX_GROUP]


def remove_old_mix(mix_out, mix_manifests):
    """A rerun may write fewer mix parts than last time; a stale part left
    behind would be picked up as data. Only this script writes mix_* files."""
    for folder, pattern in ((mix_out, "mix_*.npz"), (mix_manifests, "mix_*.json")):
        for p in Path(folder).glob(pattern):
            p.unlink()


class SplitWriter:
    """Writes rows into the split's memory-mapped arrays at shuffled
    positions, so the whole set never has to fit in RAM."""

    def __init__(self, out_dir, n_rows, rng, x_dtype):
        out_dir.mkdir(parents=True, exist_ok=True)
        L = CFG["signal"]["window_len"]
        self.X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=x_dtype, shape=(n_rows, 2, L))
        self.y = np.lib.format.open_memmap(out_dir / "y.npy", mode="w+", dtype=np.uint8,
                                           shape=(n_rows, len(DATASET_CLASSES)))
        self.snr = np.zeros(n_rows, dtype=np.float32)
        self.out_dir, self.perm, self.next = out_dir, rng.permutation(n_rows), 0
        self.class_counts, self.n_rows = Counter(), n_rows

    def write(self, iq, y, snr_db):
        pos = self.perm[self.next:self.next + len(iq)]
        self.next += len(iq)
        order = np.argsort(pos)
        pos = pos[order]
        self.X[pos] = merge.normalize_batch(iq[order])
        self.y[pos] = y[order]
        self.snr[pos] = snr_db[order]
        for c, n in zip(DATASET_CLASSES, y.sum(axis=0)):
            if n:
                self.class_counts[c] += int(n)

    def close(self):
        if self.next != self.n_rows:
            raise RuntimeError(f"wrote {self.next} rows into a {self.n_rows}-row split")
        self.X.flush()
        self.y.flush()
        np.save(self.out_dir / "snr_labels.npy", self.snr)
        del self.X, self.y


def merge_split(split, manifests, args, split_seed):
    limit = args.limit_per_shard
    n_in = sum(min(m["n_windows"], limit or m["n_windows"]) for m in manifests)
    n_mix = int(round(args.mix_fraction * n_in))
    rng = np.random.default_rng(split_seed)
    out = SplitWriter(Path(args.out) / split, n_in + n_mix, rng, args.x_dtype)

    pool = defaultdict(list)
    for m in manifests:
        data = load_shard(Path(args.shards) / m["group"] / m["shard"], m)
        if limit:
            # Shards are written class by class, SNR by SNR: evenly spaced
            # rows keep every class and SNR in a quick check, a prefix would not.
            rows = np.linspace(0, len(data["iq"]) - 1, min(limit, len(data["iq"]))).astype(int)
            data = {k: v[rows] for k, v in data.items()}
        snr = merge.fill_noise_floor_snr(data["snr_db"], data["y"], np.random.default_rng([m["seed"], m["part"]]))
        for s in range(0, len(snr), CHUNK):
            out.write(data["iq"][s:s + CHUNK], data["y"][s:s + CHUNK], snr[s:s + CHUNK])
        # Mixture components: single-label, high-SNR, never NOISE_FLOOR.
        usable = (data["y"].sum(axis=1) == 1) & (snr >= args.pool_min_snr) & (data["y"][:, NOISE_IDX] == 0)
        pool_iq, labels = data["iq"][usable], data["y"][usable].argmax(axis=1)
        for idx in np.unique(labels):
            pool[DATASET_CLASSES[idx]].append(pool_iq[labels == idx])
        print(f"  {m['shard']}: {len(snr)} windows")
    pool = {c: np.concatenate(parts) for c, parts in pool.items()}

    emitters = [c for c in merge.EMITTERS if c in pool]
    jammers = [c for c in merge.JAMMERS if c in pool]
    if n_mix and len(emitters) < 3:
        raise RuntimeError(f"{split}: only {len(emitters)} classes have windows at >= {args.pool_min_snr} dB; "
                           "mixtures need at least 3 (lower --pool-min-snr or raise --limit-per-shard)")
    bins = CFG["dataset_snr_bins_db"]
    notes = {"script": "scripts/merge_shards.py", "pool_min_snr_db": args.pool_min_snr,
             "mix_weights": dict(zip(merge.MIX_KINDS, args.mix_weights)),
             "inputs_version": merge.combined_version(manifests)}
    with ShardWriter(args.mix_out, MIX_GROUP, split, split_seed, manifest_dir=args.mix_manifests,
                     notes=notes) as writer:
        buf_iq, buf_y, buf_snr = [], [], []
        for i in range(n_mix):
            wseed = window_seed(split_seed, i)
            r = np.random.default_rng(wseed)
            kind = merge.MIX_KINDS[r.choice(len(merge.MIX_KINDS), p=args.mix_weights)]
            classes = merge.pick_mixture_classes(r, kind, emitters, jammers)
            comps = [(c, pool[c][r.integers(len(pool[c]))]) for c in classes]
            snr_db = float(r.choice(bins))
            iq = merge.mix_windows(comps, snr_db, r)
            writer.add(iq, classes, snr_db, "mix", wseed)
            buf_iq.append(iq)
            buf_y.append(np.isin(DATASET_CLASSES, classes).astype(np.uint8))
            buf_snr.append(snr_db)
            if len(buf_iq) == CHUNK or i == n_mix - 1:
                out.write(np.stack(buf_iq), np.stack(buf_y), np.asarray(buf_snr, dtype=np.float32))
                buf_iq, buf_y, buf_snr = [], [], []
    out.close()
    print(f"  {split}: {n_in} shard windows + {n_mix} mixtures = {n_in + n_mix}")
    return out, writer.written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", default=str(MANIFEST_DIR), help="folder of the teammates' manifest JSONs")
    p.add_argument("--shards", default=str(REPO_ROOT / "data" / "shards"), help="folder holding <group>/<shard>.npz")
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "combined"))
    p.add_argument("--mix-out", default=str(REPO_ROOT / "data" / "shards" / MIX_GROUP))
    p.add_argument("--mix-manifests", default=str(MANIFEST_DIR), help="where the mix/ manifests go")
    p.add_argument("--mix-fraction", type=float, default=0.25,
                   help="mixtures per split as a fraction of its shard windows (WEEK1_PLAN: ~250k on ~960k)")
    p.add_argument("--mix-weights", type=float, nargs=3, default=[0.4, 0.4, 0.2],
                   metavar=("SIG_JAM", "SIG_SIG", "THREE"))
    p.add_argument("--pool-min-snr", type=float, default=16,
                   help="only windows at or above this SNR are used as mixture components")
    p.add_argument("--seed", type=int, default=4100, help="train seed for mixtures; val/test use +1/+2")
    p.add_argument("--x-dtype", choices=["float16", "float32"], default="float16",
                   help="float16 halves X (3.9 -> 2.0 GB for train); its rounding is ~66 dB below "
                        "the signal, far under the noise of even a +20 dB window")
    p.add_argument("--limit-per-shard", type=int, default=0, help="quick check: first N windows of each shard")
    args = p.parse_args()

    if not SEED_BLOCK[0] <= args.seed <= SEED_BLOCK[1] - 2:
        p.error(f"--seed must be inside Eavan's block {SEED_BLOCK}")
    if abs(sum(args.mix_weights) - 1) > 1e-6:
        p.error("--mix-weights must add up to 1")

    manifests = read_manifests(args.manifests)
    missing = [str(Path(args.shards) / m["group"] / m["shard"]) for m in manifests
               if not (Path(args.shards) / m["group"] / m["shard"]).exists()]
    if missing:
        sys.exit("Missing shards (download them first):\n  " + "\n  ".join(missing))
    print(f"{len(manifests)} shards from groups {sorted({m['group'] for m in manifests})}")

    remove_old_mix(args.mix_out, args.mix_manifests)
    t0 = time.time()
    summary, mix_manifests = {}, []
    for k, split in enumerate(SPLITS):
        print(f"{split}:")
        out, written = merge_split(split, [m for m in manifests if m["split"] == split], args, args.seed + k)
        mix_manifests += written
        summary[split] = {"n_windows": out.n_rows, "class_counts": dict(out.class_counts)}

    commit, dirty = _git_commit()
    info = {
        "combined_version": merge.combined_version(manifests + mix_manifests),
        "class_list_version": class_list_version(),
        "classes": DATASET_CLASSES,
        "git_commit": commit,
        "git_dirty": dirty,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "splits": summary,
        "shards": [{"shard": m["shard"], "sha256": m["sha256"]} for m in manifests + mix_manifests],
    }
    Path(args.out, "combined.json").write_text(json.dumps(info, indent=2))
    print(f"Done in {time.time() - t0:.0f}s. combined_version {info['combined_version']} -> {args.out}")


if __name__ == "__main__":
    main()
