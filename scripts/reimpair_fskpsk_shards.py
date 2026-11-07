"""
Rewrite the existing FSK / PSK / QAM shards with the impairment chain, without
the external datasets (SED-39, 2026-11-07).

make_fskpsk_shards.py used to generate these 18 classes with no impairments,
so every generated window sat exactly at 0 Hz and the 60-class model learned
"centred -> FSK / PSK / QAM" (BPSK right 100% at 0 Hz, 4% at +300 kHz). The
script is fixed, but rerunning it needs RadioML 2018 and HisarMod 2019.1. This
one doesn't: it reads the shards already made and

  * re-makes every "gen" row from its stored seed through
    impairments.generate_impaired(digital, cls, snr, rng(seed)) -- the same
    window make_fskpsk_shards.py now writes for that seed;
  * copies every external row (RadioML / HisarMod) unchanged: the merge places
    and re-noises those itself (merge.harmonize_external, on by default).

Same rows, order, classes, SNRs, seeds and shard names; new SHA-256s and
manifests. Before re-making a row it checks that its seed still reproduces
the stored window (digital.generate_window), so a wrong seed recipe stops the
run instead of silently mislabelling data.

    python scripts/reimpair_fskpsk_shards.py --shards data/shards --manifests data/manifests \\
        --out data/shards_v2/fskpsk --manifests-out data/manifests_v2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_CLASSES  # noqa: E402
from src.data import impairments  # noqa: E402
from src.data.shards import SPLITS, ShardWriter, load_shard  # noqa: E402
from src.generators import digital  # noqa: E402

GROUP = "fskpsk"
CHECK_EVERY = 997          # verify one stored generated window in this many against its seed
CHECK_ATOL = 1e-4          # the stored windows are complex64


def reimpair_split(manifests, shards_dir, out_dir, manifests_out, split):
    stats = dict(gen=0, ext=0, checked=0)
    first = manifests[0]
    notes = {**first.get("notes", {}), "impairments": True,
             "reimpaired_by": "scripts/reimpair_fskpsk_shards.py",
             "reimpaired_from": {m["shard"]: m["sha256"] for m in manifests}}
    with ShardWriter(out_dir, GROUP, split, first["seed"], manifest_dir=manifests_out, notes=notes) as w:
        for m in manifests:
            data = load_shard(Path(shards_dir) / m["shard"], m)
            for i in range(len(data["iq"])):
                classes = {DATASET_CLASSES[j] for j in np.flatnonzero(data["y"][i])}
                src, seed, snr = str(data["source"][i]), int(data["seed"][i]), float(data["snr_db"][i])
                window = data["iq"][i]
                if src == "gen":
                    if len(classes) != 1:
                        raise ValueError(f"{m['shard']} row {i}: generated row with labels {sorted(classes)}")
                    cls = next(iter(classes))
                    if stats["gen"] % CHECK_EVERY == 0:
                        old, _ = digital.generate_window(cls, snr, np.random.default_rng(seed))
                        if not np.allclose(old, window, atol=CHECK_ATOL):
                            raise ValueError(f"{m['shard']} row {i}: seed {seed} no longer reproduces the stored "
                                             f"{cls} window; the generator changed since the shard was made")
                        stats["checked"] += 1
                    window, _ = impairments.generate_impaired(digital, cls, snr, np.random.default_rng(seed))
                    stats["gen"] += 1
                else:
                    stats["ext"] += 1
                w.add(window, classes, snr, src, seed)
    return stats, w.written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="folder holding fskpsk/<shard>.npz")
    p.add_argument("--manifests", required=True, help="folder holding the fskpsk_*.json manifests")
    p.add_argument("--out", required=True, help="folder the new fskpsk shards go into")
    p.add_argument("--manifests-out", required=True, help="folder the new fskpsk manifests go into")
    args = p.parse_args()

    manifests = [json.loads(f.read_text()) for f in sorted(Path(args.manifests).glob(f"{GROUP}_*.json"))]
    if not manifests:
        sys.exit(f"no {GROUP}_*.json manifests in {args.manifests}")
    for split in SPLITS:
        ms = sorted((m for m in manifests if m["split"] == split), key=lambda m: m["part"])
        if not ms:
            continue
        t0 = time.time()
        stats, written = reimpair_split(ms, Path(args.shards) / GROUP, args.out, args.manifests_out, split)
        print(f"{split}: {stats['gen']} generated rows re-made ({stats['checked']} checked against their seed), "
              f"{stats['ext']} external rows kept, {len(written)} shard(s), {time.time() - t0:.0f}s", flush=True)
    print(f"Shards in {args.out}\nManifests in {args.manifests_out}")


if __name__ == "__main__":
    main()
