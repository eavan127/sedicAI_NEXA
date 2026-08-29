"""Retrain one ensemble member with the SHIPPED architecture.

Why this exists
---------------
results/ensemble_0.pt was overwritten on 2026-08-29 05:31 with a model trained
while model.stft_freq_summary was switched on -- fc1 (256, 195) and attention
(1, 196, 1) against the shipped (256, 192) / (1, 193, 1). That happened during
the window when the flag was sitting `true` uncommitted in configs/default.yaml,
and anything running scripts/train_ensemble.py in that window writes straight
to ensemble_{i}.pt.

The consequence is not cosmetic: the ensemble is what carries the benchmark
margin. It contributes +2.8 points on LFM_RADAR and +2.7 on FHSS over the mean
single member, and two of five members fail the 80% gate on their own. A
member that will not load leaves four.

No backup survived -- results/ in the worktree is hardlinked to the main
checkout, so both paths were the same inode. The original is regenerated
rather than recovered: same seed, same data, same config, and train_one calls
set_seed(seed) so a CPU rerun reproduces it closely.

The displaced freq-summary model was preserved first, at
results/experiments/stft_freqsummary_0529_0531.pt, so nothing is lost.

Guards
------
Refuses to run if model.stft_freq_summary is enabled, which is the exact
condition that caused the damage. Writes through the same save-on-improve path
as the experiment runner, so a killed run leaves a usable partial checkpoint
rather than nothing.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG  # noqa: E402
from src.train import load_data, stratified_split  # noqa: E402
from scripts.train_ensemble import train_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--member", type=int, default=0)
    args = ap.parse_args()

    if CFG.get("model", {}).get("stft_freq_summary", False):
        raise SystemExit(
            "model.stft_freq_summary is enabled. This script restores the "
            "SHIPPED architecture; running it with the flag on would rewrite "
            "the member with the very architecture that broke it.")

    seed = 2000 + args.member
    out = ROOT / "results" / f"ensemble_{args.member}.pt"
    history = ROOT / "docs" / "experiments" / f"ensemble_{args.member}_restore_history.json"
    print(f"restoring member {args.member}  ·  seed {seed}  ·  "
          f"stft_freq_summary={CFG.get('model', {}).get('stft_freq_summary', False)}")
    print(f"writing {out}\n")

    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    tr, va, _ = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])

    history.parent.mkdir(parents=True, exist_ok=True)
    model = train_one(X, y, snr_labels, tr, va, seed=seed,
                       history_path=history, ckpt_path=out)
    torch.save(model.state_dict(), out)

    sd = torch.load(out, map_location="cpu")
    print(f"\nsaved {out}")
    print(f"fc1.weight {tuple(sd['fc1.weight'].shape)}  "
          f"attn {tuple(sd['attn_pool.score.weight'].shape)}  "
          f"(shipped architecture is (256, 192) / (1, 193, 1))")


if __name__ == "__main__":
    main()
