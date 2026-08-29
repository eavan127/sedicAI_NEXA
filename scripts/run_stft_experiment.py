"""Train ONE model with model.stft_freq_summary enabled, for the JSR experiment.

The hypothesis
--------------
STFTBranch ends with `f.mean(dim=2)`, averaging the frequency axis away, so
"peak at bin 2 then bin 6" (FHSS hopping) and "peak at bin 4 then bin 4" (tone
jamming) become the same features. Measured consequences: FHSS recall falls
0.962 -> 0.035 as the jammer goes from equal power to +10 dB, 46.5% of
held-out jamming is predicted as FHSS, and jamming's seed spread is 10.8
points against FHSS's 1.1.

With the flag on, the branch pools time only (keeping 200 kHz frequency bins
instead of 400) and adds three per-frame features computed from the STFT
magnitude directly: frequency max, spectral flatness, and peak-frequency
delta. The pre-registered bar is FHSS recall at +10 dB JSR rising above 0.25,
from a baseline of 0.048 for a single member.

Why one member and not five
---------------------------
The effect being tested is roughly 5x a single model's seed spread, so one
member answers it -- 2.5 hours instead of 13. docs/experiments/
jsr_baseline_1model.json is the matching single-member baseline, measured with
the same probe, sample size and seed.

Why the config file is not edited
---------------------------------
The flag is switched on in memory here rather than in configs/default.yaml,
because with it on the fused tensor shape changes and NONE of the five
checkpoints in results/ will load. Leaving it enabled on disk would break the
console and the submission. This script cannot leave that landmine behind.

The checkpoint is written to a NEW file. Nothing in results/ is overwritten.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES  # noqa: E402

# Switch the experiment on BEFORE anything constructs a model, and only in
# this process's memory.
CFG.setdefault("model", {})["stft_freq_summary"] = True

from src.train import load_data, stratified_split  # noqa: E402
from scripts.train_ensemble import train_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_stft_freq_summary.pt"
HISTORY = ROOT / "docs" / "experiments" / "stft_experiment_history.json"
SEED = 2000       # member 0's seed -- the one the pinned baseline was measured on


def main():
    assert CFG["model"]["stft_freq_summary"] is True, "flag failed to enable"
    print(f"stft_freq_summary = {CFG['model']['stft_freq_summary']}")
    print(f"seed {SEED}  ·  writing to {OUT.name}  ·  results/ untouched otherwise\n")

    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    tr, va, _ = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    print(f"train {len(tr)}  val {len(va)}  epochs {CFG['training']['epochs']}\n")

    model = train_one(X, y, snr_labels, tr, va, seed=SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
