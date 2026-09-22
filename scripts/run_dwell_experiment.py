"""Train ONE model with model.stft_dwell_feature enabled.

The hypothesis
--------------
LFM_RADAR precision is 51%, and 46% of its false positives are windows that
are really FHSS (docs/experiments/c2_baseline_eval.json, radar_fhss_confusion).
Both classes are visible to the model -- this is not the masking problem
stft_freq_summary/stft_keep_rows target -- they just look alike to it: a
radar chirp sweeps continuously across frequency rows, FHSS holds one row per
hop then jumps to an arbitrary new one. `_peak_freq_delta` already reports
frame-to-frame jumps but not whether consecutive jumps keep going the same
way, which is what turns a jump into a sweep.

With the flag on, the branch keeps frequency resolution (as stft_freq_summary
does) and adds one per-frame feature, `_sweep_consistency`: whether this
frame's step has the same sign as the previous one's. High and sustained for
a chirp, near chance for FHSS. See src/models/amc_cnn.py and
configs/default.yaml for the full docstring.

Why one member and not five
---------------------------
Same reasoning as run_stft_experiment.py: one member is enough to see whether
the effect is real before spending 13 hours on an ensemble.

Why the config file is not edited
----------------------------------
The flag is switched on in memory here rather than in configs/default.yaml,
because with it on the fused tensor shape changes and none of the checkpoints
in results/ will load. This script cannot leave that landmine behind.

The checkpoint is written to a NEW file. Nothing in results/ is overwritten.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES  # noqa: E402

# Switch the experiment on BEFORE anything constructs a model, and only in
# this process's memory.
CFG.setdefault("model", {})["stft_dwell_feature"] = True

from src.train import load_data, stratified_split  # noqa: E402
from scripts.train_ensemble import train_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiment_stft_dwell_feature.pt"
SEED = 2000       # same seed used to pin the freq_summary/keep_rows baselines


def main():
    assert CFG["model"]["stft_dwell_feature"] is True, "flag failed to enable"
    print(f"stft_dwell_feature = {CFG['model']['stft_dwell_feature']}")
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
