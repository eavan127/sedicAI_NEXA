"""Train ONE model for the jammed-FHSS experiment (any combination of the switches below).

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

Switches (2026-09-20). With no arguments this is exactly what it always was: flag on, seed
2000, writing results/experiment_stft_freq_summary.pt. Every cell of the 2x2-plus plan in
docs/POST_STAGE1_FIXES.md (E5) is one invocation:

    python scripts/run_stft_experiment.py --no-freq-summary --divisor 40          # A
    python scripts/run_stft_experiment.py                                          # C  (flag)
    python scripts/run_stft_experiment.py --no-freq-summary --keep-rows            # C2 (keep the rows)
    python scripts/run_stft_experiment.py --keep-rows --divisor 40                 # C2 + flag + A
    python scripts/run_stft_experiment.py --no-freq-summary                        # baseline re-run

    --keep-rows        variant C2: replace the frequency average with a learned layer
    --dwell-feature     branch fix_radar-fhss-confusion: add _sweep_consistency, targeting
                        LFM_RADAR precision / the radar-called-FHSS confusion, not masking
                        (see model.stft_dwell_feature in configs/default.yaml)
    --no-freq-summary  turn the original flag off
    --divisor N        training.snr_weight_divisor (20 = today; 40 samples -10 dB ~3x, not ~10x)
    --smoke            1 epoch on a tiny slice, to prove the whole pipeline in about a minute
    --out-dir DIR      where to write (default results/)

Score a finished run with the probes, passing the same model switches:

    python scripts/probe_jsr.py --n 600 --seed 0 --checkpoint results/experiment_rows.pt --stft-keep-rows
    python scripts/high_snr_probe.py --n 300 --checkpoint results/experiment_rows.pt --stft-keep-rows
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 2000       # member 0's seed -- the one the pinned baseline was measured on


def _tag(freq_summary, keep_rows, divisor, dwell_feature=False):
    """Output name. The default run keeps its historical name so the notebook still finds it."""
    if freq_summary and not keep_rows and not dwell_feature and divisor == 20:
        return "stft_freq_summary"
    parts = ((["freq"] if freq_summary else []) + (["rows"] if keep_rows else [])
             + (["dwell"] if dwell_feature else []))
    if divisor != 20:
        parts.append(f"div{divisor:g}")
    return "_".join(parts) or "baseline"


def run_tag(freq_summary, keep_rows, divisor, seed=SEED, dwell_feature=False):
    """The full output name for a run: the architecture/training cell, plus the seed when it is
    not the baseline seed, so a second seed can never overwrite the first. The Colab notebook
    calls this same function, so the file names cannot drift apart."""
    tag = _tag(freq_summary, keep_rows, divisor, dwell_feature)
    return tag if seed == SEED else f"{tag}_seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-freq-summary", action="store_true", help="turn model.stft_freq_summary OFF")
    ap.add_argument("--keep-rows", action="store_true", help="turn model.stft_keep_rows ON (variant C2)")
    ap.add_argument("--dwell-feature", action="store_true",
                    help="turn model.stft_dwell_feature ON (branch fix_radar-fhss-confusion, "
                          "targets LFM_RADAR precision / radar-called-FHSS confusion)")
    ap.add_argument("--divisor", type=float, default=None, help="training.snr_weight_divisor (default: config, 20)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on a tiny slice: prove the pipeline, not the model")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    # Switch the experiment on BEFORE anything constructs a model, and only in this
    # process's memory (configs/default.yaml is never edited: with a flag on, the
    # fused shape or the layer set changes and NONE of the five checkpoints in
    # results/ would load).
    freq_summary = not args.no_freq_summary
    CFG.setdefault("model", {})["stft_freq_summary"] = freq_summary
    CFG["model"]["stft_keep_rows"] = args.keep_rows
    CFG["model"]["stft_dwell_feature"] = args.dwell_feature
    if args.divisor is not None:
        CFG["training"]["snr_weight_divisor"] = args.divisor
    divisor = CFG["training"].get("snr_weight_divisor", 20)
    if args.smoke:
        CFG["training"]["epochs"] = 1

    from src.models.amc_cnn import AMC_CNN
    from src.train import load_data, stratified_split
    from scripts.train_ensemble import train_one

    # Prove the switches took, the way the notebook does, before spending hours.
    probe = AMC_CNN(num_classes=len(CLASSES), input_len=CFG["signal"]["window_len"])
    assert probe.stft_branch.freq_summary == freq_summary, "stft_freq_summary did not apply"
    assert probe.stft_branch.keep_rows == args.keep_rows, "stft_keep_rows did not apply"
    assert probe.stft_branch.dwell_feature == args.dwell_feature, "stft_dwell_feature did not apply"
    n_params = sum(p.numel() for p in probe.parameters())
    del probe

    tag = run_tag(freq_summary, args.keep_rows, divisor, args.seed,
                  args.dwell_feature) + ("_smoke" if args.smoke else "")
    out = args.out_dir / f"experiment_{tag}.pt"
    hist = args.out_dir / f"experiment_{tag}_history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"stft_freq_summary = {freq_summary}   stft_keep_rows = {args.keep_rows}   "
          f"stft_dwell_feature = {args.dwell_feature}   snr_weight_divisor = {divisor:g}")
    print(f"parameters {n_params:,}   seed {args.seed}   writing {out.name} (+ history); results/ otherwise untouched\n")

    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    tr, va, _ = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])
    if args.smoke:
        # stratified_split returns indices grouped by class, so a plain [:N] slice would
        # be one class only. Take a random subset.
        rng = np.random.default_rng(0)
        tr, va = rng.choice(tr, 2000, replace=False), rng.choice(va, 500, replace=False)
    print(f"train {len(tr)}  val {len(va)}  epochs {CFG['training']['epochs']}\n")

    # history_path / ckpt_path: rewritten after every epoch and on every validation
    # improvement, so a run killed at epoch 25 leaves a usable checkpoint.
    model = train_one(X, y, snr_labels, tr, va, seed=args.seed, history_path=hist, ckpt_path=out)
    torch.save(model.state_dict(), out)
    (args.out_dir / f"experiment_{tag}_config.json").write_text(json.dumps({
        "stft_freq_summary": freq_summary, "stft_keep_rows": args.keep_rows,
        "stft_dwell_feature": args.dwell_feature,
        "snr_weight_divisor": divisor, "seed": args.seed, "parameters": n_params,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
