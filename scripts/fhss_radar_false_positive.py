"""
Direct test of the asymmetric confusion evaluate.py's confusion_between()
found: LFM_RADAR's false positives are disproportionately true-FHSS windows
(36% vs a ~25% base rate on the checkpoint tested so far), but FHSS's false
positives are not disproportionately true-LFM_RADAR (24.5%, at base rate).

scripts/duty_cycle_breakdown.py ruled out one mechanism: real radar's own
duty cycle inside a composite. Recall for actual radar stayed 94-99%
regardless of how little of the window it occupied, so the model isn't
missing brief radar pulses -- that's not the leak.

This tests the failure mode directly: generate PURE standalone FHSS windows
(zero radar, nothing else) across FHSS's own parameter ranges (hop_rate_hz,
n_channels, channel_spacing_hz -- configs/default.yaml), run them through the
model, and check what fraction get FALSELY flagged as LFM_RADAR. Broken down
by each FHSS parameter, to see whether a specific configuration (e.g. a fast
hop rate sweeping a wide channel comb, which can visually resemble a
chirp's continuous frequency sweep in a spectrogram) is what the model
mistakes for radar, or whether the false-positive rate is flat across
parameters (in which case the leak isn't about FHSS's own shape at all, and
the real driver is something else, e.g. how LFM_RADAR was trained).

--ensemble averages predictions over ensemble_0.pt..ensemble_{n-1}.pt instead
of loading best_model.pt -- necessary to get a trustworthy answer once the
ensemble (not a single model) is what's actually being submitted and
calibrated for. Same reasoning as calibrate_thresholds.py's --ensemble flag:
a single model's threshold/behaviour doesn't transfer cleanly to the
ensemble's averaged probabilities.

Usage:
    python scripts/fhss_radar_false_positive.py --n 800
    python scripts/fhss_radar_false_positive.py --n 800 --ensemble --n-models 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
from src.data.preprocess import add_awgn, preprocess_window  # noqa: E402
from src.generators.fhss import generate_fhss  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LEN = CFG["signal"]["window_len"]


def _load_models(ckpt_paths):
    models = []
    for ckpt in ckpt_paths:
        model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


def _predict_probs(models, X):
    """Average sigmoid probabilities over one or more models -- same
    averaging train_ensemble.py's _predict uses, so this diagnostic sees
    exactly what the submitted ensemble actually outputs."""
    summed = None
    with torch.no_grad():
        for model in models:
            p = torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy()
            summed = p if summed is None else summed + p
    return summed / len(models)


def noise_control_check(models, threshold, n=500, seed_offset=777):
    """Sanity check before trusting anything else in this script: does the
    model flag LFM_RADAR on signals that are neither FHSS nor anything else --
    pure complex Gaussian noise, nothing transmitting?

    If this comes back high, the checkpoint's LFM_RADAR calibration is broken
    in general -- most likely the threshold in config was calibrated for a
    DIFFERENT checkpoint than the one currently at results/best_model.pt (e.g.
    calibrate_thresholds.py was run against a retrained model, but this script
    is loading a stale local copy). A model that fires on everything will of
    course also fire on FHSS -- that says nothing FHSS-specific, and the
    breakdown below would be reporting a mismatch artifact, not a real
    confusion pattern. Caught exactly this once already: a stale local
    checkpoint paired with a threshold from a different run gave a 100%
    false-positive rate on pure noise.
    """
    rng = np.random.default_rng(CFG["dataset"]["seed"] + seed_offset)
    windows = []
    for _ in range(n):
        noise = rng.standard_normal(WINDOW_LEN) + 1j * rng.standard_normal(WINDOW_LEN)
        windows.append(preprocess_window(noise, WINDOW_LEN))
    X = np.stack(windows).astype(np.float32)
    probs = _predict_probs(models, X)
    radar_prob = probs[:, CLASS_TO_IDX["LFM_RADAR"]]
    return float((radar_prob > threshold).mean()), float(radar_prob.mean())


def _bucket_report(values, false_positive, radar_prob, label, bins, bin_labels):
    print(f"--- by {label} ---")
    idx = np.digitize(values, bins) - 1
    for i, blabel in enumerate(bin_labels):
        mask = idx == i
        if not mask.any():
            print(f"  {blabel:<18}{'0':>6}  (no examples in this bin)")
            continue
        print(f"  {blabel:<18}{int(mask.sum()):>6}  fp_rate={false_positive[mask].mean():>6.1%}"
              f"  mean_prob={radar_prob[mask].mean():.3f}")
    r = np.corrcoef(values, radar_prob)[0, 1]
    print(f"  correlation({label}, radar_prob) = {r:+.3f}\n")


def main(n, ensemble, n_models):
    rng = np.random.default_rng(CFG["dataset"]["seed"] + 199)
    fs = CFG["signal"]["fs"]
    total_duration = CFG["signal"]["total_duration"]
    cfg = CFG["fhss"]

    ckpt_dir = REPO_ROOT / CFG["paths"]["checkpoints"]
    if ensemble:
        ckpt_paths = [ckpt_dir / f"ensemble_{i}.pt" for i in range(n_models)]
        missing = [p for p in ckpt_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing ensemble checkpoints: {missing} -- run "
                f"train_ensemble.py --models {n_models} first")
        ckpt_desc = f"{n_models}-model ensemble average ({ckpt_dir}/ensemble_*.pt)"
    else:
        ckpt_paths = [ckpt_dir / "best_model.pt"]
        ckpt_desc = str(ckpt_paths[0])
    models = _load_models(ckpt_paths)
    threshold = resolve_multilabel_thresholds()[CLASS_TO_IDX["LFM_RADAR"]]

    noise_fp_rate, noise_mean_prob = noise_control_check(models, threshold)
    print(f"Checkpoint: {ckpt_desc}")
    print(f"LFM_RADAR threshold used: {threshold:.3f}")
    print(f"\n--- Noise control (sanity check, not FHSS) ---")
    print(f"  pure noise false-positive rate: {noise_fp_rate:.1%}  (mean prob {noise_mean_prob:.3f})")
    if noise_fp_rate > 0.15:
        print("  *** WARNING: this checkpoint flags LFM_RADAR on pure noise well above")
        print("  chance. The threshold in config likely doesn't match this checkpoint")
        print("  (e.g. calibrated for a different, retrained model). The FHSS breakdown")
        print("  below is NOT trustworthy as-is -- fix the checkpoint/threshold pairing")
        print("  first (retrain + calibrate_thresholds.py against THIS checkpoint, or")
        print("  load the checkpoint calibrate_thresholds.py actually used) before")
        print("  drawing any conclusion about FHSS specifically. ***\n")
    else:
        print("  OK -- low false-positive rate on noise, so a high FHSS-specific rate")
        print("  below would be a real signal, not a general miscalibration artifact.\n")

    hop_rates, n_channels_list, spacings, windows = [], [], [], []
    for _ in range(n):
        hop_rate = rng.uniform(*cfg["hop_rate_hz"])
        hop_duration = 1 / hop_rate
        n_channels = int(rng.integers(*cfg["n_channels"]))
        spacing = rng.uniform(*cfg["channel_spacing_hz"])
        hop_freqs = (np.arange(n_channels) - n_channels / 2) * spacing

        sig = generate_fhss(fs, total_duration, hop_duration, hop_freqs, rng=rng)
        snr_db = rng.choice(CFG["snr_bins_db"])
        noisy = add_awgn(sig, float(snr_db), rng=rng)
        windows.append(preprocess_window(noisy, WINDOW_LEN))

        hop_rates.append(hop_rate)
        n_channels_list.append(n_channels)
        spacings.append(spacing)

    hop_rates = np.array(hop_rates)
    n_channels_arr = np.array(n_channels_list, dtype=float)
    spacings = np.array(spacings)
    X = np.stack(windows).astype(np.float32)

    probs = _predict_probs(models, X)
    radar_prob = probs[:, CLASS_TO_IDX["LFM_RADAR"]]
    false_positive = radar_prob > threshold  # ground truth is "absent" for every example here

    print(f"--- {n} PURE standalone FHSS windows (zero radar) ---")
    print(f"Overall false-positive rate (FHSS mistaken for LFM_RADAR): {false_positive.mean():.1%}\n")

    hop_bins = np.linspace(*cfg["hop_rate_hz"], 6)
    _bucket_report(hop_rates, false_positive, radar_prob, "hop_rate_hz", hop_bins,
                    [f"{hop_bins[i]/1000:.0f}-{hop_bins[i+1]/1000:.0f} kHz" for i in range(5)])

    ch_bins = np.linspace(*cfg["n_channels"], 6)
    _bucket_report(n_channels_arr, false_positive, radar_prob, "n_channels", ch_bins,
                    [f"{ch_bins[i]:.0f}-{ch_bins[i+1]:.0f}" for i in range(5)])

    sp_bins = np.linspace(*cfg["channel_spacing_hz"], 6)
    _bucket_report(spacings, false_positive, radar_prob, "channel_spacing_hz", sp_bins,
                    [f"{sp_bins[i]/1000:.0f}-{sp_bins[i+1]/1000:.0f} kHz" for i in range(5)])

    # A fast hop rate sweeping a WIDE comb (many channels * wide spacing) moves
    # across frequency the fastest overall -- the closest visual analogue to a
    # chirp's continuous sweep, so this composite quantity is the most direct
    # test of the "looks like a chirp" theory, not just each parameter alone.
    sweep_rate = n_channels_arr * spacings * hop_rates
    sw_bins = np.quantile(sweep_rate, [0, .2, .4, .6, .8, 1.0])
    sw_bins[-1] *= 1.0001  # nudge past the max so digitize includes it in the last bin
    _bucket_report(sweep_rate, false_positive, radar_prob,
                    "comb_width x hop_rate (effective sweep speed)", sw_bins,
                    [f"quintile {i+1}" for i in range(5)])

    print("Interpretation: a false-positive rate or radar_prob that climbs")
    print("across bins (or a correlation clearly away from 0) in any of these")
    print("means that parameter drives the confusion -- fix the generator/model")
    print("there. Flat rates and near-zero correlations everywhere mean FHSS's")
    print("own shape isn't the story, and the leak likely comes from how")
    print("LFM_RADAR itself was trained (e.g. the class_weight_multipliers")
    print("boost being asymmetric, LFM_RADAR 1.6x vs FHSS 1.3x -- see")
    print("configs/default.yaml), not from any specific FHSS configuration.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--ensemble", action="store_true",
                    help="average over ensemble_*.pt instead of best_model.pt")
    p.add_argument("--n-models", type=int, default=5)
    a = p.parse_args()
    main(a.n, a.ensemble, a.n_models)
