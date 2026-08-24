"""
Tests the specific mechanism proposed for the asymmetric LFM_RADAR/FHSS
confusion found by evaluate.py's confusion_between() check: LFM_RADAR's false
positives are disproportionately true-FHSS windows (36% vs a ~25% base rate
on the checkpoint tested so far), but FHSS's false positives are not
disproportionately true-LFM_RADAR (24.5%, at base rate). The leak only runs
one direction.

Hypothesis: LFM_RADAR is pulsed (src/generators/radar.py's embed_pulse_train
-- silent between pulses, PRI up to 10ms against a ~160us window), while FHSS
is continuous (src/generators/fhss.py -- never silent). In a LFM_RADAR+FHSS
composite window, a low radar duty cycle means most of the window is
pure-FHSS-looking with only a brief radar blip. Training on many such
windows -- labelled "LFM_RADAR present" despite looking mostly like FHSS --
could teach the model a spurious FHSS -> radar association, which would then
misfire on windows that are ACTUALLY pure FHSS. FHSS has no equivalent silent
gap for radar to dominate, so the reverse association has no mechanism to
form -- consistent with the asymmetry already measured.

This script generates fresh LFM_RADAR+FHSS composites (same mixture_combos
ordering and mixture_sir_db range configs/default.yaml uses for real
training data), computes each one's TRUE radar duty cycle within the
512-sample window actually fed to the model, and buckets LFM_RADAR detection
recall by that duty cycle. If recall drops sharply at low duty cycle, that
supports the hypothesis -- the model IS specifically worse at "briefly
visible" radar, which is exactly the failure mode "mostly FHSS, briefly
radar" training examples would produce.

Runs against whatever checkpoint is at results/best_model.pt. Swap in the
current trained one for a result that reflects the live model, not a stale
local copy -- the duty-cycle TREND is what matters here, not the absolute
recall numbers, but a stale checkpoint's numbers shouldn't be quoted as if
they were current.

Usage:
    python scripts/duty_cycle_breakdown.py --n 800
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, resolve_multilabel_thresholds  # noqa: E402
from src.data.composite import mix_components  # noqa: E402
from src.data.preprocess import add_awgn, preprocess_window  # noqa: E402
from src.generators.fhss import random_fhss_example  # noqa: E402
from src.generators.radar import random_radar_example  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_LEN = CFG["signal"]["window_len"]
DUTY_BINS = [0.0, 0.05, 0.15, 0.35, 0.60, 1.0001]  # last edge > 1 so duty=1.0 falls in the last bin
BIN_LABELS = ["0-5%", "5-15%", "15-35%", "35-60%", "60-100%"]


def radar_duty_cycle(radar_iq, window_len):
    """Fraction of the FIRST window_len samples (what preprocess_window keeps,
    so what the model actually sees) that are radar-active -- same -20dB-
    below-peak convention as composite.py's active_power, for consistency
    with how "on" is defined everywhere else in this pipeline."""
    windowed = radar_iq[:window_len]
    mag_sq = np.abs(windowed) ** 2
    peak = mag_sq.max()
    if peak == 0:
        return 0.0
    return float((mag_sq > 0.01 * peak).mean())


def main(n):
    rng = np.random.default_rng(CFG["dataset"]["seed"] + 99)

    ckpt = REPO_ROOT / CFG["paths"]["checkpoints"] / "best_model.pt"
    model = AMC_CNN(num_classes=len(CLASSES), input_len=WINDOW_LEN).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    threshold = resolve_multilabel_thresholds()[CLASS_TO_IDX["LFM_RADAR"]]

    duties, windows = [], []
    for _ in range(n):
        radar_iq = random_radar_example(rng=rng)
        fhss_iq = random_fhss_example(rng=rng)
        duties.append(radar_duty_cycle(radar_iq, WINDOW_LEN))

        mixed, _ = mix_components([("LFM_RADAR", radar_iq), ("FHSS", fhss_iq)], rng=rng)
        snr_db = rng.choice(CFG["snr_bins_db"])
        noisy = add_awgn(mixed, float(snr_db), rng=rng)
        windows.append(preprocess_window(noisy, WINDOW_LEN))

    duties = np.array(duties)
    X = np.stack(windows).astype(np.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy()
    radar_prob = probs[:, CLASS_TO_IDX["LFM_RADAR"]]
    detected = radar_prob > threshold

    print(f"Generated {n} LFM_RADAR+FHSS composites, checkpoint: {ckpt}")
    print(f"LFM_RADAR threshold used: {threshold:.3f}\n")
    print(f"{'duty cycle':<12}{'n':>6}{'radar recall':>15}{'mean prob':>12}")
    print("-" * 45)

    bin_idx = np.digitize(duties, DUTY_BINS) - 1
    for i, label in enumerate(BIN_LABELS):
        mask = bin_idx == i
        if not mask.any():
            print(f"{label:<12}{0:>6}{'n/a':>15}{'n/a':>12}")
            continue
        recall = detected[mask].mean()
        print(f"{label:<12}{int(mask.sum()):>6}{recall:>15.3f}{radar_prob[mask].mean():>12.3f}")

    print(f"\nMedian duty cycle across all {n} generated composites: {np.median(duties):.3f}")
    print("(shows how common the 'mostly silent radar' case is in the")
    print("generator's own natural distribution, not just a constructed edge case)")

    overall_recall = detected.mean()
    low, high = duties < 0.15, duties >= 0.60
    print(f"\nOverall recall: {overall_recall:.3f}")
    if low.any() and high.any():
        print(f"Low duty (<15%, n={int(low.sum())}) recall: {detected[low].mean():.3f}")
        print(f"High duty (>=60%, n={int(high.sum())}) recall: {detected[high].mean():.3f}")
        gap = detected[high].mean() - detected[low].mean()
        print(f"\nGap: {gap:+.3f}. A large positive gap supports the hypothesis --")
        print("the model is specifically worse at briefly-visible radar, consistent")
        print("with 'mostly FHSS, briefly radar' training examples teaching a leak.")
        print("A small/negative gap means duty cycle isn't the driver -- look elsewhere.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=800)
    main(p.parse_args().n)
