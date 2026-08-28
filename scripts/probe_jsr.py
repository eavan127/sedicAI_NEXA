"""Measure how a victim emitter survives a jammer of increasing strength.

Why this exists
---------------
FHSS recall on the test split is 0.963 when the emitter has the window to
itself and 0.489 when a jammer shares it. That averaged number cannot say
whether the loss is physics -- a jammer 20 dB up leaves the victim 1% of the
power -- or a fixable defect. Sweeping jammer-to-signal ratio separates them,
and it turns out to be a cliff rather than a slope: FHSS holds at 0.960 with
the jammer at equal power, halves by +5 dB, and is gone by +10 dB.

That matters because the information is NOT gone at +10 dB. FHSS occupies
10-48 kHz channels while barrage jamming spreads over 200 kHz - 1.2 MHz, so a
frequency-selective detector has 10-20 dB of processing gain available. The
model cannot use it, because STFTBranch averages the frequency axis away.

This script is the before/after measurement for that hypothesis. Run it once
against the current checkpoints, change the architecture, retrain, run it
again with the same seed and sample size, and compare the whole curve.

Reading the output
------------------
Recall alone is not enough, and this is the trap the FHSS pipeline doc already
walked into once: across three successive fixes FHSS recall rose 82.5 -> 89.7
-> 92.2 while JAMMING recall fell 80.0 -> 73.3 -> 67.5 in the same runs. The
two classes share a decision boundary and trade against each other. So every
row also reports what the jammer costs the JAMMING class, and what a jammer
ALONE gets called -- if FHSS recall improves because the model started calling
jammers FHSS, this table shows it instead of hiding it.

Usage
-----
    python scripts/probe_jsr.py --n 600 --out evals/jsr_baseline.json
    # ... change the architecture, retrain ...
    python scripts/probe_jsr.py --n 600 --out evals/jsr_freqsummary.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, resolve_multilabel_thresholds  # noqa: E402
from src.data.composite import unit_power  # noqa: E402
from src.data.preprocess import add_awgn, preprocess_window  # noqa: E402
from src.generators.fhss import random_fhss_example  # noqa: E402
from src.generators.jamming import random_jamming_example  # noqa: E402
from src.models.amc_cnn import AMC_CNN  # noqa: E402
from src.ui.app_models import ensemble_paths  # noqa: E402

# The sweep points. None means "no jammer at all", which is the control: it
# should reproduce the standalone recall from the scorecard, and if it does not
# the probe is out of distribution and nothing below can be trusted.
JSR_POINTS = (None, 0, 5, 10, 15, 20)


def _load_models(members=None):
    """Load the ensemble, or the first `members` of it.

    A hypothesis test does not need five models. The pre-registered bar for
    the frequency-summary experiment is FHSS recall at +10 dB JSR rising from
    0.035 to above 0.25 -- an effect far larger than the 3-point seed spread
    of a single model, so one member answers it. That is one 2.6-hour retrain
    instead of five, and the comparison stays like-for-like as long as the
    baseline is measured with the same number of members.
    """
    models = []
    for path in ensemble_paths()[:members] if members else ensemble_paths():
        model = AMC_CNN(num_classes=len(CLASSES), input_len=CFG["signal"]["window_len"])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models.append(model)
    if not models:
        raise SystemExit("No ensemble checkpoints found in results/.")
    return models


def _window(generator, rng):
    """One window, generated the way build_dataset does.

    Generators are asked for the config's full total_duration and the first
    window is taken, because several of them behave differently when asked for
    exactly one window's worth of signal -- a radar with a long PRI, for
    instance, may place no pulse at all in a 160 us request.
    """
    window_len = CFG["signal"]["window_len"]
    x = np.asarray(generator(fs=CFG["signal"]["fs"],
                              total_duration=CFG["signal"]["total_duration"],
                              rng=rng))[:window_len]
    if len(x) < window_len:
        x = np.pad(x, (0, window_len - len(x)))
    return unit_power(x)


def _predict(models, windows):
    window_len = CFG["signal"]["window_len"]
    batch = np.stack([preprocess_window(w, window_len) for w in windows]).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(batch)
        return np.mean([torch.sigmoid(m(tensor)).numpy() for m in models], axis=0)


def probe(models, n, seed, snr_db):
    thresholds = np.array(resolve_multilabel_thresholds())
    fhss_i, jam_i = CLASSES.index("FHSS"), CLASSES.index("JAMMING")
    rng = np.random.default_rng(seed)
    rows = []

    for jsr in JSR_POINTS:
        victim_windows, jammer_only_windows = [], []
        for _ in range(n):
            victim = _window(random_fhss_example, rng)
            if jsr is not None:
                victim = victim + _window(random_jamming_example, rng) * (10 ** (jsr / 20.0))
            victim_windows.append(add_awgn(victim, snr_db=snr_db, rng=rng))
            jammer_only_windows.append(
                add_awgn(_window(random_jamming_example, rng), snr_db=snr_db, rng=rng))

        pv = _predict(models, victim_windows)
        pj = _predict(models, jammer_only_windows)

        rows.append({
            "jsr_db": jsr,
            # Does the victim survive?
            "fhss_recall": float((pv[:, fhss_i] > thresholds[fhss_i]).mean()),
            # Is the jammer still seen in the same window? (Not applicable
            # when there is no jammer -- there it is a false alarm instead.)
            "jamming_in_mixed": float((pv[:, jam_i] > thresholds[jam_i]).mean()),
            # The zero-sum guard: a jammer on its own must not be called FHSS.
            "jamming_recall_alone": float((pj[:, jam_i] > thresholds[jam_i]).mean()),
            "jammer_called_fhss": float((pj[:, fhss_i] > thresholds[fhss_i]).mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600,
                    help="windows per condition. 150 leaves +/-8 points of binomial "
                          "noise at p=0.5, which is wider than some real effects; "
                          "600 halves that to +/-4.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--members", type=int, default=None,
                    help="use only the first N ensemble members. Use 1 for a "
                          "single-model hypothesis test; the baseline must be "
                          "measured with the same value.")
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = ap.parse_args()

    models = _load_models(args.members)
    print(f"{len(models)} ensemble members  ·  n={args.n} per condition  "
          f"·  SNR {args.snr_db:+.0f} dB  ·  seed {args.seed}\n")

    rows = probe(models, args.n, args.seed, args.snr_db)

    print(f"{'JSR':>9s} {'FHSS recall':>12s} {'JAM in mix':>11s} "
          f"{'JAM alone':>10s} {'jammer->FHSS':>13s}")
    for r in rows:
        label = "none" if r["jsr_db"] is None else f"{r['jsr_db']:+d} dB"
        print(f"{label:>9s} {r['fhss_recall']:12.3f} {r['jamming_in_mixed']:11.3f} "
              f"{r['jamming_recall_alone']:10.3f} {r['jammer_called_fhss']:13.3f}")

    print("\nFHSS recall  — does the victim survive the jammer?")
    print("JAM in mix   — is the jammer also detected in that same window?")
    print("JAM alone    — jammer with no victim: recall must not fall.")
    print("jammer->FHSS — jammer with no victim called FHSS: must not rise.")
    print("\nA real fix moves FHSS recall up at +5 and +10 dB while the last two "
          "columns hold. FHSS recall rising alongside jammer->FHSS is the two "
          "classes trading boundary space, not an improvement.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "n": args.n, "seed": args.seed, "snr_db": args.snr_db,
            "stft_freq_summary": bool(CFG.get("model", {}).get("stft_freq_summary", False)),
            "members": len(models),
            "rows": rows,
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
