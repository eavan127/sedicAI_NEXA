"""
Evaluate ONE trained experiment model on everything, and judge it against a baseline.

Why this exists. `src/evaluate.py` scores the fixed files in results/ and writes into
evals/. An experiment run (scripts/run_stft_experiment.py) produces a single checkpoint
somewhere else, with its own architecture flags and -- crucially -- its own probability
calibration. This script scores that checkpoint the way the team scores the submission,
using the SAME metric functions from src/evaluate.py, so the numbers are comparable with
the report.

The one thing that must not be skipped: thresholds. The thresholds in configs/default.yaml
were calibrated for the five-model ensemble. A retrained model has different probabilities,
so it gets its own, chosen on the VALIDATION split by scripts/calibrate_thresholds.py's
rule (highest threshold that still keeps recall >= benchmark + margin). The baseline is
recalibrated the same way, so the comparison is like for like. Because calibration pins
each judged class's recall near the same value (~83% on validation), a better model shows
up as HIGHER PRECISION at that recall and as better behaviour in company (see
recall_in_context), not as a higher headline recall.

What it reports (all on the held-out test split, calibrated thresholds):
    per class      precision, recall, F1, accuracy, balanced accuracy, specificity,
                   support, and the TP / FP / FN / TN counts
    benchmark      the >80% recall gate on LFM_RADAR, FHSS, JAMMING
    comms vs CEMA  discrimination accuracy, jamming recall, false alarm rate
    tiers          coarse tier accuracy and per-tier recall; dense-QAM recall
    confusion      radar <-> FHSS false positives
    in company     every class's recall when alone / with another emitter / with a jammer
    by SNR         recall and mean probability per class, and FHSS/JAMMING/radar in company

And with --baseline it prints the side-by-side deltas and a PASS/FAIL verdict on the
pre-registered checks in `judge()`: FHSS must go UP where it is masked, JAMMING must not
move, nothing else may break.

Usage:
    # evaluate a trained run (calibrates on validation, writes eval + thresholds JSON)
    python scripts/evaluate_experiment.py --checkpoint results/experiment_rows.pt \\
        --stft-keep-rows --name c2 --baseline docs/experiments/c2_baseline_eval.json

    # final verdict once the probes have also been run with the calibrated thresholds
    python scripts/evaluate_experiment.py --verdict-only --eval-json results/eval_c2.json \\
        --baseline docs/experiments/c2_baseline_eval.json \\
        --high-snr results/high_snr_c2.json --baseline-high-snr docs/experiments/c2_baseline_high_snr.json \\
        --jsr results/jsr_c2.json --baseline-jsr docs/experiments/c2_baseline_jsr.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CFG, CLASSES, CLASS_TO_IDX  # noqa: E402

BENCHMARK = CFG["benchmark_recall"]
JUDGED = list(CFG["judged_classes"])
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- helpers

def _py(obj):
    """numpy -> plain python, recursively, so the result is JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _py(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _pct(x, digits=1):
    return "   n/a" if x is None else f"{100 * x:{5 + digits}.{digits}f}%"


def _delta(new, base, digits=1):
    if new is None or base is None:
        return "   n/a"
    return f"{100 * (new - base):+{4 + digits}.{digits}f}"


# --------------------------------------------------------------------------- model + data

def load_models(ckpts, input_len):
    import torch
    from src.models.amc_cnn import AMC_CNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for path in ckpts:
        m = AMC_CNN(num_classes=len(CLASSES), input_len=input_len).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    return models, device


def predict(models, device, X, batch=512):
    """Average sigmoid probabilities over the models, in batches (a whole split at once
    would materialise every intermediate activation and run a GPU out of memory)."""
    import torch

    summed = None
    with torch.no_grad():
        for m in models:
            chunks = [torch.sigmoid(m(torch.from_numpy(X[i:i + batch]).float().to(device))).cpu().numpy()
                      for i in range(0, len(X), batch)]
            p = np.concatenate(chunks, axis=0)
            summed = p if summed is None else summed + p
    return summed / len(models)


def calibrate(probs_val, y_val, margin):
    """Per-class thresholds on the validation split, by the team's own rule."""
    from scripts.calibrate_thresholds import _best_threshold

    target = BENCHMARK + margin
    out = {}
    for cls in CLASSES:
        i = CLASS_TO_IDX[cls]
        t, precision, ok = _best_threshold(probs_val[:, i], y_val[:, i], target)
        out[cls] = {"threshold": float(t),
                    "val_precision_at_threshold": None if precision is None else float(precision),
                    "val_recall_target_met": bool(ok)}
    return out


# --------------------------------------------------------------------------- metrics

def compute_metrics(y, probs, thr_vec, snr):
    from sklearn.metrics import classification_report
    from src.evaluate import (coarse_tier_metrics, comms_vs_jamming, confusion_between,
                              dense_qam_recall, recall_in_context)

    y = y.astype(int)
    preds = (probs > thr_vec).astype(int)          # strict '>' like src/evaluate.py and calibration
    n = len(y)
    rep = classification_report(y, preds, labels=range(len(CLASSES)), target_names=CLASSES,
                                output_dict=True, zero_division=0)

    per_class = {}
    for cls in CLASSES:
        i = CLASS_TO_IDX[cls]
        p_i, t_i = preds[:, i] == 1, y[:, i] == 1
        tp, fp = int((p_i & t_i).sum()), int((p_i & ~t_i).sum())
        fn, tn = int((~p_i & t_i).sum()), int((~p_i & ~t_i).sum())
        recall = rep[cls]["recall"]
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[cls] = {
            "precision": rep[cls]["precision"], "recall": recall, "f1": rep[cls]["f1-score"],
            "support": int(rep[cls]["support"]),
            "accuracy": (tp + tn) / n, "balanced_accuracy": (recall + specificity) / 2,
            "specificity": specificity,
            # what a model that never predicts this class would score: shows how little
            # per-class accuracy proves when the class is rare
            "trivial_accuracy": (tn + fp) / n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
    averages = {k: {"precision": rep[k]["precision"], "recall": rep[k]["recall"], "f1": rep[k]["f1-score"]}
                for k in ("micro avg", "macro avg", "weighted avg", "samples avg")}

    benchmark = {"benchmark_recall": BENCHMARK,
                 "judged": {c: {"recall": per_class[c]["recall"], "passed": bool(per_class[c]["recall"] >= BENCHMARK)}
                            for c in JUDGED}}
    benchmark["passed"] = all(v["passed"] for v in benchmark["judged"].values())

    by_snr, context_by_snr = {}, {}
    for b in sorted(set(snr.tolist())):
        m = snr == b
        by_snr[str(float(b))] = {}
        for cls in CLASSES:
            i = CLASS_TO_IDX[cls]
            pos = m & (y[:, i] == 1)
            by_snr[str(float(b))][cls] = {
                "recall": float(preds[pos, i].mean()) if pos.any() else None,
                "mean_probability": float(probs[pos, i].mean()) if pos.any() else None,
                "support": int(pos.sum()),
            }
        ric = recall_in_context(y[m], preds[m])
        context_by_snr[str(float(b))] = {c: ric[c] for c in JUDGED}

    return _py({
        "per_class": per_class, "averages": averages, "benchmark": benchmark,
        "comms_vs_jamming": comms_vs_jamming(y, preds),
        "coarse_tier": coarse_tier_metrics(y, preds),
        "dense_qam_recall": dense_qam_recall(y, preds),
        "radar_fhss_confusion": {
            "LFM_RADAR_fp_that_are_true_FHSS": confusion_between(y, preds, "LFM_RADAR", "FHSS"),
            "FHSS_fp_that_are_true_LFM_RADAR": confusion_between(y, preds, "FHSS", "LFM_RADAR"),
        },
        "recall_in_context": recall_in_context(y, preds),
        "by_snr": by_snr, "context_by_snr": context_by_snr,
    })


# --------------------------------------------------------------------------- the verdict

def _ctx(ev, cls, bucket):
    try:
        return ev["recall_in_context"][cls][bucket]["recall"]
    except (KeyError, TypeError):
        return None


def _probe_at(hs, key, snr):
    """Overlay outcome column from a high_snr_probe JSON, as a FRACTION, at one SNR."""
    try:
        return hs["overlay_outcomes"][str(float(snr))][key] / 100.0
    except (KeyError, TypeError):
        return None


def judge(new, base, high_new=None, high_base=None, jsr_new=None, jsr_base=None):
    """The pre-registered checks (fixed before any run; see docs/POST_STAGE1_FIXES.md, E5).

    Returns a list of dicts {group, name, rule, new, base, passed}. `passed` is None when the
    data for a check was not supplied (so it is shown as SKIPPED, not silently counted).
    A run passes only if every required check that could be evaluated passed.

    Three groups. FHSS must go UP where it is masked; JAMMING must not move; nothing else
    may break. Thresholds are absolute points (0.03 = 3 points) and are deliberately larger
    than the ~6-point run-to-run noise of a single training run wherever a gain is claimed.
    """
    checks = []

    def add(group, name, rule, nv, bv, ok):
        checks.append({"group": group, "name": name, "rule": rule, "new": nv, "base": bv,
                       "passed": None if ok is None else bool(ok)})

    def pc(ev, cls, key):
        return ev["per_class"][cls][key]

    G1, G2, G3 = "FHSS goes up", "JAMMING stays intact", "Nothing else breaks"

    # ---- FHSS goes up ----
    nv, bv = _ctx(new, "FHSS", "with_jammer"), _ctx(base, "FHSS", "with_jammer")
    add(G1, "FHSS recall with a jammer present (test split)", "at least +10 points over baseline",
        nv, bv, None if nv is None or bv is None else nv >= bv + 0.10)
    if high_new is not None:
        for snr, floor in ((6, 0.55), (10, 0.45)):
            nv, bv = _probe_at(high_new, "both", snr), _probe_at(high_base, "both", snr) if high_base else None
            add(G1, f"jammed-FHSS reported as BOTH, +{snr} dB", f"at least {int(floor * 100)}%",
                nv, bv, None if nv is None else nv >= floor)
    add(G1, "FHSS precision (no false-alarm blow-up)", "not more than 3 points below baseline",
        pc(new, "FHSS", "precision"), pc(base, "FHSS", "precision"),
        pc(new, "FHSS", "precision") >= pc(base, "FHSS", "precision") - 0.03)

    # ---- JAMMING stays intact ----
    nv, bv = pc(new, "JAMMING", "recall"), pc(base, "JAMMING", "recall")
    add(G2, "JAMMING recall", f">= {int(BENCHMARK * 100)}% and not >3 points below baseline",
        nv, bv, nv >= max(BENCHMARK, bv - 0.03))
    nv, bv = pc(new, "JAMMING", "precision"), pc(base, "JAMMING", "precision")
    add(G2, "JAMMING precision", "not more than 2 points below baseline", nv, bv, nv >= bv - 0.02)
    nv, bv = new["comms_vs_jamming"]["false_alarm_rate"], base["comms_vs_jamming"]["false_alarm_rate"]
    add(G2, "false alarm rate on civilian-only windows", "at most 0.1%", nv, bv, nv <= 0.001)
    nv, bv = new["comms_vs_jamming"]["accuracy"], base["comms_vs_jamming"]["accuracy"]
    add(G2, "comms-vs-jamming discrimination accuracy", "not more than 1 point below baseline",
        nv, bv, nv >= bv - 0.01)
    nv, bv = _ctx(new, "JAMMING", "alone"), _ctx(base, "JAMMING", "alone")
    add(G2, "JAMMING recall when alone", "not more than 3 points below baseline",
        nv, bv, None if nv is None or bv is None else nv >= bv - 0.03)
    if jsr_new is not None and jsr_base is not None:
        rows_n = {r["jsr_db"]: r for r in jsr_new["rows"]}
        rows_b = {r["jsr_db"]: r for r in jsr_base["rows"]}
        nv, bv = rows_n[10]["jammer_called_fhss"], rows_b[10]["jammer_called_fhss"]
        add(G2, "jammer (no victim) wrongly called FHSS", "not more than 3 points above baseline",
            nv, bv, nv <= bv + 0.03)
        nv, bv = rows_n[10]["fhss_recall"], rows_b[10]["fhss_recall"]
        add(G1, "FHSS recall at JSR +10 dB (pre-registered bar)", "above 25%", nv, bv, nv > 0.25)
    if high_new is not None and high_base is not None:
        for snr in (6, 10):
            nv, bv = _probe_at(high_new, "class_only", snr), _probe_at(high_base, "class_only", snr)
            add(G2, f"jammer MISSED (FHSS reported without JAMMING), +{snr} dB",
                "not more than 3 points above baseline", nv, bv,
                None if nv is None or bv is None else nv <= bv + 0.03)

    # ---- nothing else breaks ----
    for cls in ("LFM_RADAR", "FHSS"):
        nv, bv = pc(new, cls, "recall"), pc(base, cls, "recall")
        add(G3, f"{cls} recall", f">= {int(BENCHMARK * 100)}% (benchmark)", nv, bv, nv >= BENCHMARK)
    nv, bv = pc(new, "LFM_RADAR", "precision"), pc(base, "LFM_RADAR", "precision")
    add(G3, "LFM_RADAR precision", "not more than 3 points below baseline", nv, bv, nv >= bv - 0.03)
    if high_new is not None:
        rs = high_new["recall"]["standalone"]
        floor_ok = all(v >= 97.0 for s_, v in zip(high_new["snr_bins"], rs) if s_ >= -6)
        add(G3, "standalone FHSS recall, SNR >= -6 dB", "at least 97% at every level",
            min(v for s_, v in zip(high_new["snr_bins"], rs) if s_ >= -6) / 100, None, floor_ok)
    return checks


def print_verdict(checks):
    print("\n" + "=" * 100)
    print("VERDICT  (pre-registered checks; see docs/POST_STAGE1_FIXES.md, E5)")
    print("=" * 100)
    order = {"FHSS goes up": 0, "JAMMING stays intact": 1, "Nothing else breaks": 2}
    checks = sorted(checks, key=lambda c: order.get(c["group"], 9))       # stable: keeps each group's own order

    def fmt(x):
        if x is None:
            return "  n/a"
        return f"{100 * x:5.2f}%" if abs(x) < 0.01 else f"{100 * x:5.1f}%"     # tiny rates need the decimals

    group = None
    for c in checks:
        if c["group"] != group:
            group = c["group"]
            print(f"\n  {group}")
        mark = {True: "PASS", False: "FAIL", None: "skip"}[c["passed"]]
        nv, bv = fmt(c["new"]), fmt(c["base"])
        print(f"    [{mark}] {c['name']:<58s} new {nv}  base {bv}   needs: {c['rule']}")
    ran = [c for c in checks if c["passed"] is not None]
    failed = [c for c in ran if not c["passed"]]
    print()
    if not failed:
        print(f"  OVERALL: PASS  ({len(ran)} checks). FHSS went up and JAMMING held.")
    else:
        print(f"  OVERALL: FAIL  ({len(failed)} of {len(ran)} checks failed):")
        for c in failed:
            print(f"     - {c['group']}: {c['name']}")
    skipped = len(checks) - len(ran)
    if skipped:
        print(f"  ({skipped} check(s) skipped: their data was not supplied)")
    return not failed


# --------------------------------------------------------------------------- printing

def print_report(ev):
    print(f"\n{'=' * 100}\nEVALUATION: {ev['name']}   ({ev['parameters']:,} parameters, "
          f"{ev['n_test']:,} test windows)\n{'=' * 100}")
    print(f"flags: stft_freq_summary={ev['flags']['stft_freq_summary']}  stft_keep_rows={ev['flags']['stft_keep_rows']}"
          f"  stft_dwell_feature={ev['flags'].get('stft_dwell_feature', False)}"
          f"   checkpoint(s): {', '.join(Path(c).name for c in ev['checkpoints'])}")

    print(f"\nTHRESHOLDS calibrated on the validation split ({ev['n_val']:,} windows), "
          f"target recall {BENCHMARK:.0%} + {ev['margin']:.0%} margin")
    print(f"  {'class':<12}{'threshold':>10}{'val precision':>15}   {'met target'}")
    for cls in CLASSES:
        t = ev["thresholds"][cls]
        vp = "n/a" if t["val_precision_at_threshold"] is None else f"{t['val_precision_at_threshold']:.3f}"
        print(f"  {cls:<12}{t['threshold']:>10.2f}{vp:>15}   {'yes' if t['val_recall_target_met'] else 'NO (fallback)'}")

    print("\nPER CLASS (test split)")
    print(f"  {'class':<12}{'precision':>10}{'recall':>9}{'F1':>8}{'accuracy':>10}{'bal.acc':>9}{'specif.':>9}"
          f"{'support':>9}{'TP':>7}{'FP':>7}{'FN':>7}{'TN':>8}")
    for cls in CLASSES:
        r = ev["per_class"][cls]
        star = " *" if cls in JUDGED else "  "
        print(f"  {cls:<12}{r['precision']:>10.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}{r['accuracy']:>10.3f}"
              f"{r['balanced_accuracy']:>9.3f}{r['specificity']:>9.3f}{r['support']:>9}{r['tp']:>7}{r['fp']:>7}"
              f"{r['fn']:>7}{r['tn']:>8}{star}")
    print("  (* = judged class.  accuracy = (TP+TN)/all; a model that never predicts the class scores "
          "'trivial accuracy', see JSON.)")
    for k, v in ev["averages"].items():
        print(f"  {k:<12}{v['precision']:>10.3f}{v['recall']:>9.3f}{v['f1']:>8.3f}")

    b = ev["benchmark"]
    print(f"\nBENCHMARK (> {BENCHMARK:.0%} recall on the judged classes)")
    for cls, r in b["judged"].items():
        print(f"  {cls:<12} recall {r['recall']:.4f}   {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if b['passed'] else 'FAIL'}")

    c = ev["comms_vs_jamming"]
    print("\nCOMMS vs HOSTILE CEMA (the 'competitive advantage' criterion)")
    print(f"  discrimination accuracy {c['accuracy']:.4f}   jamming recall {c['jamming_recall']:.4f}   "
          f"false alarm rate {c['false_alarm_rate']:.5f}   (n={c['n_evaluated']})")
    ct = ev["coarse_tier"]
    print(f"COARSE TIER accuracy {ct['accuracy']:.4f}   per tier: "
          + "  ".join(f"{k} {v:.3f}" for k, v in ct["per_tier_recall"].items()))
    if ev["dense_qam_recall"]:
        print(f"DENSE QAM (16QAM+64QAM) recall {ev['dense_qam_recall']['recall']:.4f} "
              f"(n={ev['dense_qam_recall']['n_evaluated']})")
    rf = ev["radar_fhss_confusion"]
    for k, v in rf.items():
        if v:
            print(f"  {k}: {v['fraction_that_are_true_' + k.split('_that_are_true_')[1]]:.3f} of {v['false_positives']} false positives")

    print("\nRECALL IN COMPANY  (alone / with another emitter / with a jammer)   recall (support)")
    for cls in CLASSES:
        r = ev["recall_in_context"][cls]
        cells = []
        for bucket in ("alone", "with_emitter", "with_jammer"):
            x = r[bucket]
            cells.append("     n/a      " if x["recall"] is None else f"{x['recall']:6.3f} ({x['support']:>5})")
        print(f"  {cls:<12}" + "   ".join(cells))

    print("\nRECALL by SNR, judged classes  (recall / mean probability of the class on its own positives)")
    snrs = sorted(ev["by_snr"], key=float)
    print(f"  {'':<12}" + "".join(f"{float(s_):>+11.0f} dB" for s_ in snrs))
    for cls in JUDGED:
        row = "".join(f"{100 * ev['by_snr'][s_][cls]['recall']:>8.1f}%/{ev['by_snr'][s_][cls]['mean_probability']:.2f}"
                      for s_ in snrs)
        print(f"  {cls:<12}{row}")
    print("\n  FHSS recall when a jammer is in the window, by SNR")
    row = "".join(("     n/a   " if ev["context_by_snr"][s_]["FHSS"]["with_jammer"]["recall"] is None
                   else f"{100 * ev['context_by_snr'][s_]['FHSS']['with_jammer']['recall']:>10.1f}% ")
                  for s_ in snrs)
    print(f"  {'':<12}{row}")


def print_compare(new, base):
    print(f"\n{'=' * 100}\nSIDE BY SIDE   baseline ({base['name']}, {base['parameters']:,} params)  ->  "
          f"new ({new['name']}, {new['parameters']:,} params)\n{'=' * 100}")
    print(f"  {'class':<12}| {'precision':^22} | {'recall':^22} | {'F1':^22} | {'accuracy':^22}")
    print(f"  {'':<12}| {'base':>6}{'new':>7}{'delta':>8} | {'base':>6}{'new':>7}{'delta':>8} | "
          f"{'base':>6}{'new':>7}{'delta':>8} | {'base':>6}{'new':>7}{'delta':>8}")
    for cls in CLASSES:
        cells = []
        for key in ("precision", "recall", "f1", "accuracy"):
            bv, nv = base["per_class"][cls][key], new["per_class"][cls][key]
            cells.append(f"{100 * bv:>5.1f}%{100 * nv:>6.1f}%{_delta(nv, bv):>8}")
        star = " *" if cls in JUDGED else ""
        print(f"  {cls:<12}| " + " | ".join(cells) + star)
    print("  deltas are percentage points; * = judged class")

    print("\n  in company, recall (alone / with emitter / with jammer):  base -> new")
    for cls in CLASSES:
        cells = []
        for bucket in ("alone", "with_emitter", "with_jammer"):
            b_, n_ = _ctx(base, cls, bucket), _ctx(new, cls, bucket)
            cells.append("        n/a         " if b_ is None else f"{100 * b_:5.1f}% ->{100 * n_:5.1f}% ({_delta(n_, b_)})")
        print(f"  {cls:<12}" + "  ".join(cells))

    print("\n  comms vs jamming:  base -> new")
    for key in ("accuracy", "jamming_recall", "false_alarm_rate"):
        b_, n_ = base["comms_vs_jamming"][key], new["comms_vs_jamming"][key]
        print(f"    {key:<18} {100 * b_:8.3f}% -> {100 * n_:8.3f}%")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", action="append", default=None, help="trained checkpoint (repeat to average several)")
    ap.add_argument("--stft-freq-summary", action="store_true")
    ap.add_argument("--stft-keep-rows", action="store_true")
    ap.add_argument("--stft-dwell-feature", action="store_true",
                    help="branch fix_radar-fhss-confusion: model.stft_dwell_feature")
    ap.add_argument("--name", default="experiment")
    ap.add_argument("--margin", type=float, default=0.03, help="safety margin above the benchmark when calibrating")
    ap.add_argument("--out", type=Path, default=None, help="evaluation JSON (default results/eval_<name>.json)")
    ap.add_argument("--thresholds-out", type=Path, default=None, help="calibrated thresholds JSON for the probes")
    ap.add_argument("--baseline", type=Path, default=None, help="baseline evaluation JSON to compare with")
    ap.add_argument("--limit", type=int, default=None, help="debug: random subset of N windows per split")
    ap.add_argument("--verdict-only", action="store_true", help="skip evaluation; judge existing JSON files")
    ap.add_argument("--eval-json", type=Path, default=None, help="(verdict-only) the new run's evaluation JSON")
    ap.add_argument("--high-snr", type=Path, default=None)
    ap.add_argument("--baseline-high-snr", type=Path, default=None)
    ap.add_argument("--jsr", type=Path, default=None)
    ap.add_argument("--baseline-jsr", type=Path, default=None)
    args = ap.parse_args()

    def load(path):
        return None if path is None else json.loads(Path(path).read_text())

    if args.verdict_only:
        assert args.eval_json and args.baseline, "--verdict-only needs --eval-json and --baseline"
        new, base = load(args.eval_json), load(args.baseline)
        ok = print_verdict(judge(new, base, load(args.high_snr), load(args.baseline_high_snr),
                                 load(args.jsr), load(args.baseline_jsr)))
        sys.exit(0 if ok else 1)

    assert args.checkpoint, "--checkpoint is required (or use --verdict-only)"
    CFG.setdefault("model", {})["stft_freq_summary"] = bool(args.stft_freq_summary)
    CFG["model"]["stft_keep_rows"] = bool(args.stft_keep_rows)
    CFG["model"]["stft_dwell_feature"] = bool(args.stft_dwell_feature)

    from src.train import load_data, stratified_split

    X, y, snr = load_data()
    d = CFG["dataset"]
    _, val_idx, test_idx = stratified_split(y, snr, d["val_frac"], d["test_frac"], d["seed"])
    if args.limit:
        rng = np.random.default_rng(0)
        val_idx = rng.choice(val_idx, min(args.limit, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, min(args.limit, len(test_idx)), replace=False)

    models, device = load_models(args.checkpoint, X.shape[-1])
    n_params = sum(p.numel() for p in models[0].parameters())
    print(f"evaluating {args.name}: {len(models)} model(s), {n_params:,} parameters, device {device}")

    print("predicting validation split ...")
    probs_val = predict(models, device, X[val_idx])
    cal = calibrate(probs_val, y[val_idx].astype(int), args.margin)
    thr_vec = np.array([cal[c]["threshold"] for c in CLASSES], dtype=np.float32)

    print("predicting test split ...")
    probs_test = predict(models, device, X[test_idx])
    result = compute_metrics(y[test_idx], probs_test, thr_vec, snr[test_idx])
    result.update({
        "name": args.name, "checkpoints": [str(c) for c in args.checkpoint], "parameters": n_params,
        "flags": {"stft_freq_summary": bool(args.stft_freq_summary), "stft_keep_rows": bool(args.stft_keep_rows),
                 "stft_dwell_feature": bool(args.stft_dwell_feature)},
        "margin": args.margin, "thresholds": cal, "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
    })

    out = args.out or ROOT / "results" / f"eval_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_py(result), indent=2))
    thr_out = args.thresholds_out or ROOT / "results" / f"thresholds_{args.name}.json"
    thr_out.write_text(json.dumps({c: cal[c]["threshold"] for c in CLASSES}, indent=2))

    print_report(result)
    print(f"\nwrote {out}\nwrote {thr_out}   <- pass this to the probes with --thresholds")

    if args.baseline:
        base = load(args.baseline)
        print_compare(result, base)
        print_verdict(judge(result, base))
        print("\n(Only the test-split checks are judged here. Run the probes with the thresholds file above, "
              "then re-run with --verdict-only to add the probe checks.)")


if __name__ == "__main__":
    main()
