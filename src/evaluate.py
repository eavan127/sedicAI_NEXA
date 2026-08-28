"""
Evaluation: per-class recall, confusion matrix, accuracy-vs-SNR curve, and a
scorecard that states plainly whether the organiser's benchmark
(`configs/default.yaml: benchmark_recall`) is met.

--ensemble evaluates the AVERAGE of ensemble_0.pt..ensemble_{n-1}.pt
(train_ensemble.py's checkpoints) instead of the single best_model.pt --
use this to get the full 8-class report (accuracy-vs-SNR, confusion matrix,
comms-vs-jamming) for the ensemble actually being submitted, not just the
judged-class recall/precision train_ensemble.py's own scorecard reports.

Usage:
    python -m src.evaluate
    python -m src.evaluate --ensemble --n-models 5
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, multilabel_confusion_matrix

from src.config import (CFG, CLASSES, CLASS_TO_IDX, REPO_ROOT, TIERS,
                         resolve_multilabel_thresholds)
from src.models.amc_cnn import AMC_CNN
from src.train import load_data, stratified_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The official rules (SEDIC 2026 RF track, 11 Aug public release) require
# >80% accuracy on the High Priority (Military/CEMA) and Jamming classes.
# Earlier drafts of our docs targeted 90% as a stricter internal bar; keep
# reporting against the actual published number so the brief states the real
# margin, not a self-imposed one.
BENCHMARK_RECALL = 0.80

# Coarse tiers. The 7-class number is what the gate is scored on, but the tier
# call is what matters operationally: mistaking a distant phone for an attack
# (civilian -> hostile) is a false alarm, whereas confusing 16QAM with 64QAM at
# low SNR still yields the correct decision, "this is ordinary traffic".
# TIERS now lives in src/config.py and is re-exported above, so importing it
# no longer drags matplotlib and sklearn into dependency-light callers like
# src/timeline.py. Existing `from src.evaluate import TIERS` call sites keep
# working unchanged.


def _tier_of(class_name):
    for tier, members in TIERS.items():
        if class_name in members:
            return tier
    raise KeyError(f"{class_name} is in no tier — update TIERS")


TIER_CLASS_IDX = {t: np.array([CLASS_TO_IDX[c] for c in members]) for t, members in TIERS.items()}


def coarse_tier_metrics(y_true, y_pred):
    """Presence-based recall per tier, over multi-hot (N, num_classes) arrays.

    A tier counts as "present" in a window when ANY of its member class bits
    is 1 -- independently per tier, so a JAMMING-overlaid-on-LFM_RADAR window
    correctly reads as BOTH Military and Hostile present, not forced into
    picking one. This replaces the old single "predicted tier" lookup, which
    assumed exactly one class (and therefore exactly one tier) per window --
    an assumption composite examples break.

    "accuracy" here is tier-bit (Hamming) accuracy across all four tiers and
    all examples, analogous to train.py's val_bit_acc -- not "did every tier
    call in this window match", which would be a much harsher subset-exact
    metric.
    """
    tier_names = list(TIERS)
    presence_true = np.stack([y_true[:, TIER_CLASS_IDX[t]].any(axis=1) for t in tier_names], axis=1)
    presence_pred = np.stack([y_pred[:, TIER_CLASS_IDX[t]].any(axis=1) for t in tier_names], axis=1)

    out = {"accuracy": float((presence_true == presence_pred).mean()), "per_tier_recall": {}}
    for i, t in enumerate(tier_names):
        m = presence_true[:, i]
        out["per_tier_recall"][t] = float(presence_pred[m, i].mean()) if m.any() else None
    return out


def comms_vs_jamming(y_true, y_pred):
    """The metric the rules single out for 'significantly higher technical scores':

        "Models that can successfully distinguish between standard communication
         signals and hostile CEMA interference (e.g., RF Jamming)"

    Reported as its own headline number so the panel does not have to dig it out
    of the per-class table.

    Presence-based: "is jamming present" is now an independent yes/no bit, not
    "which single class was picked" -- a civilian window that is ALSO jammed
    (a composite example) correctly counts as a civilian example that IS
    jammed, evaluated on whether the JAMMING bit was caught, rather than
    forcing an either/or choice that composite data no longer represents.
    """
    civ = np.array([CLASS_TO_IDX[c] for c in TIERS["Civilian"]])
    jam = CLASS_TO_IDX["JAMMING"]

    is_civilian = y_true[:, civ].any(axis=1)
    true_is_jam = y_true[:, jam] == 1
    mask = is_civilian | true_is_jam
    if not mask.any():
        return None

    true_is_jam_m = true_is_jam[mask]
    pred_is_jam_m = y_pred[mask, jam] == 1

    tp = int((true_is_jam_m & pred_is_jam_m).sum())
    fn = int((true_is_jam_m & ~pred_is_jam_m).sum())
    fp = int((~true_is_jam_m & pred_is_jam_m).sum())

    return {
        "accuracy": float((true_is_jam_m == pred_is_jam_m).mean()),
        "jamming_recall": tp / (tp + fn) if tp + fn else None,
        "false_alarm_rate": fp / int((~true_is_jam_m).sum()) if (~true_is_jam_m).any() else None,
        "n_evaluated": int(mask.sum()),
    }


def recall_in_context(y_true, y_pred):
    """Per-class recall split by what ELSE is present in the same window --
    "alone", "with another emitter" (no jammer), and "with a jammer" -- instead
    of one number that averages over all three.

    That average is not a neutral summary, it is actively misleading. Roughly
    a third of test windows are composites, and standalone recall is even
    across every class -- the damage is entirely concentrated "in company",
    and each class fails in a DIFFERENT way there: some survive a benign
    second emitter and only collapse under a jammer, others collapse under
    any company at all. A single per-class recall figure blends a class that
    is actually fine alone with whatever is happening in company and reports
    the blend as if it were one homogeneous number -- which has previously
    made at least one class look like an off-trend anomaly in the overall
    column when its standalone performance was unremarkable and the entire
    story was "in company". This is reported for EVERY class, not just the
    judged ones -- the judged classes have exactly the same alone/company
    split hiding inside their single recall number, and nobody had looked.

    Buckets (mutually exclusive, by presence in the true multi-hot row):
      - "alone": exactly one positive label in the window (the class itself).
      - "with_emitter": more than one positive label, and JAMMING is NOT one
        of them -- company from another comms/military emitter.
      - "with_jammer": JAMMING IS one of the positive labels, and the class
        under test is not JAMMING itself.

    JAMMING's own "with_jammer" bucket is meaningless (a jammer can't be "in
    company with a jammer" relative to itself, since there is only one
    JAMMING class) and reads as an empty bucket, not a folded-in zero --
    JAMMING co-occurring with another emitter lands in "with_emitter" instead,
    since from JAMMING's point of view that IS just "another emitter present".

    A bucket can legitimately be empty (JAMMING's "with_jammer", and every
    bucket but "alone" for NOISE_FLOOR, which by construction of the mixture
    generator never co-occurs with anything). An empty bucket's recall is
    None -- never 0.0 or NaN -- so a reader cannot mistake "this situation
    does not occur in the data" for "the model failed at it every time".

    The support count travels with every recall for the same reason: the
    buckets are NOT equally sized (which classes end up "with a jammer" at
    all, and how often, is purely a function of which mixture combinations
    exist in configs/default.yaml), so two classes' "with_jammer" recall
    numbers are not directly comparable difficulty-for-difficulty unless the
    reader can also see how much evidence backs each one.
    """
    jam = CLASS_TO_IDX["JAMMING"]
    has_jam = y_true[:, jam] == 1
    n_pos = y_true.sum(axis=1)

    def _bucket(mask, idx):
        support = int(mask.sum())
        recall = float(y_pred[mask, idx].mean()) if support else None
        return {"recall": recall, "support": support}

    out = {}
    for cls in CLASSES:
        idx = CLASS_TO_IDX[cls]
        is_pos = y_true[:, idx] == 1
        # For JAMMING itself there is no "other jammer" to be in company
        # with -- its own presence bit IS the global jamming bit, so using
        # `has_jam` unmodified would make "with_emitter" vacuously empty too.
        jam_present_other = has_jam if cls != "JAMMING" else np.zeros(len(y_true), dtype=bool)

        alone_mask = is_pos & (n_pos == 1)
        with_emitter_mask = is_pos & (n_pos > 1) & (~jam_present_other)
        with_jammer_mask = is_pos & jam_present_other

        out[cls] = {
            "alone": _bucket(alone_mask, idx),
            "with_emitter": _bucket(with_emitter_mask, idx),
            "with_jammer": _bucket(with_jammer_mask, idx),
        }
    return out


def confusion_between(y_true, y_pred, class_a, class_b):
    """Among class_a's false positives (predicted present, truly absent), what
    fraction are windows where class_b is truly present?

    A high fraction is direct evidence the model is substituting class_b for
    class_a -- not just "unsure", but specifically reaching for the other
    class -- which points at a targeted fix (e.g. separating the two in the
    composite/overlay examples where both can co-occur) rather than more
    generic loss-weight tuning. A low fraction means class_a's false
    positives are spread across everything else, and this pairing isn't the
    story.
    """
    a, b = CLASS_TO_IDX[class_a], CLASS_TO_IDX[class_b]
    fp_mask = (y_pred[:, a] == 1) & (y_true[:, a] == 0)
    n_fp = int(fp_mask.sum())
    if n_fp == 0:
        return None
    overlap = int((y_true[fp_mask, b] == 1).sum())
    return {"false_positives": n_fp, "fraction_that_are_true_" + class_b: overlap / n_fp}


EVAL_BATCH_SIZE = 256


def _predict_probs_one(model, X):
    """Batched, not one giant forward pass -- X can be the whole test split
    (thousands of windows). A single unbatched call tries to materialize
    every intermediate activation (attention pooling's (batch, 193, time)
    tensor especially) for the entire split at once, which OOMs on a real
    GPU once the dataset is full-sized rather than a smoke/dry-run subset."""
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), EVAL_BATCH_SIZE):
            batch = torch.tensor(X[i:i + EVAL_BATCH_SIZE]).to(DEVICE)
            chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _predict_probs(models, X):
    """Average sigmoid probabilities over one or more models -- same
    averaging train_ensemble.py's _predict uses, so this evaluation matches
    what's actually submitted when given the 5 ensemble checkpoints."""
    summed = None
    for model in models:
        p = _predict_probs_one(model, X)
        summed = p if summed is None else summed + p
    return summed / len(models)


def evaluate(ensemble=False, n_models=5):
    X, y, snr_labels = load_data()
    d = CFG["dataset"]
    _, _, test_idx = stratified_split(y, snr_labels, d["val_frac"], d["test_frac"], d["seed"])

    X_test, y_test, snr_test = X[test_idx], y[test_idx], snr_labels[test_idx]

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

    models = []
    for p in ckpt_paths:
        model = AMC_CNN(num_classes=len(CLASSES), input_len=X.shape[-1]).to(DEVICE)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()
        models.append(model)
    print(f"Evaluating: {ckpt_desc}\n")

    thresholds = resolve_multilabel_thresholds()

    probs = _predict_probs(models, X_test)
    preds = (probs > thresholds).astype(int)   # thresholds broadcasts (8,) against probs (N, 8)
    y_test = y_test.astype(int)

    evals_dir = REPO_ROOT / CFG["paths"]["evals"]
    evals_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_test, preds, labels=range(len(CLASSES)),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    print(classification_report(y_test, preds, labels=range(len(CLASSES)),
                                 target_names=CLASSES, zero_division=0))

    # Scorecard — the number the judges actually check
    scorecard = {"benchmark_recall": BENCHMARK_RECALL, "judged_classes": {}, "passed": True}
    for cls in CFG["judged_classes"]:
        recall = report[cls]["recall"]
        passed = recall >= BENCHMARK_RECALL
        scorecard["judged_classes"][cls] = {"recall": recall, "passed": bool(passed)}
        scorecard["passed"] &= bool(passed)

    coarse = coarse_tier_metrics(y_test, preds)
    cvj = comms_vs_jamming(y_test, preds)
    ric = recall_in_context(y_test, preds)

    # LFM_RADAR and FHSS are the two classes needing the most aggressive
    # threshold/loss-weight help (see configs/default.yaml) -- check whether
    # that's because they're specifically confused with EACH OTHER (fixable
    # by separating them in the composite data) or just individually hard
    # (a different problem, threshold/weight tuning is closer to the right
    # tool for that).
    radar_fhss_confusion = {
        "LFM_RADAR_fp_that_are_true_FHSS": confusion_between(y_test, preds, "LFM_RADAR", "FHSS"),
        "FHSS_fp_that_are_true_LFM_RADAR": confusion_between(y_test, preds, "FHSS", "LFM_RADAR"),
    }

    with open(evals_dir / "scorecard.json", "w") as f:
        json.dump({"per_class": report, "benchmark": scorecard,
                    "coarse_tier": coarse, "comms_vs_jamming": cvj,
                    "radar_fhss_confusion": radar_fhss_confusion,
                    "recall_in_context": ric}, f, indent=2)

    # Flat CSVs alongside the JSON/PNG artifacts — Power BI (and Excel) read
    # CSV directly via Get Data > Text/CSV, no JSON connector needed. Same
    # numbers as scorecard.json, just reshaped for a BI tool instead of code.
    csv_dir = evals_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_dir / "per_class_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support", "is_judged_class"])
        for cls in CLASSES:
            r = report[cls]
            w.writerow([cls, r["precision"], r["recall"], r["f1-score"], r["support"],
                        cls in CFG["judged_classes"]])

    print(f"\n--- Benchmark (>{BENCHMARK_RECALL:.0%} recall on judged classes) ---")
    for cls, r in scorecard["judged_classes"].items():
        print(f"  {cls:<12} recall={r['recall']:.4f}  {'PASS' if r['passed'] else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if scorecard['passed'] else 'FAIL'}")

    # Headline numbers for the brief — see docs/WINNING_STRATEGY.md
    if cvj:
        print("\n--- Comms vs Hostile CEMA (the 'Competitive Advantage' criterion) ---")
        print(f"  discrimination accuracy : {cvj['accuracy']:.4f}")
        if cvj["jamming_recall"] is not None:
            print(f"  jamming recall          : {cvj['jamming_recall']:.4f}")
        if cvj["false_alarm_rate"] is not None:
            print(f"  false alarm rate        : {cvj['false_alarm_rate']:.4f}"
                  "   (civilian wrongly flagged as jamming)")

    print("\n--- LFM_RADAR / FHSS cross-confusion ---")
    for label, r in radar_fhss_confusion.items():
        if r is None:
            print(f"  {label}: no false positives to check")
        else:
            frac = next(v for k, v in r.items() if k.startswith("fraction"))
            print(f"  {label}: {frac:.1%} of {r['false_positives']} false positives")

    print("\n--- Coarse tier (Civilian / Military / Hostile) ---")
    print(f"  tier accuracy: {coarse['accuracy']:.4f}")
    for tier, rec in coarse["per_tier_recall"].items():
        print(f"    {tier:<10} recall={rec:.4f}" if rec is not None
              else f"    {tier:<10} recall=n/a")

    # Per-class recall split by company -- see recall_in_context's docstring
    # for why the overall column alone hides this. Every class, not just the
    # judged ones.
    def _cell(bucket):
        return "n/a" if bucket["recall"] is None else f"{bucket['recall']:.3f} (n={bucket['support']})"

    print("\n--- Recall by company: alone / with another emitter / with a jammer ---")
    print(f"  {'class':<12}{'alone':<18}{'with emitter':<18}{'with jammer':<18}")
    for cls, buckets in ric.items():
        print(f"  {cls:<12}{_cell(buckets['alone']):<18}"
              f"{_cell(buckets['with_emitter']):<18}{_cell(buckets['with_jammer']):<18}")

    # Confusion matrix — a single 8x8 no longer makes sense once more than one
    # class can be true in the same window (which of the 2 true classes would
    # a mismatched cell even mean?). One small present/absent 2x2 per class
    # instead, via sklearn's multilabel_confusion_matrix.
    mcm = multilabel_confusion_matrix(y_test, preds, labels=range(len(CLASSES)))
    ncols = 4
    nrows = int(np.ceil(len(CLASSES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    for i, cls in enumerate(CLASSES):
        ax = np.atleast_1d(axes).flat[i]
        ConfusionMatrixDisplay(mcm[i], display_labels=["absent", "present"]).plot(
            ax=ax, colorbar=False)
        ax.set_title(cls, fontsize=10)
    for j in range(len(CLASSES), nrows * ncols):
        np.atleast_1d(axes).flat[j].axis("off")
    plt.tight_layout()
    plt.savefig(evals_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Accuracy vs SNR: overall, every judged class (bold dashed — these are
    # the ones the >80% line applies to), and every civilian class (thin
    # solid — reported for completeness/mandatory-classification, not judged
    # against the benchmark line).
    unique_snrs = sorted(np.unique(snr_test))

    def _class_recall_by_snr(idx):
        """Recall of one class's bit at each SNR bin -- of windows at that
        SNR where this class is truly present (standalone or composite),
        what fraction did we correctly flag as present."""
        accs = []
        for s in unique_snrs:
            m = (snr_test == s) & (y_test[:, idx] == 1)
            accs.append(preds[m, idx].mean() if m.any() else np.nan)
        return accs

    # Same grid feeds the plot below and the CSV — compute once.
    accs_by_class = {cls: _class_recall_by_snr(CLASS_TO_IDX[cls]) for cls in CLASSES}
    with open(csv_dir / "accuracy_by_class_snr.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "snr_db", "recall", "is_judged_class"])
        for cls in CLASSES:
            for s, acc in zip(unique_snrs, accs_by_class[cls]):
                w.writerow([cls, s, acc, cls in CFG["judged_classes"]])

    plt.figure()
    plt.plot(unique_snrs, [(preds[snr_test == s] == y_test[snr_test == s]).mean()
                            for s in unique_snrs],
              marker="o", color="black",
              label=f"overall (per-class bit accuracy, all {len(CLASSES)} classes)", linewidth=2)
    for cls in CFG["judged_classes"]:
        plt.plot(unique_snrs, accs_by_class[cls],
                  marker=".", linestyle="--", linewidth=2, label=f"{cls} (judged)")
    for cls in CLASSES:
        if cls in CFG["judged_classes"]:
            continue
        plt.plot(unique_snrs, accs_by_class[cls],
                  marker="", linestyle="-", linewidth=1, alpha=0.6, label=cls)
    plt.axhline(BENCHMARK_RECALL, color="red", linestyle=":",
                label=f"{BENCHMARK_RECALL:.0%} benchmark (judged classes only)")
    plt.xlabel("SNR (dB)")
    # Per-class lines are RECALL (of windows where that class is truly
    # present, standalone or composite, how many were correctly flagged) --
    # the black "overall" line is per-class-bit ACCURACY (includes true
    # negatives too). Different quantities, same 0-1 axis; see legend.
    plt.ylabel("Accuracy / Recall")
    plt.title("Accuracy vs. SNR — all classes")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(evals_dir / "accuracy_vs_snr.png", dpi=150)
    plt.close()

    print(f"\nArtifacts written to {evals_dir} (JSON/PNG) and {csv_dir} (CSV, for Power BI/Excel)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", action="store_true",
                    help="evaluate the ensemble_*.pt average instead of best_model.pt")
    p.add_argument("--n-models", type=int, default=5)
    a = p.parse_args()
    evaluate(a.ensemble, a.n_models)
