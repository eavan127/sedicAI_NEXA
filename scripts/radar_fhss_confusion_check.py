"""Judge a stft_dwell_feature run on the ONE problem it targets: LFM_RADAR
precision, driven by radar-called-FHSS confusion -- not the FHSS-under-jammer
masking checks in evaluate_experiment.py's --verdict-only, which are
pre-registered for a different experiment (C2 / stft_keep_rows) and will
happily PASS a run that does nothing for this problem, as the seed-2000
'dwell' run did: it passed all 10 of those checks while LFM_RADAR precision
went DOWN 1.3 points and the radar-called-FHSS fraction went UP 3.4 points.

Reads the eval JSON(s) `evaluate_experiment.py` already writes
(results/eval_<name>.json) -- nothing here re-runs inference.

Usage:
    python scripts/radar_fhss_confusion_check.py \
        --eval-json results/eval_dwell.json \
        --eval-json results/eval_dwell_seed2001.json \
        --baseline docs/experiments/c2_baseline_eval.json

Multiple --eval-json means multiple seeds of the SAME cell: reports the
range as well as the mean, since a single seed's move can be noise (single
runs on this codebase wobble up to about 6 points on related metrics --
docs/POST_STAGE1_FIXES.md, E5 baselines section).
"""
import argparse
import json
from pathlib import Path


def _confusion(ev):
    """LFM_RADAR's false positives that are really FHSS, as a fraction --
    the number this whole experiment exists to bring down from 46%."""
    return ev["radar_fhss_confusion"]["LFM_RADAR_fp_that_are_true_FHSS"]["fraction_that_are_true_FHSS"]


def summarize(evals, baseline):
    rows = []
    for ev in evals:
        rows.append({
            "lfm_radar_precision": ev["per_class"]["LFM_RADAR"]["precision"],
            "lfm_radar_recall": ev["per_class"]["LFM_RADAR"]["recall"],
            "fhss_precision": ev["per_class"]["FHSS"]["precision"],
            "radar_called_fhss": _confusion(ev),
            "seed": ev.get("checkpoints", ["?"])[0],
        })
    base = {
        "lfm_radar_precision": baseline["per_class"]["LFM_RADAR"]["precision"],
        "lfm_radar_recall": baseline["per_class"]["LFM_RADAR"]["recall"],
        "fhss_precision": baseline["per_class"]["FHSS"]["precision"],
        "radar_called_fhss": _confusion(baseline),
    }
    return rows, base


def _fmt_pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-json", action="append", required=True,
                    help="results/eval_<name>.json from evaluate_experiment.py; repeat per seed")
    ap.add_argument("--baseline", type=Path, default=Path("docs/experiments/c2_baseline_eval.json"))
    args = ap.parse_args()

    evals = [json.loads(Path(p).read_text()) for p in args.eval_json]
    baseline = json.loads(args.baseline.read_text())
    rows, base = summarize(evals, baseline)

    print(f"\n{'=' * 88}\nRADAR / FHSS CONFUSION CHECK  ({len(rows)} run(s) vs {args.baseline.name})\n{'=' * 88}")
    print(f"{'':>14}{'LFM_RADAR prec':>16}{'LFM_RADAR rec':>16}{'FHSS prec':>12}{'radar->FHSS':>14}")
    print(f"{'baseline':>14}{_fmt_pct(base['lfm_radar_precision']):>16}"
          f"{_fmt_pct(base['lfm_radar_recall']):>16}{_fmt_pct(base['fhss_precision']):>12}"
          f"{_fmt_pct(base['radar_called_fhss']):>14}")
    for i, r in enumerate(rows):
        print(f"{'run ' + str(i):>14}{_fmt_pct(r['lfm_radar_precision']):>16}"
              f"{_fmt_pct(r['lfm_radar_recall']):>16}{_fmt_pct(r['fhss_precision']):>12}"
              f"{_fmt_pct(r['radar_called_fhss']):>14}")

    print("\nLFM_RADAR prec  -- the 51% this experiment targets. Must go UP.")
    print("LFM_RADAR rec   -- guardrail: must not drop below 80% (the benchmark).")
    print("FHSS prec       -- guardrail: fixing radar must not just push the confusion onto FHSS.")
    print("radar->FHSS     -- the mechanism: share of radar's false positives that are true FHSS. Must go DOWN from 46%.")

    prec_deltas = [100 * (r["lfm_radar_precision"] - base["lfm_radar_precision"]) for r in rows]
    conf_deltas = [100 * (r["radar_called_fhss"] - base["radar_called_fhss"]) for r in rows]
    mean_prec_delta = sum(prec_deltas) / len(prec_deltas)
    mean_conf_delta = sum(conf_deltas) / len(conf_deltas)

    print(f"\nmean LFM_RADAR precision delta: {mean_prec_delta:+.1f} points"
          + (f"  (range {min(prec_deltas):+.1f} to {max(prec_deltas):+.1f})" if len(rows) > 1 else "  (single run -- confirm with a second seed before trusting the direction)"))
    print(f"mean radar->FHSS delta:         {mean_conf_delta:+.1f} points"
          + (f"  (range {min(conf_deltas):+.1f} to {max(conf_deltas):+.1f})" if len(rows) > 1 else "  (single run -- confirm with a second seed before trusting the direction)"))

    # No formal pre-registered bar exists for this feature yet (unlike C2's
    # FHSS-under-jammer +10-point bar) -- this is a plain, honest read of the
    # two numbers the hypothesis makes a claim about, not a PASS/FAIL gate.
    if mean_prec_delta > 1.0 and mean_conf_delta < -1.0:
        verdict = "PROMISING -- both numbers move the right way, outside single-run noise. Worth a full run."
    elif mean_prec_delta < -1.0 or mean_conf_delta > 1.0:
        verdict = "NOT WORKING -- precision and/or the confusion moved the wrong way."
    else:
        verdict = "INCONCLUSIVE -- both deltas are inside single-run noise (~6 points on related metrics). Need more seeds."
    print(f"\nREAD: {verdict}\n")


if __name__ == "__main__":
    main()
