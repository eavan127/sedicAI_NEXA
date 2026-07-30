# 07 — Evaluation & Benchmark

**Owner:** Person D · **Day:** 3

## The number that decides everything

> *"Must achieve an Accuracy/Recall of > 90% specifically on the High Priority
> (Military/CEMA) and Jamming classes."*

Not overall accuracy — **per-class recall on LFM_RADAR, FHSS, JAMMING**.

**Recall** = of all the real instances of a class, what fraction did we catch?
It penalises misses, not false alarms — the right priority for threat detection,
where failing to flag a hostile emitter is worse than a false warning.

A model can post 85% overall accuracy and still fail if one judged class sits at
70%. Watch the three, not the average.

## Tools

| Tool | Use | Licence |
|---|---|---|
| [scikit-learn](https://scikit-learn.org/) | `classification_report`, `confusion_matrix` | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org/) | Confusion matrix, accuracy-vs-SNR plots | PSF/BSD-compatible |
| PyTorch | Inference | BSD-3-Clause |
| NumPy | Metric slicing by SNR | BSD-3-Clause |

## Running

```bash
python -m src.evaluate
```

Evaluates on the **held-out test split** — never train or validation.

## Outputs (`evals/`)

| File | Contents |
|---|---|
| `scorecard.json` | Per-class metrics + explicit PASS/FAIL per judged class |
| `confusion_matrix.png` | Which classes get confused with which |
| `accuracy_vs_snr.png` | Overall + per-judged-class accuracy vs SNR, with the 90% line drawn |

Console prints a plain verdict:

```
--- Benchmark (>90% recall on judged classes) ---
  LFM_RADAR    recall=0.9800  PASS
  FHSS         recall=0.9100  PASS
  JAMMING      recall=0.9400  PASS
  OVERALL: PASS
```

No ambiguity about whether we cleared the bar — and `scorecard.json` is
copy-paste evidence for the brief.

## Reading the confusion matrix

Rows = truth, columns = prediction. Off-diagonal cells are the failures.

Check in this order:

1. **Sweep-jamming ↔ LFM radar** — both are chirps; the predicted trouble spot
   (see [04](04-jamming-generation.md))
2. **Jamming ↔ noisy civilian** — barrage jamming is broadband noise; at low SNR
   a noisy QAM signal can look similar
3. **FHSS ↔ anything** — should be cleanly separable; confusion here suggests a
   generation problem, not a model problem

## Reading accuracy-vs-SNR

The expected shape is a curve that falls off at low SNR — that is normal and
well documented in the AMC literature, not a defect. What matters:

- **Where does it cross 90%?** Say so explicitly in the brief: *"clears the
  benchmark above X dB."* That is an honest, quantified claim.
- **A flat line near 100% across all SNRs is a red flag**, not a triumph — it
  usually means the classes are separable by some artefact (length, amplitude,
  padding) rather than by signal content. Investigate before celebrating.

That second point matters for us specifically: our synthetic classes come from
different pipelines than RadioML/RadChar, so an artefact leak is plausible. See
[05](05-preprocessing.md).

## Two headline metrics beyond the benchmark

`evaluate.py` reports both automatically.

### Comms vs hostile CEMA

The rules single this out for higher technical scores:

> *"Models that can successfully distinguish between standard communication
> signals and hostile CEMA interference (e.g., RF Jamming)"*

That is a **binary** task, so it is reported as its own number rather than left
buried in a 7×7 confusion matrix — discrimination accuracy, jamming recall, and
false alarm rate (civilian wrongly flagged as jamming). Put it in the brief using
the rules' own vocabulary; a panel should not have to extract it.

### Coarse tier — Civilian / Military / Hostile

Not all confusion is equal:

| Confusion | Verdict |
|---|---|
| 16QAM ↔ 64QAM | Harmless — the operational call ("ordinary traffic") is still right |
| Civilian ↔ Jamming | **Serious** — a false alarm |
| Radar ↔ Sweep jamming | **Serious** — different threat types |

Coarse-tier accuracy captures that distinction, and it is the false-alarm story
that matters operationally. Expect it to be higher than the 7-class number.

## Prove generalisation — the test that matters most

Our judged classes are synthetic or third-party. A good score on our own split
only shows we learned our own data.

**Train on one parameter subset, evaluate on a disjoint one.** For example, train
on radar PRIs of 1–5 ms and test on 6–10 ms. If accuracy holds, the model learned
*"chirp"*; if it collapses, we have found the fatal flaw ourselves rather than
having the organisers find it.

Report this as a **held-out parameter generalisation test**. It costs one extra
training run and directly answers the question a technical panel is silently
asking about synthetic training data.

## Ablation table — measure the design, don't assert it

Retrain with one component removed at a time:

| Configuration | Radar recall | FHSS recall | Jamming recall |
|---|---|---|---|
| Full pipeline | | | |
| − class weighting | | | |
| − augmentation | | | |
| − SNR sweep (high-SNR only) | | | |

A handful of short training runs, and it demonstrates every choice was measured.

## Cheap accuracy, once the model is final

Only after the gate is cleared — these are polish, not fixes:

| Technique | Cost | Notes |
|---|---|---|
| **Overlapping inference windows** | compute only | `infer.py --stride 512`; catches bursts split across a boundary |
| **Test-time augmentation** | a few lines | Average predictions over random phase rotations — label-preserving, so mathematically free |
| **Seed ensemble** | 2–3 short runs | Average 3 models trained with different seeds |

## What to report in the brief

- Per-class precision/recall/F1 table
- Confusion matrix figure
- Accuracy-vs-SNR curve with the 90% line
- The SNR at which each judged class crosses 90%
- **Limitations**: FHSS and jamming are synthetic, validated by internal
  consistency tests rather than by an independent expert; parameter ranges come
  from general literature, not a specific emitter

That last bullet is not a weakness to hide. A team that names its own limitation
reads as more credible than one that claims 99% and cannot explain why.

## Definition of done

- [ ] `python -m src.evaluate` runs clean on the real dataset
- [ ] `scorecard.json` shows PASS on all three judged classes
- [ ] Both figures exported at brief/poster quality
- [ ] Confusion matrix inspected, not just generated
- [ ] Accuracy-vs-SNR shape sanity-checked (falls off at low SNR — not suspiciously flat)
