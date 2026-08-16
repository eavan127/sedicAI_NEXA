# SEDIC 2026 — Project Overwatch: Status & Next-Phase Plan

**Updated:** 16 Aug 2026 · **Team:** Eavan (radar + model/training), Eileen (civilian + data pipeline),
Jessy (FHSS + evaluation), Chua (jamming + inference/submission)
**Runway:** 28 days to the 13 Sept 2026 preliminary deadline (Top 10 announced 18 Sept, Grand Finale 7 Oct)

This replaces the original 4-day sprint playbook, which was written before the dataset existed and
targeted an outdated 90% bar. Both have moved — this reflects where the project actually stands today.

---

## 1. What has actually changed since the original playbook

| Then | Now |
|---|---|
| Benchmark target: >90% recall | Official rules (RF track PDF, 11 Aug release): **>80%** recall on Military/CEMA + Jamming |
| Civilian loader: not started | Implemented — real RadioML data, all 4 civilian classes present |
| Dataset: didn't exist | 42,000 examples, 7 classes × 6,000, balanced, 6 SNR bins (−10 to +10 dB) |
| Model: not trained | Trained, evaluated, checkpoint exists |
| Timeline framing: 4 days | Actually ~4 weeks from today to submission |

---

## 2. Where we stand today

**Dataset** — complete and balanced. Civilian classes (BPSK/QPSK/16QAM/64QAM) come from RadioML 2018.01A;
LFM_RADAR blends real RadChar pulses with our generator; FHSS and JAMMING are fully synthetic. Everything
is standardized to 3.2 MHz / 512 samples (160 µs) per window, swept across −10, −6, −2, +2, +6, +10 dB.
Eileen's loader also caught and fixed a real bug: RadioML's shipped class-index file doesn't match the
actual data — worth stating in the brief, since it's a correctness issue most teams using this dataset
won't have caught.

**Model & results** — a small 1D-CNN (~4.26M parameters, two conv blocks + two linear layers), trained
from scratch. Latest full run on the real 7-class dataset:

| Class | Recall | vs. 80% bar |
|---|---|---|
| LFM_RADAR | 87.4% | PASS, +7.4 pts |
| FHSS | 84.3% | PASS, +4.3 pts |
| JAMMING | 86.3% | PASS, +6.3 pts |

Comms-vs-jamming discrimination accuracy: 96.2%. False alarm rate (civilian wrongly flagged as jamming):
1.3%. Coarse-tier civilian recall: 93.7% (the operational "is this ordinary traffic" call), even though
raw 16QAM/64QAM confusion is high — see limitations below.

**Known, understood weaknesses** — not blockers, but worth tracking:
- 16QAM/64QAM confuse heavily with each other. This is a well-documented, expected limit of automatic
  modulation classification at this window length/model size, not a pipeline bug, and it doesn't touch
  the judged classes.
- Jamming sub-type breakdown (probed via `diagnose_jamming.py`): sweep jamming is handled best (~92%),
  barrage (wideband noise) is the weakest (~72–82%). This is a named, specific target if anyone wants to
  spend time improving JAMMING recall further.
- **Single-seed results wobble.** A five-seed variance run showed JAMMING's worst-case recall sits only
  ~1.7 points above the 80% bar — thin, though still passing. The ensemble result (5 seeds averaged) is
  the safety margin, not the single-run number — this needs to be re-run and the numbers recorded (see
  Week 1 below).

**Tooling** — smoke config for fast dry runs, full pytest suite (96 tests, all green), two working Colab
notebooks (fast single-model path and the fuller ensemble+variance+diagnostics path), a variance
measurement script that quantifies the seed-to-seed noise floor, and per-class/per-SNR/per-jamming-type
evaluation artifacts (JSON, PNG, and CSV for anyone who wants to chart them elsewhere).

**Not yet done**: ensemble+variance numbers not yet recorded from a run on the real dataset; technical
brief not started; video not recorded; submission form not filled; team name not finalized; and — the
single biggest untested risk — `infer.py`'s assumption about the organizer's Qualifier IQ Stream file
format (interleaved float32) has never been checked against a real file, because the organizers haven't
released it yet.

---

## 3. Plan for the next four weeks

### Week 1 (16–23 Aug) — lock down the evidence
- **Eavan**: run the ensemble (5 seeds) + variance measurement on Colab against the real dataset; record
  the actual worst-case numbers per judged class. This is the credibility evidence for the brief — "our
  worst observed run still clears 80%" is a much stronger claim than one lucky run.
- **Chua**: draft the dtype-verification procedure for the Qualifier IQ Stream now, before the file
  exists, so it's ready to run the moment it's released (spectrogram check, sample-count check, amplitude
  sanity check — same steps as the original playbook's P4 section, still valid).
- **Everyone**: skim `docs/pipeline/` and this document once — no need to re-read code, the reasoning is
  already written down.
- **Decide as a team**: is anyone spending time on the QAM confusion or barrage-jamming weakness, or are
  we treating both as named, accepted limitations? Either is defensible — just decide deliberately instead
  of by default.

### Week 2 (24–31 Aug) — write the brief, script the video
- Draft the technical brief in full (structure in section 5 below). Each person drafts the section for
  the part of the system they own — Eileen writes data/dataset, Eavan writes architecture/training, Jessy
  writes evaluation/results, Chua assembles and writes the submission/limitations sections.
- Draft the video script, sized to the suggested split below.
- If Week 1's decision was to improve something: try it now, and re-run `measure_variance.py` before
  believing any result — a 2-point change is meaningless against a 4-point noise floor.

### Week 3 (1–7 Sept) — record and assemble
- Record and edit the video (≤5 min, everyone narrates their own part — it's both easier to record and
  more convincing than one person explaining work they didn't do).
- Finalize the brief PDF: proofread, add the team name, confirm dataset citations/licenses are present.
- Full dry run of the actual submission: Google Form, PDF attachment, YouTube link, all in one sitting,
  timed.

### Week 4 (8–13 Sept) — buffer
- Submit with days to spare, not hours.
- Freeze the checkpoint and code once submitted — no further changes.

---

## 4. Things that would make the project more well-rounded

Not urgent fixes — the model already clears the bar — but each of these strengthens the submission on its
own terms:

- **Ensemble + variance numbers in the brief**, not just a single run (see Week 1).
- **Named limitations, stated plainly**: 16QAM/64QAM confusion, synthetic FHSS/jamming validated only by
  internal consistency tests, RadioML's single symbol rate. A team that names its own limits reads as more
  credible than one claiming a suspiciously clean 99%.
- **Held-out parameter generalization test**: train on one radar PRI range, evaluate on a disjoint one. If
  accuracy holds, the model learned "chirp," not "our specific training distribution" — one extra training
  run, and it pre-empts the question a technical panel is likely to ask about synthetic training data.
- **Ablation table**: retrain with one design choice removed at a time (class weighting, augmentation, SNR
  sweep) and report the recall shift for each. Demonstrates the choices were measured, not asserted.
- **Comms-vs-jamming metric front and center** — the rules explicitly call this out for higher technical
  scores, so it shouldn't be buried inside a 7×7 confusion matrix.
- **Verify the Qualifier Stream format assumption** the moment it's released — this is the one step in the
  entire pipeline that has never been tested against anything real.
- **Team name**, decided and used consistently across the brief, video, and form.

---

## 5. Technical report — structure and how to write it

This is the standard shape a technical report/brief in this kind of engineering competition takes. Most
of the content already exists somewhere in `docs/` — assembling is most of the work, not writing from
scratch.

1. **Cover** — team name, project name ("Project Overwatch"), track, date.
2. **Executive summary** — 3–5 sentences: what was built, the headline result (recall numbers against the
   80% bar), stated plainly.
3. **Problem statement** — what the challenge requires, in your own words, citing the rules.
4. **Dataset & methodology** — sources, licenses (RadioML/RadChar, both CC BY-NC-SA 4.0), class counts,
   SNR coverage, how the RadioML class-order bug was found and fixed, how synthetic classes were
   validated (spectrogram comparison against literature).
5. **Signal processing / preprocessing** — windowing (512 samples, 3.2 MHz, 160 µs), normalization, why
   the format contract (same shape/statistics regardless of class or source) matters.
6. **Model architecture** — diagram, layer-by-layer explanation, parameter count, why a 1D-CNN over
   alternatives (see section 6 below for the speed argument).
7. **Training methodology** — loss (class-weighted cross-entropy), optimizer, schedule, why the
   class-and-SNR-stratified split matters.
8. **Evaluation & results** — per-class precision/recall/F1 table, confusion matrix, accuracy-vs-SNR
   curve, the benchmark scorecard stated in exact numbers against the exact 80% threshold — never round up
   or use vague language like "very high accuracy."
9. **Discussion & limitations** — named honestly (section 4 above has the list).
10. **Conclusion & roadmap** — what's next if selected for Phase 2; the original playbook's idea of a
    roadmap paragraph (RadChar's regression labels — pulse width, PRI, pulse count — as a path to emitter
    fingerprinting, with ELINT/attribution named as explicitly out of scope) is still a good, low-cost way
    to show awareness of the wider problem without overclaiming.
11. **References** — datasets, licenses, any literature cited for parameter ranges.
12. **Appendix** (optional) — ablation table, extra figures.

**How to actually execute it**: draft incrementally through Weeks 1–2, not in one sitting at the end —
that's the point of assigning sections by ownership now. State every number precisely and say what it's
being compared against. Every claim about noise robustness should point at the accuracy-vs-SNR figure,
not just be asserted in prose.

---

## 6. Which model is faster?

Measured directly on this machine (CPU, single thread, no GPU):

- **2.0 ms** per single-example prediction
- **~1,000 examples/second** when batched

The model is small by deep-learning standards — about 4.26 million parameters, two convolutional blocks
and two linear layers. That's the direct payoff of the 1D-CNN choice over a Transformer-style
architecture: a Transformer's self-attention cost grows quadratically with sequence length, so it would be
slower per prediction, slower and more data-hungry to train, and — for a signal window this short (512
samples) — buys no accuracy the CNN doesn't already capture. This is also useful evidence for the
"real-time capable" claim relevant to a Phase 2 live demo: on GPU or with batched inference over a
streaming input, this model comfortably keeps up with the 160 µs windows it classifies.
