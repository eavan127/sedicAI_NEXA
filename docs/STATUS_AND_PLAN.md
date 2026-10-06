# SEDIC 2026 — Project Overwatch: Status & Next-Phase Plan

**Updated:** 6 Sept 2026 · **Team:** Eavan (radar + model/training), Eileen (civilian + data pipeline),
Jessy (FHSS + evaluation), Chua (jamming + inference/submission)
**Runway:** 7 days to the 13 Sept 2026 preliminary deadline (Top 10 announced 18 Sept, Grand Finale 7 Oct)

This replaces the original 4-day sprint playbook, which was written before the dataset existed and
targeted an outdated 90% bar. Both have moved — this reflects where the project actually stands today.

> **Corrections applied 6 Sept 2026.** Three claims in the 16 Aug version were wrong and have been
> fixed in place:
>
> 1. **There is no organiser-provided Qualifier IQ Data Stream.** Confirmed with the organiser; it
>    appears nowhere in either official PDF. There is no classification-log requirement either. All
>    training and evaluation data is our own. See [`rules/`](rules/) for the verbatim rules.
> 2. **The model is not a 1D-CNN.** It is a two-branch fusion network (IQ + STFT) with 148,938
>    parameters — see §2 below and the technical brief §4.
> 3. **The §2 results were from a superseded run.** Replaced with the shipped 5-model ensemble.

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

**Model & results** — a **two-branch fusion CNN, 148,938 parameters**, trained from scratch. An IQ
branch (one standard convolution plus two dilated convolutions, widening the receptive field from ~19
to ~43 samples) runs in parallel with an STFT branch (small 2-D CNN over the spectrogram); the two are
fused to 192 channels, pooled with energy-gated attention, and read out through a 192 → 256 → 8 head
with **independent sigmoid outputs**, because a window can carry more than one signal at once.

Shipped figures are the **5-model ensemble** on the held-out test split (19,260 windows), thresholds
calibrated on validation data only:

| Class | Recall | Precision | vs. 80% bar |
|---|---|---|---|
| LFM_RADAR | 85.00% | 0.54 | PASS, +5.0 pts |
| FHSS | 82.84% | 0.59 | PASS, +2.8 pts |
| JAMMING | 84.44% | 0.99 | PASS, +4.4 pts |

Comms-vs-jamming discrimination accuracy: **97.05%**. False alarm rate (civilian wrongly flagged as
jamming): **0.03%** — four windows out of 13,230.

Two qualifications that belong next to those numbers, not in a footnote:

- **The pass is a test-set average, not a per-condition guarantee.** At −10 dB all three judged
  classes fall below the bar: LFM_RADAR 67.8%, FHSS 58.6%, JAMMING 47.8%.
- **Radar and FHSS trade precision for recall** by design — at 0.54 and 0.59, roughly half their
  positive flags are false alarms. Deliberate: a missed emission is the costlier error against this
  benchmark. JAMMING does not share the trade-off.

**Known, understood weaknesses** — not blockers, but worth tracking:
- **Dense QAM.** 16QAM/64QAM are separated by a fixed statistical rule applied *after* the network, so
  their individual recalls are one dense-QAM decision split across two labels; the meaningful figure is
  their combined 65.58% over 7,290 windows. Precision is the real cost (0.43 and 0.46 — both record
  more false positives than true positives). The literature cross-check in brief §7.3 shows sub-0 dB
  dense-QAM recognition below 50% is normal even for dedicated classifiers, so this is a property of
  the problem, not a pipeline bug — and it doesn't touch the judged classes.
- **Jamming sub-types** on the shipped ensemble: sweep 95.7%, tone 82.3%, **barrage 69.3%** — barrage
  is the only sub-type that fails the bar on its own. FHSS is the dominant confusion for all three.
- **LFM_RADAR ↔ FHSS confusion**, concentrated in slow-hopping FHSS: 27.6% false-positive rate overall,
  falling from 39.6% at 25–50 kHz hop rates to 18.6% at 125–150 kHz, against 2.4% on pure noise. Hop
  rate drives it (correlation −0.200); channel count and spacing do not.
- **Retrain variance is larger than seed variance within a run.** Two complete retrains of the same
  ensemble on identical code and data gave **77.29% vs 84.05%** LFM_RADAR recall — the first would have
  failed the benchmark. This is random weight initialisation, not a defect, but it means the shipped
  figures come from one retained set of five checkpoints evaluated in a single run. **Any future
  retrain must be re-verified, never assumed to reproduce these numbers.**
- **Out-of-distribution recall is well below in-distribution.** Against an independently generated GNSS
  jamming dataset (Zenodo 4629685, MATLAB-simulated, not ours), never trained on: 57.6% recall at a 0.0% false-alarm rate. SingleChirp at 20.0% is
  the instructive case — our own synthetic sweep scores 95.7%, so the model learned some of the
  generator's signature rather than the general concept.

**Tooling** — smoke config for fast dry runs, full pytest suite (96 tests, all green), two working Colab
notebooks (fast single-model path and the fuller ensemble+variance+diagnostics path), a variance
measurement script that quantifies the seed-to-seed noise floor, and per-class/per-SNR/per-jamming-type
evaluation artifacts (JSON, PNG, and CSV for anyone who wants to chart them elsewhere).

**Not yet done**: video not recorded; submission form not filled; report §8 needs a proofread pass
(several broken sentences) before PDF export.

**Closed since 16 Aug**: ensemble + variance numbers recorded; technical brief drafted through §9;
team name finalised as **NEXA**; and the item previously listed here as "the single biggest untested
risk" — `infer.py`'s interleaved-float32 assumption about the organiser's stream — **is no longer a
risk at all, because no organiser stream exists.** The organiser confirmed the data is entirely our
own. Nothing needs to be verified against a file that will never arrive.

---

## 3. Plan for the next four weeks

### Week 1 (16–23 Aug) — lock down the evidence
- **Eavan**: run the ensemble (5 seeds) + variance measurement on Colab against the real dataset; record
  the actual worst-case numbers per judged class. This is the credibility evidence for the brief — "our
  worst observed run still clears 80%" is a much stronger claim than one lucky run.
- ~~**Chua**: draft the dtype-verification procedure for the Qualifier IQ Stream.~~ **Dropped 6 Sept** —
  there is no organiser stream to verify against. The effort went into out-of-distribution validation
  against an independently generated GNSS jamming dataset instead (brief §7.2), which serves the same purpose better: it
  tests generalisation against data we did not produce.
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
- ~~**Verify the Qualifier Stream format assumption.**~~ **Done differently, 6 Sept** — no organiser
  stream exists. Generalisation is now tested against an independently generated GNSS jamming dataset instead (brief §7.2):
  57.6% recall, 0.0% false alarm, on data from outside this project entirely.
- ~~**Team name**, decided and used consistently.~~ **Done** — NEXA.

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
6. **Model architecture** — diagram, layer-by-layer explanation, parameter count, why two branches
   (IQ for amplitude/phase, STFT for time-frequency) rather than one representation doing both jobs.
7. **Training methodology** — loss (BCEWithLogits per class, inverse-frequency weights), low-SNR
   oversampling, optimizer, schedule, why the class-and-SNR-stratified split matters.
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

## 6. How fast is the model?

> **The old numbers here were withdrawn on 6 Sept 2026.** This section previously quoted **2.0 ms per
> prediction** and **~1,000 examples/second**, alongside a 4.26M-parameter 1D-CNN. Both the
> architecture and the parameter count were wrong, which means the timings were measured on a
> different network and say nothing about what we ship. **Do not quote them** in the brief, the video,
> or a Phase 2 pitch.

What we can state today, from `web/data/model_card.json` and the brief:

- **148,938 parameters** — small by any deep-learning standard, and small enough that the shipped
  five-model ensemble runs entirely **in a browser via ONNX**, which is the OMNI console.
- The browser demo is itself the real-time evidence, and it needs no benchmark number attached to it.

**If a timing figure is wanted** — for a Phase 2 "real-time capable" claim, say — re-measure it on the
shipped ensemble and record the conditions (CPU/GPU, thread count, batch size, single model vs.
ensemble). Until then the honest position is that we have not measured it on this architecture.

The architecture argument still holds and does not depend on a timing number: a Transformer's
self-attention cost grows quadratically with sequence length, so over a 512-sample window it would be
slower and hungrier to train while buying no accuracy the two-branch CNN doesn't already capture.
