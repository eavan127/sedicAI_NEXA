# Accuracy plan: what limits the 60-class accuracy, and what really helps

Branch `eavan-accuracy-plan` (on top of Eileen's SED-39 branch: spectrum branch + fixed FSK/PSK/QAM data).
Every recommendation below comes with a measurement. Ideas that did **not** help are listed too, so
nobody spends a training run on them again.

Measured on 7 Nov 2026 with seed 4 (`results/classifier/seed4_best.pt`), the local test set
(`data/combined`, version `3fae376e92d3`: single labels, so the labels-v2 twins still count as errors
here) and the full YOLO detector. Scripts: `scripts/probe_window_length.py`, `scripts/eval_pipeline.py`.

## 1. Error budget: where the accuracy goes

30,000 test windows; 24,006 have one emitter (the rest are mixtures).

| What happened to a window | Share | Fixable? |
|---|---|---|
| Correct | **46.2%** | |
| Wrong **family** | 30.2% | partly (sections 3.1-3.5) |
| Other mix-up inside the right family | 10.9% | partly (voting) |
| **Size** mix-up (QAM / APSK / 8-16-32PSK / ASK) | 9.6% | physics limit -> group answers |
| Signal vs noise | 2.0% | confidence rule |
| RADAR_CW vs JAM_TONE | 1.2% | identical waveforms -> pair answer (labels-v2) |

By signal strength (a third of the test windows are below 0 dB):

| SNR | Share of windows | Correct | Main error |
|---|---|---|---|
| < 0 dB | 31% | 25% | wrong family 53% (the signal is buried) |
| 0-9 dB | 31% | 53% | wrong family 22%, size 11% |
| >= 10 dB | 37% | 58% | wrong family 18%, size 12% |

Biggest cross-family mix-ups at >= 10 dB (true -> called): AM_SSB_SC -> JAM_SPOT (90), LTE_LIKE ->
JAM_BARRAGE (87), RADAR_CW <-> JAM_TONE (121, twins), NBFM / PM / AFSK -> JAM_TONE or RADAR_CW (184),
LORA_CSS -> JAM_SWEPT (46, twins), 32/64APSK -> JAM_PARTIAL_BAND (72), JAM_DRFM -> its radar (103,
labels-v2 pair), NOISE_FLOOR -> OFDM / SSR_IFF (52).

**Reading it:** the model does not mostly fail on "hard maths". It fails where **0.16 ms is too short
a look**: a voice-modulated carrier looks like a tone jammer, a few OFDM symbols look like noise
jamming, a few APSK rings look like a jammer band.

## 2. Problems, causes, fixes, evidence

### P1. Weak signals (31% of windows are below 0 dB)
- **Cause:** physics. At -10 dB the pattern is mostly gone (report figure 5).
- **Fix:** listen longer (the tracker's vote), and say "uncertain" instead of guessing.
- **Evidence:** vote length on val/test (Eileen, SED-39): 1 window 49.7% -> 8 windows 72.7% -> 32 windows 74.1% (>= 10 dB: 62.9% -> 78.4% -> 80.0%).
- **Status:** voting is in the pipeline ✅. Confidence bands are in the answers ✅ (high >= 0.40, medium >= 0.10, below = "uncertain").
- **80% on confident answers: NOT supported yet.** `eval_pipeline.py` now prints coverage vs accuracy per cut-off. On 2 x 40 captures (seed 4, choose set / check set), at >= 10 dB a 0.40 cut-off gave **80% on one set and 62% on the other** (only ~35-40 strong-signal tracks each). Re-measure with the new model on >= 200 captures before claiming any operating point; Eileen's 400-capture run with the spectrum model found 79% above 0.40.

### P2. Sizes inside QAM / APSK / PSK / ASK (9.6% of all windows)
- **Cause:** 10-60 symbols per window; the perfect-receiver ceiling is ~74% (PSK) / ~45% (QAM).
- **Fix:** answer the group with the likeliest size (Eileen, `resolve.SIZE_GROUPS`) ✅.
- **Also merged (2026-11-12):** QAM and APSK are one answer, "QAM / APSK": 32% of APSK windows at >= 10 dB were called QAM. As one group, family right per window 89-90% (was APSK 56%, QAM 48%); pipeline 286 -> 295 of 542 found emitters right, no new wrong answers.
- **Checked here: would a longer-window model help?** `scripts/probe_window_length.py`, same expert features, same classifier, 15 dB, 300 windows per class:

| Group | chance | one 512 window | **vote 4 x 512** | one 2048 window |
|---|---|---|---|---|
| QAM sizes | 33% | 41% | **50%** | 53% |
| PSK sizes | 33% | 32% | **33%** | 40% |
| APSK sizes | 33% | 56% | **66%** | 68% |
| OFDM vs noise jamming | 33% | 62% | **68%** | 71% |
| APSK vs partial-band jammer | 33% | 53% | **63%** | 61% |

- **Result:** voting already gets most of the gain; a 2048-sample model adds only 0-7 points. **Do not train a long-window model.** PSK sizes stay near chance: the physics limit, confirmed again.

### P3. Voice radios read as tone / spot jammers (largest strong-signal error)
- **Cause:** a voice cycle (0.3-3 ms) is longer than the window, so AM / SSB / NBFM / PM / AFSK look like an unmodulated or narrow carrier.
- **Fix 1 ✅ (Eileen):** the spectrum branch raised analog from 30.9% to **43.3%**.
- **Fix 2 ✅ (pipeline):** the zoom rules already name the kind over the whole track (70 of 72 right at 30 dB) and the dashboard shows it.
- **Fix 3 ✅ (this branch):** when the zoom is on a clear signal (>= 15 dB in its own band), its kind **overrides** the track answer: NBFM, WBFM, AM_DSB_WC / SC, AM_SSB_WC / SC (USB/LSB), OOK (Morse CW), or the RADAR_CW / JAM_TONE pair for a plain carrier. The classifier's answer is kept as `classifier_answer`, and the reason says so. **Voice demo: 0 of 3 right -> 3 of 3** (the classifier had called all three AM_SSB).
- **Caveat:** our analog *generators* carry fast 8-60 kHz test tones, not speech, so the synthetic test set cannot score fix 3 fairly. Check it on SED-38 audio scenes and on the first RSPduo walkie-talkie recording.

### P4. OFDM / LTE read as barrage noise jamming (87 at >= 10 dB)
- **Cause:** 160 µs holds ~2 OFDM symbols, so the cyclic-prefix pattern is weak.
- **Fix:** voting (62% -> 68% in the probe) + delay-correlation features (already in) + the spectrum branch (multicarrier 0.19 -> 0.27 in Eileen's pilot).
- **Not worth it:** a 2048-sample model (only +3 over voting).

### P5. APSK read as partial-band jammer (72 at >= 10 dB)
- **Fix:** voting (53% -> 63%).
- **To check:** whether it shrinks on the fixed data (`f993d8ebc417`).

### P6. FSK / PSK / QAM only recognised at 0 Hz (training-data bug)
- **Cause:** the 18 FSK/PSK/QAM classes were never placed in frequency. Seed 4: BPSK 100% at 0 Hz, 2% at +300 kHz; DSSS 0% at 0 Hz.
- **Fix ✅ (data):** `f993d8ebc417` (Part A done).
- **Fix (model):** retrain Part B of `notebooks/colab_fskpsk_fix.ipynb`, ~2 h T4.
- **Expect:** roughly the same headline (~50%), but honest, and correct off-centre and on real signals.

### P7. Pulsed radar inside tracks (radar group 23% per track)
- **Cause 1, measured:** a radar is on only 1-9% of the time, so most windows in its track hold only noise and drag the vote down.

  Windows straight from a long pulse train (15 dB, 80-100 windows per class):

  | Class | training-like | random window | window with a pulse | pulse-centred |
  |---|---|---|---|---|
  | POLY_BARKER | 41% | 0% | 37% | 45% |
  | FRANK | 88% | 19% | 66% | 90% |
  | BARKER | 60% | 20% | 80% | 55% |
  | PULSE | 66% | 32% | 100% | 55% |
  | COSTAS | 89% | 14% | 83% | 90% |

- **Fix ✅ (this branch):** `extract.active_windows` votes only with windows where the emitter is on (in-band power >= 3 dB above the noise). Switch: `--active-windows` in `run_pipeline.py` / `eval_pipeline.py`. Pipeline, 20 captures, 70 pulsed emitters, 15 dB:

  | | all windows | active windows |
  |---|---|---|
  | track named right | 23% | **29%** |
  | RADAR_PULSE | 33% | **67%** |
  | RADAR_BARKER | 0% | **33%** |
  | RADAR_COSTAS | 14% | **29%** |
  | others (PPM, JAM_PULSED, SSR_IFF) | unchanged | unchanged |

- **Tried, not kept:** centring windows on each pulse. RADAR_PULSE went to 100%, but PPM fell 60% -> 20%; overall 27%, worse than active windows.
- **Cause 2, checked:** cutting a radar to its exact nominal band costs accuracy in isolation (POLY_BARKER 17% -> 7%, FRANK 66% -> 48%, BARKER 72% -> 51%; a 2x wider cut gives 14 / 55 / 61%). **But a wider cut does not help the real pipeline** (margin 10% -> 50%: pulsed 29% -> 27%, mixed scenes 47% -> 43%, because it pulls in neighbours). The margin stays at 10%; it can be tried with `eval_pipeline.py --cut-margin`.
- **Still open:** in the full pipeline POLY_BARKER / FRANK windows score only 4-6%, below every isolated test. So the YOLO **boxes** for these radars (time or frequency) are the next suspect: compare their boxes with the truth boxes.

### P8. Look-alike twins (RADAR_CW / JAM_TONE, LORA_CSS / JAM_SWEPT, DRFM + radar)
- **Fix ✅:** labels-v2 pairs + resolve rules (behaviour over time, context). Nothing more to gain from the classifier.

### P9. Noise called a signal (2%)
- **Fix ✅ for windows:** threshold T = 0.12 (Eileen).
- **For tracks:** do not apply T (it drops a third of real tracks). Use confidence bands instead: low (<0.05, right 22%), medium (0.10-0.20, right 50%), high (>0.40, right 79%).

## 3. What to do, in order

| # | Action | Who | Cost | Expected gain (evidence) |
|---|---|---|---|---|
| 1 | Train Part B on the fixed data (spectrum + expert, seed 4) | Eavan / Eileen | 2 h T4 | correct off-centre; same headline (P6) |
| 2 | Active-window vote as the pipeline default | Claude | ✅ done (`--all-windows` = old way) | radar tracks +6 pts, RADAR_PULSE +34; mixed scenes unchanged (P7) |
| 3 | Confidence bands + "uncertain" in answers and dashboard; coverage/accuracy table in the evaluator | Claude | ✅ done | the 80% operating point did not hold on a second capture set; re-measure on the new model, >= 200 captures (P1, P9) |
| 4 | Zoom result overrides voice / tone track answers when clear | Claude | ✅ done | voice demo 0/3 -> 3/3; score on SED-38 audio + RSPduo next (P3) |
| 5 | Find why POLY_BARKER / FRANK fail in the pipeline (wider cut ruled out; check the YOLO boxes) | Eileen / Jessy | 1/2 day | radar group, open (P7) |
| 5b | Physics branch (built; SEPARABILITY_ANALYSIS 6.1): train with `PHYSICS = True` in the notebook, compare with Part B | Eavan / Eileen | 2 h T4 | fair check +1.1 (+1.4 at >= 10 dB); PSK, spread, ASK, analog |
| 6 | Re-run this error budget on the new model + labels-v2 test set | Claude | 10 min | confirm what is left |

**Do not spend GPU time on:** a longer-window model (P2, P4), more epochs (Eileen: 15 -> 20 gave +0.2),
`n_fft` 128 (worse than 64), or the IQ-amplitude / instantaneous-frequency inputs (no gain in the pilot).
