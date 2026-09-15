# Fixes deferred until after stage 1

Decided 2026-09-11: nothing here changes before the stage-1 submission (due 2026-09-13).
Fix and retrain after stage 1, before the finale (2026-10-07). This list was accurate on
2026-09-11 — re-check each item against the code before acting on it.

## A. Needs a retrain

### 1. The sweep jammer barely sweeps inside a window (highest priority)

**What's wrong.** `generate_sweep_jamming` (`src/generators/jamming.py`) reuses the radar
chirp function and draws one upward slide across the full 2 ms every generator produces
(`signal.total_duration`). `preprocess_window` keeps only the first 512 samples (160 µs).
So there are two gaps:

- It slides once and never repeats, although its docstring says "fast repeating sweep jammer".
- Only a fraction of the slide lands inside the window.

**Measured (300 examples each, 2026-09-11):**

| Window kind | Share of the slide inside the window | Frequency rise seen |
|---|---|---|
| Standalone JAMMING | 8% | median 25 kHz (8–40) |
| Jammer over a RadioML victim | ~49% | median 141 kHz (50–247) |
| For comparison: whole slide inside the window | 100% | median 285 kHz (100–497) |

The STFT branch's input spectrogram has 200 kHz frequency rows (3.2 MHz / n_fft 16), so a
standalone sweep's 25 kHz rise stays inside one row and looks like a flat tone. Inside the
branch it gets worse (see item 2).

**Why it matters.**

- The §6.5 sweep recall of 95.7% is measured on the same near-flat sweep
  (`scripts/jamming_subtype_breakdown.py` calls the same generator with the same 2 ms), so it
  means "recognises our near-flat sweep", not "recognises a sweeping signal".
- Likely cause of the §7.2 result: 20% recall on real recorded SingleChirp jamming.
- A spectrogram of one of our sweep jammers shows a flat line.

**Fix.**

- Keep the 512-sample window. It is fixed by RadChar (report §3.5). Fix the generator, not the window.
- Make each sweep pass shorter than the window (for example 20–160 µs per pass) and repeat it
  (sawtooth), so every window shows at least one full slide.
- Check the sweep width against the 200 kHz STFT rows. With the current 100–500 kHz range,
  narrow sweeps still look flat even when the whole slide is visible. Consider widening it.
- Put the fix inside `generate_sweep_jamming`, so every caller picks it up: standalone
  windows, overlays, and the §6.5 script.
- Update the JavaScript copy in `web/generators.js` (`randomJammingExample`, sweep branch) to
  match, then re-run `web/test/generators_smoke.mjs`.
- Add a test: every generated sweep's frequency must move by at least one STFT row within a
  single 512-sample window.
- After retraining, watch LFM_RADAR ↔ JAMMING confusion. A fast chirp looks more like radar;
  radar's silent gaps should still separate them.
- Report sections to update: §3.2 (table), §3.3 (jamming parameters), §6.5, §7.2, §8.

### 2. The STFT branch throws away where energy sits in frequency

**What's wrong.** Inside `STFTBranch` (`src/models/amc_cnn.py`), a 2×2 max-pool merges the
200 kHz spectrogram rows into 400 kHz rows, and the branch's last step (`f.mean(dim=2)`)
averages the whole frequency axis away. So "the peak moves from row 2 to row 6" (an FHSS hop,
or a sweep's slope) and "the peak stays on row 4" (a tone) produce the same features. The
class docstring records the measured cost on the pre-September model: FHSS recall collapsing
as jammer strength rises, and 46.5% of held-out jamming predicted as FHSS.

**Don't just raise `n_fft`.** A bigger FFT gives thinner frequency rows but wider time
columns. With n_fft 64, each column spans 20 µs, longer than the fastest FHSS hop (~21 samples,
about 6.6 µs), so hops blur together. The team moved from 64 to 16 for exactly this reason.

**A fix was designed but never run.** The `model.stft_freq_summary` flag pools over time only
(keeping the 200 kHz rows) and adds three per-frame features computed from the spectrogram:
frequency max, spectral flatness, and peak-frequency change. `scripts/run_stft_experiment.py`
trains one model with the flag on. Its pre-registered bar: FHSS recall at +10 dB JSR above 0.25,
up from a 0.048 single-member baseline. No result is recorded in the repo, only the baselines
in `docs/experiments/`.

**Even with the flag on, many FHSS hops stay small.** Measured over 20,000 generated FHSS
windows (2026-09-11): 46% of hop-to-hop jumps are smaller than one 200 kHz row, 4% of FHSS
examples fit their whole channel comb inside a single row, and 18% fit inside one 400 kHz
pooled row. The IQ branch has to carry those hops.

**A bug in the designed fix: shift the spectrogram first.** `STFTBranch` passes the spectrogram
to its convolutions and to `_peak_freq_delta` in raw FFT order: row 0 is 0 Hz, rows 1–7 are
positive frequencies and rows 8–15 are negative (verified in the Lesson 5 trace, 2026-09-11).
`_peak_freq_delta` takes the loudest row and differences it between frames with no wrap-around.
So a signal drifting across 0 Hz (row 15 → row 0) scores a jump of 15/16, the maximum possible,
and every civilian signal sits at 0 Hz. The same row order splits any signal centred on 0 Hz
between the top and bottom edges of the convolution input. Fix: apply `torch.fft.fftshift` over
the frequency axis inside `STFTBranch.forward`, before the convolutions and the features. Mirror
it in `src/models/onnx_export.py` (`compute_stft_mag`) and `web/dsp.js`, so the web parity tests
still hold.

**Run it alongside item 1**, since both need a full retrain. The peak-frequency-change feature
measures exactly what a repeating sweep does. Turning the flag on changes the fused width from
192 to 195 channels and changes the parameter count. Report §4.5, §4.7 and the architecture
figure would all need updating, and none of the current checkpoints will load.

### 3. Jammers laid over radar are much weaker than their label says

**What's wrong.** `apply_jamming` (`src/generators/jamming.py`) scales the jammer against the
victim's average power over the whole signal (`np.mean(|signal|²)`), not its power while it is
on. Overlays apply it to the full 2 ms raw victim, before the window is cut. A pulsed radar is
silent most of that time, so its average is tiny, and the jammer is scaled down to match.

**Measured (400 examples each, 2026-09-11).** This is the effective JSR inside the kept
512-sample window, measured against the victim while it is on, minus the labelled JSR:

| Victim | Median | 10th–90th percentile |
|---|---|---|
| LFM_RADAR (pulsed) | −13.8 dB | −17.6 to −9.5 dB |
| FHSS (continuous) | 0.0 dB | −0.2 to +0.2 dB |

So radar overlays labelled 0–20 dB JSR really span about −14 to +6 dB against the pulse. The
radar is usually louder than the jammer that is supposed to drown it.

**Why it matters.** Report §3.2 says power is measured only over the parts of the window that
contain signal. That is true for the noise (`add_awgn`) and for mixture SIR (`unit_power`), but
not for the jammer in overlays. Radar-plus-jammer results, such as the `recall_in_context`
scorecard entries, were measured on these weaker jammers. Three-way mixtures containing radar
were not measured.

**Fix.** Measure the victim with `active_power` inside `apply_jamming` (the helper already
exists in `src/data/composite.py`), or apply the jammer after windowing. Extend the JSR test to
a pulsed victim. Then run the full retrain chain.

## B. Report only (no retrain)

4. **The §5.1 step figures (pp. 23–27) use an untrained model.** For a QPSK + jammer input,
   Step 9 answers QPSK + 16QAM + LFM_RADAR, with the Hostile box empty. The Civilian and
   Military boxes in Step 9 are also missing their labels. Rerun the same window through the
   trained ensemble (`results/ensemble_0.pt` to `ensemble_4.pt`). The script that made these
   figures is not in the repo. A trained-model trace (input → both branches → fusion → attention → head
   → ensemble → status) was produced on 2026-09-11 for Lesson 5; that script can become
   `scripts/trace_window.py` to regenerate these figures.
5. **Figures show a threshold of 0.5** (pp. 21, 22, 26). The shipped system uses per-class
   calibrated thresholds (`configs/default.yaml`, `multilabel_thresholds_per_class`).
6. **§6.5 says tone jamming is "single or multi-carrier".** §3.3 and the config
   (`jamming.max_tones: 1`) say single-tone only.
7. **§8.1 repeats a phrase:** "generated benchmark datasets. Not field captures under" appears twice.
8. **The architecture figure (p. 22)** says the first dilated layer "sees ~22 samples". The
   receptive field after that layer is 19 (7 + 12).
9. **"Clean and faded conditions" (§1.1, §3.3).** The generators add white noise (AWGN) only;
   no multipath fading is modelled on the synthetic classes. Either model fading or reword,
   and add it to §8.

10. **§6.1 and §7.3 say 16QAM and 64QAM are split by "a fixed statistical rule applied after
   the network".** In the code, the reported 16QAM and 64QAM scores come straight from the
   network's own two outputs, and the network cannot tell them apart: per `src/measure.py`, it
   picks the right one 51.4% of the time on a single window. The |C42| rule (`src/measure.py`)
   is only used for a measured caption in the analysis panel (`src/ui/plots.py` and
   `web/constellation.js`), pooled over at least 8 windows at SNR ≥ +2 dB. It never changes an
   evaluated label. Reword both sections: the network detects dense QAM, 16 vs 64 is not
   reliably separable per window, and the combined dense-QAM recall (65.58%) is the meaningful figure.

11. **§7.3 says indices "17 to 21" sit at roughly ten times the amplitude.** Measured from the
    file (200 frames per class at +30 dB, 2026-09-11): only index 17 (AM-SSB-WC, RMS 11.8) and
    index 18 (AM-SSB-SC, RMS 8.3) do; indices 19 and 20 sit at about 1.0, like the digital classes.
    §3.1 already says "classes 17 and 18" correctly, so make §7.3 match. Index 21's near-constant
    envelope holds (envelope variation 0.005).
12. **Optional: this closes a stated uncertainty (§3.1).** The report says indices 3 and 4 (BPSK vs
    QPSK) were not confirmed, and that doing so would need full carrier recovery. A power-law test
    doesn't need it. Measured at +30 dB (spike strength = peak ÷ average of the spectrum): index 3
    squared gives 89.6, against about 14 for indices 4, 5 and 12, which is the signature of BPSK.
    Index 4's spike doubles at the 4th power (13.8 → 27.3), which is the signature of QPSK. This is
    consistent with the paper's order, which the team used. Worth adding as a third independent check.

13. **§8.2 quotes a seed spread that doesn't match the shipped checkpoints.** The report says FHSS
    "ranged 76.3% to 84.0% across five seeds". Measured from `evals/ensemble_scorecard.json`
    (2026-09-12), the five submitted members give **76.83% to 84.48%** (spread 7.65 points), with
    member 4 at 76.83% failing the benchmark on its own while the ensemble passes at 82.84%.
    Update the numbers to the ensemble actually submitted. Worth adding alongside: the ensemble
    beats every individual member on LFM_RADAR (85.00% vs a best member of 83.89%), but on JAMMING
    it sits below four of the five (84.44% vs 83.79–86.57%), because averaging compresses
    probabilities toward the mean against a high threshold — the same effect as §7.1's third fault.
    Ensembling buys consistency, not a higher score on every class.

## C. Code and outputs (re-run evaluation, no retrain)

14. **`src/evaluate.py` line 525** titles the figure "Accuracy vs. SNR — all classes", but the
   per-class lines are recall; only the overall line is accuracy. Rename it, then re-run
   `python -m src.evaluate --ensemble`. The report's §6.2 caption already says "Recall against SNR".
15. **Optional:** add per-class accuracy to `evals/scorecard.json`, since the rules literally
   say "Accuracy > 80%".
16. **Two Claude-made deliverables carry a wrong number:** the published brief page and
    `docs/overwatch_walkthrough.pptx` say "14 of 16,290" false alarms. The correct figure is
    4 of 13,230 civilian-only windows (0.03%). The report itself is right.

17. **Optional diagnostic:** check whether barrage misses (§6.5, 69.3% recall) concentrate at
    narrow bandwidths. A 200 kHz barrage is as thin as a tone in the spectrogram.
    `scripts/jamming_subtype_breakdown.py` could split its barrage results by bandwidth.

18. **Check whether the model leans on absolute phase (diagnostic, no retrain).** Our generators
    start every synthetic radar pulse, every sweep and every FHSS hop at phase 0
    (`generate_lfm_chirp_iq`, `generate_fhss`), and training applies no phase-rotation
    augmentation. A real receiver's absolute phase is arbitrary, so any reliance on it would not
    survive real data. Test: evaluate the shipped ensemble on the test split rotated by a fixed
    angle (for example 90°, with `phase_rotate_batch`) and compare per-class recall against the
    unrotated run. Same model, same windows, so a shift of more than about 2 points is real. If
    scores move, randomise phase in the generators and add rotation augmentation (option F3).
    A hint to check at the same time: in the Lesson 5 trace (one QPSK + tone window), the single
    strongest attention weight sat on the window's first sample (0.3 µs). One window proves
    nothing. But if attention piles onto the first few samples across many windows, that fits this
    shortcut, or a padding edge effect.

19. **Test whether the "energy gate" does anything (diagnostic, no retrain).** In each trained model,
    the attention scorer's weight on the raw-power channel is small and mixed in sign: 0.058,
    −0.084, 0.136, 0.092 and 0.122 (Lesson 5 trace). Set that weight to zero in all five models
    and re-evaluate. If recall doesn't move, §4.6 should say the trained network relies on its
    learned features, and that the energy input is an option it barely uses.

## D. After-retrain checklist (in order)

0. Decide on item 2 (the STFT frequency features) first. It changes the model's shape, so it
   must be settled before anything is trained.

1. Rebuild the dataset with `python -m src.data.build_dataset`, then upload it to Google Drive as before.
   Check the build log's `sources:` line reports 6,000 real RadChar windows. `src/data/radchar.py`
   silently uses `RadChar-Tiny.h5` when `RadChar-Small.h5` is missing (for example on Colab), which
   would make most radar synthetic without any error.
2. Retrain the 5-model ensemble on Colab with `notebooks/colab_retrain_2026-09.ipynb`. This
   produces `results/ensemble_0.pt` to `ensemble_4.pt`.
3. Recalibrate with `python scripts/calibrate_thresholds.py --ensemble --n-models 5`. The
   script only prints the values; copy them into `configs/default.yaml` by hand.
4. Evaluate with `python -m src.evaluate --ensemble` and `python scripts/train_ensemble.py --eval-only`.
   Results go to `evals/`.
5. Re-run the diagnostics: `scripts/jamming_subtype_breakdown.py` (§6.5),
   `scripts/validate_external_jamming.py` (§7.2), and optionally `scripts/measure_variance.py`.
6. Run `pytest`.
7. Update the web demo: `python scripts/export_onnx.py`, then `python web/build.py`, then run
   the `web/test/` parity checks, then redeploy on Vercel.
8. Update every number in the report (§3, §6, §7, §8), re-upload `evals/csv/` to Power BI,
   and send the new `evals/` to the team.

## E. The report's own future work (§9.2), with concrete steps

All of E1–E5 need the full retrain chain in section D. E6 is optional research.

### E1. Dense-QAM precision (the largest remaining weakness)

**Evidence.** 16QAM precision 0.434 (3,184 false positives against 2,440 true positives); 64QAM
0.462 (2,296 against 1,970); combined dense-QAM recall 65.58% over 7,290 windows.

**Cheapest lever first, no retrain.** The civilian thresholds were calibrated for best F1
(BPSK 0.33, QPSK 0.265, 16QAM 0.265, 64QAM 0.275). Raising the two dense-QAM thresholds trades
recall for precision. `scripts/calibrate_thresholds.py` has no precision target, so either sweep
by hand and re-run `python -m src.evaluate --ensemble`, or add a `--min-precision` option beside
the existing `--margin`. Confirm `comms_vs_jamming` is unchanged: it depends on the JAMMING
output, not on these thresholds.

**Data lever (retrain).** `dataset.civilian_examples_per_snr` is already 3,000, and RadioML holds
up to 4,096 per class-SNR block. Evidence for going higher is weak: 400 → 1,000 helped, 1,000 →
2,000 did not.

**Weighting lever (retrain).** `class_weight_multipliers` in `configs/default.yaml`. The config
records that more civilian weight did *not* fix civilian recall, so don't repeat it without a
diagnostic that says why it should work this time.

**Structural lever (retrain, biggest).** A dedicated civilian path (F5), reusing the matched
filter and carrier recovery already in `src/ui/plots.py` and `web/constellation.js`. Remember the
ceiling: about 56 recovered symbols per window is not enough to separate 16 from 64 (measured at
a coin flip), so pooling across windows (F2) is what actually breaks that tie.

**Test.** Per-class precision and recall, `dense_qam_recall`, and the confusion counts.

### E2. LFM_RADAR ↔ FHSS confusion, concentrated in slow hopping

**Evidence.** 51.2% of radar's 2,230 false positives land on windows where FHSS is genuinely
present. The false-positive rate against radar is 39.6% at 25–50 kHz hop rates against 18.6% at
125–150 kHz, versus 2.4% on pure noise.

**Fix A, data (retrain).** Hop-rate-aware augmentation: oversample the slow end of
`fhss.hop_rate_hz` so the model sees more of the hard case, rather than sampling the range evenly.

**Fix B, features (retrain).** Make "discrete versus continuous" explicit: item 2's
`stft_freq_summary` peak-frequency-change feature, or F4's instantaneous-frequency input channel.
A chirp is a smooth ramp; FHSS is a staircase.

**Test.** `scripts/fhss_radar_false_positive.py` reproduces the by-hop-rate table, and
`radar_fhss_confusion` in the scorecard gives the headline fractions.

### E3. Civilian–civilian mixtures (closing the one untrained overlap)

**What to do.** Add pairs to `dataset.mixture_combos` in `configs/default.yaml`, for example
`['BPSK','QPSK']`, `['QPSK','16QAM']`, `['16QAM','64QAM']`. No new code is needed:
`build_mixture_examples` already draws civilian components from the RadioML clean pool and skips a
combo only when that pool is missing.

**Cost and risk.** Each combo adds 300 × 6 = 1,800 windows. A 16QAM + 64QAM pair may make the
dense-QAM boundary harder, so measure `dense_qam_recall` before and after.

**Test.** Rebuild, retrain, compare per-class precision and the combined dense-QAM figure.

### E4. Low-SNR robustness

**Evidence.** At −10 dB: LFM_RADAR 67.8%, FHSS 58.6%, JAMMING 47.8%, civilian near zero.

**Already in place.** `compute_snr_weights` in `src/train.py` oversamples low-SNR examples about
10×, so "just oversample more" is not the untried idea it looks like.

**Option A, no retrain.** SNR-conditional thresholds. `src/measure.py` already has
`noise_floor_power()` and `estimate_snr_db()`. Gate the per-class thresholds on that estimate, and
guard it the way `constellation_order` guards |C42|: refuse to adjust when the estimate falls
outside the range it was validated on.

**Option B, retrain.** An SNR-estimation head predicting SNR from the shared features, then used
as in Option A. It also gives the demo a reportable SNR readout.

**Option C, evaluation only.** Held-out parameter-range testing, still listed as outstanding in
§9: train on part of the hop-rate and PRI ranges, test on the held-out part. This measures
generalisation beyond your own generator without needing any new data source.

**Test.** `evals/csv/accuracy_by_class_snr.csv`, the −10 and −6 dB rows.

### E5. The high-SNR decline in FHSS and radar recall

**Evidence.** FHSS falls from 90.6% at +2 dB to 81.2% at +10 dB, and its mean probability falls
with it (0.736 → 0.690), so this is genuine uncertainty rather than a threshold artefact. Radar
shows the same pattern mildly (0.776 → 0.746).

**Diagnose before fixing.** Split the +10 dB misses by generator parameter (hop rate, channel
count and spacing for FHSS; pulse width, PRI and duty cycle for radar), using
`scripts/fhss_radar_false_positive.py` and `scripts/duty_cycle_breakdown.py` as templates. Expect
an interaction with items 1 and 3: at high SNR, a near-flat sweep and an under-strength jammer
both look different from what the labels claim.

**Then fix on the data side**: more diverse high-SNR examples for these two classes, aimed at
whatever the diagnostic implicates.

**Test.** The same SNR CSV at +6 and +10 dB, and the mean-probability column alongside recall.

### E6. Emitter fingerprinting (a first step toward attribution)

**Why it is reachable.** RadChar ships per-waveform labels for `number_of_pulses`, `pulse_width`,
`time_delay` and `pulse_repetition_interval` (`src/data/radchar.py`).

**What to do.** Add regression outputs beside the 8 classification outputs, predicting pulse width
and PRI, and apply that loss only on RadChar windows (mask it elsewhere, since our synthetic
radar's parameters are drawn rather than measured). The shared trunk stays as it is.

**Scope.** Research direction, not needed for the competition. It is what turns "a radar is
present" into "a radar with these characteristics", which §9.1 lists as out of scope today.

**Test.** Mean absolute error in microseconds on held-out RadChar windows, with the classification
metrics required not to regress.

## F. Improvement options (not decided)

**How to test them.** Single training runs vary by up to about 6 points (the config records a
6.2-point noise floor; §8.2 reports 2.2–8.9). Test each change with single-model runs (~2.5 h)
against a single-model baseline, two seeds each, and only trust large effects. Retrain the full
ensemble (~13 h) once, at the end.

**No retrain**

- **F1. Test-time phase averaging** (`scripts/train_ensemble.py --tta N`). It's built in but was
  never measured: it averages each prediction over N phase rotations. `scripts/calibrate_thresholds.py`
  has no TTA option, so thresholds must be calibrated with the same TTA before using it (the
  §7.1 fault #3 again: averaging compresses probabilities). Run item 18 first. If the model
  depends on absolute phase, rotated views are out of distribution and TTA could hurt.
- **F2. Multi-window decisions for the operator.** Pooling consecutive windows is the only real
  lever at −10 dB and for 16 vs 64QAM (|C42| accuracy: 59% for one window, 98% for 64). Report
  it as an operational figure next to the per-window benchmark, never instead of it.

**Retrain: inputs and architecture**

- **F3. Phase realism.** A random starting phase per pulse and per hop in the generators, plus
  random phase rotation during training.
- **F4. Extra IQ-branch inputs: amplitude and instantaneous frequency** (how fast the phase turns,
  measured at every sample) as input channels 3 and 4. Sweeps show as ramps, FHSS as staircases
  and tones as flat lines, all at full time resolution, with no time–frequency trade-off. This
  targets items 1 and 2, the small-hop note, and the §9 "discrete vs continuous" radar/FHSS idea.
  Caveats: it is noisy at low SNR and follows the stronger signal when two overlap, so add it as
  extra channels, not a replacement. About +900 parameters.
- **F5. A dedicated civilian path** (report §9). Recover symbols (the matched filter and carrier
  recovery already exist in `src/ui/plots.py` and `web/constellation.js`) and feed them to a
  small classifier. Per window, 16 vs 64QAM stays limited by ~56 symbols; pooling (F2) is what
  beats that.

**Retrain: data realism**

- **F6. Channel impairments on the synthetic classes:** multipath fading, carrier-frequency
  offset, phase noise and IQ imbalance. None exist today. This addresses item 9 ("faded
  conditions") and likely part of the §7.2 real-data gap (57.6% recall).
- **F7. Put RadChar radar into overlays and mixtures** (the report's own stated next step). It
  raises real radar above today's 29% of radar-bearing windows.
- **F8. Civilian–civilian mixtures** (§8.1 scope gap). A generator change only.
- **F9. Restore multi-tone jamming** once item 2 keeps frequency position: tones stay put while
  FHSS dashes move.
- **F10. Radar duty cycles of 15–44%** (currently no coverage). Re-run
  `scripts/duty_cycle_breakdown.py` after F4 or item 2, and extend coverage only if radar/FHSS
  confusion stays low.

**Already tried; don't repeat**

- More data per class: 400 → 1,000 helped, 1,000 → 2,000 didn't (within the noise floor).
  Standalone civilian data is already at 3,000.
- More loss weight on the civilian classes: it didn't fix civilian recall.
- Raising n_fft: it blurs FHSS hops (the team moved from 64 to 16 for this).

**Caution:** training on the external GNSS jamming set (§7.2) would stop it being an independent
test. If it's used, split it by category: train on some, test on others.

**Suggested order after stage 1:** item 18 and F1 first (no retrain) → build items 1–2 plus
F3/F4/F6 with tests → single-model experiments → one final ensemble retrain → checklist D.
