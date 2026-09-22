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

**DONE 2026-09-20 — the spectrogram is now shifted (guarded by the flag).** `STFTBranch` passes the spectrogram
to its convolutions and to `_peak_freq_delta` in raw FFT order: row 0 is 0 Hz, rows 1–7 are
positive frequencies and rows 8–15 are negative (verified in the Lesson 5 trace, 2026-09-11).
`_peak_freq_delta` takes the loudest row and differences it between frames with no wrap-around.
So a signal drifting across 0 Hz (row 15 → row 0) scores a jump of 15/16, the maximum possible,
and every civilian signal sits at 0 Hz. The same row order splits any signal centred on 0 Hz
between the top and bottom edges of the convolution input. Fix: apply `torch.fft.fftshift` over
the frequency axis inside `STFTBranch.forward`, before the convolutions and the features. Mirror
it in `src/models/onnx_export.py` (`compute_stft_mag`) and `web/dsp.js`, so the web parity tests
still hold.
Implemented in `STFTBranch.forward` as `torch.fft.fftshift(mag, dim=2)` under `if self.freq_summary`,
so the flag-off path is unchanged: the shipped `ensemble_0.pt` produces logits identical to 0.000e+00
after the edit. Verified effect: a tone drifting across 0 Hz used to report a frame-to-frame jump of
0.938 (maximum possible is 1.0); it now reports 0.062. `tests/test_amc_cnn.py` and
`tests/test_pipeline.py` pass. Mirrored in `src/models/onnx_export.py` on branch `c2-keep-stft-rows` (2026-09-20): the exported copy
applies the same shift with `torch.roll`, which is `fftshift` for an even `n_fft`. `web/dsp.js` needs no
change, because the shift is inside the exported graph and the browser still feeds it raw-order magnitudes.

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

**BUILT and MEASURED 2026-09-22, branch `fix_radar-fhss-confusion` (a version of Fix B).**
`model.stft_dwell_feature` (`src/models/amc_cnn.py`, default false): one more per-frame feature,
`_sweep_consistency` — whether this frame's frequency step keeps going the same direction as the
previous frame's, i.e. whether jumps compound into a sweep rather than landing as isolated hops.
Where `stft_freq_summary`'s existing `_peak_freq_delta` says *a* jump happened, this says whether
consecutive jumps *agree*: high and sustained for a radar chirp, near chance for FHSS hopping to
an arbitrary new channel each time. Adds 1 output channel, +257 parameters (149,195 vs the
148,938-parameter baseline). Composes with `stft_keep_rows` and `stft_freq_summary` — neither is
required, but note `_sweep_consistency` reads the raw STFT magnitude directly (same as
`_peak_freq_delta`), so `stft_keep_rows`'s richer *conv-path* representation does not reach it;
they are independent mechanisms, not one subsuming the other. Flag off is proven byte-identical
(`tests/test_stft_dwell_feature.py`), so every checkpoint in `results/` keeps loading strict=True.

**Result is mixed, not a fix for LFM_RADAR precision — but a real, repeated effect on the specific
confusion mechanism.** Two single-model seeds (2000, 2001) on two datasets:

| Run | vs. | LFM_RADAR precision | radar's FPs that are true FHSS |
|---|---|---|---|
| old data (`eavan-retrain`), seed 2000 | old single-model baseline (51.4% / 46.4%) | 50.2% (−1.3) | 49.8% (+3.4, worse) |
| old data, seed 2001 | same | 51.3% (−0.1, flat) | 46.1% (−0.3, flat) |
| `radar-fix-data`, seed 2000 | same-data flag-off baseline (47.0% / 47.7%) | 46.5% (−0.5, flat) | 43.4% (−4.3, better) |
| `radar-fix-data`, seed 2001 | same-data flag-off baseline | 44.3% (−2.7) | 35.7% (−12.0, better) |

On the old dataset the effect is inside noise (the two seeds disagree on direction) — no real
effect. On `radar-fix-data` (same shape, `(128400, 2, 512)`, unclear yet how its generation differs
from `eavan-retrain` — ask before treating it as the new standard set) the confusion fraction drops
in the same direction both times, averaging **−8.2 points**, which is outside the ~6-point
single-run noise band this codebase has measured elsewhere. LFM_RADAR precision itself does **not**
rise on either dataset — at best flat, at worst down a few points, same direction both new-data
seeds. Reading it plainly: the feature makes the model mistake FHSS for radar less often, but
total false positives on LFM_RADAR do not fall by a matching amount, so something else is
contributing false alarms this feature does not touch. Not the precision fix E2 is looking for;
a real, narrower, partial one.

**Why it plausibly works less than hoped: `_sweep_consistency` tracks the single loudest row per
frame.** In a composite window (radar or FHSS *plus* a jammer — most of the confused cases), the
loudest row is often the jammer's, not the victim emitter's own movement, so the feature may be
reporting "is the jammer steady or erratic" rather than "is the underlying emitter sweeping or
hopping" in exactly the cases that matter. Untried: restricting the peak search to the top few rows
per frame, or masking out a detected jammer block first (via `_spectral_flatness`/`_frequency_max`)
before picking which row to track.

**Not yet done, before this is a decided result:** a third seed or the full 5-model ensemble; a
same-data baseline+flag comparison on old data too (only `radar-fix-data` got one); clarity on
`radar-fix-data`'s provenance and whether it should replace `eavan-retrain` generally, which would
also mean re-baselining `main`'s existing checkpoints against it, not just this branch.

**Tooling added alongside this:** `scripts/run_dwell_experiment.py` and
`scripts/run_stft_experiment.py --dwell-feature` (single-member training runs, same pattern as the
C2/`stft_freq_summary` runners); `--stft-dwell-feature` on `evaluate_experiment.py`;
`scripts/radar_fhss_confusion_check.py`, which reads `evaluate_experiment.py`'s JSON and reports
only the two numbers this hypothesis makes a claim about (LFM_RADAR precision, the FHSS-confusion
fraction) plus guardrails, across multiple seeds — built because
`evaluate_experiment.py --verdict-only`'s 10 checks are pre-registered for the *other* experiment
(C2 / masking) and can PASS a `stft_dwell_feature` run that does nothing for this problem, which is
exactly what happened with the first old-data seed-2000 run before a second seed corrected it.

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

**DIAGNOSED 2026-09-20 (`scripts/high_snr_probe.py`, 200 windows per cell, 5-model ensemble).**
The decline is not in the class itself. Splitting FHSS windows by kind:

| FHSS recall | −2 dB | +6 dB | +10 dB |
|---|---|---|---|
| standalone | 100% | 99.5% | 100% |
| mixture (with radar) | 96.5% | 94% | 94% |
| overlay (with a jammer) | 83.5% | 47.5% | **33%** |

Clean FHSS is perfect at high SNR. The loss is entirely in **jammed** windows, and on those misses
the model reports JAMMING instead: as noise drops, the jammer gets crisper and takes over the
window. Jammed-FHSS windows are about 17.6% of all FHSS windows (1,800 overlay + 1,800 three-way of
20,400), so losing ~50 points on them costs ~8.8 points overall — which accounts for the 9.4-point
aggregate drop in §6.2.

This is the same failure `scripts/probe_jsr.py` was written for, and its pinned baseline shows the
cliff directly: FHSS recall 0.96 at JSR 0 dB, 0.56 at +5, **0.035 at +10**. That script's own
analysis names the cause: FHSS occupies 10–48 kHz channels while barrage jamming spreads over
200 kHz–1.2 MHz, so 10–20 dB of frequency-selective processing gain is available and the model
cannot use it, because the STFT branch averages the frequency axis away. **So E5 is item 2, and the
fix is the `stft_freq_summary` experiment that was never run.**

**Is the victim really "gone" under a strong jammer? No (measured 2026-09-20).** For each jammer
subtype and JSR, take the STFT the model receives and, per time frame, compare the FHSS energy
with the jammer's energy in the one cell the FHSS occupies. Share of hops where FHSS is at least
as loud as the jammer in its own cell (200 windows per cell, dataset generators, no noise):

| Jammer | JSR +0 | +5 | +10 | +15 | +20 dB |
|---|---|---|---|---|---|
| barrage | 96% | 85% | 72% | 63% | 57% |
| tone | 92% | 74% | 65% | 60% | 57% |
| sweep (as generated: near-flat) | 83% | 52% | 41% | 35% | 31% |

So even at +20 dB most hops are still recoverable from where they sit in frequency: a coherent
16-sample tone gains about 10 dB over wideband noise in one cell, and a barrage covers only 1–6 of
the 16 rows, so many hops land outside it. Total power says the victim is buried; per-cell
position says it is mostly still there. The current model cannot use that, because the STFT
branch averages the frequency axis away. This argues for item 2 (`stft_freq_summary`) *before*
narrowing `jamming.jsr_db`, which would make the task easier rather than the model better.
It is a physical proxy, not proof the model can learn it: the experiment is the proof.

If `jsr_db` is narrowed later, it lives in four places: `configs/default.yaml`
(`jamming.jsr_db`), read by `overlay_jamming` (`src/data/composite.py:47`), by `mix_components`
(`:120`, three-way mixtures containing JAMMING) and by `src/scenarios.py:266`; and the JavaScript
copy `web/generators.js:38` and `:357`. Change all of them together or the web demo drifts from the
training data.

**The exact plan for the jammed-FHSS failure (decided 2026-09-20).**

*Metric.* `scripts/high_snr_probe.py` now prints, for overlay windows that truly hold FHSS and
JAMMING, four columns: both (correct), JAMMING only, FHSS only, neither. The aim is to move
"JAMMING only" into "both" without growing "FHSS only" (that would mean the jammer got missed).

*Baselines (single member, `ensemble_0.pt`, seed 2000; 300 windows per cell; command:
`python scripts/high_snr_probe.py --n 300 --class FHSS --ensemble --n-models 1 --seed 7`).*

| SNR | both | JAMMING only | FHSS only | neither |
|---|---|---|---|---|
| −2 dB | 62.0% | 26.3% | 11.7% | 0.0% |
| +2 | 49.0% | 44.0% | 7.0% | 0.0% |
| +6 | 25.3% | 72.3% | 2.0% | 0.3% |
| **+10** | **16.3%** | **79.7%** | 4.0% | 0.0% |

(The 5-model ensemble gives 23.0% at +10 dB, 43.5% at +6 dB.)

*Pass marks, fixed before running.* Primary: "both" at +10 dB reaches at least 45% (baseline 16.3%)
and at +6 dB at least 55% (baseline 25.3%), a gain larger than the run-to-run noise (single runs
vary by up to about 6 points). Guardrails: "FHSS only" at +6 and +10 dB grows by no more than
3 points; standalone FHSS recall stays at least 97% for SNR ≥ −6 dB; standalone JAMMING recall
stays above 80% after recalibration; the comms-vs-jamming false alarm rate stays at or below 0.1%.

*Design: a 2×2 of single-model runs (seed 2000, about 2.5 h each on Colab).*

| | divisor 20 (today) | divisor 40 |
|---|---|---|
| **flag off** | baseline (already measured above) | run A |
| **flag on** | run C | run A+C |

`training.snr_weight_divisor` (added 2026-09-20, default 20 = unchanged) sets how strongly
training oversamples low SNR: 20 samples −10 dB about 10× more than +10 dB; 40 gives about 3×.
Why it is a candidate: from −2 dB to +10 dB, the "both" rate falls in step with the sampling
weight (72.5 → 61.0 → 43.5 → 23.0% against relative weights 1.26 → 0.79 → 0.50 → 0.32).
That is correlation, not proof, so the run decides. The flag (`stft_freq_summary`) is item 2.
`scripts/run_stft_experiment.py` currently hard-wires flag on and divisor 20, so it needs
arguments for the flag and the divisor to run all four cells.

*Fallback if no cell reaches the pass mark:* narrow `jamming.jsr_db` (see the note above).

*A decision-rule fix was measured and rejected.* Among windows where JAMMING is detected, the
FHSS probability separates "jammed FHSS" from "jammer only" with an AUC of only 0.74–0.76
(400 windows each, 5-model ensemble). Lowering the FHSS threshold trades found hops for false
alarms almost one for one: at +10 dB, threshold 0.25 finds 28.0% of jammed FHSS with 1.8% false
FHSS on jammer-only windows, 0.15 finds 47.6% with 7.3%, 0.10 finds 64.5% with 27.5%. At +2 dB
the model already raises FHSS on 44.8% of jammer-only windows at the current threshold. So the
fix has to change what the model can see, not where the line is drawn.

**What the STFT branch does for jamming (measured 2026-09-20, 5-model ensemble).** Switching the
branch off at inference (never trained that way) on 6,000 held-out test windows, thresholds
unchanged, drops JAMMING recall 85.5 → 49.6% and LFM_RADAR 85.2 → 68.7%, so it cannot be removed.
By jammer type, standalone at +6 dB (300 windows each):

| Jammer | JAMMING recall, STFT on → off | Called FHSS, STFT on → off |
|---|---|---|
| barrage | 95.3% → 73.0% | 18.3% → 80.3% |
| tone | 99.3% → 57.3% | 2.3% → 81.0% |
| sweep (near-flat) | 100% → 98.0% | 9.0% → 62.0% |

Reading: the IQ branch alone reads any narrowband, steady energy as FHSS; the STFT branch is what
says "steady or wide, so a jammer". Restricting the switch-off to windows where JAMMING fired is
worse than lowering the FHSS threshold (AUC 0.73 → 0.67; at +10 dB, 51.7% found for 26.4% false
against 47.6% for 7.3% by threshold alone). So the fix must give the STFT branch position
information (item 2), not remove it.

**A guess that did not hold.** "FHSS windows with only a few distinct frequencies look like a tone
and get missed" is not supported: standalone FHSS at +10 dB is found 94.3% of the time with 3–4
distinct frequencies and 99.8% with 5–8. What the check did show is a data gap: only 73 of 3,000
generated FHSS windows (2.4%) contain four or fewer distinct frequencies, and none contains one, so
slow hopping (report §3.3 limitation) is barely represented. Belongs with E2.

**Correction: what `stft_freq_summary` really does (read from the code, 2026-09-20).** The line
`f = f.mean(dim=2)` still runs with the flag on. The flag (a) pools time only, so the conv path
keeps 16 rows of 200 kHz instead of 8 of 400 kHz before that average, and (b) attaches three
per-frame numbers computed straight from the spectrogram: frequency max (the *level* of the
loudest row, not its position), spectral flatness (how peaky), and peak-frequency change (how far
the loudest row moved since the previous frame). None of them says *which* row. So the flag adds
"did it move" and "how peaky", which is hop evidence, but not "where relative to the jammer's
block". It is a side channel, not a redesign.

The `stft_freq_summary` history: `f.mean(dim=2)` arrived with the dual-branch model (commit
`aaa77b4`, 2026-08-18); the flag followed ten days later (`186c3e7`, 2026-08-28), after the
jammer-overlay probes measured the cost.

**Variant C2 — BUILT on branch `c2-keep-stft-rows` (2026-09-20), not yet trained.** Keep the rows through a
learned layer.
- Code: `STFTBranch(keep_rows=True)` in `src/models/amc_cnn.py`; config key `model.stft_keep_rows` (default false).
  With it on, `f.mean(dim=2)` becomes `flatten(1, 2)` → `Conv1d(64·rows → 64, kernel 1)` → BatchNorm → ReLU,
  behind `_collapse_rows()`, and the spectrogram rows are put in frequency order first. Layers exist only when
  the flag is on, so the flag-off `state_dict` is unchanged and every checkpoint still loads strict.
- Cost: 181,898 parameters (+32,960; the report's 148,938 is the flag-off model). Combined with `stft_freq_summary`
  (which keeps 16 rows): 215,437. Fused width stays 192 (195 with the flag), so the head is untouched.
- Proof it is safe: flag off gives a logit difference of exactly 0.0 on the shipped `ensemble_0.pt`; 17 new tests
  in `tests/test_stft_keep_rows.py` (356 pass overall), which I checked fail when the code is sabotaged three
  ways; the exported ONNX graph matches PyTorch to about 1e-7 for all four flag combinations, run through
  onnxruntime (the repo `.venv`; export a C2 checkpoint by setting `stft_keep_rows: true` first, since
  `scripts/export_onnx.py` builds its model from the config).
- Runner: `scripts/run_stft_experiment.py` now takes `--keep-rows`, `--no-freq-summary`, `--divisor N`, `--seed`,
  `--smoke`, `--out-dir`. No arguments = the original run, byte for byte in what it writes. It saves the best
  checkpoint on every validation improvement and a history file every epoch. Cells:
  `--no-freq-summary --divisor 40` (A) · no arguments (C, the flag) · `--no-freq-summary --keep-rows` (C2) ·
  `--keep-rows` (C2 + flag) · `--keep-rows --divisor 40` (C2 + flag + A) · `--no-freq-summary` (baseline re-run,
  which also measures run-to-run noise). Each writes `results/experiment_<name>.pt`.
- Scoring: `scripts/probe_jsr.py` (the pre-registered bar) and `scripts/high_snr_probe.py` (the four-column
  table) both take `--checkpoint PATH` plus `--stft-keep-rows` / `--stft-freq-summary`. A wrong flag fails
  loudly (a `state_dict` error), it does not mis-score.
- Smoke-tested end to end (train → save → both probes) on CPU; the smoke numbers mean nothing.

*Original design note for C2 (kept for reference):*
Replace `f.mean(dim=2)` with `f.flatten(1, 2)` (64 channels × 8 rows = 512 channels; 1,024 if the
time-only pool is also used) followed by `Conv1d(512, 64, 1)` + BatchNorm + ReLU, behind its own
flag (for example `model.stft_keep_rows`, default false so every checkpoint still loads). Output
stays 64 channels, so fusion is still 192 wide. Cost: about +33k parameters (+65k with 16 rows),
so 148,938 becomes about 182k. Why it is the more direct test: in the branch-bottleneck test the
tensor *before* the average held more evidence than the one after (linear-probe AUC 0.92 against
0.84 at +10 dB, jammed FHSS versus jammer only), and C2 gives the network access to exactly that
tensor. The flag and C2 attack the same gap in two ways; if both are built, run them as separate
cells beside A.

**Already on the team's remote: `origin/eileen-stft-experiment` (Eileen, 2026-08-29 to 08-31).** Found while
building C2, and it corrects two statements above.
- It holds `notebooks/colab_stft_freq_summary.ipynb`, a ready Colab run of the `stft_freq_summary` experiment
  (train seed 2000 with `ckpt_path`/`history_path`, then `probe_jsr.py --stft-freq-summary`, then a PASS/FAIL cell
  against the 0.25 bar). It has **no saved outputs and no result files**, so whether it was ever run is a question
  for Eileen. So "never run" should read "no result recorded". Ask before spending 2.5 h on the same cell.
- It also adds two expert-feature flags in `src/models/amc_cnn.py`, both default off: `model.cumulant_features`
  (|C40|, |C42|, |C63| after an RRC matched filter, aimed at 16QAM vs 64QAM) and `model.if_features`
  (max|d²phase| / median|d²phase| of the unwrapped phase, pooled AUC 0.887 on the radar/FHSS confusion). The
  second is a scalar cousin of option F4 (instantaneous frequency), so F4 is partly prototyped. Neither targets
  jammed-FHSS masking.
- Merging: her edits sit after `STFTBranch` and in `AMC_CNN.__init__` (extra `cumulant_features`, `if_features`
  arguments); C2 edits `STFTBranch` and adds one `stft_keep_rows` argument in the same constructor. Expect one
  small conflict there, resolved by keeping all three arguments in one signature. Do **not** merge the
  `results/*.pt` checkpoints she force-added for Render.

LFM_RADAR is a separate story: under a jammer it holds up (90% at +10 dB). Its milder loss is in
mixtures with FHSS (96% at −6 dB → 85% at +10 dB), where 30 of its misses were called FHSS. That is
E2, not this.

**Remaining diagnosis, if wanted.** Split the +10 dB misses by generator parameter (hop rate, channel
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
