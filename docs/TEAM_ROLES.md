# Team Roles — 4 People, 4 Days

## The standard we build to

The rules require **>90% recall on Military/CEMA and Jamming**. Treat that as the
**minimum to qualify**, not the target.

Here is why the distinction matters for us specifically: **none of our judged
classes came from the organisers.** Radar is RadChar, FHSS and jamming are our own
synthesis. So a high score on our own test split proves very little — it only
shows the model learned *our* data. The number that counts is measured on the
Qualifier IQ Stream, which none of us has seen.

Everything we do is therefore aimed at one question: **does this hold up on
signals we did not generate?**

### Targets

| Metric | Minimum | Our target |
|---|---|---|
| Recall on LFM_RADAR, FHSS, JAMMING | 90% | **95%, with margin across SNR** |
| Coarse-tier accuracy (Civilian / Military / Hostile) | — | **95%+** |
| Comms vs hostile CEMA discrimination | — | reported as its own number |
| Civilian classes | mandatory to classify | **90%+ at high SNR**, degradation documented |
| Held-out parameter generalisation | — | **accuracy holds on unseen parameters** |

### The four standards

1. **Randomise widely.** Every generated signal draws its parameters from a broad
   range, not a narrow "realistic-looking" one. A wider training distribution is
   more likely to contain the organisers' actual signal. Nyquist is the only
   limit, and `tests/test_config.py` enforces it.
2. **Prove generalisation, do not claim it.** Train on one parameter subset,
   evaluate on a disjoint one. If accuracy holds, the model learned the concept
   rather than our specific instance of it.
3. **Measure design choices, do not guess them.** Every component (class
   weighting, augmentation, SNR sweep) gets an ablation row showing what happens
   without it.
4. **State limitations before anyone else finds them.** A claimed 99% with no
   explanation invites a technical panel to go looking. Naming our own limits is
   more credible and takes one paragraph.

---

## The principle: own a signal category end-to-end

Each person owns **one signal category completely** — the science behind it, its
data, its quality checks, its performance in the results, its section of the
brief, its segment of the video.

You become the team's only expert on your category. Nobody else studies it.
Nobody studies anyone else's.

Each person also owns **one shared component**, because the pipeline that joins
the categories still needs owners.

| Person | Signal category | Shared component |
|---|---|---|
| **P1** | Civilian comms — BPSK, QPSK, 16QAM, 64QAM | Data pipeline (`build_dataset`, preprocessing) |
| **P2** | Radar — LFM | Model + training |
| **P3** | FHSS | Evaluation + metrics |
| **P4** | Jamming — barrage, tone, sweep | Inference + submission packaging |

---

## P1 — Civilian Comms + Data Pipeline

**Signal category:** BPSK, QPSK, 16QAM, 64QAM (4 of the 7 classes)
**Owns:** `src/data/`, RadioML loading, the assembled dataset

### Your signal, in one paragraph
These are the ordinary modulations of civilian spectrum — phones, WiFi, data
links. They encode bits by moving a point around a 2D constellation: BPSK uses 2
positions, QPSK 4, 16QAM 16, 64QAM 64. More positions means more bits per symbol
but less noise tolerance. They are **not** scored at 90%, but the rules mandate
classifying them, and they are the contrast that makes threat classes detectable.

### What you learn
`h5py` (about an hour), NumPy boolean indexing, `scipy.signal.resample_poly`.
Read [`pipeline/01-data-sources.md`](pipeline/01-data-sources.md) and
[`pipeline/05-preprocessing.md`](pipeline/05-preprocessing.md).

### Never need to learn
Chirp maths, hop sequences, jamming types, PyTorch.

### Tasks
| Day | Work |
|---|---|
| 0 | Start the RadioML download (~21 GB) — wall-clock, not effort |
| 1 | Implement `load_radioml_civilian()`. **Verify class order against the file** |
| 2 | Sample-rate reconciliation across all sources. Run `build_dataset`, publish per-class counts to the team |
| 3 | Rebalance on request from P2/P3 |
| 4 | Hand dataset table + attribution lines to P4 |

### Your traps
- `f['X'][:]` loads 21 GB into RAM and kills the process — slice, never load whole
- RadioML frames are `(1024, 2)`; pipeline wants `(2, 1024)`. Transposed data runs fine and is silently wrong
- RadioML SNRs are **even only**. `snr_bins_db` must stay even or your classes vanish from odd bins

### You write
Brief section: datasets, licences, attribution, class counts, SNR coverage.

---

## P2 — Radar + Model & Training

**Signal category:** LFM_RADAR (judged at >90%)
**Owns:** `src/generators/radar.py`, `src/models/`, `src/train.py`

### Your signal, in one paragraph
An LFM pulse is a **chirp** — frequency sweeps linearly across the pulse, so it
appears as a diagonal streak in a spectrogram. Radar is a *sensor*, not a
comms system: it transmits a short pulse then goes **silent to listen** for the
echo. Those silent gaps, repeating at the Pulse Repetition Interval, are its
signature — and the thing that separates it from sweep jamming.

### What you learn
The chirp concept (above — that is most of it), PyTorch training loop (already
written, you read and run it), recall vs accuracy, class weighting. Read
[`pipeline/02-radar-generation.md`](pipeline/02-radar-generation.md) and
[`pipeline/06-model-training.md`](pipeline/06-model-training.md).

### Never need to learn
HDF5 internals, hop sequences, jamming types.

### Tasks
| Day | Work |
|---|---|
| 0 | Start the RadChar download (Tiny or Small variant) |
| 1 | Extract RadChar LFM pulses. Plot spectrograms, confirm the diagonal. Run the smoke training config end to end so you are ready before real data lands |
| 2 | First real training run the moment P1 delivers. Report per-class recall to the team |
| 3 | Tune until the three judged classes clear 90%. **Data fixes before model changes** |
| 4 | Final run, export checkpoint + training curves |

### Your advantage
Radar is the one judged class with a **real published dataset** (RadChar, ICASSP
2023). Use it as your primary source — our generator is for augmentation and for
topping up thin SNR bins.

### You write
Brief sections: radar signal characterisation, architecture, training setup.

---

## P3 — FHSS + Evaluation

**Signal category:** FHSS (judged at >90%)
**Owns:** `src/generators/fhss.py`, `src/evaluate.py`

### Your signal, in one paragraph
Frequency Hopping Spread Spectrum jumps its carrier between channels hundreds of
times a second, following a pseudorandom sequence both ends agreed in advance.
It exists to resist interception and jamming — an eavesdropper without the
sequence catches only fragments. In a spectrogram it looks like scattered
rectangular blocks at different frequencies.

### What you learn
The hop concept (above), reading a spectrogram, `pytest`, and the metrics side:
recall, confusion matrices, accuracy-vs-SNR. Read
[`pipeline/03-fhss-generation.md`](pipeline/03-fhss-generation.md) and
[`pipeline/07-evaluation.md`](pipeline/07-evaluation.md).

### Never need to learn
HDF5, chirp maths, PyTorch model internals.

### ⚠️ Your category is the highest-risk one
Radar has RadChar. Jamming has distinctive broadband signatures. **FHSS has
neither** — no public raw-IQ dataset exists, so it is entirely our synthesis,
validated only by our own tests.

It also already had a real aliasing bug: 64 channels at 50 kHz spanned ±1.6 MHz
against a 500 kHz Nyquist limit, so outer hops folded back to wrong frequencies
entirely. Fixed, and `tests/test_config.py` now blocks it — **do not raise
`n_channels` or `channel_spacing_hz` without re-running the tests.**

### Tasks
| Day | Work |
|---|---|
| 1 | Generate FHSS, plot spectrograms, verify block pattern matches configured hop rate. Confirm tests pass |
| 2 | Tune parameters, produce publication-quality figures |
| 3 | Own the evaluation run. Read the confusion matrix, report which pairs confuse, drive fixes |
| 4 | Export confusion matrix, accuracy-vs-SNR, scorecard |

### You write
Brief sections: FHSS generation methodology and its honest limitation
(literature-derived ranges, not a specific emitter); results and metrics.

---

## P4 — Jamming + Inference & Submission

**Signal category:** JAMMING (judged at >90%)
**Owns:** `src/generators/jamming.py`, `src/infer.py`, brief assembly, video

### Your signal, in one paragraph
Jamming is defined by **intent, not form** — it borrows whatever waveform denies
the spectrum. Barrage floods wideband noise; tone parks continuous carriers;
sweep runs a fast repeating chirp. It is the only class that is inherently
hostile: nobody jams by accident. **JSR** (Jammer-to-Signal Ratio) sets how much
stronger the jammer is than its victim.

### What you learn
The three jamming types (above), JSR, and enough of everyone else's category to
narrate it — which the daily sync gives you for free. Read
[`pipeline/04-jamming-generation.md`](pipeline/04-jamming-generation.md) and
[`pipeline/08-inference-submission.md`](pipeline/08-inference-submission.md).

### ⚠️ Your hardest problem
Sweep jamming calls **the radar chirp function** — same maths. They differ only
in behaviour: radar pulses then listens; a jammer runs continuously and sweeps
faster. Put your sweep spectrogram beside P2's radar spectrogram. **If you two
cannot tell them apart by eye, the model will not either.** Sort this out with P2
on Day 2, not Day 4.

### Tasks
| Day | Work |
|---|---|
| 1 | Generate all three jamming types, plot, verify against reference figures. Start brief skeleton and video script |
| 2 | Tune parameters with P2 to separate sweep from radar. Draft brief sections as P1/P3 deliver |
| 3 | Run inference on the qualifier stream. **Confirm the file dtype first.** Record video draft |
| 4 | Assemble brief, final video edit, package, **submit with hours to spare** |

### Your trap
`load_iq_file()` assumes interleaved float32. If the organisers ship `complex64`
or `int16`, it still runs and produces a confident, worthless log. Before
trusting it: plot a spectrogram of the loaded data — does it look like *any*
radio signal? Check sample count against stated duration.

### You write
Brief: introduction, jamming methodology, limitations, Phase-2 roadmap, and
final assembly of everyone's sections.

---

## Decisions already made — do not relitigate

| Question | Decision |
|---|---|
| Representation | Raw IQ, not spectrogram (spectrogram destroys phase, killing P1's classes) |
| Jamming class | Standalone baseline, not overlay |
| GUI | Not built — Phase 2 only |
| Docker / TorchSig | Not used |
| Classes | 7, fixed |

### Effort that looks productive but is not

| Tempting | Why not |
|---|---|
| Transformer instead of CNN | Our bottleneck is data quality, not model capacity. Costs a day, changes little |
| Building a GUI now | Phase 2 only, explicitly not part of Phase 1 scoring |
| Chasing 90% on 64QAM at −10 dB | Physically impossible — the constellation sits inside the noise floor. Document the limit instead |
| Switching to spectrograms | Destroys phase, kills the civilian classes |
| TorchSig / Docker / GNU Radio setup | Does not cover our judged classes |
| Multi-label output | Correct in principle, needs full relabelling. Phase 2 |

## Order of work

Only start each once the previous is solid.

1. **Clear the gate** — >90% on all three judged classes
2. **Widen parameter randomisation** and retrain *(largest robustness gain)*
3. **Held-out parameter generalisation test** *(largest credibility gain)*
4. **Ablation table** *(brief quality)*
5. **Overlapping windows, test-time augmentation, seed ensemble** *(polish)*

Step 1 qualifies us. Steps 2 and 3 are what make the result defensible.

---

## What stays shared

| Item | Rule |
|---|---|
| `configs/default.yaml` | Everyone's parameters live here. **Announce before editing** |
| `pytest` | Green before every push, no exceptions |
| Class list / label order | Fixed Day 1. Changing it invalidates everything upstream |

---

## The daily sync — 15 minutes, not optional

Each evening, every person explains **in plain language** what their category is
and what they found. No jargon, no code.

Three jobs at once:

1. **P4 gets brief material** without reading anyone's code
2. **It is rehearsal for the video and jury Q&A** — if you can explain your
   category to a teammate who knows nothing about it, you can explain it to a judge
3. **It catches integration mistakes early** — "wait, your SNR values are odd
   numbers?" is a five-minute fix on Day 1 and a lost day on Day 3

---

## Combining at the end

### Technical brief

| Section | Author |
|---|---|
| Datasets, licences, attribution | P1 |
| Radar characterisation, architecture, training | P2 |
| FHSS methodology, results, metrics | P3 |
| Introduction, jamming, limitations, roadmap, assembly | P4 |

### Video (≤5 min) — each person narrates their own category

| Segment | Time | Who |
|---|---|---|
| Problem and the 7 classes | 0:30 | P4 |
| Civilian comms + data sources | 0:45 | P1 |
| Radar + model architecture | 1:15 | P2 |
| FHSS + results | 1:15 | P3 |
| Jamming + live inference | 1:00 | P4 |
| Limitations and roadmap | 0:15 | P4 |

Everyone speaking about their own category is easier to record and far more
convincing than one person narrating work they did not do.

---

## Honest risks with vertical ownership

**The seams are where this breaks.** Splitting by category means nobody owns the
*joins* — sample rates, label ordering, `configs/default.yaml`. Those are exactly
where the silent bugs have already appeared (aliasing, odd SNR bins, transposed
frames). The daily sync exists to catch them; treat it as load-bearing.

**P2 and P4 must coordinate directly.** Radar and sweep jamming share the same
chirp maths and are the predicted confusion pair. They cannot each tune in
isolation — they need a side-by-side comparison by Day 2.

**Single points of failure.** If one person drops out, nobody knows their
category. Pair up as backup: P1↔P4, P2↔P3. Each should be able to *run* the
other's scripts even if they could not have written them.

**P4 has the most calendar-sensitive load.** The brief and video are not Day-4
tasks. Anything not drafted by end of Day 3 will not be good. Start Day 1 with
placeholders.
