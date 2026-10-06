# Video Narration Script — word for word

**Group NEXA · Track I, Project Overwatch · SEDIC 2026 Phase 1**
Companion to [`VIDEO_PLAN.md`](VIDEO_PLAN.md). Cross-checked line by line against `SEDIC REPORT.docx`
(Technical Brief, Phase 1) on 2026-09-06 — every figure below appears in that report.

**Total 4:40** against the 5:00 cap. Read at ~145 words per minute; word counts are given per segment
so each person can time their own take. If a read runs long, cut adjectives, never numbers.

> **Read this first:** the cross-check section lists six places where the report contradicts the
> older repo docs. The
> script follows the report. Do not "correct" it back against `STATUS_AND_PLAN.md` — that file is
> stale, and one of its errors is a completely different model architecture.

---

## Runtime map

| # | Segment | Mode | In–Out | Len | Words | Who |
|---|---|---|---|---|---|---|
| 1 | Cold open | C | 0:00–0:22 | 22 s | 53 | Eileen |
| 2 | Data & DSP | B + C cut-in | 0:22–0:52 | 30 s | 73 | Eileen |
| 3 | Architecture | B | 0:52–1:30 | 38 s | 92 | Eavan |
| 4 | **Live demo — many signals in one window** | A | 1:30–1:52 | 22 s | 55 | Jessy |
| 5 | **The Performance page, top to bottom** | A + 2×C | 1:52–4:04 | **132 s** | 320 | Jessy |
| 6 | Limitations | C | 4:04–4:30 | 26 s | 63 | Chua |
| 7 | Close | C | 4:30–4:40 | 10 s | 27 | Chua |

**Per person:** Jessy 154 s · Eileen 52 s · Eavan 38 s · Chua 36 s.

> **External validation was cut from the video on 7 Sept.** The out-of-distribution test against an
> independently generated GNSS jamming dataset **remains in the technical brief, §7.2**, unchanged. The video simply does not
> cover it — a five-minute video cannot cover everything in the brief, and omitting a topic is not the
> same as reporting half of it.
>
> **Consequences to be aware of, not to hide:**
> - Every number spoken in the video is now measured on data we generated and evaluated ourselves.
>   There is no external check anywhere in the video.
> - **If a judge asks** — in Phase 2 or in writing — the answer is direct: we ran it, it is in brief
>   §7.2, recall on that external jamming set was 57.6% aggregate with zero false alarms, and the chirp
>   category showed our generator's fingerprint. Never suggest the test was not run.
> - Eavan loses his only face-to-camera moment; he now appears only as the head inset over the
>   architecture slides.

**The live demo was compressed on 7 Sept**, from two segments and 54 s to one segment and 26 s. Phase 1
requires no GUI or live demo — both are Phase 2 items — and everything the demo showed is proven better
by the benchmark numbers and the accuracy-vs-SNR curve. It stays because it proves the system runs end
to end, and because a working interface is a named Phase 2 advantage. §6 then gained 6 s of its own on
7 Sept to carry the overlay result, bringing the total to 4:42. The 28 seconds went to the two
credibility segments, which were carrying more weight than their runtime reflected.

**Screen recording is Jessy's, all of it** — every Mode A segment (4, 5, 6) is narrated by whoever
drives the console, so the narration follows what actually happens on screen instead of being synced
to someone else's cursor afterwards. Segments 4–6 are one continuous 1:52 block of her voice; the
Mode C cut-in inside §5 is the only break in it, so protect that cut.

**Bookends, decided 6 Sept — no group shots.** The opening goes to whoever narrates the segment that
follows it; the close to whoever narrated the segment before it. Eileen opens straight into her own
data segment; Chua closes straight out of her own limitations segment. No voice change at either seam,
and no shot needing four people in one room.

**Face to camera:** Eileen — cold open, RadioML cut-in. Eavan — external validation. Chua —
limitations, close. Jessy — metric cut-in. Everyone appears; nobody appears alongside anyone else.

**Briefing requirement.** Jessy narrates Eileen's signal work (inside §4) and Chua's jamming work
(§6). Both owners should walk her through their part before recording — even the one-sentence version
should sound like someone who understands the physics.

---

## 1 · Cold open (0:00–0:22) — Eileen, face to camera, 55 words

Eileen takes this because segment 2 is hers — she opens and continues straight into the data segment
with no voice change at the seam.

**EILEEN**
> In a contested band, civilian traffic, tactical emissions and deliberate interference all occupy the
> same spectrum at the same time. So the operator's first question is simple. Who is transmitting, and
> are they hostile?
>
> We answer that from raw IQ, across clean and faded conditions, over eight classes — and where the
> rules ask for radar or FHSS, we deliver both.
>
> We're Group NEXA. This is Project Overwatch.

**Record this separately from segment 2**, even though both are Eileen. Sixty-four continuous seconds
is too long for one read, and the cut gives the edit somewhere to breathe.

**On screen:** taxonomy assembling under the narration —
`BPSK · QPSK · 16QAM · 64QAM` / `LFM_RADAR · FHSS` / `JAMMING` / `NOISE_FLOOR`, grouped as
Civilian · High Priority · Hostile · Empty. Title card lands on the last line: **NEXA · Project
Overwatch · Track I**.

---

## 2 · Data & DSP logic (0:22–0:52) — Eileen, 73 words

**Mode B** (slides, head inset) except where marked.

> Our dataset is 128,400 windows across eight classes and six SNR bins, from minus ten to plus ten
> decibels. Civilian modulations come from RadioML. Radar comes from RadChar — every standalone radar
> window is real third-party data. FHSS, jamming and the empty channel we generate ourselves, because
> no public raw-IQ dataset covers them.
>
> Everything is forced through one DSP contract: 512 samples at 3.2 megahertz — a 160-microsecond
> window — normalised to zero mean and unit variance. That normalisation matters. Raw window power
> varies by eighteen decibels across our SNR range, and the slope differs per class.

**[CUT TO MODE C — Eileen face to camera, ~9 s of the above budget]**

> Left alone, the model would learn loudness instead of signal. And one more thing: the labels
> shipped with RadioML are wrong. We found it, and we fixed it against the source paper.

**On screen (Mode B):** `128,400 windows · 8 classes · 6 SNR bins` as large numerals; the format
contract table; the power-vs-SNR figure on the normalisation line. On the cut-in, a single pop-up:
**"classes.txt is wrong — corrected against O'Shea et al."**

**Direction:** the RadioML bug is the strongest credibility beat in the first half. Deliver it flat
and factual, not triumphantly. Don't explain the method on camera — it's in the brief, §3.1.

---

## 3 · Model architecture (0:52–1:30) — Eavan, 92 words

**Mode B.** *(See §9.1 — this segment was rewritten. Do not use the old 1D-CNN description.)*

> The classifier is a two-branch fusion network, trained entirely from scratch — there's no
> pretrained backbone for raw RF.
>
> The two branches exist because no single representation does both jobs. Raw IQ keeps amplitude and
> phase, which is what separates BPSK from 64QAM. A spectrogram throws phase away but makes frequency
> movement explicit — a chirp sweeping, a radio hopping. So we read the same window through both.
>
> The IQ branch uses dilated convolutions to widen its receptive field from nineteen samples to
> forty-three, enough to span multiple FHSS hops. The STFT branch is a small 2-D CNN over the
> spectrogram. We fuse them, pool with energy-gated attention, and output eight independent sigmoid
> probabilities — because a real window can be QPSK *and* jamming at once.
>
> The whole network is 148,938 parameters.

**On screen:** two-branch diagram with tensor shapes, fusion at 192 channels (128 IQ + 64 STFT),
head 192 → 256 → 8. The parameter count as a large standalone numeral at the end.

**Direction:** the last line is the punch — a network this small doing this job is the architectural
argument. Pause before it.

---

## 4 · Live demo — many signals in one window (1:30–1:52) — Jessy, 55 words

**Mode A** — console screen capture, opening on a stream that is **already running**. No loading, no
file picker, no walkthrough. The job is to show the multi-label capability working on a real stream;
the numbers that prove it are in §5.

> A mixed sequence, already running — several emitters sharing the band, none of them labelled.
>
> The model reports every signal it finds in a window, with a confidence for each. Here: QPSK and
> jamming, together, in the same 160 microseconds.
>
> And in Signal Analysis, window by window — a chirp here, a hop pattern there.

**Shot notes — the sequence is already loaded and playing when the segment opens.** Do not film the
load, the file picker, or the first empty seconds; the segment starts on a console that is already
working.

1. **RF Replay, mid-stream.** Detection events populating. Frame a window that carries **two labels at
   once** with their confidences visible — that is the whole point of the shot.
2. **Cut to Signal Analysis.** Step through two or three windows that resolve to *different* signals —
   a chirp with its listening gap, then a hop pattern. Let the picture do the explaining.
3. **The graph: a glance, not a tour.** Do not explain axes, colour scales or window length. Segment 5
   is where the evidence lives; this is only proof the thing runs.

**Pick your window carefully before recording.** Per-class thresholds differ a lot — JAMMING is 0.87,
LFM_RADAR 0.22, FHSS 0.25 — so a window can show radar flagged positive at 0.30 while a higher-looking
number elsewhere is negative. On camera that reads as a bug. Choose a window where the displayed
confidences and the flags agree at a glance, or the segment raises a question it has no time to answer.

**Direction:** twenty-two seconds. The temptation with a working console is to demonstrate it; resist.
The one idea this segment plants is *one window, several signals, each with a confidence* — which is
what segment 5 closes on with the single-vs-multi numbers.

---

## 5 · The Performance page, top to bottom (1:52–4:04) — Jessy, 320 words

**Mode A, one continuous scroll down the console Performance page, with two Mode C cut-ins.**
126 seconds — 45% of the video. It absorbs the old comms-vs-jamming segment, because the CEMA figures
live in this page's own category table; splitting them out meant showing the same table twice.

It carries **65 of the 100 rubric points**: Performance Benchmark 25, Mandatory Classification 30 (the
multi-condition half), Competitive Advantage 10.

> This is the Performance page. Everything on it is the five-model ensemble, on a held-out test split
> of 19,260 windows, with per-class thresholds calibrated on separate validation data.
>
> Benchmark: pass. One card per judged class.

**[CUT TO MODE C — Jessy face to camera, ~30 s]**

> Each card states four metrics, in a deliberate order.
>
> Recall is primary — it is what the rule is written against, and what the thresholds were calibrated
> for. 85.0, 82.8, 84.44, against a bar of 80.
>
> Balanced accuracy is second, because the organiser's word is "accuracy" and it is never defined.
> This is the reading that survives the class imbalance: 85.6, 86.0, 92.2. We clear the bar either way.
>
> Precision and F1 are shown for transparency, not because they are judged. Radar and FHSS sit near 54
> and 59 percent — the direct cost of buying recall margin.
>
> And plain accuracy is printed last and de-emphasised, with the reason underneath it: a model that
> predicts nothing at all already scores 84.1.

**[BACK TO MODE A — scroll to the category table]**

> By category — and the CEMA row, the discrimination the rules weigh most heavily. Comms versus
> jamming, 97.05 percent across 16,290 windows, at 84.44 jamming recall and a 0.03 percent false alarm
> rate: four civilian windows out of 13,230.
>
> Per-class recall across all eight. And the scorecard under it is per class, per window, ungated and
> unsmoothed — no pooling, no temporal smoothing, one 160-microsecond window judged alone.
>
> Recall against SNR, per class, against that same 80 percent line.
>
> All three judged classes clear it from minus six decibels upward — five of the six bins. Only minus
> ten fails, and there radar is 67.8, FHSS 58.6, jamming 47.8.
>
> Civilian isn't benchmark-gated, but on the same line: BPSK clears from minus two up, 16QAM and 64QAM
> from plus two up. QPSK never reaches it — it peaks at 74.3.

**[CUT TO MODE C — Jessy, ~24 s]**

> Dense QAM is the one place we do not decide per window. A 512-sample window carries about 56
> symbols, and 16QAM and 64QAM are closer together than the estimator's own spread at that count — so
> this is not something a better model fixes.
>
> Pooled across windows it resolves: 75 percent at eight windows, 84 at sixteen, 92 at thirty-two, 98
> at sixty-four. Below eight windows, or below 2 dB, it refuses to decide rather than guessing.

**[BACK TO MODE A — the last panel]**

> The last panel splits single-signal windows from overlapping ones — and this matters more than it
> looks. Seven in ten of the windows behind each judged class have more than one signal in them. The
> headline number already is the hard case.
>
> Radar and FHSS drop to 79.7 and 78.0. Jamming goes the other way, up to 85.2 — a jammer is easier to
> see against a victim than in empty spectrum.
>
> And a single-label model can't score here at all. One label per window means that on a two-signal
> window, you are capped at half.

---

### Page flow — every section, in the page's own order

| # | On screen | Say | Mode |
|---|---|---|---|
| 1 | Provenance line | ensemble · 19,260-window held-out split · thresholds calibrated on validation | A |
| 2 | **Benchmark verdict** | "Benchmark: pass" | A |
| 3 | The three judged-class cards | **the four-metric hierarchy** | **C** |
| 4 | **By category** table + CEMA row | tiers, then comms-vs-jamming — 97.05% / 84.44% / 0.03% | A |
| 5 | **Per-class recall** bar chart | all eight classes | A |
| 6 | **Scorecard** | "per class, per window, ungated and unsmoothed" | A |
| 7 | Confusion matrix | show, do not narrate — radar/FHSS is §6's story | A |
| 8 | **Recall vs SNR** | **per-class pass ranges** — see table below | A |
| 9 | **Dense-QAM pooling** | resolves with window count; refuses below 8 or below 2 dB | **C** |
| 10 | **Single- vs multi-signal** | the multi-label payoff | A |

### The four metrics, and what each represents

The page orders them deliberately. Say the order, not just the numbers.

| Metric | Status on the page | Why |
|---|---|---|
| **Recall** | **Primary** | What the rule is written against; what thresholds were calibrated for |
| **Balanced accuracy** | **Supporting** | The defensible reading of the organiser's "accuracy" — corrects for class imbalance, cannot be gamed by silence |
| Precision, F1 | Transparency | Stated rather than omitted. Low precision on radar/FHSS is the cost of recall margin; a reader finding it unannounced is worse than us naming it |
| Plain accuracy | **De-emphasised** | Included for completeness only. Printed with its trivial baseline beside it |

| Judged class | Recall | Balanced acc. | Precision | F1 | Plain acc. | Trivial baseline |
|---|---|---|---|---|---|---|
| LFM_RADAR | **85.0%** | **85.6%** | 53.8% | 65.9% | 86.0% | 84.1% |
| FHSS | **82.8%** | **86.0%** | 58.9% | 68.8% | 88.1% | 84.1% |
| JAMMING | **84.44%** | **92.2%** | 99.5% | 91.3% | 97.5% | 84.1% |

**The argument this makes:** we pass under both readings of an undefined requirement, and plain
accuracy is disqualified on its own terms — 84.1% is what a model outputting nothing already scores.

### Recall vs SNR — where each class clears the bar

| Class | −10 dB | −6 dB | −2 dB | +2 dB | +6 dB | +10 dB | Clears 80% from |
|---|---|---|---|---|---|---|---|
| **LFM_RADAR** *(judged)* | 67.8 | **92.2** | **86.9** | **88.6** | **89.6** | **84.9** | **−6 dB up** |
| **FHSS** *(judged)* | 58.6 | **91.8** | **89.0** | **90.6** | **85.9** | **81.2** | **−6 dB up** |
| **JAMMING** *(judged)* | 47.8 | **83.5** | **90.0** | **94.5** | **94.3** | **96.5** | **−6 dB up** |
| BPSK | 8.9 | 63.8 | **85.7** | **88.4** | **87.3** | **88.3** | −2 dB up |
| 16QAM | 7.6 | 41.4 | 71.4 | **86.3** | **90.0** | **90.5** | +2 dB up |
| 64QAM | 1.9 | 16.4 | 63.6 | **83.6** | **86.2** | **85.1** | +2 dB up |
| QPSK | 0.2 | 1.6 | 57.3 | 72.9 | 74.3 | 72.9 | **never — peaks 74.3 at +6 dB** |
| NOISE_FLOOR | 48.0 | 52.7 | 49.3 | 54.7 | 60.0 | 62.7 | never — bookkeeping label |

Recall %, from `evals/csv/accuracy_by_class_snr.csv`. Bold = at or above 80%. The 80% bar is only
binding on the three judged classes; the civilian rows are measured against the same line for
comparison, not because the rules apply it to them.

**What to say, and what it buys.** Stating a *range* per class is stronger than "it works at high SNR":
all three judged classes hold above the bar across five of six bins, which is the multi-condition
requirement evidenced rather than asserted. And naming the −10 dB failure in the same breath means the
panel hears it from us rather than reading it off the figure.

**QPSK never clears 80% at any SNR.** It is not benchmark-gated, so this is not a failure against the
rules — but say it plainly rather than letting the curve be scanned. It peaks at 74.3% at +6 dB. The
brief names QPSK as the civilian exception in §6.2; the video should not appear to have skipped it.

### Overlapping signals — how to frame it convincingly

| Judged class | Single-signal | Overlapping | Share overlapping | Headline (blend) |
|---|---|---|---|---|
| LFM_RADAR | 97.7% *(900 win.)* | 79.7% *(2,160 win.)* | **70.6%** | **85.0%** |
| FHSS | 94.6% *(900)* | 78.0% *(2,160)* | **70.6%** | **82.84%** |
| JAMMING | 82.7% *(900)* | **85.2%** *(2,160)* | **70.6%** | **84.44%** |

**Three things make this convincing, in this order.**

**1. The headline already is the hard case — this is the fact to lead with.** Seven in ten of the test
windows behind every judged class contain more than one signal. The benchmark was not passed on clean
isolated emitters and then quietly averaged; 85.0 / 82.84 / 84.44 *are* the blend, and they reproduce
exactly from the two columns above. Nothing here is a second, easier scorecard — the console labels
this panel "additive to the scorecard, not a competing one," and that is literally true.

**2. A single-label model cannot compete on this subset at all.** A softmax classifier emits one label
per window. On a two-signal window it can be right about at most one of them, so its recall is capped
at **50%**; on a three-signal window, **33%**. Our 79.7% and 78.0% are not being measured against 100 —
they are measured against a ceiling most published AMC architectures cannot reach by construction.
That is the payoff of the independent-sigmoid head from §3, and it is worth one sentence on camera.

**3. Jamming gets *better* under overlap** — 82.7% alone, 85.2% with a victim present. That is the
operationally meaningful direction. A jammer with something to jam is the real case; a jammer alone in
empty spectrum is the artificial one, and it is the one we do slightly worse on.

**Say the drop, don't dress it.** Radar and FHSS fall to 79.7% and 78.0% in overlapping windows —
fractionally under the bar *in that slice*. The benchmark is measured on the full test set and passes;
this panel is the diagnostic that shows where the difficulty sits. Naming a number the panel can read
off the chart themselves is what makes the rest of the segment credible.

### Dense-QAM pooling — and the limitation to say out loud

| Windows pooled | 16QAM vs 64QAM accuracy |
|---|---|
| 1 | 59.3% — *below minimum, refused* |
| 8 | 75.4% |
| 16 | 84.3% |
| 32 | 92.1% |
| 64 | 98.0% |

**State the limitation, don't bury it.** The model classifies **per window**. Naming *which* dense QAM
is not done by the network at all — it is a statistical estimator applied afterwards, and it needs
multiple windows of the same emitter to work. Per window, combined dense-QAM recall (did it notice
dense QAM was present, regardless of which it named) is **65.6% over 7,290 windows**.

Two refusals worth mentioning as strengths, not caveats:

- **Below 8 windows it refuses to report an order** rather than guessing.
- **Below 2 dB SNR it refuses too** — the channel pulls the estimator toward zero, so no order is
  reported at all.

Also true and worth knowing if asked: pooling fixes the estimator only. Averaging the *model's* own
16QAM/64QAM probabilities stays at chance however many windows you use, because that error is a bias
rather than noise.

### This puts a number in the video that the brief does not contain

The dense-QAM pooling table is on the console but **not in the technical brief**, which reports only
the 65.58% combined figure. With the pooling result now spoken on camera, **the brief must be updated
to match before the PDF is exported.** This is no longer optional — it is the one place the two
submitted artefacts would contradict each other.

**Direction:** 126 seconds is long. The two Mode C cut-ins are the only breaks in it and they are
load-bearing — protect both. Every number exactly as written: 84.44, not "about 84."

---

## 6 · Limitations (4:04–4:30) — Chua, Mode C face to camera, 63 words

> Four limits we'd rather state than have you find.
>
> Radar and FHSS are tuned for recall, so roughly half their positive flags are false alarms.
> Dense QAM is weak at low SNR — which the literature says it is for everyone. Barrage jamming,
> at 69.3 percent, is our weakest sub-type.
>
> And at minus ten decibels, all three judged classes fall below the bar. The pass is a test-set
> average, not a per-condition guarantee.

**On screen:** four lines appearing as spoken. No decoration.

**Direction:** flat delivery, no hedging words — no "unfortunately," no "a little." Owning the limits
in person is the point of the shot.

---

## 7 · Close (4:30–4:40) — Chua, face to camera, 27 words

Chua carries straight on from her own limitations segment — same framing, no voice change at the seam.

**CHUA**
> Next is emitter characterisation — RadChar's pulse-width and PRI labels point that way. Attribution
> is out of scope, and we say so.
>
> We're NEXA. Thanks for watching.

**End card, held to last frame — 4 seconds minimum:** **NEXA** · Eavan Tan · Jessy Pang Xin Yuan ·
Chua Xin Ying · Eileen Yeoh Jing Han · Universiti Teknologi PETRONAS · Track I, Project Overwatch ·
SEDIC 2026.

With no group shot anywhere in the video, this card is the only place the panel sees the whole team.
Hold it long enough to read comfortably, and spell every name exactly as on the brief cover.

---

## Cross-check against the report — what changed and why

Six differences between the report and the older repo docs. **The report wins in every case**, and the
script above already reflects that. Listing them because anyone editing this script from memory will
otherwise reintroduce them.

### 9.1 The architecture in the old docs is a different model

`STATUS_AND_PLAN.md:33` describes "a small 1D-CNN (~4.26M parameters, two conv blocks + two linear
layers)." The report §4 and `web/data/model_card.json` both describe a **two-branch fusion CNN — IQ
branch with dilated convolutions plus an STFT branch, energy-gated attention pooling, 148,938
parameters**. That is a 29× difference in parameter count and a different architecture entirely.

The first draft of the video plan carried the old description. It is corrected here. **Nobody should
describe this model as a 1D-CNN on camera.**

### 9.2 The inference-speed claim has been dropped

The old docs quote 2.0 ms per prediction and ~1,000 windows/second. Those were measured on the 4.26M
1D-CNN, not on this network, and neither the report nor the model card contains a speed figure for
the current architecture. The claim is therefore unverified and has been cut from the script.

If you want it back, re-measure it on the shipped ensemble first. The console running in-browser is
already sufficient evidence of "small and fast" without a number attached to it.

### 9.3 The low-SNR limitation is broader than previously stated

The earlier plan said jamming is weak at −10 dB. The report §6.2 is blunter: **all three judged
classes fall below the bar at −10 dB** — LFM_RADAR 67.8%, FHSS 58.6%, JAMMING 47.8%. The script says
this. It is also the honest framing, and it matches what a judge will read in §8.2.

### 9.4 Barrage jamming is 69.3%, not "72–82%"

`STATUS_AND_PLAN.md` gives a range from an older diagnostic. Report §6.5 measures 69.3% on the shipped
ensemble — below the 80% bar as a sub-type. The script uses 69.3%.

### 9.5 The dense-QAM pooling claim was cut

An earlier draft used "pooled over 32 windows, dense QAM resolves at 92.1%," from
`web/data/performance.json`. The report does not make that claim; it reports **combined dense-QAM
recall of 65.58% across 7,290 windows**, with a literature cross-check (§7.3) showing sub-0 dB
performance under 50% is normal even for dedicated classifiers. The script follows the report. Don't
reintroduce the pooling number unless it goes into the brief first.

### 9.6 What the report added that the plan didn't have

The out-of-distribution test (§7.2) did not exist when the video plan was written, and it is the
single most valuable credibility asset in the submission — an independently generated GNSS jamming dataset, never trained
on, 57.6% recall at 0.0% false alarm. It earned its own segment (§7), and gained ten seconds on
7 Sept when the live demo was compressed.

The three methodology faults in §7.1 — threshold leakage, zero-margin calibration, and a single-model
threshold applied to the ensemble — are equally strong, but there is no runtime for them. They stay
in the brief. If a segment overruns and something must be cut, cut §4 further, then §2's dataset
detail. Never §7.

---

## Before recording

- [ ] **Proofread report §8.0 and §8.1** — several sentences have missing spaces and typos
      ("They are states openly", "not field captures under generated benchmark datasets.not field
      captures under operational interference", "Civilian-civilian overlaps are out of scope.The").
      §8 duplicates a clause. That section is worth 20 points as a PDF; fix before export.
- [ ] Every figure spoken above diffed against the report one final time on recording day
- [ ] Console rebuilt so the on-screen numbers match the script (`python web/build.py`)
- [ ] Cold open (Eileen) and close (Chua) shot in matching framing and lighting, so the bookends pair
      even though they are different people
- [ ] Eileen's cold open recorded as a separate take from segment 2
- [ ] Rough cut timed before any colour or motion work
- [ ] If over 5:00: trim §4 further, then §2. Never §5.
- [ ] **Add the dense-QAM pooling table to the technical brief** — the video now states it; the
      PDF must match. Blocking.
- [ ] **Decide the dense-QAM panel.** The Performance page shows dense-QAM order resolving across an
      event — 59.3% on one window rising to 92.1% pooled over 32 — but **the technical brief does not
      make that claim**; it reports combined dense-QAM recall of 65.58%. Showing the panel on camera
      would put a number in the video that the PDF does not support. Either add it to the brief or
      scroll past it.
