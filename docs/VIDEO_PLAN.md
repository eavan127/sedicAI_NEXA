# Video Demonstration — Plan

**Group NEXA · Track I, Project Overwatch: Cognitive CEMA · SEDIC 2026 Phase 1**
Hard cap 5:00. Target runtime **4:40**, leaving 20 s of margin.

> **Structure locked 6 Sept 2026** against the technical brief. The word-for-word narration is
> [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md); this document and that one describe the same ten segments. If
> they ever disagree, the script wins — it was written against the report.

---

## 1. Why the shape of this video is what it is

The rubric routes more points through the video than through the technical brief:

| Criterion | Pts | Where it is earned |
|---|---|---|
| Mandatory Classification | 30 | Brief + video (multi-condition evidence) |
| Performance Benchmark | 25 | **"Submitted as a proven *video* performance benchmark"** |
| Competitive Advantage | 10 | Comms-vs-jamming, best *demonstrated* |
| Technical Brief | 20 | PDF |
| Video Demonstration | 15 | "Explain model functionality and results" |

**40–50 of 100 points depend on this five-minute video.** Runtime is therefore allocated against
the points table, not split evenly between four narrators.

**No organiser IQ stream exists** (confirmed with the organiser). Every number shown is measured on
our own data, with no external check anywhere in the loop. The panel knows this. Credibility of the
evaluation is therefore itself a scored property — held-out split, ensemble, and named limitations
are not modesty, they are the substitute for external validation.

---

## 2. Production modes

| Mode | Description | Used for |
|---|---|---|
| **A** | OMNI console screen recording + voice-over | Live demo, benchmark, comms-vs-jamming |
| **B** | Slides + voice-over, small talking-head inset | Data/DSP, architecture |
| **C** | Face to camera with pop-up content | Cold open, transitions, close |

Mode C stays short and is used at structural joins — opening, the seam into the benchmark, and the
close. Pop-ups carry content in or out; they do not run under a talking head for long stretches.

---

## 3. Runtime plan

| # | Segment | Mode | In–Out | Len | Narrator | Serves |
|---|---|---|---|---|---|---|
| 1 | Cold open — the mission | C | 0:00–0:22 | 22 s | Eileen | Framing |
| 2 | Data & DSP logic | B + C | 0:22–0:52 | 30 s | Eileen | MC 30 · TB 20 |
| 3 | Model architecture | B | 0:52–1:30 | 38 s | Eavan | VD 15 · TB 20 |
| 4 | **Live demo — many signals in one window** | A | 1:30–1:52 | 22 s | Jessy | VD 15 |
| 5 | **The Performance page, top to bottom** | A + 2×C | 1:52–4:04 | **132 s** | Jessy | **PB 25 · MC 30 · CA 10** |
| 6 | Limitations | C | 4:04–4:30 | 26 s | Chua | Credibility |
| 7 | Close | C | 4:30–4:40 | 10 s | Chua | — |

**Per person:** Jessy 154 s · Eileen 52 s · Eavan 38 s · Chua 36 s. Total **4:40**.

### Two changes made 7 Sept

**External validation was cut from the video.** The out-of-distribution test against real recorded
GNSS jamming **remains in the technical brief, §7.2**, unchanged. A five-minute video cannot cover
everything in a brief, and omitting a topic is not the same as reporting half of it — but be clear
about what it costs:

- Every number spoken in the video is now measured on data we generated and evaluated ourselves.
  There is no external check anywhere in the video.
- **If a judge asks**, the answer is direct: we ran it, it is in brief §7.2, recall on real recorded
  jamming was 57.6% aggregate with zero false alarms, and the chirp category showed our generator's
  fingerprint. Never suggest the test was not run.
- **Eavan loses his only face-to-camera moment.** He now appears only as the head inset over the
  architecture slides, and drops to 42 s. If you want him back on camera, the cheapest fix is a short
  Mode C cut-in inside segment 3 — the 148,938-parameter line is the natural place for it.

**The freed 28 s went into the Performance page**, taking segment 5 from 58 s to 86 s. That segment is
now nearly a third of the video, which is defensible: it is the only place the 25-point Performance
Benchmark row and the multi-condition half of the 30-point Mandatory Classification row are evidenced.

### The live demo is a feature, not evidence

**Decided 7 Sept.** The console demo was two segments and 54 seconds; it is now one segment and 26.
The reasoning, stated so nobody re-expands it by accident:

- **Phase 1 does not require a GUI or live demo.** Both are Phase 2 items, named as such in the
  announcement. Nothing in the Phase 1 rubric asks for one.
- **Everything the demo showed is proven better elsewhere.** "Model functionality" is proven by the
  benchmark numbers. "Multi-condition" is proven by the accuracy-vs-SNR curve across eight classes and
  six SNR bins — which is what the 30-point Mandatory Classification row actually asks for. One
  low-SNR replay of one file is an anecdote next to that curve.
- **What it does uniquely** is show the system runs end to end rather than existing as a notebook of
  metrics, which makes a panel more inclined to believe the numbers. And it plants the GUI early: a
  working interface is named a "significant Competitive Advantage" in Phase 2 judging. Both are worth
  26 seconds. Neither is worth 54.

**The 28 seconds went to the two credibility segments** — external validation 18 → 28 s, limitations
16 → 26 s — which is where the video was thinnest relative to how much it was carrying. Total runtime
dropped to 4:36. **Segment 6 then took 6 s back on 7 Sept** to carry the jamming-overlay result, so the
total is 4:42 — the same runtime as before the compression, but spent on evidence rather than on a
console walkthrough. That trade is the whole point of the change.

**Screen recording is Jessy's, all of it.** Every Mode A segment — 4, 5, 6 and 7 — is narrated by
whoever is driving the console, which removes any need to sync a voice-over to someone else's cursor
and lets the narration follow what actually happens on screen rather than a rehearsed guess.

### Bookends: no group shots

**Decided 6 Sept.** The cold open and the close are delivered by one person each, not by all four
together. The rule is continuity: **the opening goes to whoever narrates the segment that follows it,
and the close to whoever narrated the segment before it.** So Eileen opens straight into her own data
segment, and Chua closes straight out of her own limitations segment — no voice change across either
seam, and no shot requiring four people in one room at one time.

Every member still appears on camera; there is simply no shot with all four in frame.

| Member | Face to camera | Voice over console/slides |
|---|---|---|
| Jessy | **Two cut-ins inside §5** | Live demo, and the whole Performance page walkthrough |
| Eileen | **Cold open**, RadioML cut-in | Data & DSP (head inset) |
| Eavan | External validation | Architecture (head inset) |
| Chua | Limitations, **close** | — |

**Two consequences of these two decisions, stated plainly so nobody is surprised in the edit.**

1. **Jessy narrates 2:20 of a 4:42 video — 50% of it**, and segments 4 through 6 are one continuous
   console block sitting across the middle of the video. The **two** Mode C cut-ins inside §5 are the
   only breaks in it, so protect both cuts in the edit. This is the price of putting the Performance
   page at the centre and giving every console segment to the person driving the console; it is
   defensible, but it should be a decision rather than a drift.
2. **Chua is at 36 seconds** and **Eavan at 42**, and neither narrates work the other segments cover.
   Chua built the jamming generators and does not explain them; Eavan lost his only face-to-camera
   moment when external validation was cut. A short Mode C cut-in inside segment 3 — on the
   148,938-parameter line — is the cheapest way to put him back on camera.

**If you want to even that out** without touching the screen-recording rule: give segment 7 (external
validation) to Chua instead of Eavan — it is a jamming-generalisation result as much as an architecture
one, and it would put her at 64 s against Eavan's 42 s. One swap, no other knock-ons.

**Ownership note.** Segment 8a is Eavan's because out-of-distribution behaviour is an
architecture-generalisation result. Since Jessy now narrates four segments covering three other
people's work, she needs to be briefed well enough to answer a Phase 2 question about any of it — or
to hand that question to the person who owns it.

---

## 4. Segment detail

### 1 · Cold open (0:00–0:22) — Mode C, Eileen

Single narrator, face to camera. Eileen takes it because segment 2 is hers — she opens and then
continues straight into the data segment with no voice change at the seam.

**Beats:** the problem (three kinds of signal sharing one band) → the operator's question (who is
transmitting, and are they hostile) → what we built (eight classes, raw IQ, clean and faded) → team
and track.

**Pop-up under the narration.** The eight-class taxonomy assembling as it is described:
`BPSK · QPSK · 16QAM · 64QAM | LFM_RADAR · FHSS | JAMMING | NOISE_FLOOR`
grouped as Civilian / High Priority / Hostile / Empty.

**Must land:** we deliver radar **and** FHSS where the rules ask for radar *or* FHSS, plus an eighth
empty-channel class that isn't required. One clause, don't dwell.

Title card at the end of the read: **NEXA · Project Overwatch · Track I**.

**Recording note:** shoot the open and segment 2 as separate takes even though both are Eileen —
64 continuous seconds is too long for one read, and the cut gives the editor somewhere to breathe.

### 2 · Data & DSP logic (0:22–0:52) — Mode B, Eileen

The rubric names "DSP logic" explicitly in both the brief row and the announcement. Use that word.

**Beats, in order:**

1. Sources — RadioML 2018.01A for the four civilian modulations; RadChar real pulses blended with our
   own generator for LFM; FHSS and jamming fully synthetic. Licences on screen (both CC BY-NC-SA 4.0).
2. **The DSP contract** — every window is 512 samples at 3.2 MHz, i.e. **160 µs**, normalised, same
   shape and statistics regardless of class or source. This is what makes eight heterogeneous sources
   comparable to one model.
3. SNR sweep: −10, −6, −2, +2, +6, +10 dB. This is the "multi-condition" requirement, planted early.
4. **The RadioML class-index bug** — the shipped class-order file does not match the actual data. We
   found it and fixed it. Say this plainly; with no organiser data, it is the strongest available
   evidence that we audited our inputs rather than trusting them.

**On screen:** 128,400 windows total · 8 classes · 6 SNR bins. One spectrogram strip showing LFM,
FHSS and jamming side by side.

### 3 · Model architecture (0:52–1:30) — Mode B, Eavan

> **Rewritten 6 Sept 2026.** This segment previously described a 1D-CNN with ~4.26M parameters, taken
> from `STATUS_AND_PLAN.md`. That is not the model we ship. See
> [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) §9.1.

**Beats:**

1. **Two-branch fusion network**, trained from scratch — no pretrained backbone exists for raw RF.
2. Why two branches: raw IQ keeps amplitude and phase (what separates BPSK from 64QAM); a spectrogram
   discards phase but makes frequency movement explicit (a chirp sweeping, a radio hopping). No single
   representation does both jobs, so the same window is read through both.
3. **IQ branch** — one standard convolution plus two dilated convolutions, widening the receptive
   field from ~19 to ~43 samples, enough to span multiple FHSS hops. **STFT branch** — small 2-D CNN
   over the spectrogram. Fused to 192 channels (128 IQ + 64 STFT), pooled with energy-gated attention.
4. Head is 192 → 256 → 8 with **independent sigmoid outputs**, not softmax — a window can be QPSK
   *and* jamming at once.
5. **148,938 parameters** total. Pause before this line; it is the punch.

**On screen:** two-branch diagram with tensor shapes, the 192-channel fusion point, and the parameter
count as a large standalone numeral at the end.

**Do not state an inference-speed figure.** The 2.0 ms / 1,000-per-second numbers in the old docs were
measured on the superseded architecture and have been withdrawn (`STATUS_AND_PLAN.md` §6). The console
running the ensemble in a browser is the real-time evidence, and it needs no number attached.

### 4 · Live demo — many signals in one window (1:30–1:52) — Mode A, Jessy

**22 seconds, opening on a stream that is already running.** No loading, no file picker, no
walkthrough. This segment plants one idea — **one window, several signals, each with a confidence** —
and hands straight to the Performance page for the evidence.

**Three beats:**

1. **RF Replay, mid-stream.** Several emitters sharing the band, none labelled, detections populating
   per window. Frame a window carrying **two labels at once** with their confidences visible.
2. **Signal Analysis, window by window.** Two or three windows resolving to *different* signals — a
   chirp with its listening gap, then a hop pattern. Let the picture explain.
3. **The graph: a glance, not a tour.** No axes, no colour scales, no window-length explanation.

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

**Why it is only 22 seconds.** Phase 1 requires no GUI or live demo — both are Phase 2 items — and
everything a longer demo would show is proven properly in §5. What this segment uniquely does is make
the multi-label design visible before §5 quantifies it, and plant the working interface that counts as
a "significant Competitive Advantage" in Phase 2 judging.

### 5 · The Performance page, top to bottom (1:52–4:04) — Mode A + 2×C, Jessy

**One continuous scroll down the console Performance page, 126 seconds — 45% of the video.**

**It absorbs the old comms-vs-jamming segment.** The CEMA figures live in this page's own category
table, so keeping them as a separate segment meant putting the same table on screen twice. Folding
them in also means the video's structure now matches the console's, which is easier to record and
easier to follow.

It carries **65 of the 100 rubric points** — Performance Benchmark 25, the multi-condition half of
Mandatory Classification 30, and Competitive Advantage 10.

**Flow — every section of the page, in the page's own order. Do not reorder:**

| # | On screen | Say | Mode |
|---|---|---|---|
| 1 | Provenance line | ensemble · 19,260-window held-out split · thresholds calibrated on validation | A |
| 2 | **Benchmark verdict** | "Benchmark: pass" | A |
| 3 | The three judged-class cards | **the four-metric hierarchy** — first cut-in | **C** |
| 4 | **By category** table + CEMA row | tiers, then comms-vs-jamming: 97.05% / 84.44% / 0.03% | A |
| 5 | **Per-class recall** bar chart | all eight classes | A |
| 6 | **Scorecard** | say the phrase: "per class, per window, ungated and unsmoothed" | A |
| 7 | Confusion matrix | show, do not narrate — radar/FHSS is §6's story | A |
| 8 | **Recall vs SNR** | clears by −6 dB, all three fall below at −10 dB | A |
| 9 | **Dense-QAM pooling** | resolves with window count; refuses below 8 windows or 2 dB | **C** |
| 10 | **Single- vs multi-signal** | the multi-label payoff | A |

#### The four metrics, and what each represents

The page orders them deliberately, and the order *is* the argument. Say the order, not just the
numbers.

| Metric | Status on the page | Why |
|---|---|---|
| **Recall** | **Primary** | What the rule is written against; what the thresholds were calibrated for |
| **Balanced accuracy** | **Supporting** | The defensible reading of the organiser's "accuracy" — corrects for class imbalance, cannot be gamed by silence |
| Precision, F1 | Transparency | Stated rather than omitted. Low precision on radar/FHSS is the direct cost of recall margin; a reader finding it unannounced is worse than us naming it |
| Plain accuracy | **De-emphasised** | For completeness only, printed with its trivial baseline beside it |

| Judged class | Recall | Balanced acc. | Precision | F1 | Plain acc. | Trivial baseline |
|---|---|---|---|---|---|---|
| LFM_RADAR | **85.0%** | **85.6%** | 53.8% | 65.9% | 86.0% | 84.1% |
| FHSS | **82.8%** | **86.0%** | 58.9% | 68.8% | 88.1% | 84.1% |
| JAMMING | **84.44%** | **92.2%** | 99.5% | 91.3% | 97.5% | 84.1% |

**What this establishes:** we pass under *both* readings of an undefined requirement, and plain
accuracy is disqualified on its own terms — 84.1% is what a model outputting nothing already scores.
That closes the metric question rather than deflecting it.

#### Comms vs. hostile CEMA — now stop 4, not its own segment

| Tested | Windows | Result |
|---|---|---|
| Jamming present | **3,060** | **84.44%** caught |
| — overlaid on a real transmission | 2,160 | **85.19%** |
| — alone in otherwise empty spectrum | 900 | 82.67% |
| Civilian only | **13,230** | **4** false alarms — 0.03% |
| **Combined** | **16,290** | **97.05%**, 99% precision |

Give the false-alarm figure its count, not only the percentage — "four" is concrete, and *0.03% of
what* is the question a judge asks. If anyone cross-checks the confusion matrix it shows **14** jamming
false positives: four in the civilian-only windows, ten on windows containing radar, FHSS or noise but
no civilian signal.

#### Recall vs SNR — where each class clears the bar

Give a **range per class**, not a general statement. "It works at high SNR" is weak; "all three judged
classes hold above the bar from −6 dB upward, five of the six bins" is the multi-condition requirement
evidenced.

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

**The three judged classes all clear 80% from −6 dB up** — five of six bins — and all three fail at
−10 dB (67.8 / 58.6 / 47.8). Say the failure in the same breath as the range, so the panel hears it
from us rather than reading it off the figure. §6 repeats it as a named limitation.

**Civilian, on the same line** (not gated by the rules, shown for comparison): BPSK from −2 dB up;
16QAM and 64QAM from +2 dB up; **QPSK never clears it at any SNR**, peaking at 74.3% at +6 dB.
NOISE_FLOOR sits at 48–63% throughout, consistent with its role as a bookkeeping label rather than a
fading signal.

**Say the QPSK result rather than leaving it to be scanned.** It is not a failure against the rules —
civilian classes are not benchmark-gated — but the brief names it as the civilian exception in §6.2,
and a video that shows the curve without mentioning it looks like it hoped nobody would look.

#### Dense-QAM pooling — and the limitation to say out loud

| Windows pooled | 16QAM vs 64QAM accuracy |
|---|---|
| 1 | 59.3% — *below minimum, refused* |
| 8 | 75.4% |
| 16 | 84.3% |
| 32 | 92.1% |
| 64 | 98.0% |

**The limitation is the point of the section, not a caveat on it.** The model classifies **per
window**. Naming *which* dense QAM is not done by the network at all — it is a statistical estimator
applied afterwards, and it needs several windows of the same emitter to work. Per window, combined
dense-QAM recall (did it notice dense QAM was present, regardless of which it named) is **65.6% over
7,290 windows**.

Why it is not a fixable modelling failure: a 512-sample window carries about **56 symbols**, and the
two constellations are closer together than the estimator's own spread at that count. A better model
does not solve it; more windows do.

Two refusals, which are strengths rather than caveats:

- **Below 8 windows** the resolver refuses to report an order rather than guessing.
- **Below 2 dB SNR** it refuses too — the channel pulls the estimator toward zero, so no order is
  reported at all.

If asked: pooling fixes the estimator only. Averaging the *model's* own 16QAM/64QAM probabilities
stays at chance however many windows are used, because that error is a bias rather than noise.

#### Overlapping signals — how to frame it convincingly

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

Close the segment here. It is the payoff of the independent-sigmoid head from §3: one window, as many
labels as there are signals in it.

#### ⚠ Blocking item

The dense-QAM pooling table is on the console but **not in the technical brief**, which reports only
the 65.58% combined figure. Now that the pooling result is spoken on camera, **the brief must be
updated to match before the PDF is exported.** This is the one place the two submitted artefacts would
otherwise contradict each other.

**Direction:** 126 seconds is long for one voice. The two Mode C cut-ins are the only breaks and they
are load-bearing — protect both in the edit.

### 6 · Limitations (4:04–4:30) — Mode C, Chua face to camera

Delivered on camera, not over a chart. Owning the limits in person is the point of the segment; a
voice-over reads as a disclaimer, a face reads as a team that knows where its own edges are.

This and the close are now Chua's only on-air time, so they carry weight — the limitations segment is
the video's credibility beat, which makes it a reasonable thing to be the one voice she brings.

**Three, fast — no hedging language.** *(Revised 6 Sept against brief §8 — an earlier draft used a
dense-QAM pooling figure the report does not make, and understated the low-SNR limit.)*

1. **Radar and FHSS are tuned for recall**, so roughly half their positive flags are false alarms
   (precision 0.54 and 0.59). Deliberate: a missed emission is the costlier error here.
2. **Dense QAM is weak at low SNR** — combined recall 65.58%, and the literature cross-check in brief
   §7.3 shows sub-0 dB dense-QAM recognition below 50% is normal even for dedicated classifiers. A
   property of the problem, not a pipeline fault, and it does not touch the judged classes.
3. **Barrage jamming, at 69.3%, is our weakest sub-type** — the only one that fails the bar on its
   own. Sweep is strongest at 95.7%.

**Fourth limit, added 7 Sept** with ten of the seconds taken from the live demo. It was previously cut
for runtime and left to the brief alone, which was the wrong call for the video's most honest segment:
at −10 dB **all three** judged classes fall below the bar (67.8 / 58.6 / 47.8), so the aggregate pass
is a test-set average rather than a per-condition guarantee. Say it. A judge who finds it in §8.2 after
not hearing it will wonder what else was left out.

### 7 · Close (4:30–4:40) — Mode C, Chua

Chua takes it because segment 8b is hers — she carries straight on from the limitations, same framing,
no voice change at the seam. Ten seconds.

One line of Phase 2 roadmap — RadChar's regression labels (pulse width, PRI, pulse count) as a path
toward emitter characterisation, with attribution named as out of scope — then hand to the end card.
Do not overclaim in the last ten seconds.

**End card, held to the last frame:** **NEXA** · all four names · Universiti Teknologi PETRONAS ·
Track I, Project Overwatch · SEDIC 2026.

With no group shot anywhere in the video, this card is the only place the panel sees the whole team.
Hold it long enough to read comfortably — 4 seconds minimum, not a half-second flash — and spell every
name exactly as it appears on the technical brief cover.

---

## 5. Rules the whole video follows

1. **Every figure to one decimal, against the stated bar.** No rounding up, no "over 85", no "very
   high accuracy." Precision is the credibility signal.
2. **One source of truth for numbers:** the technical brief, which agrees with the delivered
   scorecard behind `web/data/performance.json` (5-model ensemble, 2026-09-03). `STATUS_AND_PLAN.md`
   §2 was corrected to match on 6 Sept; anything quoting 87.4 / 84.3 / 86.3 or a 4.26M-parameter
   1D-CNN is a pre-correction copy — most likely one of the `.worktrees/` checkouts.
3. **Say "DSP."** The rubric uses the term twice; use its vocabulary.
4. **Never imply organiser data was used.** There is none. Say "held-out test split" every time.
5. **The video and the brief must not disagree on a single number.** Final check before upload is a
   line-by-line diff of every figure spoken against the brief PDF.

---

## 6. Recording checklist

- [ ] Console rebuilt and numbers refreshed (`python web/build.py`) before any screen capture
- [ ] Browser full-screen, 1920×1080, no bookmarks bar, no notifications, no dev tools
- [ ] One take per segment, recorded separately — never one continuous 5-minute take
- [ ] Cold open (Eileen) and close (Chua) shot in the **same framing and lighting as each other**, so
      the video's bookends match even though they are different people
- [ ] Eileen's cold open and segment 2 recorded as **two separate takes**, not one 64-second read
- [ ] End card held **4 seconds minimum** — with no group shot in the video, it is the only place the
      panel sees the whole team
- [ ] Every member's name spelled on the end card exactly as on the brief cover
- [ ] Audio: same room and mic per person across all their segments; record 3 s of room tone
- [ ] Slide deck exported at 1920×1080; on-screen numerals large enough to read on a phone
- [ ] Segments 4–6 recorded in one console session — same machine, same audio setup
- [ ] Eileen and Chua brief Jessy on the segments covering their work before she records them
- [ ] Rough cut timed **before** colour/motion work — if it runs over 5:00, trim segment 4 further,
      then segment 2. Never 5, 6 or 7.
- [ ] Final export ≤ 5:00 by the file's own duration, not by the edit timeline
- [ ] YouTube: unlisted or public (**not** private — judges must be able to open it), title
      `NEXA — SEDIC 2026 Project Overwatch (RF Track) — Phase 1`, link tested logged-out
- [ ] Link pasted into the submission form and opened once more after submitting

---

## 7. Open items

- [x] ~~Correct the docs describing a non-existent Qualifier IQ Data Stream~~ — done 6 Sept across
      `SEDIC2026_Track1_Documentation.md`, `STATUS_AND_PLAN.md`, `TEAM_ROLES.md`, `README.md` and
      `pipeline/08-inference-submission.md`. The `.worktrees/` branch checkouts still carry the old
      text and will need the same fix if they merge.
- [ ] Decide whether to move segment 7 (external validation) to Chua, to bring her above 36 seconds
- [ ] **Add the dense-QAM pooling table to the technical brief** — the video now states it, so the PDF must match. **Blocking.**
- [ ] Confirm the judged-class figures one last time against the scorecard on recording day
- [ ] Decide whether the ablation table appears in the video or only in the brief (recommend: brief
      only — there is no runtime for it)
