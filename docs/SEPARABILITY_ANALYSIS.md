# Separability analysis: the tiny differences between the signals we confuse

Branch `eavan-accuracy-plan`, 7 Nov 2026. Companion to `docs/ACCURACY_PLAN.md`.

**Question.** For every group of classes the 60-class model confuses, what is the exact physical
difference between them? Can that difference be seen in one 160 µs window (512 samples at
3.2 MS/s)? If not in one window, which combination of conditions (several measurements, more time,
context) separates them? And would a team of specialised models ("agents") beat one big model?

**Method.**
1. Read how each class is defined in our own generators (`src/generators/*.py`).
2. Write down the physical difference.
3. Measure it with blind physics features (`scripts/probe_specialists.py`: nothing from the generator is used).
4. Compare a small specialist trained on those features with the CNN on the same classes.

## 1. The theory: what really differs, class by class

### 1.1 PSK order: BPSK / QPSK / OQPSK / 8PSK / 16PSK / 32PSK
| Pair | The tiny difference | Visible in 160 µs? |
|---|---|---|
| BPSK vs QPSK | Squaring BPSK removes the data (the phase is 0 or π, so 2x the phase is always 0): **y² is a pure tone**. QPSK needs the **4th power** for that. | Yes, if >= ~16 symbols and SNR >= ~5 dB. The line height grows with the number of symbols. |
| QPSK vs OQPSK | Same 4 points, but OQPSK shifts Q by half a symbol, so the signal **never passes through zero amplitude**. | Yes: the share of near-zero amplitudes (envelope dips). |
| 8 / 16 / 32PSK | The 8th, 16th or 32nd power gives the line. Each step needs ~4x more symbols at the same SNR, and the phase spacing (45°, 22.5°, 11.25°) quickly falls below the phase noise plus the error from an unknown frequency offset. | **No** for 16 vs 32PSK in 10-60 symbols; ceiling ~74% even with perfect sync (measured 4 Nov). |

### 1.2 QAM and APSK sizes
- **Difference:** the number of amplitude rings / grid levels (16QAM: 3 rings; 64QAM: 9; 256QAM: 32 distinct amplitudes).
- **Problem:** with 10-60 symbols you get fewer samples than rings, and noise at 15 dB blurs neighbouring rings into one.
- **Ceiling:** ~45% for QAM sizes even with perfect sync.
- **Verdict:** one window cannot separate them; they need **hundreds of symbols**, i.e. the whole track at symbol rate.

### 1.3 APSK vs QAM vs partial-band noise jammer
| Pair | The tiny difference | Visible in 160 µs? |
|---|---|---|
| APSK vs QAM | APSK rings are circles with **equal-spaced phase on each ring**; QAM is a square grid. The amplitude histogram has **2-4 sharp ring peaks** for APSK and a broad cluster for QAM. | Partly: needs ~30+ symbols. |
| APSK/QAM vs JAM_PARTIAL_BAND | The digital signal has a **symbol clock**: its loudness \|y\|² repeats at the symbol rate, giving a **spectral line** (cyclostationarity). Band-limited noise has **no line**, a Gaussian (Rayleigh) amplitude and a flat spectrum with sharp edges. | **Yes**: the symbol-rate line and the amplitude shape differ fundamentally. |

### 1.4 FSK family: 2FSK / 4FSK / 8FSK / GFSK / GMSK / MSK / CPFSK
| Pair | The tiny difference | Visible in 160 µs? |
|---|---|---|
| 2 / 4 / 8FSK | The instantaneous frequency jumps between **2, 4 or 8 levels**: a histogram with that many peaks. | Yes for 2 vs 4; 8 levels need more symbols per level. |
| MSK vs GMSK | Both continuous-phase with h = 0.5. GMSK first smooths the data with a **Gaussian filter (BT ~0.3)**: frequency transitions are **ramps, not steps**. | Yes: the smoothness of the frequency track. |
| GFSK vs GMSK | Both Gaussian-smoothed; GFSK has a **larger modulation index** (wider frequency swing for the same symbol rate). | Yes, if the symbol rate can be estimated. |
| CPFSK vs MSK | CPFSK has an arbitrary h; MSK is exactly 0.5. | Only with the symbol rate measured precisely. |

### 1.5 Analog voice / message carriers vs tone and spot jammers
In our generators the analog "message" is **8-60 kHz noise-like** (on purpose, so a 160 µs window shows 1-10 message cycles).
| Pair | The tiny difference | Visible in 160 µs? |
|---|---|---|
| NBFM / PM vs JAM_TONE | Both have a **constant envelope**. The tone's instantaneous frequency is **flat**; FM/PM's **wobbles** with the message. PM's frequency follows the message's **derivative** (more high-frequency content than FM). | **Yes** for FM/PM vs tone (the frequency spread). FM vs PM only partly (spectral tilt of the frequency track). |
| AM_DSB_WC vs tone | AM has a carrier line plus **symmetric sidebands**, and its envelope varies. | Yes. |
| AM_SSB_SC vs JAM_SPOT (largest error) | Both are narrowband noise-like bands. SSB is **one-sided** relative to its (missing) carrier and its envelope follows the message; spot noise is symmetric, Gaussian and flat. Without the carrier, "one-sided" has no reference inside the window. | **Barely.** Spectral tilt / asymmetry and envelope statistics are weak at 512 samples. |
| AFSK vs tone | AFSK is FM with audio tones (1200 / 2200 Hz): in 160 µs the frequency moves less than one audio cycle, so it looks like a slightly wobbling tone. | **Mostly no**; needs >= 1 ms (the zoom-in). |
| OOK vs tone | OOK keys the carrier on/off: the window shows **gaps**. | Yes, if a gap falls inside the window. |

### 1.6 OFDM / LTE_LIKE / NR_LIKE vs noise jamming
| Pair | The tiny difference | Visible in 160 µs? |
|---|---|---|
| OFDM-like vs JAM_BARRAGE / PARTIAL_BAND | Both look like Gaussian noise. OFDM has a **cyclic prefix**: the end of each symbol is copied in front, so the signal **correlates with itself at a delay of one FFT length**. Noise never does. | Yes, but weakly: 160 µs holds only **~2 LTE symbols**, so the echo is measured on 1-2 copies. |
| LTE vs NR vs generic OFDM | **Subcarrier spacing** (LTE 15 kHz, NR 30/60 kHz, generic 12.5/25/50 kHz) sets the delay of that echo (213 vs 107/53 samples...) and the **prefix length** (4.7 / 2.3 / 1.2 µs). Pilots every 6th vs 2nd subcarrier. | Yes, via the echo's delay, when the echo is visible. |

### 1.7 Radar shapes (one pulse)
- **RADAR_PULSE:** plain carrier pulse.
- **LFM:** linear chirp (frequency ramps).
- **BARKER:** 13 chips of 0/π phase.
- **POLY_BARKER:** multi-level phase chips.
- **FRANK:** quadratic phase steps (a stepped chirp).
- **COSTAS:** frequency hops in a Costas order.
- **FMCW:** continuous triangle sweep.

The difference lies **inside one pulse** (phase or frequency track), so it is visible **only if a whole pulse is inside the window**. Duty cycles are 1-9% and repetition intervals up to 2 ms: **most windows contain no pulse at all**.

### 1.8 Jammer kinds (defined by what one window shows)
- **BARRAGE / PARTIAL_BAND / SPOT:** noise, differing **only in bandwidth** (>= 0.9 MHz / 200-600 kHz / 10-60 kHz, with deliberate gaps between the ranges).
- **MULTITONE:** 2-8 lines.
- **CHIRP:** sawtooth repeating every 16-56 samples.
- **SWEPT:** a slow piece of a sweep.
- **PULSED:** noise bursts.
- **REACTIVE:** **silence, then noise switches on once**.

REACTIVE (and partly PULSED) is defined by a **transient**. Over a long track only one window contains the switch-on; every later window is plain noise of some width. So JAM_REACTIVE scores 0% in tracks: **it is a behaviour, not a waveform**, and must be decided over time (the tracker sees noise appear, after a victim signal, and stay on).

### 1.9 Pulse comms vs radar
- **PPM (ADS-B-like):** 0.5 µs pulses in a fixed preamble pattern (0, 1, 3.5, 4.5 µs).
- **SSR_IFF:** interrogation pulse pairs.
- **AFSK:** continuous.
- **RADAR_PULSE:** single longer pulses at a fixed interval.

**Difference:** pulse width and the **pattern of pulse spacings** inside a few µs; visible when the pattern falls in the window.

## 2. Measured: physics specialists vs the CNN

`scripts/probe_specialists.py`, 200 windows per class, our generators + the impairment module.
- **Specialist:** gradient boosting on ~90 blind physics features (power-law lines, symbol-rate line, instantaneous-frequency shape, envelope, delay echo, spectral shape, on/off timing) + the 104 expert features. Tested on windows **placed anywhere** in the band.
- **CNN (seed 4):** argmax among the group's classes only, on windows **centred at 0 Hz** (its most favourable condition, because of the FSK/PSK/QAM training bug).

| Group | chance | 15 dB: CNN | 15 dB: specialist | 5 dB: CNN | 5 dB: specialist |
|---|---|---|---|---|---|
| PSK order (6) | 17% | 49% | **57%** | 44% | **49%** |
| QAM sizes (5) | 20% | 23% | 28% | 23% | 21% |
| APSK vs QAM vs band noise (6) | 17% | 22% | **46%** | 24% | **33%** |
| FSK family (7) | 14% | **86%** | 83% | **80%** | 74% |
| Analog vs tones (11) | 9% | 32% | **58%** | 31% | **53%** |
| OFDM vs noise jamming (5) | 20% | 62% | **68%** | 58% | 59% |
| Radar shapes (7) | 14% | **80%** | 75% | **77%** | 57% |
| Jammer kinds (9) | 11% | 89% | **95%** | 85% | 86% |
| Pulse comms vs radar (5) | 20% | 79% | **86%** | 70% | 73% |

**Per class, the two are complementary** (15 dB, CNN vs specialist):

| Specialist much better | CNN much better |
|---|---|
| NBFM 4% -> 42% | JAM_SPOT 55% vs 32% |
| PM 2% -> 38% | 8PSK 66% vs 39% |
| LTE_LIKE 4% -> 56% | 4FSK 87% vs 64% |
| 64APSK 0% -> 30% | FRANK 87% vs 69% |
| 128APSK 0% -> 25% | BARKER 94% vs 78% |
| 16PSK 4% -> 28% | OFDM 56% vs 44% |
| SSR_IFF 69% -> 90% | |

What the physics found that the CNN missed:
- the **frequency wobble** of FM/PM against a pure tone;
- the **symbol-rate line** that separates APSK from noise jamming;
- the **cyclic-prefix echo** of LTE;
- the **power-law lines** for PSK order.

What the CNN does better than these features: **shapes in time** (radar phase codes, FSK level patterns at low SNR) and **band shape** (spot noise).

## 3. Fusion: the "team of specialists" design, measured before building it

`scripts/probe_fusion.py`: same placed windows for all three variants (only groups where the CNN is not hurt by the 0 Hz bug).

| Group (15 dB) | CNN | physics | **CNN + physics fused** |
|---|---|---|---|
| APSK vs QAM vs band noise | 35% | 44% | **49%** (+14 over CNN) |
| Analog vs tones | 49% | 60% | **59%** (+10) |
| OFDM vs noise jamming | 62% | 71% | **70%** (+8) |
| Radar shapes | 82% | 78% | **87%** (+5) |
| Jammer kinds | 92% | 94% | **98%** (+6) |
| Pulse comms vs radar | 88% | 83% | **92%** (+4) |

Fusing is never worse than the better of the two, and is up to 14 points better than the CNN.

### 3.1 One fusion over all 60 classes (no router)
120 windows per class, all 60 classes (FSK/PSK/QAM centred, the CNN's training condition; others placed).
**Control** = the same gradient boosting trained on the CNN's scores alone, so any gain beyond it comes
from the physics features, not from a better last layer.

| | CNN | control (CNN scores re-learned) | **CNN + physics** | family: CNN -> fused |
|---|---|---|---|---|
| 15 dB | 55.9% | 56.6% | **62.8%** (+6.2) | 76.0% -> **82.6%** |
| 5 dB | 48.4% | 51.1% | **54.6%** (+3.5) | 66.5% -> **75.6%** |

## 4. The multi-agent idea: verdict

**Idea.** One "router" agent picks the family; a specialised agent per family (PSK agent, radar agent,
voice agent...) decides the exact class; an arbiter combines them.

**What the measurements say.**
1. **The knowledge the specialists add is real**: +4 to +14 points inside the confused groups, and the CNN and the physics features are wrong on different classes.
2. **But a router adds its own mistakes.** The family is right only 66-83% of the time (CNN, or fused at best). A hard router throws away every window it sends to the wrong specialist: the overall result is capped at router accuracy x specialist accuracy.
3. **One fused step over all 60 classes already captures the gain** (+6.2 points at 15 dB, +3.5 at 5 dB; family +7 to +9) **without any routing errors**.

**Recommendation.**
- **Do not build separate routed agents for classification.** Build the specialists' knowledge into the model as one more input branch: a **physics branch** next to the expert-feature branch (the same pattern that gave +2.6 for the expert features and +2.6 for the spectrum branch), trained once (~2 h T4).
- **Agents DO make sense where they already exist**, because those tasks really are different and work on different time scales:

| Agent | Looks at | Decides | Status |
|---|---|---|---|
| Detector | 10 ms spectrogram picture | where / when each transmitter is | ✅ YOLO + narrow-signal energy |
| Classifier (+ expert + spectrum + physics branches) | 160 µs windows | what it is, per window | ✅ (+ physics branch: to build) |
| Tracker / voter | the whole track | one answer per transmitter; skips empty windows | ✅ |
| Resolver | behaviour over time | twins, sizes as groups, reactive / pulsed behaviour | ✅ (reactive: to add) |
| Zoom (analog) | 50 kS/s over the whole track | voice / Morse kind; overrides on clear signals | ✅ |
| Listener | demodulated audio | Morse text / speech | ✅ |
| Reader (local LLM) | text | summary, grounded | ✅ |

A local LLM is **not** useful for classifying the signal itself: the decision is numerical and lives in
the features above. Its place is the last step (reading text, explaining the answer to the operator).

## 5. Multiple conditions: how to separate what one window cannot

For pairs one window cannot separate, the answer is a **chain of conditions over time**, decided by
the tracker / resolver, not by the classifier:

| Pair | Condition chain |
|---|---|
| RADAR_CW vs JAM_TONE | Same waveform. Tone appears ON TOP of an existing signal -> jammer; alone for a long time -> CW radar; else the honest pair (built) |
| JAM_REACTIVE vs spot / barrage noise | Noise band that **switches on after a victim signal appeared** at that frequency and then stays on -> reactive; otherwise by width: <= 60 kHz spot, 200-600 kHz partial, >= 0.9 MHz barrage (to build: an onset rule in the resolver) |
| LORA_CSS vs JAM_SWEPT | Width <= 0.5 MHz and wraps within one symbol -> LoRa; >= 0.9 MHz or no wrap for > 1.2 ms -> swept jammer (built) |
| QAM / APSK / 8-16-32PSK sizes | Group answer + likeliest size (built); a true size needs hundreds of symbols: symbol-rate demodulation of the whole track (future) |
| AFSK / voice AM / SSB vs tone / spot | Zoom over the whole track; override on a clear signal (built) |
| Pulsed radars | Vote only with windows that contain a pulse (built); intra-pulse code from the pulse itself |
| OFDM / LTE / NR vs noise jamming | Echo at the FFT length over several symbols: the vote over the track adds them up (built); the physics branch adds the echo measure per window |

## 6. What to do, in order (with the evidence for each)

| # | Action | Evidence it helps | Cost |
|---|---|---|---|
| 1 | Train on the fixed data (Part B), the base for everything below | off-centre FSK/PSK/QAM work (BPSK 2% -> expected ~90% at +300 kHz) | 2 h T4 |
| 2 | **Physics branch in the model** (power-law lines, symbol-rate line, instantaneous-frequency shape, envelope near-zero share, delay-echo peak, spectral peaks/symmetry, on/off timing) | +6.2 points top-1 at 15 dB, +3.5 at 5 dB, family +7-9 (section 3.1) | 1 day code (torch versions of `probe_specialists.physics`) + 2 h T4 |
| 3 | Reactive-jammer onset rule in the resolver | JAM_REACTIVE is 0% in tracks by definition (section 1.8) | 1/2 day |
| 4 | Score the voice path on real audio scenes (SED-38) and the RSPduo | zoom override 0/3 -> 3/3 on the demo; real speech untested | with SED-38 |
| — | **Not worth it:** routed per-family agents, a long-window model, more epochs, `n_fft` 128 | sections 3.1 and 4; ACCURACY_PLAN P2 | — |

## Sources
- Our measurements: `results/accuracy_plan/specialists_15dB.json`, `specialists_5dB.json`, `fusion_15dB.json`, `fusion_60class.json`, `window_length.json`.
- Feature families (higher-order statistics, cyclostationary features) in the AMC literature: [Green-ML AMC, 2026](https://arxiv.org/pdf/2604.10317); [Dobre et al., cyclostationarity-based classification of linear digital modulations](https://researchwith.njit.edu/en/publications/cyclostationarity-based-modulation-classification-of-linear-digit/).
- Mixture-of-experts AMC routes by **SNR**, not by family: [MoE-AMC, 2023](https://arxiv.org/abs/2312.02298). We found no published evidence for routed per-family experts, which matches our measurement.
