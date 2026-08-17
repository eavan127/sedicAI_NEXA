# External validation of the civilian classes — CSPB.ML.2018R2

**Run by:** P1 · **Reproduce:** `scripts/eval_cspb.py` · **Raw output:** `evals/cspb_external.json`

---

## Why

Our civilian classes come from RadioML alone. Scoring them on a RadioML test
split cannot separate *"learned modulation"* from *"learned RadioML"* — the
question [`TEAM_ROLES.md`](TEAM_ROLES.md) says all our work is aimed at.

CSPB.ML.2018R2 (Spooner, CSP Blog) covers the same four modulations but
randomises **symbol rate, carrier offset and pulse-shaping roll-off** — the
parameters RadioML holds fixed. We never train on it.

**Scale:** all 28 batches, **56,000 signals** in our four classes, one centred
512-sample window each so samples are independent. Recall precision ≈ ±0.4%.

| | RadioML (ours) | CSPB.ML.2018R2 |
|---|---|---|
| samples/symbol | fixed ~8 | **2.0 – 22.5** |
| roll-off | narrow, ~fixed | 0.10 – 1.00 |
| SNR | −10…+10 dB (our AWGN) | 0.0 – 13.1 dB (inband) |

> SNR definitions differ — CSPB's inband SNR is not our `add_awgn` scale. Do not
> compare the two SNR axes directly.

---

## Headline: the classifier generalises, but its false alarms do not

| Class | RadioML own split | CSPB (56k signals) |
|---|---|---|
| BPSK | 0.75 | 0.842 |
| QPSK | 0.68 | 0.516 |
| 16QAM | 0.36 | 0.252 |
| 64QAM | 0.21 | 0.287 |
| **overall** | ~0.50 | **0.474** |

Aggregate accuracy on third-party data essentially matches our own split. The
model did **not** merely memorise RadioML.

### But the false-alarm rate is concentrated almost entirely at slow symbol rates

Civilian misclassified as JAMMING — the metric the rules single out:

| samples/symbol | n | jamming FA |
|---|---|---|
| **0–4** | 7,316 | **0.328** |
| 4–6 | 7,124 | 0.129 |
| 6–8 | 6,400 | 0.012 |
| **8–10** | 6,416 | **0.001** |
| 10–14 | 14,048 | 0.001 |
| 14+ | 14,696 | 0.002 |

**At RadioML-like symbol rates the false-alarm rate is effectively zero. At slow
symbol rates, one civilian signal in three is flagged as jamming.**

Our reported figure of 0.0142 is not mildly optimistic — it measures the one
regime where the model is near-perfect. Overall on CSPB it is **0.0615**.
Accuracy shows the same shape, peaking at 8–10 sps, exactly RadioML's value.

This is the concrete, measured form of the limitation recorded in
[`BRIEF_P1_DATA.md`](BRIEF_P1_DATA.md): RadioML carries one symbol rate, and the
model leaned on it.

---

## 16QAM and 64QAM are not separately identifiable at a 512-sample window

**This is the most consequential result here, and it changes what the brief may
claim.** It surfaced by accident, from comparing three training configurations.

| config | 16QAM | 64QAM | **sum** |
|---|---|---|---|
| baseline | 0.252 | 0.287 | **0.539** |
| full jitter (100% resampled) | 0.284 | 0.047 | 0.331 |
| mixed jitter (50%) | 0.066 | 0.476 | **0.542** |

Baseline and mixed jitter have **identical combined QAM performance** (0.539 vs
0.542) while their individual columns are almost exactly swapped. In the mixed
model, 16QAM is classified as 64QAM 42% of the time.

The mass is conserved. The model reliably detects *"dense QAM constellation"* —
that total is stable — but which of the two labels it assigns is close to
arbitrary and flips with configuration.

The same behaviour appeared in isolated 4-class civilian experiments during this
work, where runs collapsed to "BPSK plus one arbitrary QAM class" depending on
the random seed. That was initially read as a broken experiment; at n = 14,000
per class on third-party data it is clearly a property of the task.

**Why:** at ~8 samples/symbol a 512-sample window spans roughly 51-64 symbols.
16QAM has 16 constellation points, 64QAM has 64. At about one observation per
point, a 64-point constellation is not statistically distinguishable from a
sparser one. This also explains the otherwise puzzling fact that **64QAM recall
is flat across SNR** — 0.087 at -10 dB and 0.273 at +10 dB on our own data.
More SNR does not buy more symbols.

### Consequences

- **Do not quote 16QAM and 64QAM recalls as separate meaningful numbers.** Any
  single value is close to a coin flip. Report the combined high-order-QAM
  detection rate (~0.54), or report both with this caveat attached.
- `TEAM_ROLES.md` attributes 64QAM's weakness to *"the constellation sits inside
  the noise floor"* at -10 dB. That rationale is wrong — the limit holds at
  +10 dB too. The advice not to chase it stands; the stated reason does not.
- This is a **window-length** limit, not a data-volume or model-capacity limit.
  More civilian examples cannot fix it, which is consistent with the recorded
  1000 -> 2000 experiment producing nothing measurable.
- The clean way to confirm it is CSPB's 32,768-sample records: score the same
  signals at 512 / 1024 / 2048 / 4096 windows. Our own format contract fixes
  the window at 512, so this cannot be tested on RadioML.

---

## The fix, and what it costs

`configs/civ_jitter.yaml` resamples civilian frames by a log-uniform factor
(0.55–1.9×) before windowing, synthesising the symbol-rate diversity RadioML
lacks. Opt-in; `configs/default.yaml` is untouched and its behaviour is
byte-identical.

The lower bound is set by the format contract, not preference: a 1024-sample
frame resampled below ~0.5× falls under the 512-sample window, and padding back
up would manufacture the flat-tail artefact the contract exists to prevent.

Two variants were measured, differing only in `civilian.jitter_fraction` — what
share of civilian frames get resampled.

| Metric (CSPB, 56k) | Baseline | full jitter (1.0) | **mixed (0.5)** |
|---|---|---|---|
| jamming FA @ 0–4 sps | 0.328 | **0.026** | 0.129 |
| **jamming FA overall** | 0.0615 | **0.0155** | 0.0261 |
| **any-threat FA** | 0.206 | 0.246 | **0.129** |
| accuracy @ 0–4 sps | 0.052 | **0.268** | 0.134 |
| accuracy @ 6–8 sps | 0.561 | 0.535 | **0.632** |
| accuracy @ 8–10 sps | 0.642 | 0.550 | **0.645** |
| accuracy @ 14+ sps | 0.540 | 0.431 | **0.575** |
| **overall accuracy** | 0.474 | 0.450 | **0.524** |
| BPSK | 0.842 | **0.921** | 0.857 |
| QPSK | 0.516 | 0.550 | **0.696** |
| high-order QAM (sum) | 0.539 | 0.331 | **0.542** |

**Resampling 100% of frames is the wrong setting.** It removes RadioML's native
rate from training entirely, and accuracy at 8–10 sps — where the model was
strongest — falls 0.642 → 0.550. Mixing keeps that regime represented and
recovers it fully (0.645), while still improving the slow-rate bands.

**Mixed is the better configuration overall**: best accuracy, best any-threat
false-alarm rate (down 37% from baseline), native-rate performance intact, and
2.4× better jamming false alarms than baseline. Full jitter still wins on the
jamming-FA metric alone (0.0155 vs 0.0261) by sacrificing everything else.

Under every variant the jamming leakage partly **moves to FHSS** rather than
disappearing (full jitter: QPSK→FHSS 0.38, 16QAM→FHSS 0.25, 64QAM→FHSS 0.25) —
the same FHSS-attractor behaviour found by the held-out parameter test.

### On our own RadioML split, mixed looks worse — unresolved

JAMMING 0.7522 against baseline 0.8344 (−8.2 points, right at the 7.2-point
noise floor) and FHSS 0.8244. These are single runs. Whether that is real
degradation or seed noise is **not established** and needs `measure_variance.py`
on each config before anyone acts on it.

---

## Seed noise — what may NOT be claimed

`scripts/measure_variance.py --runs 3` on the jitter config:

| class | mean | min | max | spread |
|---|---|---|---|---|
| LFM_RADAR | 0.9122 | 0.8944 | 0.9278 | 0.033 |
| FHSS | 0.9026 | 0.8722 | 0.9200 | 0.048 |
| JAMMING | 0.8400 | 0.8022 | 0.8744 | 0.072 |

**Noise floor: 7.2 points.**

A single jitter run reported LFM_RADAR = 0.9244, i.e. "PASS". Across three seeds
the mean is 0.9122 and the **minimum is 0.8944 — a FAIL**. Whether that class
clears the gate depends on the seed. Reporting the single run as a pass would
not have survived scrutiny.

Against the baseline's single-run judged figures:

| class | baseline (1 run) | jitter (3-run mean) | Δ | vs 7.2-pt floor |
|---|---|---|---|---|
| LFM_RADAR | 0.8722 | 0.9122 | +4.0 | **within noise — no claim** |
| FHSS | 0.8078 | 0.9026 | +9.5 | exceeds floor — *suggestive only* |
| JAMMING | 0.8344 | 0.8400 | +0.6 | **within noise — no claim** |

Even FHSS's +9.5 is **not** a solid claim: the baseline is a single run and
carries its own ±7.2 uncertainty, so this compares a mean against a point
estimate. Establishing it needs `measure_variance.py` on the baseline config
too. There is also no plausible mechanism by which civilian-only preprocessing
should move FHSS by 9 points — treat it as unexplained until measured.

---

## Recommendations

1. **If the jitter is adopted, use `configs/civ_mix.yaml` (fraction 0.5), not
   the full-replacement variant.** It is better on every measure except the
   jamming-FA number in isolation. Either way it changes the dataset everyone
   trains on, so it is the team's call, not P1's — and the own-split regression
   above should be resolved first.
2. **Stop quoting 0.0142 as the false-alarm rate.** It is measured only where
   the symbol rate matches training. Quote the CSPB figure, or state the
   condition explicitly.
3. **Report high-order QAM as one number, or caveat the split.** 16QAM and
   64QAM individually are not identifiable at this window length — see above.
3. **The FHSS attractor is now confirmed three ways** — held-out jamming
   bandwidth, held-out radar duty cycle, and now real third-party civilian
   traffic. It is a class-boundary problem, not a jamming-generator problem.
   Owners: P3 and P4. See [`HELDOUT_GENERALISATION.md`](HELDOUT_GENERALISATION.md).
4. **If 64QAM matters**, the jitter is the wrong tool for it — investigate
   separately. CSPB's 32,768-sample records allow varying the window length
   directly, which our 512-sample format contract does not.

---

## Provenance

- Cite as **CSPB.ML.2018R2**. Use R2, not the original CSPB.ML.2018: the
  original has an RNG flaw that duplicated parameter sets across signals.
- ⚠️ The CSP Blog states a citation requirement but **no explicit licence
  terms** — confirm permitted use before relying on this in the submission.
- Batch 8's zip ships with 3999 entries; `signal_31986.tim` was omitted in
  upload and is posted separately. `eval_cspb.py` accounts for both and splices
  the file back in (it is a QPSK example, so dropping it would lose real data).
