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

## The fix, and what it costs

`configs/civ_jitter.yaml` resamples civilian frames by a log-uniform factor
(0.55–1.9×) before windowing, synthesising the symbol-rate diversity RadioML
lacks. Opt-in; `configs/default.yaml` is untouched and its behaviour is
byte-identical.

The lower bound is set by the format contract, not preference: a 1024-sample
frame resampled below ~0.5× falls under the 512-sample window, and padding back
up would manufacture the flat-tail artefact the contract exists to prevent.

| Metric (CSPB, 56k) | Baseline | + jitter | |
|---|---|---|---|
| jamming FA @ 0–4 sps | 0.328 | **0.026** | **12.8× better** |
| jamming FA @ 4–6 sps | 0.129 | **0.011** | 11× better |
| **jamming FA overall** | **0.0615** | **0.0155** | **4× better** |
| accuracy @ 0–4 sps | 0.052 | **0.268** | 5× better |
| accuracy @ 8–10 sps | 0.642 | 0.550 | worse |
| overall accuracy | 0.474 | 0.450 | worse |
| BPSK | 0.842 | **0.921** | better |
| **64QAM** | 0.287 | **0.047** | **collapses** |
| any-threat FA | 0.206 | 0.246 | worse |

**It is a real trade, not a free win.** It buys a 4× reduction in the
false-alarm metric the rules reward, and pays with peak accuracy and a 64QAM
collapse that is certain at n = 14,000, not noise.

The jamming leakage does not vanish so much as **move to FHSS** (QPSK→FHSS 0.38,
16QAM→FHSS 0.25, 64QAM→FHSS 0.25), which is why any-threat false alarms rise.

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

1. **Do not enable the jitter silently.** It changes the dataset everyone trains
   on and trades 64QAM for false-alarm robustness. That is the team's call, with
   the numbers above in front of them.
2. **Stop quoting 0.0142 as the false-alarm rate.** It is measured only where
   the symbol rate matches training. Quote the CSPB figure, or state the
   condition explicitly.
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
