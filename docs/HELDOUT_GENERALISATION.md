# Held-out parameter generalisation test — findings

**Run by:** P1 · **Owners of the fixes:** P3 (FHSS) and P4 (jamming)
**Reproduce:** `scripts/heldout_generalisation.py` · **Raw output:** `evals/heldout/generalisation.json`

---

## Why this test exists

Our three judged classes are synthetic. A high score on our own test split only
proves the model learned *our* data. This test asks the question a technical
panel is silently asking:

> Did the model learn the **concept** — chirp, frequency hopping, denial — or did
> it memorise the particular parameter values we happened to generate?

**Method.** Train only on the lower half of each judged class's defining
parameter range, then score against the disjoint upper half, which the model has
never seen.

| Class | Split parameter | Train half | Held-out half |
|---|---|---|---|
| LFM_RADAR | `radar.bandwidth_hz` | 50–775 kHz | 775–1500 kHz |
| FHSS | `fhss.hop_rate_hz` | 25–87.5 kHz | 87.5–150 kHz |
| JAMMING | `jamming.sweep_bandwidth_hz`, `barrage_bandwidth_hz` | lower halves | upper halves |

The split was verified to have actually taken effect before any result was
trusted: measured chirp sweep width is 54–767 kHz in the training half against
744–1486 kHz in the held-out half.

**Not PRI**, despite `pipeline/07-evaluation.md` suggesting it as the example.
PRI is drawn *conditional* on pulse width (clamped up to
`pulse_width / max_duty_cycle` to keep duty under 100%). With a lowered ceiling
that clamp pushes PRI back above the ceiling for wide pulses, so the two halves
would silently **overlap** and the test would prove nothing. Bandwidth has no
such coupling.

---

## Result

| Class | In-range recall | Held-out recall | Delta | Reading |
|---|---|---|---|---|
| LFM_RADAR | 0.888 | 0.787 | **−10.0 pts** | inconclusive — within noise |
| FHSS | 0.801 | 0.818 | **+1.7 pts** | holds |
| JAMMING | 0.774 | 0.399 | **−37.6 pts** | **fails** |

Judged against the documented noise floor of **10.8 points** — the spread seen
across five *identical* runs differing only in random seed
(`scripts/measure_variance.py`).

- **JAMMING fails unambiguously.** 37.6 points is 3.5× the noise floor.
- **LFM_RADAR is borderline, not a pass.** −10.0 against a 10.8-point floor is
  *not distinguishable from noise* — it is not evidence that radar generalises,
  only that we cannot tell from one run. Report it as inconclusive.
- **FHSS genuinely holds.** It even gains slightly, which is itself a clue — see
  below.

---

## Mechanism: FHSS is acting as an attractor class

Recall alone says a class failed. Where the predictions *go* says why:

```
JAMMING (held-out)   FHSS=0.465   JAMMING=0.399   QPSK=0.040   16QAM=0.029
```

**More held-out jamming is classified as FHSS than as jamming.** This is not
diffuse confusion — it is a directed leak into one specific class.

This same failure has now appeared **three times**, each time recorded in
`configs/default.yaml` as a comment next to whichever parameter was tuned to
suppress it:

| When | Symptom | Mitigation applied |
|---|---|---|
| Tone jamming | *"85 of 200 tone examples were predicted as FHSS"* | `max_tones` 3 → 1 |
| High-duty radar | *"90% duty → 16.7% ... and 116/150 called FHSS"* | `max_duty_cycle` capped at 0.15 |
| Wide-band jamming (this test) | 46.5% of held-out jamming predicted FHSS | none yet |

The common thread: **any signal occupying many frequencies across the window
gets absorbed into FHSS.** Multiple simultaneous tones do it. A near-continuous
chirp train does it. And now a wide sweep or wide barrage does it.

Each previous fix narrowed a *generator* range until the symptom disappeared.
That suppresses the symptom without addressing the cause, and it carries a cost:
every narrowing shrinks the training distribution, which is precisely what makes
the model brittle on parameters it has not seen. The two earlier mitigations and
this failure are the same problem viewed from different angles.

**The underlying issue is that the discriminating feature is weak at this window
length.** What genuinely separates FHSS from a sweep jammer is *structure*, not
occupancy: FHSS holds a frequency for a dwell then jumps discretely, whereas a
sweep is continuous and barrage is simultaneous. At 512 samples / 160 µs with
4–24 hops per window, a single dwell is only ~21–40 samples. There is little
room for the model to see "held, then jumped" rather than merely "several
frequencies present".

---

## What this result does and does not claim

**Stated plainly, because it would be easy to overstate:**

- This test deliberately creates a train/test parameter mismatch that **does not
  exist in the production pipeline** — `configs/default.yaml` trains across the
  *full* bandwidth range, so the deployed model does see wide-band jamming.
  **Do not quote 0.399 as an expected jamming recall.**
- What it *does* establish is which feature the model leans on. A model that
  recognised the denial concept would not lose two-thirds of its jamming recall
  when only the bandwidth changes. It is keying on spectral occupancy, and
  occupancy is shared with FHSS.
- The risk this implies for the Qualifier stream is real but indirect: jamming
  whose bandwidth sits outside our training distribution is liable to be called
  FHSS. Both are threat classes, so the coarse tier call survives — but the
  judged per-class recall would not.
- **The jamming figure is a lower bound.** Tone jamming has no bandwidth
  parameter, so roughly a third of JAMMING examples were drawn identically in
  both halves. The true effect on the splittable sub-types is worse than −37.6.
- In-range jamming recall here (0.774) is below the full-range main run (0.834)
  because this model trained on deliberately narrowed data. Compare 0.399
  against 0.774, never against 0.834.
- Civilian classes cannot be parameter-split at all (RadioML is real capture),
  so their columns compare different random *rows* and are not a generalisation
  measurement. They are reported for completeness only.
- One training run per arm. The JAMMING effect is far too large for seed noise
  to explain, but LFM_RADAR's marginal result would need `--runs 3` to resolve.

---

## Recommended next steps — for P3 and P4, not P1

Ordered by cost. **None of these are P1's to decide**; the FHSS/jamming boundary
belongs to P3 and P4, and `configs/default.yaml` is shared.

1. **Do not narrow another generator range to make this go away.** That is the
   reflex the previous two mitigations followed, and it trades hidden-stream
   robustness for a better score on our own data. This test exists to catch
   exactly that trade.

2. **Probe jamming by sub-type** (barrage / tone / sweep) against the trained
   model, as was done when `max_tones` was set. This test cannot separate them
   because sub-type is not carried into the dataset labels. Knowing whether the
   leak is sweep-driven or barrage-driven determines the fix.

3. **Reconsider the FHSS hop-rate ceiling.** At 150 kHz a dwell is ~21 samples.
   If dwell structure is unresolvable at that rate, high-rate FHSS and wide
   sweep jamming may be genuinely inseparable at a 512-sample window — in which
   case the honest move is to narrow FHSS's *upper* hop rate and document the
   scoping decision, rather than to keep trimming jamming.

4. **Consider whether the class boundary needs a structural feature.** Raw IQ
   into a 1D-CNN may not surface "held then jumped" at this window length. This
   is a Phase-2 architectural question, not a four-day one — noting it, not
   proposing it.

---

## For the technical brief

This belongs in the limitations section, and it is worth including rather than
omitting. A team that finds its own failure mode and quantifies it reads as more
credible than one reporting a clean sweep it cannot account for.

> We tested generalisation by training on one half of each synthetic class's
> parameter range and evaluating on the disjoint other half. FHSS held
> (+1.7 points). Radar was inconclusive against our measured 10.8-point seed
> noise floor (−10.0). Jamming fell 37.6 points, with 46.5% of held-out jamming
> classified as FHSS — indicating the model separates these classes primarily by
> spectral occupancy, a feature the two share, rather than by hop structure. We
> report this as a known limitation of the synthetic jamming class.
