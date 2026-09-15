# Brief section — Datasets, licences, attribution, class counts

**Author:** P1 · **Status:** draft for P4 to assemble
**Covers:** the technical-brief material owned by P1 per [`TEAM_ROLES.md`](TEAM_ROLES.md)

---

## Attribution (required — CC BY obligation)

Both external datasets are **CC BY-NC-SA 4.0**, which requires citing the source
and naming the licence. Verbatim lines for the brief:

> Training data sourced from RadioML 2018.01A (DeepSig Inc.) and RadChar
> (Huang et al., ICASSP 2023), both licensed CC BY-NC-SA 4.0.

Use of these is endorsed by the rules — Section 3 directs participants to
*"open-source datasets (e.g., RadioML, DeepSig)"* by name. RadChar sits under the
identical licence, and "e.g." marks that list as examples rather than exhaustive.

Non-commercial use only, and derivatives share alike — both satisfied by a
competition submission.

---

## Sources

| Class | Source | Nature |
|---|---|---|
| BPSK, QPSK, 16QAM, 64QAM | RadioML 2018.01A | real, published |
| LFM_RADAR | RadChar + our generator | real, augmented synthetically |
| FHSS, JAMMING | our generators | synthetic |

### RadioML 2018.01A (civilian classes)

24 modulations × 26 SNR levels (−20…+30 dB, 2 dB steps) × 4096 frames, 1024
complex samples per frame. HDF5, ~21 GB. Stored as `/X` (IQ), `/Y` (one-hot
class), `/Z` (SNR), laid out class-major then SNR-major.

We filter to the four civilian modulations and discard the other 20.

### ⚠️ The class-index mapping shipped with RadioML is wrong

**This is worth stating in the brief — it is a correctness issue most teams
using this dataset will not have caught.**

The `classes.txt` file distributed alongside the archive does **not** match the
actual index order inside `GOLD_XYZ_OSC.0001_1024.hdf5`. Taking it at face value
silently mislabels every civilian example — the pipeline runs fine and produces
confident, meaningless numbers.

We established the correct order from the dataset paper's own class listing
(O'Shea, Roy & Clancy, *Over the Air Deep Learning Based Radio Signal
Classification*, arXiv:1712.04578) rather than the side file, and verified it
against the data:

| Check | Expected under paper order | Measured |
|---|---|---|
| Index 0 | OOK — QAM/PSK spectrum plus midband impulse | matches |
| Indices 17–21 | the five analog classes | sit at ~10× the amplitude scale of every digital class |
| Index 21 | FM — constant envelope by definition | amplitude range 0.99–1.01, effectively constant |
| Indices 22–23 | GMSK, OQPSK | the remaining constant-modulus pair |

An independent third-party analysis of this exact file reaches the same
conclusion ([Spooner, *DeepSig's 2018 Dataset*, Cyclostationary Signal
Processing, 2020](https://cyclostationary.blog/2020/09/24/deepsigs-2018-data-set-2018-01-osc-0001_1024x2m-h5-tar-gz/)).

**Indices used:** `BPSK=3, QPSK=4, 16QAM=12, 64QAM=14`.

Confirmed downstream by the confusion matrix: 16QAM and 64QAM confuse chiefly
with *each other*, and per-class difficulty orders BPSK → QPSK → 16QAM → 64QAM.
That is the ordering the modulation-classification literature predicts, and it is
not what a wrong label mapping would produce.

---

## Format contract

Every example is identical in shape and statistics regardless of class or
origin. **If any property correlates with class, the model learns that property
instead of the signal.**

| Property | Value |
|---|---|
| Window length | 512 samples (every class) |
| Sample rate | 3.2 MHz nominal (every class) |
| Window duration | 160 µs |
| Final shape | `(2, 512)` float32 |
| Normalisation | zero-mean, unit-std |
| Labels | class index + SNR (dB) |

### Sample-rate reconciliation

The three sources do not share a native rate, and reconciling them needed a
decision rather than a conversion:

| Source | Native | Action |
|---|---|---|
| RadChar | 512 samples @ 3.2 MHz (absolute) | untouched — it anchors the contract |
| RadioML | 1024 samples, **no physical sample rate** | truncate to first 512 |
| Our generators | produced at 3.2 MHz | take first 512 samples |

**RadChar anchors the contract** because it is the only source with both a fixed
length and a fixed absolute rate.

**RadioML carries no physical sample rate at all.** It is distributed at complex
baseband in normalised time and frequency — the meaningful quantity is its
symbol rate of roughly 1/8 of the sampling rate (~8 samples per symbol), not any
figure in Hz. So there is **no rate mismatch to resample away**: we adopt 3.2 MHz
as the nominal rate for the assembled dataset, under which each civilian window
represents 160 µs and ~64 symbols. `scipy.signal.resample_poly` is available if a
future source ever does need true rate conversion.

**Truncation, never padding.** RadioML frames are cut 1024 → 512. Padding RadChar
*up* to 1024 would leave half of every radar example flat, and the model would
learn *"flat tail ⇒ radar"* — manufacturing the exact artefact the contract
exists to prevent. `tests/test_format_contract.py` enforces this.

---

## Class counts and SNR coverage

**SNR bins:** −10, −6, −2, +2, +6, +10 dB (6 bins).

These **must stay even**. RadioML samples SNR in 2 dB steps from an even base, so
an odd bin returns zero civilian examples — leaving that bin populated only by
radar/FHSS/jamming and teaching the model *"odd SNR ⇒ threat class"*. Our own
generators accept any value, so RadioML is the binding constraint.
`tests/test_config.py` enforces it.

**Assembled dataset:** 1000 examples per class per SNR bin.

| Class | Count | Source |
|---|---|---|
| BPSK | 6,000 | RadioML |
| QPSK | 6,000 | RadioML |
| 16QAM | 6,000 | RadioML |
| 64QAM | 6,000 | RadioML |
| LFM_RADAR | 6,000 | RadChar + generator |
| FHSS | 6,000 | generator |
| JAMMING | 6,000 | generator |
| **Total** | **42,000** | |

Perfectly balanced by construction — no class weighting is needed to correct a
source-size imbalance, though `compute_class_weights()` remains in place.

> **Note for whoever regenerates this:** the counts above are from a build in
> which RadChar was not yet present locally, so LFM_RADAR was fully synthetic.
> The class totals are unchanged when RadChar is added (`build_dataset` reduces
> the synthetic radar count to compensate), but the real/synthetic split within
> LFM_RADAR does change. Re-state that split once the final build is made.

---

## Stated limitations

Naming our own limits is more credible than being asked about them.

1. **Civilian accuracy is SNR-bound, and that is physics.** 64QAM packs 64
   constellation points into the same plane BPSK uses for 2. Below roughly
   −6 dB those points sit inside the noise floor, and no amount of training data
   separates them — the information is not present in the signal. We report
   accuracy-vs-SNR per class rather than a single headline number for this
   reason. The civilian classes are mandatory to classify under the rules but are
   not part of the >80% judged benchmark.

2. **16QAM/64QAM confusion is expected, and it is benign operationally.**
   Mistaking one QAM order for another still yields the correct coarse call —
   "ordinary civilian traffic". The costly error is civilian → hostile, measured
   separately as the false-alarm rate.

3. **One symbol rate, one pulse shape.** RadioML's digital signals appear to use
   a single symbol rate and a narrower range of root-raised-cosine roll-off than
   its documentation claims. Our civilian classes therefore represent less
   parameter diversity than the class names suggest, which is a generalisation
   risk on the organisers' unseen stream.

4. **Non-commercial licence.** CC BY-NC-SA 4.0 permits competition use; any
   commercial deployment would require different data.
