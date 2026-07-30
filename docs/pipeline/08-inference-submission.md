# 08 — Inference & Submission

**Owner:** Person C + Person A · **Day:** 3–4

## Goal

Run the trained model on the organisers' **"Qualifier IQ Data Stream"** and
produce the classification log, then package everything the rules require.

## ⚠️ The highest-risk line of code in the repo

```python
raw = np.fromfile(path, dtype=np.float32)
iq = raw[0::2] + 1j * raw[1::2]
```

`load_iq_file()` assumes **interleaved float32 I,Q,I,Q…**. If the organisers ship
`complex64`, `int16`, a different endianness, or a file with a header, this
produces **garbage that still runs to completion** — a full, plausible-looking
CSV of meaningless predictions.

**Confirm the format the moment the file is released.** Ask the organisers
directly if it is not documented. Then verify before trusting the output:

1. Plot a spectrogram of the loaded IQ — does it look like *any* radio signal?
2. Check the sample count against the stated duration and rate
3. Check amplitudes are sane (not `1e38`, not all zeros)
4. Try `complex64` and `int16` as alternatives and see which yields a sensible
   spectrogram

A wrong dtype is the single most likely way to submit a confident, worthless log.

## Tools

| Tool | Use | Licence |
|---|---|---|
| NumPy | `fromfile`, deinterleaving, windowing | BSD-3-Clause |
| PyTorch | Batched inference | BSD-3-Clause |
| SciPy + Matplotlib | Spectrogram sanity check on the input | BSD-3-Clause |
| Python `csv` | Log output (stdlib) | PSF |

## Running

```bash
python -m src.infer --input data/raw/qualifier_iq_stream.bin --output evals/classification_log.csv
```

Slides a `window_len` window across the stream, classifies each, writes:

| Column | Meaning |
|---|---|
| `window_index` | Sequential index |
| `sample_start` | Offset into the stream |
| `predicted_class` | One of the seven |
| `confidence` | Softmax probability of the winner |
| `is_threat` | True for LFM_RADAR / FHSS / JAMMING |

`is_threat` is the "cognitive CEMA" framing the mission name asks for — the log
does not just classify, it flags. Cheap to add, reads well to judges.

Overlapping windows (`--stride` less than `window_len`) catch short bursts that
straddle a boundary. Costs only compute.

## Submission package

Per Section 4 of the rules:

| Item | Where it comes from | Owner |
|---|---|---|
| **Model source code** | This repo — `src/`, `configs/`, `tests/`, README | A |
| **Classification log & results** | `evals/classification_log.csv` | C |
| **Performance benchmark** (>90% recall on Military/CEMA + Jamming) | `evals/scorecard.json` + confusion matrix + accuracy-vs-SNR | D |
| **Technical brief** (PDF) | Dataset, architecture, DSP logic, limitations | B |
| **Video demo** (≤5 min, YouTube) | Screen recording | D |

### The brief is scored — write it accordingly

Phase 1 has five deliverables and only one is code. The brief is where the panel
forms its opinion.

**What reads as expertise**
- An **ablation table** — every design choice measured, not asserted
- A **held-out generalisation test** — proof the model transfers, not a claim
- **Comms-vs-hostile-CEMA reported as its own headline number**, in the rules'
  own vocabulary
- **Limitations named before the panel finds them**
- **Where the system stops and why** — we do detection and classification, not
  friend-or-foe, because that needs an ELINT library and IFF correlation

**What reads as inexperience**
- Uniform ~99% with no accuracy-vs-SNR curve (implies leakage or a shortcut)
- No limitations section
- Architecture diagram with no justification
- Claiming friend-or-foe capability
- Overall accuracy quoted where the rules ask for per-class recall

### Technical brief must cover

- **Datasets**: RadioML 2018.01A and RadChar, both **CC BY-NC-SA 4.0**, cited
  with attribution — exact wording in [`../TOOLS.md`](../TOOLS.md)
- **Synthetic generation**: how FHSS and jamming were produced, with parameter
  ranges and the literature they came from
- **Validation methodology**: the `tests/` suite — and honestly, that it verifies
  internal consistency rather than real-world realism
- **Architecture**: 1D-CNN, trained from scratch (no pretrained RF backbone
  exists), class weighting, augmentation
- **Results**: per-class recall, confusion matrix, accuracy-vs-SNR, and the SNR
  at which each judged class crosses 90%
- **Limitations**: named plainly
- **Phase 2 roadmap**: one line on the live GUI/waterfall we would build if we
  reach the Top 10 — shows we understand the full arc without spending days on it
  now

### Video demo (≤5 min)

Rough shape: problem and the seven classes (~30s) → architecture and data
sources (~1 min) → spectrograms of each class (~1 min) → live run of
`src.infer` on the qualifier stream (~1.5 min) → results and the benchmark table
(~1 min). Record with OBS Studio; upload unlisted if you prefer.

## Not required in Phase 1

GUI, live demo station, poster, jury presentation — all Phase 2, Top 10 only.
**Do not build a GUI this week.**

## Definition of done

- [ ] IQ file format confirmed with organisers, verified by spectrogram
- [ ] `classification_log.csv` generated and spot-checked (predictions vary
      sensibly; not one class for the entire stream)
- [ ] Scorecard shows PASS on all three judged classes
- [ ] Technical brief PDF complete, limitations section included
- [ ] Video recorded, edited, under 5 minutes, uploaded
- [ ] Repo clean: no large binaries committed, README accurate, `pytest` green
- [ ] **Submitted with hours to spare, not minutes**
