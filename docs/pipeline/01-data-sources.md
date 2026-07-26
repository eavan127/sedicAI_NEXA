# 01 — Data Sources & Acquisition

**Owner:** Person A · **Day:** 1 · **Blocks:** everything downstream

## Goal

Get labelled IQ data on disk for all seven classes. Four civilian classes and
the radar class come from published datasets; FHSS and jamming we synthesise.

## Tools

| Tool | Use | Licence |
|---|---|---|
| [h5py](https://www.h5py.org/) | Read RadioML/RadChar HDF5 files | BSD-3-Clause |
| [NumPy](https://numpy.org/) | Array handling, filtering to our classes | BSD-3-Clause |
| [Kaggle CLI](https://github.com/Kaggle/kaggle-api) | Scripted dataset download | Apache-2.0 |
| [SciPy](https://scipy.org/) | `resample_poly` for rate reconciliation | BSD-3-Clause |

```bash
pip install h5py kaggle
```

## Sources

### RadioML 2018.01A — civilian classes

24 modulations × 26 SNR levels (-20…+18 dB) × 4096 frames, 1024 complex samples
each. HDF5, **~21 GB**. Filter to BPSK, QPSK, 16QAM, 64QAM and discard the rest.

- [DeepSig official](https://www.deepsig.ai/datasets/) (registration) or
  [Kaggle mirror](https://www.kaggle.com/datasets/pinxau1000/radioml2018)
- Licence: **CC BY-NC-SA 4.0**

Start the download on Day 1 morning. 21 GB is not a background task you can
begin at 6 pm.

### RadChar — radar class

5 radar types (**LFM**, Barker, polyphase Barker, Frank, unmodulated pulse
train), SNR -20…+20 dB, 512 IQ samples at 3.2 MHz. Sizes: Tiny 50k (~400 MB) →
Large 2M (~16 GB).

- [GitHub](https://github.com/abcxyzi/RadChar) ·
  [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023)
- Licence: **CC BY-NC-SA 4.0** · Published at ICASSP 2023

**Use RadChar for LFM_RADAR rather than our own synthesis.** It is real
published labelled radar data, which removes the "we synthesised it and nobody
qualified checked it" problem for one of the three judged classes. Take the
**Tiny or Small** variant — we need thousands of examples, not two million.

Keep `src/generators/radar.py` regardless: it is tested, it is useful for
augmentation, and it is the fallback if the licence question below goes badly.

### FHSS and jamming — synthesised

No usable public raw-IQ dataset exists for either. Covered in
[03](03-fhss-generation.md) and [04](04-jamming-generation.md).

## ⚠️ Licence question — resolve on Day 1

Both datasets are **CC BY-NC-SA 4.0 (NonCommercial)**. SEDIC is a
defence-industry competition with sponsors and prizes; whether that is
"commercial" is genuinely unclear.

**Action:** email the organisers, ask plainly, keep the reply. Cite both
datasets and their licences in the technical brief either way.

Fallback if NonCommercial is refused: [TorchSig](https://github.com/TorchDSP/torchsig)
(MIT) for civilian classes, our own generator for radar. Slower and riskier —
which is exactly why this gets asked on Day 1, not Day 4.

## Layout

```
data/raw/          downloads, untouched
data/interim/      filtered/resampled intermediates
data/processed/    X.npy, y.npy, snr_labels.npy  <- what training reads
```

All gitignored. Share via Google Drive, never Git.

## Definition of done

- [ ] RadioML downloaded, filtered to the four civilian classes, SNR labels kept
- [ ] RadChar downloaded, LFM pulses extracted, SNR labels kept
- [ ] Both loaders return `(iq_complex, class_name, snr_db)` tuples
- [ ] `load_radioml_civilian()` in `src/data/build_dataset.py` no longer a stub
- [ ] Sample-rate mismatch documented and a target rate agreed (see [05](05-preprocessing.md))
- [ ] Licence question sent to organisers

## Open questions

- Which RadChar variant? (Tiny is likely enough — start there.)
- Do we keep RadChar's non-LFM radar types (Barker, Frank) as extra military
  examples, or only LFM? The rules say *"Radar Pulses (e.g., Linear Frequency
  Modulation)"* — "e.g." suggests LFM is an example, not the only accepted form,
  so including them may improve robustness on the hidden stream. Decide with the
  team; document whichever way in the brief.
