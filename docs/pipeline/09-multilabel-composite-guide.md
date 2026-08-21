# 09 — RadioML & Composite Dataset Guide (multi-label branch)

**Branch:** `eavan-multilabel-jamming-overlay` (pushed to GitHub)
**Audience:** whoever is building the dataset and training on this branch —
written so you understand *what* you're generating and *why*, not just which
commands to paste.

---

## 1. What changed vs. `main`

`main`'s model picks exactly **one** of 8 classes per window (softmax). This
branch's model can flag **more than one class at once** (sigmoid,
independent per class) — because real jamming targets something. A jammer
overlaid on top of real traffic is a normal scenario, but the old dataset
only ever contained one class per window, so the model had never even seen
that case, let alone been able to report it (softmax structurally cannot say
"both").

Two concrete effects on what you're about to generate:

1. **Labels are multi-hot**, shape `(N, 8)` — one column per class, 1 if
   present. A standalone example has one `1`; a composite example
   (jammer-on-victim) has two.
2. **The dataset now includes composite examples**, additive on top of
   everything `main` already generates — nothing standalone is removed.

---

## 2. RadioML — what it is and what happens to it

**RadioML 2018.01A**: 24 modulations × 26 SNR levels × 4096 frames, HDF5,
~21 GB. We only use 4 of its 24 classes — BPSK, QPSK, 16QAM, 64QAM — and
discard the rest. Real, published, externally-labeled — this is why the
civilian classes don't need to be synthesized. Licence CC BY-NC-SA 4.0,
citation required in the technical brief.

### Get it

```bash
pip install h5py kaggle
```

Kaggle account → Settings → **Create New API Token** → downloads
`kaggle.json` → put it at `C:\Users\<you>\.kaggle\kaggle.json`.

```bash
kaggle datasets download -d pinxau1000/radioml2018 -p data/raw --unzip
```

Confirm the result is at exactly `data/raw/GOLD_XYZ_OSC.0001_1024.hdf5` —
Kaggle sometimes unzips into a nested folder or a slightly different name;
rename/move it if so. `src/data/build_dataset.py` prints a clear "not found"
warning with the exact expected path if this is wrong — it will not error
out silently, it'll just run with civilian classes empty.

Also grab RadChar (real LFM radar data) the same way, into
`data/raw/RadChar-Tiny.h5`:

```bash
kaggle datasets download -d abcxyzi/radchar-icassp-2023 -p data/raw --unzip
```

### What `build_dataset.py` actually does with it — twice

This is the part worth understanding, not just running:

1. **Standalone pass** — `load_radioml_civilian()` reads RadioML, filters to
   the 4 target classes, subsamples `examples_per_class_per_snr` (1000,
   default config) per (class, SNR) bin. These become ordinary one-class
   examples, exactly like on `main`.
2. **Composite pass** — RadioML is loaded a **second time**, with a
   *different* random seed, as a separate pool of civilian "victim" signals.
   `build_composite_examples()` draws from that second pool, overlays a
   randomly-chosen jammer on top (`overlay_jamming()` in
   `src/data/composite.py`, using the already-tested `apply_jamming()`), and
   labels the result with **both** bits set: `{BPSK, JAMMING}`,
   `{QPSK, JAMMING}`, etc.

**Why a second, separate load instead of reusing the same rows**: so a
composite example's underlying civilian waveform isn't the literal same row
already used standalone — different seed, different sample.

**Why civilian composites specifically need RadioML present**: LFM_RADAR and
FHSS can be composite victims too (via the synthetic generators, no external
data needed), but BPSK/QPSK/16QAM/64QAM composites have no other source —
if RadioML is missing, you still get radar/FHSS composites, but zero
civilian ones, and the print output at the end will show `BPSK 0`, `QPSK 0`,
etc.

---

## 3. Run it

```bash
python -m src.data.build_dataset
```

Watch the printed summary at the end:

```
sources: <N> real RadChar, <N> synthetic/RadioML, <N> composite (jammer-overlaid)
Built <N> examples, shape per example (2, 512), labels shape (<N>, 8)
  BPSK          <N>  (present in this many windows)
  ...
  composite windows (>1 class present)   <N>
```

If any civilian class prints `0`, RadioML wasn't found or wasn't loaded —
stop and fix that before training on it.

**Sanity check before spending any GPU time:**

```bash
pytest tests/ -q
```

104 tests should pass, including composite-generator tests
(`tests/test_generators.py::TestCompositeOverlay`) and multi-label metric
tests (`tests/test_metrics.py`).

---

## 4. What you end up with

```
data/processed/X.npy           (N, 2, 512)  float32
data/processed/y.npy           (N, 8)       float32, multi-hot
data/processed/snr_labels.npy  (N,)         float
```

At the default config (`examples_per_class_per_snr: 1000`, 6 SNR bins,
`overlay_fraction: 0.3`): ~6,000 standalone examples per class, ~10,800
additional composite examples, **~58,800 total**. `overlay_fraction` isn't
independently validated yet the way `examples_per_class_per_snr` is (that
one came from a real experiment — see `configs/default.yaml`'s comments on
400→1000→2000) — treat it as a starting point, not a settled number.

**These three files are the only thing that needs to leave your machine.**
Never upload raw RadioML/RadChar (gigabytes) — Colab trains on the small
processed arrays only.

---

## 5. Where it goes next

- Upload the three `.npy` files into the shared Google Drive folder
  (`MyDrive/sedic/`), or hand them directly to whoever is running the Colab
  notebook: `notebooks/colab_training_multilabel.ipynb`.
- That notebook clones `eavan-multilabel-jamming-overlay`, sanity-checks the
  label shape is `(N, 8)` **before** training (so an accidentally-uploaded
  old single-label array fails loudly, not silently), then trains an
  ensemble + single model and prints a plain PASS/FAIL against the 80%
  judged-class recall gate.

---

## Definition of done

- [ ] RadioML present at `data/raw/GOLD_XYZ_OSC.0001_1024.hdf5`
- [ ] RadChar present at `data/raw/RadChar-Tiny.h5`
- [ ] `python -m src.data.build_dataset` run, all 4 civilian classes non-zero
      in the printed summary
- [ ] `pytest tests/ -q` passes (104 tests)
- [ ] `y.npy` confirmed shape `(N, 8)`
- [ ] Three `.npy` files uploaded to the shared Drive folder / handed off
