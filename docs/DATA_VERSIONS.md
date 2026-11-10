# Data versions: which data is the right one

The one rule: **a model is only comparable with another model trained on the same data version.**
Every training run prints `combined <version>` on its first line, and stores it in its checkpoint
(`best.pt` -> `meta["combined_version"]`). Check that before quoting or comparing any number.

```bash
python -c "import json;print(json.load(open('data/combined/combined.json'))['combined_version'])"
```
```bash
python -c "import torch;print(torch.load('results/classifier/seed4_best.pt',map_location='cpu',weights_only=False)['meta']['combined_version'])"
```

Training uses **synthetic data only** (our generators, plus RadioML 2018 / HisarMod 2019 rows
brought into our conventions). Real recordings are for validation and the demo, never training.

## Classifier data (60 classes, 512-sample windows, 960k / 120k / 120k)

| Version | Drive folder (`sedic/FOI26/...`) | What it is | Status |
|---|---|---|---|
| `de26671a5d48` | `combined_data` | First 60-class merge | ❌ superseded: RadioML windows had empty band edges and SNR labels ~10 dB too high (the model could learn "empty edges = RadioML class"); one label per window; FSK/PSK/QAM at 0 Hz |
| `3fae376e92d3` | `combined_data_new` | + external fix (`merge.harmonize_external`, 4 Nov) | ❌ superseded: one label per window; FSK/PSK/QAM at 0 Hz |
| `a5cda92d476f` | `combined_data_labels_v2` | + **labels-v2** multi-label rules (Jessy, 5 Nov): DRFM + the radar it copies, RADAR_CW / JAM_TONE together, straight LoRa-rate sweeps as LORA_CSS + JAM_SWEPT | ⚠️ **known bug**: the 18 FSK/PSK/QAM classes were never passed through the impairment chain, so every one sits at exactly 0 Hz. Models trained on it recognise them only when centred (seed 4: BPSK 100% at 0 Hz, 2% at +300 kHz; DSSS 0% at 0 Hz). **Keep it**: it is the old test set for the fair before/after comparison |
| **`f993d8ebc417`** | `combined_data_labels_v2_fskpskfix` | `a5cda92d476f` + FSK/PSK/QAM rows re-made **with** impairments (frequency placement like every other family), same merge settings and seed 4100, identical class counts (960k / 120k / 120k). Built by Eileen on 7 Nov 2026 01:02 UTC with `notebooks/colab_fskpsk_fix.ipynb` Part A, commit `34661ef`, `git_dirty: false` | ✅ **the right data from now on** |

How the fixed set is made: `scripts/reimpair_fskpsk_shards.py` re-creates each generated
FSK/PSK/QAM row from its stored seed through `impairments.generate_impaired` (spot-checking that
the seed still reproduces the stored window, so labels and SNRs cannot drift). It then re-merges
with `scripts/merge_shards.py` (external fix + labels-v2, mixtures, seed 4100). Source shards:
`FOI26/shards_data/{analog, apsk, fskpsk, jam, multicarrier, pulse, radar, spread}`.

## Classifier models and the data they were trained on

| Model | Data | Test top-1 (all SNR / >= 10 dB) | Where |
|---|---|---|---|
| seed 1 (n_fft 64, 15 ep) | `de26671a5d48` | 45.8% / 59.1% | `FOI26/members/seed1.pt` (baseline only) |
| seed 4, labels-v2 + expert features | `a5cda92d476f` | 49.4% / 62.8% | `FOI26/results/2026-11-05_nfft64_ep15_labelsv2_expert_seed4`; on the demo PC as `results/classifier/seed4_best.pt` (**current default**) |
| seed 4 + spectrum branch (Eileen, SED-39) | `a5cda92d476f` | 52.0% / 64.3% (20 ep: 52.2% / 64.6%) | `FOI26/results/2026-11-06_nfft64_ep15_labelsv2_expert_spectrum_seed4` |
| seed 4 + spectrum, fixed data | `f993d8ebc417` | *(Part B not run yet, 7 Nov)* | *(fill in when trained)*: intended next default |

Numbers on `a5cda92d476f` flatter FSK/PSK/QAM, because their test windows are centred too. When
comparing with the fixed model, use the notebook's step 11 (both models on both test sets).

## Detector (YOLO) data: separate from the classifier data

| Data | Drive folder | Model | Test result |
|---|---|---|---|
| 4,000 / 500 / 500 scenes (`scripts/make_scenes.py`, seed 2200) | `FOI26/detector_data/scenes_4000` | `results/detector_results/2026-11-05_detector_4000x5` | AP@0.5 0.39 (quick check) |
| 30,000 / 5,000 / 5,000 scenes | `FOI26/detector_data/scenes_30000` | `results/detector_results/2026-11-05_detector_30000x30` | **AP@0.5 0.70** (current, `results/detector/emitter_detector.pt`) |

Scenes are 10.24 ms x 3.2 MHz spectrogram pictures with automatic boxes from our generators; the
detector is single-class (one box per transmitter).

## Real recordings: never used for training

| Data | Where | Format | Use |
|---|---|---|---|
| Indonesia CW archive (64 files, ~378 h, 2018) | external drive `F:\12234 indonesia cw` (confidential: keep local) | 8 kHz 8-bit mono **audio** (not IQ), plus `.log`/`.txt` notes | Comms Intel page (`scripts/comms_intel.py` -> `results/comms_intel/`); demo clips (SED-37) |
| RSPduo recordings | to be recorded (SED-42) | SDRconnect IQ WAV (Recording type = IQ) | first real-signal validation of the pipeline |

## When a new version is made
1. Write it in the tables above: version, Drive folder, what changed, status.
2. Mark what it supersedes, and why.
3. Point `DATA_ON_DRIVE` in the training notebooks at it, and change `RUN_NAME`, so runs on different data never share a results folder.

## Active classifier (2026-11-09)

`results/classifier/physics_aug_best.pt` (git-ignored; Drive `FOI26/results/2026-11-09_physics_augment_seed4/best.pt`):
the same physics model retrained with rotation/flip augmentation (`--augment`), trained on **f993d8ebc417**, best epoch 15.
`physics_best.pt` (below) moved to `results/classifier/old/`. Only one `*best*.pt` stays in
`results/classifier/`, because the dashboard averages every one it finds and the average scored lower in the pipeline.

| Test | physics_best | physics_aug_best | average of both |
|---|---|---|---|
| f993d8ebc417, one window, top-1 | 48.7% | **49.8%** | 50.5% |
| f993d8ebc417, one window, >= 10 dB | 61.6% | **62.4%** | 63.3% |
| f993d8ebc417, vote of 16 windows, all SNR | 72.8% | 73.0% | 74.2% |
| f993d8ebc417, vote of 16 windows, >= 10 dB | 80.6% | 79.4% | 80.7% |
| Pipeline, 200 captures (seed 9000), track right | 46% | **50%** | 47% |
| Pipeline, same, track exact | 37% | **38%** | 38% |

The average wins on single windows but loses once the pipeline votes over a track, and costs twice the compute.
The vote rows use calibrate_week1's proxy (mean log-odds of 16 random test windows of the same class and SNR, from
different bursts); on it the augmented model gains less from voting at >= 10 dB. Quote 79.4% for the current model.
The pipeline, where a vote is one real emitter, is the deciding test.

## Previous classifier (2026-11-07)

`results/classifier/physics_best.pt` (git-ignored; Drive `FOI26/results/2026-11-07_physics_seed4/best.pt`):
seed 4, expert + spectrum + physics branches, trained on **f993d8ebc417**, best epoch 14.

| Test set | top-1 | >= 10 dB |
|---|---|---|
| f993d8ebc417 (fixed) | 48.7% | 61.6% (vote of 16 windows: 80.6%) |
| Spectrum model (old data) on the same set | 42.5% | 52.9% |

The spectrum model's 52.0% was on **a5cda92d476f**, whose FSK/PSK/QAM sit at 0 Hz like its training
data: not comparable. Always name the test set next to a number. Seed 4 moved to
`results/classifier/old/` (the dashboard averages every `*best*.pt` in `results/classifier/`).
