# Tools & Open-Source Libraries — Master List

The rules require *"Model Source Code: Utilizing standard open-source libraries
(e.g., PyTorch, TensorFlow, GNU Radio)."* Everything below is open source. This
is the single reference for what we use, what we deliberately do not, and the
licence of each — the technical brief has to cite these.

**Legend:** ✅ in use · ⚠️ optional / evaluate · ❌ considered and rejected

---

## 1. Core stack (what we actually run)

| Tool | Purpose | Licence | Status |
|---|---|---|---|
| [PyTorch](https://pytorch.org/) | Model definition, training, inference | BSD-3-Clause | ✅ |
| [NumPy](https://numpy.org/) | All IQ array maths, signal synthesis | BSD-3-Clause | ✅ |
| [SciPy](https://scipy.org/) | STFT/spectrograms, filtering, chirp maths | BSD-3-Clause | ✅ |
| [scikit-learn](https://scikit-learn.org/) | Confusion matrix, per-class recall report | BSD-3-Clause | ✅ |
| [Matplotlib](https://matplotlib.org/) | Spectrogram QA plots, accuracy-vs-SNR curve | PSF-based (BSD-compatible) | ✅ |
| [PyYAML](https://pyyaml.org/) | Config loading (`configs/*.yaml`) | MIT | ✅ |
| [pytest](https://pytest.org/) | DSP correctness + pipeline tests | MIT | ✅ |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | Training curves | Apache-2.0 | ✅ |

Install: `pip install -r requirements.txt`

**Why PyTorch over TensorFlow:** the rules name both; either is compliant. We
picked PyTorch because the RF-ML ecosystem (TorchSig, published AMC reference
implementations) is overwhelmingly PyTorch-based. Do not mix both — pick one.

---

## 2. Datasets

| Dataset | Covers | Format | Licence | Status |
|---|---|---|---|---|
| [RadioML 2018.01A](https://www.deepsig.ai/datasets/) (DeepSig) | 24 modulations incl. BPSK/QPSK/16QAM/64QAM, 26 SNR levels, 4096 frames each, 1024 samples/frame, HDF5, ~21 GB | Complex IQ | CC BY-NC-SA 4.0 | ✅ civilian classes |
| [RadChar](https://github.com/abcxyzi/RadChar) | 5 radar types: **LFM**, Barker, polyphase Barker, Frank codes, unmodulated pulse trains. SNR -20..+20 dB. 512 IQ samples @ 3.2 MHz | Complex IQ | CC BY-NC-SA 4.0 | ✅ radar class |
| Our synthetic generators (`src/generators/`) | FHSS, jamming (barrage/tone/sweep) | Complex IQ | ours | ✅ |

**RadChar is the most important find in this document.** It is a real, published,
peer-reviewed (ICASSP 2023) labelled radar dataset containing exactly the LFM
pulses the rules mandate. Using it for the radar class replaces
self-synthesised data we have no expert to validate — directly reducing our
single biggest risk. Download via [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023).

There is **no equivalent public dataset for FHSS or jamming** at raw-IQ level in
a form we can use, so those two classes stay synthetic. That is where the
residual risk now concentrates — see [`pipeline/03-fhss-generation.md`](pipeline/03-fhss-generation.md)
and [`pipeline/04-jamming-generation.md`](pipeline/04-jamming-generation.md).

### Licence & attribution

RadioML and RadChar are both **CC BY-NC-SA 4.0** (Attribution–NonCommercial–ShareAlike).

**Using them is endorsed by the rules.** Section 3 directs participants to source
training data *"using open-source datasets (e.g., RadioML, DeepSig)"* — the
organisers name RadioML and DeepSig explicitly. RadChar sits under the identical
licence, and "e.g." marks that list as examples rather than exhaustive, so it
falls under the same endorsement.

**What we still owe: attribution.** CC BY requires citing the source and naming
the licence. In the technical brief:

> Training data sourced from RadioML 2018.01A (DeepSig Inc.) and RadChar
> (Huang et al., ICASSP 2023), both licensed CC BY-NC-SA 4.0.

Two lines. Do it and it is handled.

---

## 3. RF-specific ML libraries

| Tool | What it gives us | Licence | Status |
|---|---|---|---|
| [TorchSig](https://github.com/TorchDSP/torchsig) | 60+ modulations (FSK/QAM/PSK/ASK/OFDM/analog) with channel impairments, PyTorch-native, API mirrors TorchVision. Pretrained complex-valued models. Python ≥3.10 | MIT | ⚠️ evaluate |
| [rfml](https://github.com/brysef/rfml) | Reference AMC training notebooks in PyTorch — useful as a sanity check on our architecture | BSD-3-Clause | ⚠️ reference only |
| [scikit-commpy](https://github.com/veeresht/CommPy) | Rayleigh/Rician fading channels, PSK/QAM modulation, AWGN | BSD-3-Clause | ⚠️ if we add fading |

**TorchSig — the honest assessment.** Generates civilian modulations on demand
and would remove the 21 GB RadioML download. But its documented signal families
are **communications modulations only — no radar, no FHSS, no jamming**, which
are exactly our judged classes. So it could only replace RadioML, which the rules
already endorse, and it recommends Ubuntu, 1 TB storage, and a 16 GB GPU.

**Verdict: skip it.** It cannot replace our generators, and swapping out the
four classes nobody benchmarks us on is not worth the setup cost in four days.
Docker is likewise optional for TorchSig (`pip install -e .` is the primary
route) and irrelevant to us.

---

## 4. GNU Radio (named in the rules)

| Tool | Purpose | Licence | Status |
|---|---|---|---|
| [GNU Radio](https://www.gnuradio.org/) | SDR/DSP framework, flowgraph-based signal generation | GPL-3.0 | ⚠️ optional |
| [radioconda](https://github.com/radioconda/radioconda-installer) | The practical way to install GNU Radio on **Windows** | BSD-3-Clause | ⚠️ if GNU Radio used |
| [gr-plasma](https://github.com/ShaneFlandermeyer/gr-plasma) | GNU Radio OOT module for radar — generates LFM waveforms at arbitrary bandwidth/pulse width | GPL-3.0 | ⚠️ optional |

**Do we need GNU Radio? Honestly, no.** Signal synthesis is arithmetic; NumPy
does it in milliseconds with no install risk, and our generators are already
written and tested. GNU Radio's value here is *credibility* — the rules name it,
and citing a GNU Radio cross-check in the brief reads well.

**If you do use it:** install via radioconda on Windows (a plain source build is
a known time sink), and drive it from the Python API (`gr.top_block`) rather than
clicking through GNU Radio Companion — you cannot hand-generate thousands of
examples through a GUI.

**Time-box this.** If GNU Radio is not installed and producing a plot within two
hours, drop it and note in the brief that generation was done in NumPy/SciPy with
GNU Radio used only for verification. That is an honest, defensible position.

---

## 5. Considered and rejected

| Tool | Why not |
|---|---|
| [Sionna](https://developer.nvidia.com/sionna) (NVIDIA, Apache-2.0) | Excellent TensorFlow 5G/6G link-level simulator with 3GPP channel models — but it is built for communications PHY research, not radar/jamming classification, and pulls in a whole second ML framework. Wrong tool, wrong week. |
| [scikit-rf](https://scikit-rf.org/) | RF/microwave **network** analysis (S-parameters, calibration). Unrelated to signal classification despite the name. |
| MATLAB Phased Array / Radar Toolbox | Genuinely good at radar waveforms, but proprietary — fails the open-source requirement outright. |
| Azure ML / AWS SageMaker / Vertex AI | Paid cloud platforms. Nothing here needs them; the model trains on a free Colab GPU or a laptop CPU. |

---

## 6. Compute & tooling

| Tool | Purpose | Cost |
|---|---|---|
| [Google Colab](https://colab.research.google.com/) | Free GPU for training | Free tier |
| [Kaggle Notebooks](https://www.kaggle.com/code) | Free GPU; also hosts RadChar + RadioML mirrors | Free |
| Local CPU | Our 1D-CNN is small enough to train on a laptop | Free |
| [Git LFS](https://git-lfs.com/) | Only if a checkpoint must be committed | Free |
| [OBS Studio](https://obsproject.com/) | Recording the ≤5 min video demo | GPL-2.0 |

**No paid service is required for any part of this submission.**

---

## 7. Sample-rate reconciliation — a real integration problem

The three data sources do not agree on sampling:

| Source | Samples/example | Sample rate |
|---|---|---|
| RadioML 2018.01A | 1024 | unspecified (normalised) |
| RadChar | 512 | 3.2 MHz |
| Our generators | configurable | 2 MHz (`configs/default.yaml`) |

Mixing these naively means the model can learn *"512-long ⇒ radar"* — it would
score beautifully on our data and collapse on the organisers' stream, which is
the exact failure mode we are trying to avoid.

**Resolve by resampling everything to one rate and one window length before
training** (`scipy.signal.resample_poly`), and confirm class balance is not
correlated with any preprocessing artefact. Owner: Person A + Person D jointly,
Day 2. Tracked in [`pipeline/05-preprocessing.md`](pipeline/05-preprocessing.md).

---

## Sources

- [TorchSig — TorchDSP/torchsig](https://github.com/TorchDSP/torchsig)
- [TorchSig project site](https://torchsig.com/)
- [DeepSig datasets (RadioML)](https://www.deepsig.ai/datasets/)
- [RadioML 2018.01A on Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018)
- [RadChar — abcxyzi/RadChar](https://github.com/abcxyzi/RadChar)
- [RadChar on Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023)
- [Multi-task Learning for Radar Signal Characterisation (RadChar paper)](https://arxiv.org/html/2306.13105v2)
- [gr-plasma — GNU Radio radar module](https://github.com/ShaneFlandermeyer/gr-plasma)
- [GNU Radio Windows install](https://wiki.gnuradio.org/index.php/WindowsInstall)
- [radioconda installer](https://github.com/radioconda/radioconda-installer)
- [CommPy — veeresht/CommPy](https://github.com/veeresht/CommPy)
- [rfml — brysef/rfml](https://github.com/brysef/rfml)
- [NVIDIA Sionna](https://developer.nvidia.com/sionna)
- [scikit-rf](https://scikit-rf.org/)
