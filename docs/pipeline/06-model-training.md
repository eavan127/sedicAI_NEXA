# 06 — Model & Training

**Owner:** Person D · **Day:** 2–3

## "Fine-tuning" does not apply here

There is no pretrained backbone for raw IQ the way ImageNet exists for images —
so we **train from scratch**, from random weights. Anyone writing the brief
should not describe this as fine-tuning.

The nearest thing to pretrained RF weights is
[TorchSig](https://github.com/TorchDSP/torchsig), which ships models pretrained
on complex-valued signals — but on **communications modulations only**, not
radar/FHSS/jamming. Not useful for our judged classes.

## Architecture

`src/models/amc_cnn.py` — a 1D-CNN over the `(2, N)` IQ array.

```
Input (2, 1024)
  Conv1d(2→64, k=8) → BatchNorm → ReLU → MaxPool
  Conv1d(64→128, k=8) → BatchNorm → ReLU → MaxPool
  Flatten → Dropout(0.5) → Linear(→256) → ReLU → Linear(→7)
```

The flattened width is inferred by a dummy forward pass rather than hand-computed
— padding/pooling arithmetic is easy to get subtly wrong, and
`tests/test_pipeline.py::TestModel::test_adapts_to_window_length` proves changing
`window_len` does not break it.

**Why a plain CNN, not a Transformer:** 1D-CNNs are the well-published AMC
baseline, they train in minutes on a free GPU, and with four days a bigger model
buys risk, not accuracy. Our bottleneck is data quality, not model capacity.

## Tools

| Tool | Use | Licence |
|---|---|---|
| [PyTorch](https://pytorch.org/) | Model, autograd, optimiser | BSD-3-Clause |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | Loss/accuracy curves | Apache-2.0 |
| [Google Colab](https://colab.research.google.com/) | Free GPU | free tier |
| NumPy | Splitting, class weights | BSD-3-Clause |

TensorFlow would be equally compliant with the rules — we chose PyTorch because
the RF-ML ecosystem is PyTorch-centric. **Do not mix both.**

## Training setup

| Choice | Value | Why |
|---|---|---|
| Loss | Cross-entropy, inverse-frequency class weights | Judged classes are the minority; unweighted loss optimises for classes we are not scored on |
| Optimiser | Adam, lr 1e-3 | Standard, converges fast |
| Scheduler | `ReduceLROnPlateau`, patience 3 | Drops LR when validation stalls |
| Epochs | 30 | Small model; raise if still improving |
| Batch size | 64 | |
| Checkpoint | Best validation loss → `results/best_model.pt` | Never ship the last epoch |

### The split matters more than usual

`stratified_split()` stratifies by **class *and* SNR bin jointly**. Stratifying
by class alone can put an entire SNR bin in one split, which silently puts a hole
in the accuracy-vs-SNR curve and makes the low-SNR claim unverifiable.

Guarded by `tests/test_pipeline.py::TestSplit` — disjointness, full coverage of
every class and SNR in the test split, and determinism for a given seed.

## Running

Smoke run first (seconds — proves wiring, results meaningless):

```bash
SEDIC_CONFIG=configs/smoke.yaml python -m src.train
```

Real run:

```bash
python -m src.train
```

Colab: mount Drive, `pip install -r requirements.txt`, upload `data/processed/`,
run the same command. The dataset is far smaller than RadioML raw, so uploading
processed arrays beats re-downloading 21 GB.

## If recall misses 90%

Work down this list — data first, model last:

1. **Read the confusion matrix before changing anything.** Which pair is being
   confused? Sweep-jamming vs. LFM radar is the predicted trouble spot.
2. **Check accuracy-vs-SNR.** Failing only at low SNR is expected and reportable;
   failing everywhere means the data is wrong, not the model.
3. **Raise class weights** on the failing class.
4. **Generate more examples** for it.
5. **Widen its parameter randomisation** — often an overfitting-to-one-signature
   problem.
6. **Then** touch the architecture (more filters, third conv block).

Steps 1–5 are usually the fix. Reaching for a bigger model first wastes the
little time you have.

## Definition of done

- [ ] Smoke run completes end to end
- [ ] Real run completes, `results/best_model.pt` saved
- [ ] Validation accuracy plateaus (not still climbing at the last epoch)
- [ ] `pytest tests/test_pipeline.py` passes
- [ ] Training curves exported for the brief
