# sedicAI_NEXA

**SEDIC 2026 — RF/Signal Track ("Project Overwatch")**
AI model for detecting and classifying radio signals (civilian modulations, military radar/FHSS, jamming) from raw IQ data.

Full technical documentation: [`docs/SEDIC2026_Track1_Documentation.md`](docs/SEDIC2026_Track1_Documentation.md)

## Structure

```
/data          gitignored — synthetic + RadioML arrays live here locally, never committed
/scripts       all pipeline code (generators, preprocessing, model, train/eval/infer)
/notebooks     Colab training notebooks
/results       confusion matrices, accuracy-vs-SNR plots, classification logs
/docs          technical brief + this project's documentation
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline order

1. `python scripts/gen_radar.py` / `gen_fhss.py` / `gen_jamming.py` — self-QA check (saves spectrogram plots to `results/`, compare against reference literature before trusting the data)
2. `python scripts/build_dataset.py` — assembles full dataset into `data/`
3. `python scripts/train.py` — trains `AMC_CNN`, saves best checkpoint to `results/best_model.pt`
4. `python scripts/evaluate.py` — confusion matrix + accuracy-vs-SNR curve
5. `python scripts/infer.py --input <qualifier_iq_file> --output results/classification_log.csv` — final submission log

## Team roles & 4-day timeline

See section 13 of the documentation for the full A/B/C/D day-by-day breakdown.

