# Real-world HF evaluation (KiwiSDR)

Real short-wave signals recorded on a public KiwiSDR (9M2AMH, Kuala Lumpur), labelled by a person, used to
**measure** NEXA on real signals. Never used for training: the synthetic-only training rule holds.

| Step | Command | Output |
|---|---|---|
| 1. Collect | `python scripts/kiwi_collect.py` (needs `D:\tools\kiwiclient`) | `data/kiwi/*.wav` (12 kS/s IQ, 30 s each) |
| 2. Pre-label | runs at the end of step 1, or `python scripts/kiwi_label.py data/kiwi` | `data/kiwi/labels.csv`, `suggested` column |
| 3. Confirm by ear | fill in `human_label` in `labels.csv` (guide: `docs/KIWI_LABELLING.md`) | |
| 4. Score | `python scripts/eval_kiwisdr.py` | `evals/kiwisdr/scorecard.json`, `evals/kiwisdr/clips.csv` |

**What is scored:** each clip goes through the live analysis path (the same one an SDRconnect recording takes:
channel finder → zoom rules → slow-path model). NEXA's answer for the strongest transmission is compared with
`human_label`.

| human_label | Counts as right when NEXA says |
|---|---|
| noise | nothing above the noise |
| cw | CW |
| rtty | RTTY (2FSK) |
| psk31 | PSK31 (BPSK) |
| ssb_voice | USB or LSB |
| am | AM, or CARRIER (a time station between its ticks) |
| ft8 | **not supported** yet: reported separately, never counted |
| unknown | not scored |

Only `source = human` rows are a real score. `--use-suggested` also scores the pre-labeller's guesses for a
quick look, but those numbers are not evidence.

The recordings stay in `data/kiwi/` (not in git, about 1.4 MB each). Credit the receiver when showing them:
"received on KiwiSDR 9M2AMH, Kuala Lumpur".
