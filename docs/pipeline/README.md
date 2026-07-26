# Pipeline Documentation

One document per stage. Each states what the stage does, which tools it uses and
why, how to run it, and what is still undecided.

| # | Stage | Owner | Doc |
|---|---|---|---|
| 01 | Data sources & acquisition | A | [01-data-sources.md](01-data-sources.md) |
| 02 | Radar (LFM) — classification class | A | [02-radar-generation.md](02-radar-generation.md) |
| 03 | FHSS — classification class | B | [03-fhss-generation.md](03-fhss-generation.md) |
| 04 | Jamming — classification class | C | [04-jamming-generation.md](04-jamming-generation.md) |
| 05 | Preprocessing & dataset assembly | A + D | [05-preprocessing.md](05-preprocessing.md) |
| 06 | Model & training | D | [06-model-training.md](06-model-training.md) |
| 07 | Evaluation & benchmark | D | [07-evaluation.md](07-evaluation.md) |
| 08 | Inference & submission | C + A | [08-inference-submission.md](08-inference-submission.md) |

Master tool/licence list: [`../TOOLS.md`](../TOOLS.md)

## The seven classes

| Class | Tier | Source | Judged at >90% recall |
|---|---|---|---|
| BPSK, QPSK, 16QAM, 64QAM | Civilian | RadioML 2018.01A | no |
| LFM_RADAR | Military / CEMA | RadChar dataset | **yes** |
| FHSS | Military / CEMA | our synthesis | **yes** |
| JAMMING | Hostile CEMA | our synthesis | **yes** |

The three judged classes carry the entire benchmark. Two of them are synthetic
with no external ground truth — that is where the project's risk lives, and why
`tests/` asserts the generator maths rather than trusting it.
