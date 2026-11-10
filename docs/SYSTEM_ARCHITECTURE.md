# NEXA system architecture: input → processing → audit → output

State on `main`, 10 Nov 2026. Everything runs on the local machine; the only outside links are the ones a
person clicks (KiwiSDR) and the optional SDRconnect laptop on the same private network.

## 1. The models (one of each, all current)

| Model | File | Used by | Numbers |
|---|---|---|---|
| **60-class classifier** | `results/classifier/physics_aug_best.pt` (Drive `FOI26/results/2026-11-09_physics_augment_seed4/best.pt`, epoch 15, rot90+flip augmentation, data `f993d8ebc417`) | Signal Pipeline, `run_pipeline.py`, live bridge `--mode full` | 49.8% one window, 62.4% at ≥10 dB, **79.4%** with the 16-window vote at ≥10 dB; in full busy scenes ≈42% of all emitters named right (Eileen, SED-39) |
| **Emitter detector** (YOLO) | `results/detector/emitter_detector.pt` | Signal Pipeline, `run_pipeline.py` | test AP@0.5 0.70 |
| **Slow-path model** (narrowband, 2 s at 50 kS/s) | `src/pipeline/narrowband_model.joblib` (in git) | every zoom-in: Signal Pipeline listen step, live bridge, auto-tune | names RTTY / PSK31 / AFSK / empty band; zoom rules keep the final say on AM / NBFM / USB / LSB |
| **Speech** | Whisper `medium` (cached locally, never downloaded at run time) | listen step, live bridge | English / Malay / Indonesian / Mandarin (`--language`) |
| **AI reading** | Ollama `llama3.2:3b` (local) | listen step `--llm`, Comms Intel | grounded: every quote must occur in the decoded text |
| **Browser classifier** | `web/models/*.onnx` | RF Replay / Signal Analysis / browser live receiver, Model-page fine-tune | being re-exported from `physics_aug_best.pt` (branch from the browser-model session) |

Keep **one** `*best*.pt` in `results/classifier/`: the dashboard averages every one it finds (old ones live in `old/`).

## 2. Inputs, and the path each one takes

| # | Input | Where it comes from | Processing | Results stored in |
|---|---|---|---|---|
| A | **IQ file, browser** | RF Replay page (upload) | in-browser ONNX classifier, per window | `data/local/nexa.db` (`analyses`) + raw IQ under `data/local/iq/` |
| B | **Simulated SDR / browser live receiver** | RF Replay page | same as A | same as A |
| C | **IQ file, Python pipeline** | Signal Pipeline page → `POST /api/pipeline`, or `scripts/run_pipeline.py --iq` | detector → 60-class classifier → tracker → `resolve` (pairs, size groups, reactive jammers) → **listen**: zoom → zoom rules + slow-path model → demodulate → Morse decoder / Whisper → AI reading | `results/pipeline/jobs/<id>/` (`tracks.json`, `tracks.png`, `track_N.wav`) |
| D | **Live radio** | SDRconnect (RSPduo, or a file playing) → WebSocket :5454 → `scripts/live_sdrconnect.py --host …` | records until the stream stops, then the voice path of C (or `--mode full`); `--autotune` sets SDRconnect's VFO / mode / filter | `results/pipeline/jobs/<id>/` (marked `live`) + decoded tracks into `results/comms_intel/` |
| E | **Replay as live** | `live_sdrconnect.py --replay file.wav` (demo set, KiwiSDR clips) | same as D; KiwiSDR file names give the real centre frequency | same as D |
| F | **Receiver audio archive** | `scripts/comms_intel.py --in <folder of .wav>` (Indonesia CW archive) | Morse decoder / Whisper → AI reading | `results/comms_intel/` |
| G | **KiwiSDR (real HF)** | `scripts/kiwi_collect.py` → public receiver 9M2AMH | pre-label (`kiwi_label.py`) → human confirms → `eval_kiwisdr.py` runs path D on each clip | `data/kiwi/` (clips, `labels.csv`), `evals/kiwisdr/` (scorecard) |
| H | **Analyst corrections** | History / Signal Analysis (browser) | correction → approval (four-eyes) → retrain trigger → `retrain_from_feedback.py` → candidate model → gate → approve/activate | `nexa.db` (`corrections`, `model_versions`), retrained ONNX files |

## 3. Audit (one append-only, hash-chained log: `nexa.db` → `audit_log`, page: Audit trail)

| Event | Logged by | Action names |
|---|---|---|
| Browser analyses, raw IQ, deletes, exports | server, on the request | `analysis.*`, `iq.store`, `report.export`, `client.report_export` |
| Corrections, retraining, model approval / rollback | server | `correction.*`, `retrain.*`, `model.*` |
| Accounts, sign-in, demo person switch | server | `user.*` |
| **Signal Pipeline run started** | server, on `POST /api/pipeline` (who, file, settings) | `pipeline.start` |
| **Pipeline / live result** | server, the first time it sees the finished run: source, every transmitter's answer, heard text | `pipeline.result`, `live.result` (actor "live bridge") |
| **Comms Intel recording** | server, the first time it lists it | `intel.recording` |
| **Transmitter location** | the analyst, from "Locate on KiwiSDR" → Record location | `client.tdoa_location` |

The append-only triggers and the hash chain (`/api/audit/verify`) cover every row above. Pipeline and live results
are logged once **per database** (marker `.audited-<db>` in the run folder, `.audited-<db>.json` in the Comms Intel
folder), so the demo database and any test database each get every result.

## 4. Outputs

| Output | Shows | Reads from |
|---|---|---|
| RF Replay / Signal Analysis | per-window answers, attention, waterfall | browser model, `nexa.db` |
| **Signal Pipeline** (+ "Follow live snapshots") | waterfall with boxes, every transmitter, audio, Morse / transcript, AI reading, **Locate on KiwiSDR** for HF signals with a known frequency | `results/pipeline/jobs/` |
| **Comms Intel** | decoded recordings: in short, when it was sending, AI reading with proof, facts measured, every message | `results/comms_intel/` |
| History / Report builder | stored browser analyses; exports JSON / CSV / XLSX / SigMF / IQ | `nexa.db` |
| Audit trail | everything in section 3 | `nexa.db` |
| Performance | current model per family / class, older ensemble folded | `web/data/current_model.json` (`scripts/export_current_model.py`) |
| Model | retrain trigger, candidates, approvals, rollback | `nexa.db`, `results/` |
| SDRconnect (other laptop) | the radio itself; tuned by `--autotune` | WebSocket API |

## 5. Known boundaries (true today)

* **History and exports cover the browser half only.** Pipeline / live / Comms Intel results are in the audit
  trail and on their own pages, but not in History or the report builder.
* **Location is not computed by NEXA.** It links to KiwiSDR's TDoA service and records the analyst's reading of
  its map. HF only (≤ 30 MHz), and an area, not a point.
* **Slow path:** AM vs a bare carrier and USB vs LSB direction are its weak spots; FT8 has no kind yet.
* **Live bridge voice mode skips the 60-class model** (speed); `--mode full` runs it at ~50 s CPU per second of IQ.
* **The browser half uses a different model file (ONNX)** from the Python half until the re-export lands.
