# NEXA system architecture: input → processing → audit → output

State on `main`, 10 Nov 2026. Everything runs on the local machine; the only outside links are the ones a
person clicks (KiwiSDR), the public KiwiSDRs the Live KiwiSDR panel listens to or records for a location, and the
optional SDRconnect laptop on the same private network.

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
| I | **Transmitter location (TDoA)** | Live KiwiSDR panel → "Locate transmitter" (or automatically, once per frequency, when listening starts and none is on record) → `POST /api/kiwi/locate` | `scripts/sdr_hub.py` `TdoaRunner` in the background: checks each candidate KiwiSDR's `/status` (GPS fix, position, free channel, TDoA allowed) → kiwirecorder records the 3 first usable ones at once, GPS-stamped IQ (`-m iq --kiwi-wav`, 10/15/30 s) → `src/kiwi_tdoa.py`: arrival-time difference per pair (1 s blocks, cross-correlation peak, outliers dropped) → lat/lon grid search → most likely point + 95 % area | `nexa.db` audit `client.tdoa_location` (recordings deleted) |
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
| **Transmitter location run started** | server, on `POST /api/kiwi/locate` (who, frequency, receivers, sample length) | `kiwi.locate` |
| **Transmitter location** | server, when a TDoA run finishes (source "NEXA TDoA (computed)": point, 95 % area radius and box, receivers, arrival-time differences); or the analyst, by hand with "Record location" (source "KiwiSDR TDoA") | `client.tdoa_location` |

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
| **Live KiwiSDR panel** | the listening stream's decodes; "Transmitter location": newest `client.tdoa_location` within 2 kHz of the frequency (most likely lat/lon, area radius, receivers, map link), run progress | `/api/kiwi/status`, `/api/kiwi/location` (`nexa.db`) |
| Performance | current model per family / class, older ensemble folded | `web/data/current_model.json` (`scripts/export_current_model.py`) |
| Model | retrain trigger, candidates, approvals, rollback | `nexa.db`, `results/` |
| SDRconnect (other laptop) | the radio itself; tuned by `--autotune` | WebSocket API |

## 5. Known boundaries (true today)

* **History and exports cover the browser half only.** Pipeline / live / Comms Intel results are in the audit
  trail and on their own pages, but not in History or the report builder.
* **Transmitter location is an area, not a point.** NEXA computes it itself (`src/kiwi_tdoa.py`, the method of
  the open-source hcab14/TDoA code KiwiSDR's own service runs). It does **not** call KiwiSDR's TDoA service
  (tdoa.kiwisdr.com): that needs an access string each KiwiSDR gives only to its own web page, and it was not
  reachable from either network. Limits:
  * Distances are great-circle ground distances at the speed of light; short-wave arrives via the ionosphere on
    longer paths, so every pair carries an extra 300 µs (~90 km) of uncertainty and the area is typically tens
    to a few hundred km across. The "most likely" point is the centre of that area.
  * Needs 3 GPS-timed KiwiSDRs that all hear the signal, round the transmitter (wide geometry). With 3 receivers
    the curves can cross twice; the grid is limited to ±20° round the receivers. HF only (≤ 30 MHz).
  * Each candidate is checked before recording: GPS fix, position, free channel, TDoA allowed, and that its audio
    stream really delivers from this network (about a second of one channel). A `/status` page can answer while
    the stream cannot: some proxied KiwiSDRs redirect it to `proxy2.kiwisdr.com`, or accept it and send nothing.
  * Default candidates, the first 5 that pass are recorded at once (3 are needed; 4 can show a timing error, 5 can
    name the receiver that has it): JKT Jakarta, HS0ZQS Bangkok, jplin53c Taiwan, HS0ZGZ Khon Kaen, ptlen Bandung,
    LA6LU Hua Hin, 9M2AMH Kuala Lumpur. They suit Malaysia; other regions need receivers typed into the panel.
  * When fewer than 5 of them work, NEXA looks further: the public KiwiSDR list (`rx.linkfanel.net/kiwisdr_com.js`,
    built from kiwisdr.com/public, which is often unreachable), GPS-timed + TDoA allowed + a free channel, within
    3500 km of Peninsular Malaysia, nearest first in each of 8 compass sectors, one per site (150 km), at most 12
    checked. Beyond ~3500 km multi-hop paths add milliseconds (south-west Australia was 3 ms late).
  * `TDOA_AVOID` (scripts/sdr_hub.py) keeps receivers with a known timing problem out unless typed in:
    DU6/PE1NSQ (dx6ran.ddns.net) was 0.6–0.8 ms early against every other receiver in 3 runs on 13–14 Nov.
  * Pairs that contradict the others (t_i−t_j ≠ (t_i−t_k)+(t_k−t_j)) are dropped, worst triangle first, weakest
    correlation peak first: these are wrong correlation peaks.
  * A recording counts if its file reads as GPS-stamped IQ: kiwirecorder's `--kiwi-tdoa` status code can say
    "busy" after a full recording.
  * Public receivers are shared: one job at a time, at most 5 receivers, 30 s samples by default (10/15/30),
    1–3 runs per job, 60 s pause between jobs, automatic runs once per frequency per page.
  * Each receiver is recorded by its own kiwirecorder process, all started together (GPS time stamps line them
    up). In one process, a receiver that stops ("too busy" a few seconds in) stops all the others: on 14 Nov that
    cut every 30 s recording to 9 s.
  * How a pair's arrival-time difference is measured (`src/kiwi_tdoa.py` `pair_lag`): 1 s blocks; the highest
    cross-correlation peak placed to 1/32 sample by sinc interpolation of the complex correlation (a 3-point
    parabola on a broad AM peak is pulled toward whole samples: lags jumped ~83 µs between blocks); then the lag
    that most blocks agree on (within 100 µs; at least 40 % of blocks, else the pair is not used). On a broadcast
    the runner-up peak is often 80–96 % of the best even in good blocks, so a peak-uniqueness test would throw
    good blocks away; agreement across blocks is what separates good pairs from bad (Hua Hin's lags were random).
  * Two ionospheric paths: a far receiver can hear one and two hops (~0.9 ms apart) and switch between them during
    a recording (Taiwan on 14 Nov: 20 s on the two-hop path, then 9 s direct). When at least 15 % of the blocks
    support a second lag, the one where the receiver further from the others hears the signal earliest is kept:
    an extra hop only adds path.
  * Runs (page default 3 x 15 s): the recording is repeated and each pair's median lag is used (`merge_runs`);
    when the runs disagree on a pair by more than a hop's worth (300 µs), the run where the far receiver heard it
    earliest wins, as within one run. Kept recordings of 14 Nov (A, B: Taiwan direct; C: Taiwan two hops):

    | Combined | From Kajang |
    |---|---|
    | C alone | 440 km |
    | A + C | 24 km |
    | B + C | 31 km |
    | A + B + C | 11 km |

  * Tried and not used: calibrating each receiver's delay on the known transmitter (Kajang) and applying it to
    later runs. Calibrated on A or B, run C stayed ~430 km off; calibrated on C, A and B went from 13 to ~290 km.
    The delays are not steady: Taiwan changes hops within minutes.
  * When a far receiver (> 2500 km) heard only one path, NEXA cannot tell whether it was the direct one or a
    two-hop one, so it also works out where the transmitter would be if that receiver had heard it one hop
    earlier (sky-wave path model, layer at 300 km) and shows it as "Other possible place" (`_other_hop`).
    Set C (14 Nov, 17:47, the same state as an analyst run at 17:45): Taiwan two-hop only, most likely place
    1.63° N 98.05° E (440 km from Kajang), other possible place 2.61° N 101.03° E (92 km from Kajang).
    Sets A and B saw both paths, kept the direct one, and show no other place.
  * The page says how sure an answer is: Good (4+ receivers agree, one place), Rough (they disagree, or a far
    receiver leaves two places), Not cross-checked (3 receivers). Receivers are named with the place their own
    `/status` page gives; the per-pair numbers sit under "Technical details".
  * **Accuracy, 14 Nov 2026, 15295 kHz** (Radio Malaysia, Kajang 2.99° N 101.77° E; KiwiSDR's own TDoA said
    2.50° N 102.50° E, ~95 km off). Before the fixes above (15 s, one kiwirecorder for all, parabola peaks):

    | Receivers | Answer | From Kajang |
    |---|---|---|
    | DU6, JKT, HS0ZQS | 3.32° N 104.38° E | 292 km |
    | DU6, JKT, HS0ZQS, Taiwan | 3.27° N 104.36° E | 289 km |
    | DU6, JKT, Bandung, Taiwan | 2.44° N 105.98° E | 471 km |
    | JKT, HS0ZQS, Taiwan, Bandung, Hua Hin | 1.16° N 96.14° E | 658 km |

    After (same day, the same 5 receivers JKT, HS0ZQS, Taiwan, Bandung, Hua Hin):

    | Recording | Answer | From Kajang | From KiwiSDR's answer |
    |---|---|---|---|
    | set A (9 s: cut short before the one-process-per-receiver fix) | 3.06° N 101.68° E | 13 km | 110 km |
    | set B (30 s, 2 min later) | 3.09° N 101.71° E | 13 km | 110 km |
    | A and B merged (median per pair) | 3.08° N 101.71° E | 12 km | 105 km |
    | live, "Locate transmitter" path, 30 s, 1 run | 3.05° N 102.65° E | 98 km | 63 km |

    Between runs some pairs still move by up to ~180 µs as paths change, so one run is good to roughly 100 km;
    use Runs 3 when it matters. The 95 % area (~120–150 km radius) contained Kajang in every run after the fixes.
    Still a lead to confirm, not a fix: one frequency, one day, receivers that change with the network.
  * The synthetic tests (`tests/test_kiwi_tdoa.py`) check the method end to end.
* **Slow path:** AM vs a bare carrier and USB vs LSB direction are its weak spots; FT8 has no kind yet.
* **Live bridge voice mode skips the 60-class model** (speed); `--mode full` runs it at ~50 s CPU per second of IQ.
* **The browser half uses a different model file (ONNX)** from the Python half until the re-export lands.
