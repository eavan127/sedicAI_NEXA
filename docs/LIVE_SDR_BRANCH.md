# Branch `feat/live-sdr-console`: who edits what

Two people work on this branch at the same time. To avoid merge conflicts,
each half owns its own files, and the few shared files get only small hooks
marked with `feat/live-sdr-console (SDR connection)`.

## The flow we are building

```
Antenna -> RSPduo -> SDRconnect (port 5454)
   -> NEXA server (scripts/sdr_hub.py): ONE connection, shared by every page
        |-- live dwells (3.2 MS/s) -> page: waterfall + YOLO/classifier, every dwell
        |-- background decoder: record each transmission until the air is quiet
              -> analog kind (NBFM / AM / SSB / CW / RTTY ...) -> demodulate
              -> Morse decoder or Whisper (+ optional Ollama reading)
              -> saved as a live pipeline run + Comms Intel recording + audit entry
   -> History (teammate): Alerts / Decoded review / Corrections review / Low-confidence review
```

## Half 1: SDR connection (done on this branch)

| File | What |
|---|---|
| `scripts/sdr_hub.py` | **new**. The SDRconnect connection, live dwells, and the background record-and-decode worker |
| `scripts/fake_sdrconnect.py` | **new**. A fake SDRconnect (synthetic NBFM / Morse / AM bursts), for working without the radio |
| `web/sdr_receiver.js` | **new**. `SdrConnectReceiver`: the page-side source, with the same interface as `SimulatedReceiver` / `FileReplay` |
| `tests/test_sdr_hub.py` | **new**. Hub and `/api/sdr/*` tests, run against a fake SDRconnect |
| `scripts/serve_local.py` | hook: one `head == "sdr"` line in `_route`, plus the `_hub()` / `_sdr()` methods above `_auth` |
| `web/main.js` | hooks: an import line, try/catch around `source.connect` in `rxStart`, the `"sdr"` branch in the Connect click handler, the `rxNote` text, the sample-rate readout, and the `rxSource` change listener |
| `web/index.template.html` | hooks: `<option value="sdr">` and the `#rxSdrFields` block (host, port, decode checkbox) |

## Half 2: History and page merge (teammate)

- Merge Signal Pipeline and RF Replay into one page.
- Remove the model selector (always YOLO + classifier, the default model).
- Build History with four sections: **Alerts**, **Decoded review** (opens Comms
  Intel as an expanded page), **Corrections review** and **Low-confidence review**.
  Alerts, corrections and low-confidence items open in the waterfall to approve or reject.
- Files: `web/history.js`, `web/intel.js`, `web/pipeline.js`, and the page
  layout in `web/index.template.html` (any part except the receiver panel
  hooks above). `web/main.js` is fine too: leave the marked lines as they are,
  or move them as a block.

**When you remove `modelSel`:** `rxStart` in `main.js` still logs
`model: modelSel.value` in `receiver_connect`. Replace it with a fixed name,
because that line belongs to the shared receiver code, not to the SDR hook.

## The data contract the History half reads

The SDR half writes nothing new to the database. Everything goes through
stores that already exist, so History can use the same readers for live and
uploaded runs.

1. **Every dwell** from SDRconnect goes through `analyze()` and `store()`, the
   same as for the simulated receiver. So `/api/analyses` holds it, with
   `triage.alertLevel` and `triage.needsReview` set as usual. The `caseNote`
   starts with `LIVE · SDRconnect`.
   - This feeds the **Alerts** and **Low-confidence review** sections.
2. **Every decoded transmission** is a run folder in
   `results/pipeline/jobs/<id>/` with `job.json` (`"live": true`,
   `"who": "SDRconnect receiver"`), `tracks.json`, `tracks.png` and
   `track_N.wav`. Read it through `GET /api/pipeline/...`, the same as Signal
   Pipeline runs.
   - Per track: `listen.kind`, `listen.transcript` or `listen.morse`,
     `listen.speech_segments`, `confidence` and `band`.
3. **Every decoded transmission that has text** is also a Comms Intel
   recording: `GET /api/intel/...`, `index.json`, ids in the hub's
   `intel_ids`.
   - This feeds the **Decoded review** section.
4. **Audit:** `sdr.connect` and `sdr.disconnect` (who, which SDRconnect),
   plus `live.result` and `intel.recording` for each decode, are written as
   soon as it finishes.
5. **Live status for the page:** `GET /api/sdr/status` returns `decodes`, the
   last 20 entries, each one:
   `{id, at, seconds, why, centre_hz, intel_ids, warning, tracks: [{kind, answer, f_lo_khz, f_hi_khz, text}]}`.
   - History can poll this to show "new decode" badges without re-reading the folders.

## API (`/api/sdr/*`)

| Call | Does |
|---|---|
| `POST /api/sdr/connect` `{host, port, device, decode, whisper_model, language, llm}` | Starts or joins the stream. Returns the status |
| `POST /api/sdr/disconnect` | Stops it, for every page |
| `GET /api/sdr/status` | `state` (off / connecting / streaming / error), `fs`, `centre_hz`, `on_air`, `recording_s`, `analysing`, `decodes` |
| `GET /api/sdr/dwell?s=0.05&after=<total>` | float32 I/Q interleaved at 3.2 MS/s. Returns 204 when nothing new has arrived since `after` (the `X-NEXA-SDR-Total` value of the previous dwell) |

## Running it

```
python scripts/fake_sdrconnect.py          # no radio: synthetic bursts
python scripts/serve_local.py              # RF Replay -> Receiver source "SDRconnect (live antenna)" -> Connect
```

With the real radio, start SDRconnect, turn on Settings -> WebSocket
interface, press play, and then Connect. If SDRconnect runs on another
laptop, type that laptop's address in the "SDRconnect at" box.

Known limits:
- The page classifies dwells (slices). The air between dwells is not examined,
  and the page reports that as *coverage*. The background decoder, however,
  records **every** sample of each transmission.
- If SDRconnect streams a narrow channel (below 3.2 MS/s), the dwell is
  upsampled. The band outside the stream is filler noise, which the connect
  log states.
- If a station is always on the air (an FM broadcast inside the band), a
  recording ends only at the 60 s limit, so it is decoded once a minute.
