# Live demo: SDRconnect → NEXA, two laptops

Everything stays on the local network: SDRconnect's WebSocket API (port 5454),
the bridge, Whisper and Ollama all run on the two laptops. Use a phone hotspot
**with mobile data off**, or a direct cable. Do not use the venue Wi-Fi.

```
Laptop 1 (audience screen)                 Laptop 2 (analysis, or a 3rd screen)
SDRconnect 1.0.6+                          scripts/live_sdrconnect.py  --host <laptop 1>
  RSPduo live, or an IQ file playing   →     every few s: snapshot → find channels →
  waterfall + loudspeaker                    AM/NBFM/SSB/CW → audio → Whisper (→ Ollama)
  WebSocket API on :5454                   scripts/serve_local.py --lan
                                             dashboard → Signal Pipeline → Follow live
```

The dashboard can also be opened from laptop 1's browser
(`http://<laptop 2>:8099`, sign-in required with `--lan`). Laptop 1 can then
switch between SDRconnect and the results, with no split screen needed.

## Before the day

1. Both laptops: the same SDRconnect version, 1.0.6 or newer (WebSocket API 1.0.3).
2. Laptop 2: `pip install -r requirements.txt` (includes `websockets` and `openai-whisper`).
   Run Whisper once while online so the model is cached: `python -c "import whisper; whisper.load_model('small')"`.
3. Check that SDRconnect plays the demo files (`Downloads\military_radio_IQ\<language>\iq\*.wav`, 250 kS/s int16).
   If it refuses the rate, re-export them at 2 MS/s.
4. Rehearse with no SDRconnect at all:
   `python scripts/live_sdrconnect.py --replay data/local/military_radio/military_net.wav`

## On the day

```
# laptop 2, terminal 1: the dashboard, reachable from laptop 1
python scripts/serve_local.py --lan

# laptop 2, terminal 2: the bridge (laptop 1's address from `ipconfig`)
python scripts/live_sdrconnect.py --host 192.168.x.x --whisper-model small
#   one language only on air:  --language ms   (or en / zh)
#   AI reading as well:        --llm            (slower)
```

Then open Signal Pipeline on the dashboard and tick **Follow live snapshots**.

## What to expect (measured 2026-11-08, this laptop, CPU only)

- **Voice mode** (the default) analyses a 4 s snapshot of a 3.2 MS/s stream with 3 stations in
  **24–41 s**, most of it Whisper. Results trail the radio by about half a minute. A narrower SDRconnect
  sample rate, `--whisper-model base` or `--snapshot-s 2` makes it faster.
- **Both English nets** were found, named NBFM and transcribed nearly word for word
  ("Bravo 2, this is Alpha 1. Radio check.").
- **Malay** needs `--language ms` and the `small` model. If Whisper has to guess the language, it guesses wrong.
- **Full mode** (`--mode full`: YOLO + 60-class model) costs about 50 s of CPU per second of IQ. Use
  `--snapshot-s 0.5` and treat it as "every minute or so", not live.
- A snapshot cuts a transmission at its edges, so a sentence can be split across two snapshots.

## Not yet checked with the real radio

- Whether SDRconnect's IQ stream is the whole device band or a narrower VFO channel. The bridge
  measures the rate it receives and prints it. If the stream is narrower than the device band, the
  centre frequency shown is assumed to be the VFO.
- Whether the IQ stream also flows while SDRconnect plays a recorded file.
