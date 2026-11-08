# Live demo: SDRconnect → NEXA (one or two laptops)

Everything stays on the local network: SDRconnect's WebSocket API (port 5454),
the bridge, Whisper and Ollama all run on the laptops. Use a phone hotspot
**with mobile data off**, or a direct cable. Do not use the venue Wi-Fi.

```
Laptop 1 (audience screen)                 Laptop 2 (analysis)
SDRconnect 1.0.6+                          scripts/live_sdrconnect.py --host <laptop 1>
  RSPduo live, or an IQ file playing   →     every few s: snapshot → find channels →
  waterfall + loudspeaker                    AM/NBFM/SSB/CW → audio → Whisper (→ Ollama)
  WebSocket API on :5454                   scripts/serve_local.py --lan --allow <laptop 1>  (:8099)
```

All commands: Windows PowerShell, in the repository folder.

## 1. One-time setup (every laptop that runs NEXA)

```powershell
git clone <repo> sedicAI_NEXA ; cd sedicAI_NEXA
git checkout eavan-accuracy-plan
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import whisper; whisper.load_model('medium')"   # one-time ~1.5 GB download, needs internet
```

Also needed:

- **ffmpeg on PATH**, to decode the Common Voice MP3s. Check with `ffmpeg -version`.
- **The model files**, which git does not carry. Copy them from Eavan's PC or the Drive:
  `results/classifier/physics_best.pt` and `results/detector/emitter_detector.pt`.
  Only `--mode full` and file analysis need them; the live voice mode does not.
- **Ollama**, optional, for the AI reading: install it, then run `ollama pull llama3.2:3b`.
- **The Chinese text-to-speech voice** (Windows "Microsoft Huihui"), only for regenerating the
  dataset. Get it from Settings → Time & language → Speech → add Chinese (Simplified).

## 2. Generate the data (only on the laptop that has the three Common Voice folders)

The three folders extracted from the Common Voice downloads go into one folder
(Downloads by default):

- `...cv26-malaysian-english`
- `...cv-corpus-27.0-...-ms`
- `...sps-corpus-5.0-...-ms-MY`

```powershell
# 57 transmissions (english 22 / malay 27 / chinese 8): IQ + audio + manifest.csv, ~420 MB
python scripts/make_military_voice_dataset.py --out "$env:USERPROFILE\Downloads\military_radio_IQ"
#   corpora somewhere else:   --corpora D:\datasets\commonvoice

# one 8 s capture with 3 stations at once (Alpha NBFM, Bravo NBFM, Malay AM air band)
python scripts/make_military_radio.py --seconds 8
#   -> data/local/military_radio/military_net.wav (3.2 MS/s, ~100 MB)
```

To share the data with a teammate, copy the whole `military_radio_IQ` folder (USB stick or Drive).
Nobody else needs to regenerate it.

## 3. Analyse one file (no SDRconnect)

```powershell
python scripts/run_pipeline.py --iq "$env:USERPROFILE\Downloads\military_radio_IQ\malay\iq\ms_cv_009_NBFM.wav" `
    --max-seconds 10 --hybrid --classifier results/classifier/physics_best.pt --listen `
    --whisper-model medium --language ms
```

Or use the dashboard: `python scripts/serve_local.py`, then Signal Pipeline → Analyse file.

## 4. Live on ONE laptop (rehearsal, or a solo demo)

```powershell
# terminal 1: dashboard (opens the browser)
python scripts/serve_local.py
# terminal 2a: with SDRconnect running on this laptop (radio started or file playing)
python scripts/live_sdrconnect.py --whisper-model medium
# terminal 2b: or no SDRconnect at all, replaying a file as if live
python scripts/live_sdrconnect.py --replay data/local/military_radio/military_net.wav --whisper-model medium
```

Then go to Signal Pipeline and tick **Follow live snapshots**.

## 5. Live on TWO laptops (you + teammate)

**Laptop 1 (SDRconnect, the teammate's or yours):**

1. Connect to the same hotspot as laptop 2.
2. Start SDRconnect 1.0.6+ and start the RSPduo, or play an IQ file.
3. If Windows Firewall asks about SDRconnect, allow **Private networks**.
4. Find its address: `ipconfig` → "IPv4 Address", e.g. `192.168.137.5`.

**Laptop 2 (NEXA):**

```powershell
.\.venv\Scripts\Activate.ps1
# terminal 1: dashboard, open to laptop 1 only, no sign-in
python scripts/serve_local.py --lan --allow 192.168.137.5
# terminal 2: the bridge, pointed at laptop 1
python scripts/live_sdrconnect.py --host 192.168.137.5 --whisper-model medium
#   only one language on air:  --language ms   (or en / zh)
#   AI reading too:            --llm
#   faster, less accurate:     --whisper-model small   or   --snapshot-s 2
```

- **No sign-in:** `--allow` lists the only other machines that may open the dashboard (several are
  fine: `--allow 192.168.137.5 192.168.137.9`). Everyone else on the network is refused.
  Plain `--lan` (no `--allow`) still asks for sign-in, because then anyone on the Wi-Fi could connect.
  `--lan --no-sign-in` removes it for everyone: only on your own hotspot.
- **Firewall:** if Windows asks about Python, allow **Private networks**, so laptop 1 can reach port 8099.
- **On laptop 1:** open `http://<laptop 2 IPv4>:8099` in a browser (no sign-in), go to Signal Pipeline
  and tick **Follow live snapshots**. Laptop 1 can then switch between SDRconnect and the results
  without a split screen.
- **A teammate running NEXA on their own laptop** (`python scripts/serve_local.py`) gets no sign-in
  page either, *if* their copy is up to date: `git checkout eavan-accuracy-plan` and `git pull`.
  A login page there means an old branch or download (from before 2 Oct).

## What to expect (measured 2026-11-08, Eavan's laptop: 7.3 GB RAM, CPU only)

- **Voice mode, 4 s snapshot of a 3.2 MS/s stream with 3 stations:**
  - Whisper `small`: 24–41 s per snapshot, so results trail the radio by about half a minute.
  - Whisper `medium`: about 3× slower to transcribe (13–15 s per 6 s clip, against 3–5 s for `small`),
    so expect about a minute behind.
- **Medium vs small on the radio sound:**
  - Malay: both near perfect.
  - Chinese: `medium` is clearly better (simplified characters, numbers right).
  - English: both right.
- **RAM:** `medium` uses about 3 GB. Close Chrome and other heavy apps first.
- **Both English nets** were found, named NBFM and transcribed nearly word for word.
- **Malay:** give `--language ms` when only Malay is on air. If Whisper has to guess, it often guesses wrong.
- **Full mode** (`--mode full`: YOLO + 60-class model) costs about 50 s of CPU per second of IQ. Use
  `--snapshot-s 0.5` and treat it as "every minute or so", not live.
- A snapshot cuts a transmission at its edges, so a sentence can be split across two snapshots.

## Not yet checked with the real radio

- Whether SDRconnect's IQ stream is the whole device band or a narrower VFO channel. The bridge
  measures the rate it receives and prints it. If the stream is narrower than the device band, the
  centre frequency shown is assumed to be the VFO.
- Whether the IQ stream also flows while SDRconnect plays a recorded file.
- Whether SDRconnect plays the 250 kS/s dataset files. If it refuses them, re-export at 2 MS/s.
