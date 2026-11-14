# NEXA 5-minute demo script (click by click)

One laptop runs everything: SDRconnect, the NEXA bridge, Whisper, Ollama and the dashboard.
The script has two halves:

- **Part A (about 2.5 min): signals NEXA listens to and decodes.** Morse file, Malay military voice file, KiwiSDR.
- **Part B (about 2.5 min): signals NEXA names but does not decode.** Radar, pulse, hopping, jammers, overlapping. Then correction, fine-tuning, audit trail, performance and reports.

Each step lists what to click, what to scroll to, what the audience sees, and one line to say.

---

## Before the audience arrives (10 min, not part of the 5)

1. Start Ollama (tray icon) and check: `ollama list` shows `llama3.2:3b`.
2. Shell 1, the dashboard: `python scripts/serve_local.py` (the browser opens by itself).
3. In SDRconnect, **load** the Morse file `military_radio_IQ\morse\iq\cw_sim_001_CW.wav`. Don't press play yet.
4. Shell 2, the bridge, started before play:
   `python scripts/live_sdrconnect.py --fs 250000 --whisper-model medium --llm --autotune`
5. Do one dry run of the Morse and Malay files, so Whisper and Ollama are already loaded (the first run is slow).
6. In the dashboard header, set **Acting as → Operator**.
7. Close Chrome tabs and other heavy programs.

Fallback if SDRconnect misbehaves: on **SDR Receiver**, press **Replay a file** and pick the same IQ `.wav`
(audio-only WAVs are refused: only IQ recordings have a waterfall). The waterfall, transmitters and
**Open analysis** appear in the spectrum spot of the page, without SDRconnect. Synthesize Scenario switches back to the simulator.

---

## Part A: signals NEXA decodes

### A1. Morse code from SDRconnect (about 50 s)

| Do | Audience sees |
|---|---|
| In SDRconnect, press **play** on the Morse file | SDRconnect's waterfall. The red tuning line jumps onto the signal by itself (auto-tune) |
| Switch to the dashboard. Menu: **SDR Receiver** | The **Latest intercept** card appears when the file ends: type *Morse (CW)*, frequency, the decoded text |
| Menu: **History → Captures**, scroll to **Decoded recordings** → click the newest row (or, instead of SDRconnect, press **Replay a file** on SDR Receiver and pick the same IQ `.wav`: the result appears right there in the waterfall spot) | **The waterfall with every transmitter boxed and named, with its frequency**, and the table *Every transmitter* (answer, confidence, why) |
| Under the waterfall, **Decoded signals** → **Open analysis** | The **Analysis** page |
| On the Analysis page, scroll from top to bottom | **In short** → **1. Listen** (click a dark bar: the beeps play from there) → **2. Original decoded text** + *In plain words* → **4. What it means** → **5. Facts** (speed in words per minute, call signs, short codes) |

> Say: "SDRconnect plays the radio. NEXA finds the Morse tone, decodes the letters with fixed rules, no AI,
> then the local AI explains it. Every fact on the right is measured, not guessed."

Press **← Back to SDR Receiver**.

### A2. Malay military voice from SDRconnect (about 60 s)

| Do | Audience sees |
|---|---|
| Shell 2: **Ctrl+C**. In SDRconnect, load `military_radio_IQ\malay\iq\ms_cv_009_NBFM.wav`. Start the bridge again (same command, add `--language ms`), then press **play** | SDRconnect waterfall, the walkie-talkie channel |
| Dashboard, **SDR Receiver** → **Latest intercept** → **Open analysis** | The **Analysis** page for *Narrow FM (NBFM, walkie-talkie)* |
| Press play in **1. Listen to the recording** | The Malay voice, as demodulated by NEXA |
| Scroll to **2. Original decoded text** | Whisper's Malay transcript |
| **3. In English** → **Translate to English** (up to a minute: talk over it) | The English translation, made on this laptop |
| Scroll to **4. What it means** | Summary, places, key points, and **"Why: the words it is based on"**, quoting the transcript |

> Say: "Nothing leaves this laptop. Speech to text and the AI both run offline. The AI must quote the
> exact words it relied on, so an analyst can check it."

For the NEXA waterfall with frequencies of this file: **History → Captures** → **Decoded recordings** → newest row.

### A3. KiwiSDR, a real short-wave receiver online (about 40 s)

| Do | Audience sees |
|---|---|
| Menu: **KiwiSDR**. The receiver is already `dx6ran.ddns.net:8073` and clips are 5 s. Type a frequency (e.g. `15295`), press **Start listening** | A new clip every 5 s: kind, frequency in kHz, audio player, then the text and AI reading |
| On a clip with text, press **Open analysis** | The same **Analysis** page: playback, original text, English, what it means, facts |

> Say: "Same pipeline, now on a real receiver in another country. Live, it is whatever is on air,
> so the text depends on the moment."

Press **Stop** on the KiwiSDR page before moving on (it frees the CPU).

---

## Part B: signals NEXA names but does not decode

### B1. Overlapping radar, hopping and jamming (about 50 s)

| Do | Audience sees |
|---|---|
| Menu: **SDR Receiver**. With SDRconnect not streaming, the page has **already** generated a random mix by itself (the note under the buttons says so) | The waterfall with overlapping signals from all current groups (radars, FHSS/DSSS/LoRa, FSK/PSK/QAM, jammers, IFF/PPM pulses), the dashed true boxes, and the verdict panel (**Hostile** wins) |
| For a new mix: open **More options: check a file, test scenarios, display settings** → **Scenario case** `Random mix: all signal groups…` → **Synthesize Scenario**. Each press gives a different combination | A new waterfall. The case line lists the types in it |
| Scroll to **Recommended corrections** and **Detection events** | Where the model is unsure, NEXA itself suggests what a person should check |
| For a demo of alerts and review: Scenario case `Demo 4: threat + unsure (alert and review)` → **Synthesize Scenario** | It is stored as an alert and a review item. The **Review** badge in the menu goes up |

> Say: "Radar, frequency hopping and jammers can't be 'read', but they can be recognised. NEXA names 60
> signal types and shows on the waterfall where each one is, even when they overlap."

### B2. Correct the model, approve, fine-tune (about 50 s)

| Do | Audience sees |
|---|---|
| In **Detection events**, on a wrong row press **Correct**, choose the right class, press **Submit correction** | The correction is saved, waiting for an analyst |
| Header **Acting as → Analyst**. Menu: **Review → Corrections** → **Approve** | The four-eyes rule: an operator corrects, an analyst approves |
| Menu: **Review → Alerts** and **Review → Low Confidence** | The Demo 4 capture waiting in both lists |
| Menu: **System → Model** → type a reason → **Start retraining** (skip it if time is short and show **Fine-tuning history** instead) | Progress bar. The new version is offered **only if it passes the fixed test**, and every version is kept |

> Say: "A person stays in charge. Corrections need approval, retraining is started by a person, and a
> model that gets worse is refused."

### B3. Records: history, audit, performance, reports (about 50 s)

| Do | Audience sees |
|---|---|
| Menu: **History → Captures** | Two sections: **Signal captures** (radar, hopping, jammers, comms, with verdict and classes; empty dwells hidden unless **Show empty dwells** is ticked) and **Decoded recordings** (each replayed or live recording: waterfall, transmitters, Open analysis) |
| Menu: **History → Audit Trail** → **Verify integrity** | Every action by every person, chained so that no entry can be changed without it showing |
| Menu: **System → Performance**, scroll to the bottom | Measured accuracy by family, then the **Model card** (what the model is, its data, its limits) |
| Menu: **Reports** → leave the sections ticked, **PDF** → **Build report** | A formal PDF report of the captures, corrections and audit trail |
| Menu: **System → Assistant** → click **What is waiting for review?** or **How accurate is the model?** | A plain-English answer from the local AI, using only NEXA's own records |

> Close: "From the antenna to a report, it is all on one laptop, with a person in the loop and every
> step on record."

---

## If something goes wrong

| Problem | Do this |
|---|---|
| Morse shows "no text" | The bridge was started without `--fs 250000`, or after play. Ctrl+C, restart the bridge, press play again |
| No Latest intercept after a file | **SDR Receiver** → **Replay a file**, pick the same `.wav` |
| Translate or AI reading is slow or missing | Ollama is still loading (first use). Wait 30 s and press again, or say "the local AI is warming up" |
| KiwiSDR receiver is busy or offline | Pick another frequency or receiver, or skip A3 |
| Laptop very slow | Use `--whisper-model small` for the bridge, and close the browser tabs you don't need |
