# Moving NEXA to the advisor's laptop (end-to-end)

Written 2026-11-14 from Eavan's working laptop (`D:\github clone\sedicAI_NEXA`, branch `main`,
clean, same as `origin/main` at `23a3987 "voice replay"`).

The advisor's laptop has more RAM but **nothing installed for coding**: no editor, no Git, no Python.

**Everything runs on this ONE laptop.** That means the RSPduo radio (USB), SDRconnect, the NEXA bridge,
Whisper, Ollama and the dashboard are all on the advisor's laptop. There is no second laptop and no
network between laptops. The bridge talks to SDRconnect on `localhost:5454`.
Eavan signs in to Eavan's Claude account in the Claude desktop app on that laptop. Then the prompt in
**Part C** is pasted into the Code tab, and Claude does the rest.

There are three parts:

- **Part A** (on Eavan's laptop, before the move): make a USB "transfer pack" with the files git does not carry.
- **Part B** (on the advisor's laptop, by hand, about 10 minutes): the few things Claude cannot do for itself.
- **Part C** (on the advisor's laptop): the prompt to paste, then the detailed plan Claude follows.

---

## Part A: transfer pack (on Eavan's laptop)

Git carries the code only. These files are git-ignored and must go on a USB stick (or Drive) in
one folder, `NEXA_transfer\`, **using the same folder layout as in the repo**:

| What | From (Eavan's laptop) | Size | Needed for |
|---|---|---|---|
| Classifier (current) | `results\classifier\physics_aug_best.pt` | 1.6 MB | full mode, file analysis, eval |
| Detector (full YOLO) | `results\detector\emitter_detector.pt` | 2.9 MB | full mode, dashboard pipeline |
| Old 5-class model + ensemble | `results\best_model.pt`, `results\ensemble_0..4.pt`, `results\onnx\*.onnx` | 4.5 MB | Model page / retraining |
| Local database | `data\local\nexa.db` | 0.8 MB | history, corrections, audit trail (optional; a fresh one is created otherwise) |
| Demo radio captures | `data\local\military_radio\`, `data\local\military_voice_v1\` | 0.5 GB | live replay demo |
| Voice + Morse dataset | `C:\Users\User\Downloads\military_radio_IQ\` | 0.9 GB | SDRconnect file playback, eval |
| Whisper models (saves 2 GB of downloading) | `C:\Users\User\.cache\whisper\{base,small,medium}.pt` | 2.1 GB | voice transcription |
| Claude memory (project notes) | `C:\Users\User\.claude\projects\D--github-clone-sedicAI-NEXA\memory\` | small | gives Claude on the new laptop the project history |

Do **not** copy: `.venv\`, `node_modules\`, `data\raw` (37 GB), `data\shards` (5.5 GB),
`data\combined`, `data\processed`. Those are training data only. Demos and analysis don't need them.

Total: about **3.5 GB**, so any 8 GB USB stick is enough.

Folder layout on the stick (Part D5 expects exactly this):

```
NEXA_transfer\
  results\            classifier\, detector\, onnx\, best_model.pt, ensemble_*.pt
  data\local\         nexa.db, military_radio\, military_voice_v1\
  whisper\            base.pt, small.pt, medium.pt
  military_radio_IQ\  (the whole dataset folder)
  claude_memory\      MEMORY.md + the *.md notes
```

Ollama models are not copied. The new laptop downloads them with `ollama pull` (needs internet).

---

## Part B: by hand on the advisor's laptop (before pasting the prompt)

1. Install the **Claude desktop app** (claude.ai/download), sign in with Eavan's account, and open the **Code** tab.
   On Windows the Code tab needs **Git for Windows**. If the app asks, install it from git-scm.com
   with the default options, then restart the app.
2. Plug in the USB stick. Note the drive letter (for example `E:\NEXA_transfer`).
3. Make sure the laptop has **internet** for the setup (about 4 to 6 GB of downloads). The demo itself works offline afterwards.
4. In the Code tab, choose a working folder: `C:\Users\<advisor>\` (Claude makes `sedicAI_NEXA` inside it).
5. Set the permission mode to allow edits and commands (or approve each step as it asks).
6. Paste the prompt from Part C, with the USB path filled in.

---

## Part C: the prompt to paste

```text
You are setting up the NEXA project (SEDIC 2026, RF signal intelligence: SDR → detector + classifier →
demodulator → Whisper → Ollama → local dashboard) on a NEW Windows laptop that has never been used for
coding. I am Eavan, the team lead, signed in with my own account. This laptop belongs to our advisor;
we use it because it has more RAM. EVERYTHING runs on this one laptop: the RSPduo radio plugged in by USB,
SDRconnect, the NEXA bridge, Whisper, Ollama and the dashboard. There is no second laptop. Do the whole setup end
to end yourself, step by step, and verify each step before moving on.

Facts you need:
- Code: https://github.com/eavan127/sedicAI_NEXA (branch main). Clone it to C:\Users\<this user>\sedicAI_NEXA.
- Transfer pack (git-ignored files: models, demo data, Whisper models, Claude memory) is at: E:\NEXA_transfer
  (change this if the USB drive letter differs).
- After cloning, the full plan is in docs/ADVISOR_LAPTOP_SETUP.md, "Part D". Read it and follow it.
  If cloning is not possible yet, follow the summary below.

Summary of what "done" means:
1. Tools installed (with winget where possible): Python 3.12, Git (if not already), ffmpeg on PATH,
   Ollama, and SDRplay's SDRconnect (1.0.6 or newer, with its RSPduo driver), because this laptop drives the radio.
2. Repo cloned, .venv made with Python 3.12, `pip install -r requirements.txt` done. If this laptop has an
   NVIDIA GPU, install the CUDA build of torch instead of the CPU one.
3. Transfer pack copied into the repo at the same paths (results\..., data\local\...), Whisper models into
   %USERPROFILE%\.cache\whisper, the dataset into %USERPROFILE%\Downloads\military_radio_IQ, and my Claude
   memory files into this laptop's Claude project memory folder for the new repo path.
4. Ollama running with `llama3.2:3b` pulled (and `llama3.1:8b` too if RAM ≥ 16 GB).
5. Checks pass: pytest, a Whisper transcription of one dataset file, an Ollama reply, the dashboard
   (scripts/serve_local.py) opening with the Signal Pipeline page, a live replay run of
   scripts/live_sdrconnect.py --replay with --whisper-model medium --llm, and then the real chain on
   this laptop: SDRconnect playing a dataset file (and the RSPduo live if it is plugged in) → the bridge
   on localhost → results on the dashboard.
6. A desktop shortcut / .bat file so the advisor or I can start the dashboard with a double-click.
7. A short report at the end: what was installed, versions, RAM/CPU/GPU, timing of the Whisper test
   (seconds per clip) compared with my old laptop (small 3–5 s, medium 13–15 s per 6 s clip, CPU only,
   7.3 GB RAM), and anything that failed.

Rules: never commit or push from this laptop unless I ask. Explain things in plain language (I am not
an electrical engineer). Do not use the venue Wi-Fi assumptions here; this is setup only.
```

---

## Part D: the detailed plan (for Claude doing the setup)

Read this whole part before starting. Everything here was measured on Eavan's laptop on 2026-11-14.

### D0. What the system is

- **Python 3.12** project, all commands run from the repo root. Main entry points:
  - `scripts/serve_local.py`: local dashboard + SQLite database (`data/local/nexa.db`), default port **8099**
    (it opens the browser by itself; `--no-browser`, `--port N`, `--lan --allow <ip>` for two laptops).
    Demo mode (no sign-in) is the default. Port 8080 was taken on Eavan's laptop, so don't use it.
  - `scripts/live_sdrconnect.py`: the bridge from SDRconnect's WebSocket API (port **5454**) to the
    pipeline. `--replay <wav>` works with no radio and no SDRconnect. Useful flags: `--whisper-model
    base|small|medium` (default small), `--language ms|en|zh`, `--llm` (Ollama reading), `--autotune`,
    `--fs 250000` for dataset files, `--mode full` (YOLO + classifier, slow).
  - `scripts/run_pipeline.py`: analyse one IQ file (`--iq`, `--classifier`, `--listen`, `--hybrid`).
  - `scripts/eval_military_radio.py`: score the 84-file voice/Morse set.
  - `web/start_demo.bat`: double-click launcher for the dashboard (uses `.venv` if present).
- **Whisper** (`openai-whisper` package, `src/intel/voice.py`): default model `medium`. It **never downloads
  by itself** at run time; models must already be in `%USERPROFILE%\.cache\whisper\`. Needs **ffmpeg on PATH**.
- **Ollama** (`src/intel/reason.py`, `web/ollama.js`): `http://localhost:11434`, default model `llama3.2:3b`.
  Optional everywhere: if it's missing, the system works without the AI reading.
- Full user guide for the live demo: `docs/LIVE_DEMO.md`. Architecture: `docs/SYSTEM_ARCHITECTURE.md`.

### D1. Look at the machine first

```powershell
systeminfo | findstr /C:"OS Name" /C:"Total Physical Memory"
Get-CimInstance Win32_Processor | Select Name, NumberOfCores
Get-CimInstance Win32_VideoController | Select Name, AdapterRAM
Get-PSDrive C | Select Used, Free
winget --version
```

Note RAM, CPU, GPU (NVIDIA?) and free disk (need about **15 GB** free). If `winget` is missing, ask the
user to update "App Installer" from the Microsoft Store.

### D2. Install tools

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git            # skip if the Claude app already installed it
winget install -e --id Gyan.FFmpeg
winget install -e --id Ollama.Ollama
```

- After installing, **open a new shell** (PATH changes only show in new shells) and check:
  `py -3.12 --version`, `git --version`, `ffmpeg -version`, `ollama --version`.
- If ffmpeg is still not found, find `ffmpeg.exe` under `%LOCALAPPDATA%\Microsoft\WinGet\Packages\` and add
  its `bin` folder to the user PATH.
- The installers may show Windows UAC prompts. Tell the user to click Yes.
- **SDRconnect is required** (this laptop drives the radio). It's not on winget. Open sdrplay.com → Downloads
  → SDRconnect for Windows (1.0.6 or newer) in the browser pane, and let the user run the installer (it
  includes the RSPduo driver). Check: SDRconnect opens, and with the RSPduo plugged in it lists the device.
  In its settings, **enable the WebSocket API (port 5454)** and allow control over it (for `--autotune`).

### D3. Get the code

```powershell
cd $env:USERPROFILE
git clone https://github.com/eavan127/sedicAI_NEXA.git
cd sedicAI_NEXA
git checkout main
git config user.name "Eavan Tan"     # only if a commit is ever asked for; ask for the email, don't guess
```

The repo is public, so cloning needs no sign-in. Pushing would need GitHub sign-in. Don't set that up unless asked.

### D4. Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned (ask the user first)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **NVIDIA GPU present?** Install the CUDA torch first, then the rest:
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`, then `pip install -r requirements.txt`.
  Check with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
  Whisper uses the GPU by itself when torch sees CUDA, which makes `medium` many times faster.
- Eavan's laptop: Python 3.12.10, torch 2.13.0+cpu.
- Node.js is **not** needed (it's only for the web parity tests).

### D5. Copy the transfer pack

From `<USB>\NEXA_transfer\` (layout mirrors the repo):

```powershell
$T = "E:\NEXA_transfer"                     # the USB path from the prompt
robocopy "$T\results" ".\results" /E
robocopy "$T\data\local" ".\data\local" /E
robocopy "$T\whisper" "$env:USERPROFILE\.cache\whisper" /E
robocopy "$T\military_radio_IQ" "$env:USERPROFILE\Downloads\military_radio_IQ" /E
```

Then check that these exist: `results\classifier\physics_aug_best.pt`, `results\detector\emitter_detector.pt`,
`%USERPROFILE%\.cache\whisper\medium.pt` (about 1.5 GB).

**Claude memory:** the memory folder is per machine and per repo path, so signing in with the same account
does not bring it along. Copy `$T\claude_memory\*` into
`%USERPROFILE%\.claude\projects\<this repo path, with ':' '\' ' ' turned into '-'>\memory\`
(for `C:\Users\Advisor\sedicAI_NEXA` that is `C--Users-Advisor-sedicAI-NEXA`). Check the real folder name
the app created under `%USERPROFILE%\.claude\projects\` after the first session in the repo. Then check
that `MEMORY.md` is there.

### D6. Ollama

```powershell
ollama pull llama3.2:3b                    # 2.0 GB, default for the system
ollama pull llama3.1:8b                    # 4.9 GB, only if RAM >= 16 GB (better readings)
ollama run llama3.2:3b "Say OK"            # should answer within a minute
curl http://localhost:11434/api/tags       # server is up
```

Ollama starts itself with Windows (tray icon). If `curl` fails, run `ollama serve` in its own shell.

### D7. Verify, in this order (stop and fix on the first failure)

1. Imports: `python -c "import torch, numpy, scipy, whisper, onnx, onnxruntime, websockets; print('ok')"`
2. Tests: `python -m pytest -q`. Eavan's laptop had 1 known failing test (see the memory notes); anything more is new.
3. Whisper on a real dataset file, timed:
   ```powershell
   Measure-Command { python scripts/run_pipeline.py --iq "$env:USERPROFILE\Downloads\military_radio_IQ\malay\iq\ms_cv_009_NBFM.wav" --max-seconds 10 --hybrid --classifier results/classifier/physics_aug_best.pt --listen --whisper-model medium --language ms }
   ```
   It should find an NBFM transmission and print Malay text.
4. Live replay (no radio needed):
   ```powershell
   # shell 1
   python scripts/serve_local.py
   # shell 2
   python scripts/live_sdrconnect.py --replay data/local/military_radio/military_net.wav --whisper-model medium --llm
   ```
   In the browser: **Signal Pipeline** → tick **Follow live snapshots**. Expect three stations (Alpha NBFM,
   Bravo NBFM, Malay AM) with transcripts and an Ollama reading. Use the built-in browser pane to check
   and take a screenshot as proof.
5. Note the timings and RAM use (Task Manager) to compare with Eavan's laptop: Whisper small 3–5 s and
   medium 13–15 s per 6 s clip, medium about 3 GB RAM, full mode about 50 s of CPU per 1 s of IQ.

### D8. Make it easy to start

- Desktop shortcut to `web\start_demo.bat` named "NEXA Dashboard".
- A second `.bat` in the repo root, not committed (or in the scratchpad if preferred), "NEXA Live Replay":
  it activates `.venv` and runs the replay command from D7.4.
- Add `.claude/launch.json` entries for `serve_local.py` so the browser pane can preview it.

### D9. The real chain on this one laptop (SDRconnect → bridge → dashboard)

Follows `docs/LIVE_DEMO.md` section 4. No `--host` or `--lan` is needed because everything is on localhost.

1. **SDRconnect playing a dataset file** (works with no radio):
   - Load `%USERPROFILE%\Downloads\military_radio_IQ\malay\iq\ms_cv_009_NBFM.wav` in SDRconnect (don't press play yet).
   - Shell 1: `python scripts/serve_local.py`
   - Shell 2: `python scripts/live_sdrconnect.py --fs 250000 --whisper-model medium --language ms --llm --autotune`
   - Now press play in SDRconnect. The order matters: **load → start bridge → play**. For a different file, Ctrl+C the bridge and start it again.
   - Dashboard → Signal Pipeline → **Follow live snapshots**. Expect NBFM found and the Malay text shown.
2. **RSPduo live** (only if the radio and a walkie-talkie are there): start the RSPduo in SDRconnect, then run the same
   bridge **without `--fs`** (the measured rate is right for the live radio). Add `--quiet-stop-s 10` so it analyses
   after each transmission, or press Ctrl+C to end and analyse.
3. Windows Firewall may ask about SDRconnect or Python. Choose **Private networks**. Everything is local, so nothing needs to be opened to the internet.
4. RAM is shared by everything at once: SDRconnect, Whisper medium (about 3 GB), Ollama (about 2 GB for 3b, about 5 GB for 8b) and the browser.
   Check Task Manager during the run. If RAM is above about 85%, use `--whisper-model small` or the 3b Ollama model.
5. One more desktop shortcut, "NEXA Live (SDRconnect)", that runs the D9.1 bridge command without `--fs`
   (live radio). The note in the .bat says to add `--fs 250000` when playing files.

### D10. Known pitfalls

- PATH changes need a new shell. In PowerShell 5.1, chain commands with `;` and not `&&`.
- `python -m http.server` serves `.mjs` as text/plain. Use `serve_local.py`, not http.server, for the dashboard.
- Edge can cache JS modules. Hard-refresh (Ctrl+F5) after changing web files.
- Dataset IQ files are 250 kS/s, so give `--fs 250000` when SDRconnect plays them (LIVE_DEMO.md explains why).
- Whisper `medium` needs about 3 GB RAM. `base` is poor at Malay.
- Don't commit model files, data or `web/supabase-config.js` (all git-ignored on purpose).

### D11. Final report to the user

Plain language, short: what's installed (with versions), the machine specs, the timings next to the old
laptop's, peak RAM with everything running at once, which checks passed, and what still needs a human
(for example the RSPduo live test with a walkie-talkie).
