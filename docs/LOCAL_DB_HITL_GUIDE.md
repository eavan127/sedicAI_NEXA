# NEXA local database, human-in-the-loop and retraining: teammate guide

Branch `eavan-local-db`. Everything here runs on one laptop with no internet:
the page, the model, the database, the raw IQ, and retraining.

## 1. One-time setup

```bash
git fetch origin
git switch eavan-local-db
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` now includes `onnx`, `onnxruntime` and `onnxscript`. Retraining
needs them; the rest of the app does not.

Optional, for testing with held-out data (needs `data/processed/`, which only
exists on a machine that built the dataset):

```bash
python scripts/export_verification_pack.py --out verification_pack --format both
```

Without `data/processed/`, use any `.sigmf` / `.bin` recording, or the
Simulated receiver.

## 2. Start it

Double-click `web\start_demo.bat`, or:

```bash
python scripts/serve_local.py
```

The browser opens at `http://localhost:8099/index.html`. If 8099 is busy, the
server takes the next free port and prints it. Keep the window open, and use
Ctrl+C to stop. Useful options:

| Option | What it does |
|---|---|
| `--quick-retrain` | Retrains in about 1 minute (1 epoch, 2,000 replay windows, 3,000-window exam) instead of about 4 |
| `--synthetic-exam` | Judges retrained models on the fixed synthetic exam even when `data/processed/` exists (labelled "synthetic" on the Model page) |
| `--lan` | Lets other machines on the network open it (default: this machine only) |
| `--db path\to\other.db` | Uses a different database file (handy for a clean demo) |

Everything is stored in `data\local\`: `nexa.db` (SQLite), `iq\` (raw IQ),
`models\` (retrained versions), and `jobs\` (training logs). It is git-ignored.

If the page looks broken after pulling new code (empty dropdowns, a red
"does not provide an export" error in the F12 console), your browser cached
old files. Press **Ctrl+Shift+R** once.

## 3. What to try, in order

Use two names, for example **Eavan** and **Jessy**. The four-eyes rule needs a
second person.

**A. Analyse a signal (RF Replay page)**
1. Scenario case **All three**, SNR **+2 dB**, then **Synthesize Scenario**.
2. Scroll to the timeline. Hover to read the exact time. Dashed boxes are the
   true answer.
3. **Recommended corrections** (under the timeline) lists every disagreement
   with its ms range, the important ones first.

**B. Live receiver (RF Replay page)**
1. Model **Single**, Window hop **no overlap, 512** (fastest).
2. Receiver source **Simulated receiver**, then **▶ Connect & stream**. Watch
   the connect log, then the green light and the live readouts.
3. Change Scenario case while it streams; the next dwell follows.
4. **■ Stop**. Also try **File replay** with `verification_pack\mixed_sequence.sigmf`.

**C. Correct the model (RF Replay page)**
1. Click **Review** on a recommended correction, or click a spot / drag a
   stretch on the timeline, or use **✎ Correct** in the Detection events table,
   or **＋ Report a missed signal**.
2. Tick what is really there, give a reason and your name, then **Submit
   correction**.

**D. Approve it (History page, as the other person)**
1. Type the other name in the **Operator** box (Audit trail section).
2. **Human corrections: review queue**, then **✔ Approve**. Approving your own
   correction is refused (four-eyes).
3. **Model health: retraining trigger** shows the correction rate and the
   rules.
4. **Audit trail**, then **Verify integrity** should report "Intact".

**E. Build a report (History page)**
1. Set filters (verdict, class, dates…). **Report builder** counts the
   matching captures.
2. Tick sections, pick a format (PDF, Excel, CSV, JSON, SigMF bundle, raw IQ
   bundle), edit the classification banner, then **⭳ Build report**.

**F. Retrain (Model page)**
1. Type your name and a reason. Tick **Override** (a small demo never meets
   the 100-capture rule), then **▶ Start retraining**.
2. Watch the log. When it finishes, **Model versions** shows before and after
   for each judged class and the gate result.
3. As the *other* person: **✔ Approve & activate** (only possible when the
   gate PASSED), or **✖ Reject**.
4. On RF Replay, Model becomes **Single, retrained ft-…**. Analyses now use
   it, and the saved record names the version.
5. **↺ Roll back to shipped model** returns to `best_model.pt`.

A candidate that FAILS its gate cannot be activated. That is the safety net
working, not a bug: on the real test split, one or two corrections are rarely
enough to improve every class at once.

## 4. Rules the system enforces

- Corrections need a reason, raw IQ evidence, and approval by a *different*
  named person. They are read-only once reviewed and are never deleted.
- A capture with corrections cannot be deleted (it is their evidence).
- The audit log is append-only and hash-chained. **Verify integrity** detects
  any edit or deletion made directly in the database file.
- Retraining is started by a named person with a reason. Starting it when the
  rules are not met needs an explicit, logged override.
- A retrained model replaces nothing until it passes the gate *and* a second
  person approves it. The 5-model ensemble is never changed.
- Every report export is logged with the SHA-256 of the file that was handed
  out.

## 5. Tests

```bash
python -m pytest -q
node web/test/review_check.mjs
node web/test/receiver_check.mjs
node web/test/sigmf_check.mjs
node web/test/history_check.mjs
```
