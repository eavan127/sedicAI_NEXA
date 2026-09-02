# Session Handoff — SEDIC 2026 Project Overwatch

Status as of 2026-09-02. Paste this file's content (or point Claude at it) at
the start of a new session to pick up without re-explaining everything.

## What's done and confirmed working

- **Data pipeline bug fixed**: `src/data/build_dataset.py::load_radioml_civilian()`
  now uses offset-based permutation slicing so standalone and composite-overlay
  RadioML draws never reuse the same rows (was a real bug — pigeonhole
  collision above 2048/4096 rows). Committed by you as `3f72836`.
- **NOISE_FLOOR recall regression fixed**: removed a stale `0.5x` loss-weight
  dampen in `configs/default.yaml` that made sense in the old single-label
  setup but was actively hurting recall once civilian classes grew. Verified
  on a real Colab run: recall went 0.00 → 0.51.
- **`scripts/calibrate_thresholds.py` bug fixed**: it computed thresholds for
  all 8 classes but only ever printed/exported 3 (the judged classes),
  silently leaving civilian + NOISE_FLOOR uncalibrated. Now loops over all 8.
- **OOM crashes fixed** in `src/evaluate.py`, `src/infer.py`,
  `scripts/measure_variance.py` — replaced single unbatched forward passes
  with chunked batches (`EVAL_BATCH_SIZE = 256`).
- **Colab checkpoint loss fixed**: `scripts/train_ensemble.py` now accepts
  `--sync-dir` and copies each ensemble member's checkpoint out (e.g. to a
  mounted Drive folder) the moment it finishes, instead of only at the very
  end. Notebook (`notebooks/colab_training_multilabel.ipynb`) updated to
  mount Drive early and pass `--sync-dir`.
- **Local OMNI GUI fixed**: `gradio` was declared in `requirements.txt` but
  not installed in this venv. Fixed with `pip install gradio` (installed
  gradio-6.26.0). Confirmed working end-to-end — Scenario synthesis in the
  RF Replay page runs successfully in the browser preview.

## Standing open task (highest priority — nothing else matters until this lands)

You still need to run, on Colab, **one complete pass** of:
1. `python -m scripts.train_ensemble --models 5 --sync-dir <drive-path>`
2. `python -m scripts.calibrate_thresholds --ensemble --n-models 5`
3. Paste the resulting 8-class thresholds into `configs/default.yaml`
4. `python -m src.evaluate --ensemble --n-models 5`

Every attempt so far has been a single-model run or a lost Colab session —
the fixed pipeline (disjoint data, NOISE_FLOOR fix, working calibration
script, `--sync-dir` persistence) has never actually been evaluated
end-to-end as an ensemble. This is the number the 80% benchmark claim on the
three judged classes (LFM_RADAR, FHSS, JAMMING) depends on. Until this run
completes, any recall number quoted for the ensemble is provisional.

## Not yet started

- **Power BI report** from the project's existing CSV exports
  (`evals/csv/`, produced by `src/evaluate.py`). You don't know Power BI —
  next session should be a practical, from-scratch walkthrough: import the
  CSV, build a couple of visuals (per-class recall bar chart, confusion
  matrix heatmap), not a deep tooling tour.
- **UiPath/RPA fit assessment** — you have UiPath installed and asked if it
  has a legitimate use here. Honest starting take: probably not a fit for
  the core ML pipeline (that's Python automation, not UI automation), but
  it could plausibly automate something adjacent like "watch a folder for
  new eval CSVs and refresh/export the Power BI report" — worth scoping
  concretely rather than forcing UiPath in just because it's installed.
- **Resume / career framing** — lower priority per your own phrasing, for
  when you send this to the French [recruiter/program — clarify which].

## Key facts to remember

- Repo: `eavan127/sedicAI_NEXA`, branch churn is normal — you work locally
  in parallel (VS Code), so always re-check `git status`/`git branch
  --show-current` rather than trusting what a prior session assumed.
- The deployed Render "OMNI" web app is on a free tier that spins down with
  inactivity (cold-start delay, not a real bug) and is currently serving a
  teammate's (Eileen's) Aug 29 checkpoint on `main`, not your latest work —
  worth knowing if you compare its behavior to local.
- Local checkpoints present: `results/best_model.pt` +
  `results/ensemble_0.pt` … `ensemble_4.pt` (612,917 bytes each) — these are
  what the local GUI's "ensemble" and "best model" options load from disk on
  every call (no caching, see `src/ui/app_models.py`).
- Launch the local GUI via the `omni-ui` preview config in
  `.claude/launch.json` (`python scripts/inference_ui.py`, port 7860).
