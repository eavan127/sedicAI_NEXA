import { CASES, buildScenario, caseNeedsLibrary, loadCivilianLibrary } from "./generators.js";
import { classifyCapture, loadModel, FS, WINDOW_LEN } from "./model.js";
import {
  noiseFloorPower, occupancy, powerSpectrumDb, resolveSession, scipyStft,
} from "./analysis.js";
import { drawConsole } from "./console.js";
import { eventRows, headerLine, latestBlock, printHeaderHtml, statusBlock } from "./panels.js";
import {
  drawAttention, drawBreakdown, modelCardHtml, probabilityHtml,
  provenanceHtml, scorecardHtml, windowMetadataHtml,
} from "./pages.js";
import { civilianWindows, drawConstellation } from "./constellation.js";
import { THRESHOLDS } from "./analysis.js";

const el = id => document.getElementById(id);
const statusEl = el("status"), headlineEl = el("headline");
const statusBox = el("statusBox"), latestBox = el("latestBox");
const consoleCanvas = el("console"), tbody = document.querySelector("#eventsTable tbody");
const constellationCanvas = el("constellationCanvas"), constellationBlock = el("constellationBlock");
const printHeader = el("printHeader"), printBtn = el("printBtn");
const synthBtn = el("synthBtn"), uploadBtn = el("uploadBtn"), fileInput = el("fileInput");
const caseSel = el("caseSel"), snrSel = el("snrSel"), hopSel = el("hopSel"), modelSel = el("modelSel");
const smoothingRadio = el("smoothingRadio");

for (const name of Object.keys(CASES)) {
  caseSel.add(new Option(name, name));
}
caseSel.value = "All three";

// configs/default.yaml:snr_bins_db -- the training bins, not arbitrary round
// numbers: asking the model about an SNR it never saw conflates two questions.
const SNR_BINS = [-10, -6, -2, 2, 6, 10];
for (const snr of SNR_BINS) {
  snrSel.add(new Option(`${snr >= 0 ? "+" : ""}${snr} dB`, String(snr)));
}
// rf_replay.py picks 0 if present, else the middle bin.
snrSel.value = "2";

let smoothingChoice = "Smoothed";
smoothingRadio.addEventListener("click", e => {
  const btn = e.target.closest("button");
  if (!btn) return;
  smoothingChoice = btn.dataset.value;
  for (const b of smoothingRadio.querySelectorAll("button")) b.classList.toggle("on", b === btn);
  if (session) render();          // re-render only; no re-inference needed
});

// Cached per model choice, so switching back doesn't reload from the network.
const modelCache = new Map();
let session = null;               // { capture, result, source, caseNote, truth, snrDb }
let lastDrawnWidth = 0;           // guards the ResizeObserver against redraw loops

async function getModel(which) {
  if (!modelCache.has(which)) {
    modelCache.set(which, await loadModel(which, "./models", (d, t) => {
      statusEl.textContent = `Loading ${which} model ${d}/${t}…`;
    }));
  }
  return modelCache.get(which);
}

async function init() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    // The constellation panel needs the C42 calibration constants, so the
    // card is loaded up front rather than lazily on the Model page.
    modelCard = await (await fetch("./data/model_card.json")).json();
    await getModel("ensemble");
    statusEl.textContent = "Ready. Synthesize a scenario, or select a capture file.";
    synthBtn.disabled = false;
    uploadBtn.disabled = false;
  } catch (e) {
    statusEl.textContent = `Failed to load model: ${e.message}`;
    console.error(e);
  }
}

function modelLabel(which) {
  return which === "ensemble" ? "5-model ensemble average" : "single checkpoint — best_model.pt";
}

/** Re-derives every panel from the cached raw result. Smoothing is a display
 * rule, so switching it must NOT re-run inference -- same as the Gradio page,
 * where smoothing.change only re-renders. */
function render() {
  const { capture, result, source, caseNote, truth, snrDb, which,
           snrCapped, requestedSnrDb } = session;
  const smoothed = smoothingChoice === "Smoothed";
  const resolved = resolveSession(result, smoothed);
  const events = resolved.emitterEvents;
  const tiers = resolved.tiers;
  const emptyPct = tiers.length ? tiers.filter(t => t === "Empty").length / tiers.length * 100 : 0;
  const durationMs = capture.re.length / FS * 1000;

  headlineEl.innerHTML = headerLine({
    source, snrKnown: source === "scenario", trueSnrDb: snrDb,
    snrCapped, requestedSnrDb,
    modelLabel: modelLabel(which), caseNote, durationMs,
    nWindows: result.nWindows, hop: result.hop, nEvents: events.length, tiers,
  });

  statusBox.innerHTML = statusBlock({
    occupancyValue: capture.occupancy, nEvents: events.length,
    nWindows: result.nWindows, hop: result.hop, durationMs, emptyPct,
  });
  latestBox.innerHTML = latestBlock(events, emptyPct, capture);

  drawConsole(consoleCanvas, {
    spectro: capture.spectro, spectrum: capture.spectrum,
    events, tiers, truth, durationMs,
    starts: result.starts, hop: result.hop, fs: FS,
  });
  lastDrawnWidth = consoleCanvas.clientWidth;

  printHeader.innerHTML = printHeaderHtml({
    source, caseNote, snrDb, snrKnown: source === "scenario",
    hop: result.hop, nWindows: result.nWindows, durationMs,
    modelLabel: modelLabel(which), nEvents: events.length, truth,
    thresholds: THRESHOLDS,
  });
  printBtn.disabled = false;

  // The constellation is a SEPARATE component, not another panel inside the
  // console figure: that figure's whole premise is one shared time axis, and
  // a constellation has no time axis at all. Hidden outright when the capture
  // has no civilian window, so military-only cases look exactly as they did
  // before this panel existed.
  //
  // Selected on the RESOLVED probabilities, matching
  // CaptureSession.civilian_windows(smoothed=...).
  const picks = civilianWindows(resolved.probs, result.nWindows, result.nClasses,
                                 THRESHOLDS, 4);
  const drew = modelCard && picks.length && drawConstellation(constellationCanvas, {
    picks, capture, starts: result.starts, windowLen: result.windowLen,
    fs: FS, noisePower: capture.noisePower, c42cfg: modelCard.c42,
  });
  constellationBlock.hidden = !drew;

  tbody.innerHTML = "";
  for (const row of eventRows(events)) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

/** Everything a capture needs that does NOT depend on the model or the
 * display rules -- computed once per capture, reused on every re-render. */
function measureCapture(re, im) {
  return {
    re, im,
    occupancy: occupancy(re, im, FS),
    noisePower: noiseFloorPower(re, im),
    spectro: scipyStft(re, im, 256, FS),
    spectrum: powerSpectrumDb(re, im, FS),
  };
}

async function analyze(re, im, { source, caseNote = "", truth = null, snrDb = null,
                                  snrCapped = false, requestedSnrDb = null }) {
  synthBtn.disabled = uploadBtn.disabled = true;
  try {
    const which = modelSel.value;
    const hop = Number(hopSel.value);
    const sessions = await getModel(which);

    statusEl.textContent = "Measuring capture…";
    await new Promise(r => setTimeout(r, 0));
    const capture = measureCapture(re, im);

    const t0 = performance.now();
    const result = await classifyCapture(sessions, re, im, {
      hop,
      onProgress: (d, t) => { statusEl.textContent = `Running inference… ${d}/${t} windows`; },
    });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    session = { capture, result, source, caseNote, truth, snrDb, which,
                 snrCapped, requestedSnrDb };
    render();
    statusEl.textContent = `${result.nWindows} windows classified in ${elapsed}s.`;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    console.error(e);
  } finally {
    synthBtn.disabled = uploadBtn.disabled = false;
  }
}

synthBtn.addEventListener("click", async () => {
  const caseName = caseSel.value;
  const snrDb = Number(snrSel.value);
  const script = CASES[caseName];
  try {
    // Civilian cases need the exported RadioML window library; fetched once
    // and cached, so only the first civilian case pays for it.
    let library = null, librarySnrDb = null;
    if (caseNeedsLibrary(script)) {
      statusEl.textContent = "Loading civilian capture library…";
      await new Promise(r => setTimeout(r, 0));
      library = await loadCivilianLibrary("./data");
      librarySnrDb = library.snrDb;
    }
    statusEl.textContent = "Synthesizing IQ…";
    await new Promise(r => setTimeout(r, 0));
    const scenario = buildScenario({
      totalDuration: 0.05, snrDb, seed: Math.floor(Math.random() * 1e9),
      script, library, librarySnrDb,
    });
    await analyze(scenario.re, scenario.im, {
      source: "scenario", caseNote: `case \`${caseName}\``,
      truth: scenario.segments, snrDb: scenario.trueSnrDb,
      snrCapped: scenario.snrCapped, requestedSnrDb: scenario.requestedSnrDb,
    });
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    console.error(e);
  }
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  statusEl.textContent = `Reading ${file.name}…`;
  try {
    // Interleaved float32 I,Q,I,Q,... -- the same contract as
    // src/infer.py and src/ui/session.py:load_upload.
    const buf = await file.arrayBuffer();
    let raw = new Float32Array(buf);
    if (raw.length < 2) throw new Error(
      "File contains no complex samples. Expected interleaved float32 I,Q,I,Q,... — at least 2 values.");
    if (raw.length % 2) raw = raw.subarray(0, raw.length - 1);
    const n = raw.length / 2;
    if (n < WINDOW_LEN) throw new Error(
      `Capture is ${n} complex samples; at least ${WINDOW_LEN} are needed for one window.`);
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) { re[i] = raw[2 * i]; im[i] = raw[2 * i + 1]; }
    // truth is scenario-only: never render a TRUTH overlay over data we do
    // not actually have ground truth for (session.py:analyze).
    await analyze(re, im, { source: "upload" });
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    console.error(e);
  } finally {
    fileInput.value = "";
  }
});

modelSel.addEventListener("change", async () => {
  if (!session) return;
  await analyze(session.capture.re, session.capture.im, {
    source: session.source, caseNote: session.caseNote,
    truth: session.truth, snrDb: session.snrDb,
  });
});

hopSel.addEventListener("change", async () => {
  if (!session) return;
  await analyze(session.capture.re, session.capture.im, {
    source: session.source, caseNote: session.caseNote,
    truth: session.truth, snrDb: session.snrDb,
  });
});

printBtn.addEventListener("click", () => window.print());

// The canvases size their backing store from clientWidth at draw time, so a
// figure drawn for a 1100px column is a stretched bitmap on a 186mm page.
// Redraw against the print layout, then again afterwards for the screen.
// (Chrome runs beforeprint synchronously before paginating, so a synchronous
// redraw here lands in the output.)
addEventListener("beforeprint", () => { if (session) render(); });
addEventListener("afterprint", () => { if (session) render(); });

// ResizeObserver rather than window's resize event: the canvas can also go
// from zero-width to laid-out without the window changing size -- a hidden
// pane being revealed, a font loading and reflowing the column -- and the
// figure is sized from the canvas's own width, not the window's.
new ResizeObserver(() => {
  const w = consoleCanvas.clientWidth;
  if (session && w && w !== lastDrawnWidth) render();
}).observe(consoleCanvas);

// ---------------------------------------------------------------------------
// Page switching + the other three pages
// ---------------------------------------------------------------------------

const winSlider = el("winSlider"), winReadout = el("winReadout");
const probsBox = el("probsBox"), winMetaBox = el("winMetaBox");
const attnCanvas = el("attnCanvas"), breakdownCanvas = el("breakdownCanvas");
let currentPage = "replay";
let perfData = null, modelCard = null;

function showPage(page) {
  currentPage = page;
  for (const btn of document.querySelectorAll("nav button")) {
    btn.classList.toggle("active", btn.dataset.page === page);
  }
  for (const sec of document.querySelectorAll("main section")) {
    sec.hidden = sec.id !== `page-${page}`;
  }
  // Canvases cannot be sized while hidden, so each page draws on entry.
  if (page === "signal") renderSignal();
  if (page === "performance") renderPerformance();
  if (page === "model") renderModel();
  if (page === "replay" && session) render();
}

for (const btn of document.querySelectorAll("nav button")) {
  btn.addEventListener("click", () => showPage(btn.dataset.page));
}

function renderSignal() {
  if (!session) {
    probsBox.innerHTML = `<div class="note">Load a capture on RF Replay first.</div>`;
    winMetaBox.innerHTML = "";
    return;
  }
  const n = session.result.nWindows;
  winSlider.max = String(n);
  const idx = Math.max(0, Math.min(Number(winSlider.value) - 1, n - 1));
  winReadout.textContent = `#${idx + 1} / ${n}`;
  probsBox.innerHTML = probabilityHtml(session.result, idx);
  winMetaBox.innerHTML = windowMetadataHtml(session, idx);
  drawAttention(attnCanvas, session, idx);
}

winSlider.addEventListener("input", () => { if (currentPage === "signal") renderSignal(); });

async function renderPerformance() {
  const box = el("scorecardBox"), prov = el("perfProvenance");
  if (!perfData) {
    prov.innerHTML = `<div class="note">Loading…</div>`;
    try {
      perfData = await (await fetch("./data/performance.json")).json();
    } catch (e) {
      prov.innerHTML = `<div class="note">Could not load performance data: ${e.message}</div>`;
      return;
    }
  }
  prov.innerHTML = provenanceHtml(perfData);
  box.innerHTML = scorecardHtml(perfData);
  drawBreakdown(breakdownCanvas, perfData);

  // The two figures src/evaluate.py wrote, shown as-is and captioned with
  // when they were produced.
  //
  // Deliberately NOT flagged as stale by comparing their mtime against the
  // config's: file timestamps do not survive this project's workflow. A git
  // checkout or merge rewrites configs/default.yaml's mtime, and evals
  // downloaded from Colab carry the download time rather than the time the
  // evaluation ran. Tested here, that comparison claimed the figures
  // predated the thresholds when the numbers in them demonstrably match the
  // current calibration -- a confident, wrong warning is worse than none.
  // The production time is stated so a reader can judge; the authoritative
  // check is whether the scorecard's recalls match the configured
  // thresholds, which is what the parity tests cover.
  const figs = perfData.figures ?? {};
  for (const [key, block, img, cap, what] of [
    ["confusion_matrix.png", "figConfusion", "imgConfusion", "capConfusion",
     "Counts of true versus predicted class per window, so a false positive is the off-diagonal cell in that class's column."],
    ["accuracy_vs_snr.png", "figSnr", "imgSnr", "capSnr",
     "Per-class recall across the SNR sweep, as written by the evaluation."],
  ]) {
    const meta = figs[key];
    const blockEl = el(block);
    if (!meta) { blockEl.hidden = true; continue; }
    blockEl.hidden = false;
    el(img).src = `./data/${meta.file}`;
    el(cap).innerHTML = `${what} Produced ${meta.mtime.replace("T", " ")} by ` +
      `<code>python -m src.evaluate</code>, and copied here unmodified. ` +
      `Re-run that command and <code>python web/build.py</code> after any ` +
      `retrain or recalibration.`;
  }
}

async function renderModel() {
  const box = el("modelCardBox");
  if (!modelCard) {
    box.innerHTML = `<div class="note">Loading…</div>`;
    try {
      modelCard = await (await fetch("./data/model_card.json")).json();
    } catch (e) {
      box.innerHTML = `<div class="note">Could not load model card: ${e.message}</div>`;
      return;
    }
  }
  box.innerHTML = modelCardHtml(modelCard, modelSel.value);
}

init();
