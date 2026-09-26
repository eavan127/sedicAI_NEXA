import { CASES, buildScenario, caseNeedsLibrary, loadCivilianLibrary } from "./generators.js";
import { classifyCapture, loadModel, CLASSES, FS, WINDOW_LEN } from "./model.js";
import {
  noiseFloorPower, occupancy, powerSpectrumDb, resolveSession, scipyStft,
} from "./analysis.js";
import { drawConsole } from "./console.js";
import { describe, isSigmfName, readSigmf } from "./sigmf.js";
import { FileReplay, SimulatedReceiver, dwellLevel, shouldStore, toInterleavedF32 } from "./receiver.js";
import { detectedIn, spanAt, suggestCorrections } from "./review.js";
import { eventRows, headerLine, latestBlock, printHeaderHtml, statusBlock } from "./panels.js";
import {
  breakdownTableHtml, drawAttention, drawBreakdown, animateBreakdown, drawPerClassRecall,
  denseQamHtml, modelCardHtml, probabilityHtml, provenanceHtml, scorecardHtml, summaryHtml,
  windowMetadataHtml,
} from "./pages.js";
import { civilianWindows, drawConstellation } from "./constellation.js";
import { THRESHOLDS } from "./analysis.js";
import { initAssistant, openChatLog } from "./chatbot.js";
import { isAvailable as ollamaAvailable, rewrite as ollamaRewrite, warmUp as ollamaWarmUp } from "./ollama.js";
import {
  attachCapture, buildRecord, canDelete, correctionStats, deleteAnalysis, detectServer, exportReport, retrainStatus,
  getJob, listJobs, listModels, reviewModel, rollbackModel, startRetrain,
  listAnalyses, listAudit, listCorrections, logEvent, operatorName, readConfig, reviewCorrection,
  saveAnalysis, setOperatorName, submitCorrection, supportsCorrections, usingSupabase, verifyAudit,
} from "./storage.js";
import { applyFilters, barsHtml, summarise, summaryHtml as histSummaryHtml, tableHtml } from "./history.js";
import { buildReport, buildSingle } from "./report.js";
import { installZoom, registerZoom } from "./zoom.js";

installZoom();

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
    ort.env.wasm.wasmPaths = new URL("./vendor/ort/", document.baseURI).href;   // bundled copy: the demo must run with no internet
    // Speed. proxy: inference runs in a Web Worker, so the page never
    // freezes while a capture is classified. numThreads: several cores per
    // model -- only possible when the server sends the cross-origin
    // isolation headers (scripts/serve_local.py, web/vercel.json); without
    // them the browser forces one thread and this is a no-op.
    ort.env.wasm.proxy = true;
    if (globalThis.crossOriginIsolated) {
      ort.env.wasm.numThreads = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 2) - 2));
    }
    // The constellation panel needs the C42 calibration constants, so the
    // card is loaded up front rather than lazily on the Model page.
    modelCard = await (await fetch("./data/model_card.json")).json();
    await getModel("ensemble");
    detectServer().then(info => labelSingleOption(info?.active_model ? { version: info.active_model } : null));
    statusEl.textContent = "Ready. Synthesize a scenario, or select a capture file.";
    synthBtn.disabled = false;
    uploadBtn.disabled = false;
  } catch (e) {
    statusEl.textContent = `Failed to load model: ${e.message}`;
    console.error(e);
  }
}

// The retrained single model the server is serving, if one was approved.
let activeSingleVersion = null;

function modelLabel(which) {
  if (which === "ensemble") return "5-model ensemble average";
  return activeSingleVersion ? `single checkpoint, retrained ${activeSingleVersion}` : "single checkpoint, best_model.pt";
}

/** Re-derives every panel from the cached raw result. Smoothing is a display
 * rule, so switching it must NOT re-run inference -- same as the Gradio page,
 * where smoothing.change only re-renders. */
function render() {
  const { capture, result, source, caseNote, truth, snrDb, which,
           snrCapped, requestedSnrDb, mode } = session;
  const smoothed = smoothingChoice === "Smoothed";
  const resolved = resolveSession(result, smoothed);
  const events = resolved.emitterEvents;
  const tiers = resolved.tiers;
  const emptyPct = tiers.length ? tiers.filter(t => t === "Empty").length / tiers.length * 100 : 0;
  const durationMs = capture.re.length / FS * 1000;

  headlineEl.innerHTML = headerLine({
    mode, source, snrKnown: source === "scenario", trueSnrDb: snrDb,
    snrCapped, requestedSnrDb,
    modelLabel: modelLabel(which), caseNote, durationMs,
    nWindows: result.nWindows, hop: result.hop, nEvents: events.length, tiers,
  });

  statusBox.innerHTML = statusBlock({
    occupancyValue: capture.occupancy, nEvents: events.length,
    nWindows: result.nWindows, hop: result.hop, durationMs, emptyPct,
  });
  latestBox.innerHTML = latestBlock(events, emptyPct, capture);

  const consoleOpts = {
    spectro: capture.spectro, spectrum: capture.spectrum,
    events, tiers, truth, durationMs,
    starts: result.starts, hop: result.hop, fs: FS,
  };
  consoleGeom = drawConsole(consoleCanvas, consoleOpts);
  // Re-registered on every render: the options carry this capture's data, and
  // a zoom of the previous one would be a picture of the wrong signal.
  registerZoom(consoleCanvas, target => drawConsole(target, consoleOpts));
  consoleCanvas.title = "";   // the time cursor explains itself; ⤢ Enlarge zooms
  session.suggestions = suggestCorrections({ events, truth, durationS: capture.re.length / FS });
  renderSuggestions();
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
  const constellationOpts = {
    picks, capture, starts: result.starts, windowLen: result.windowLen,
    fs: FS, noisePower: capture.noisePower, c42cfg: modelCard.c42,
  };
  const drew = modelCard && picks.length && drawConstellation(constellationCanvas, constellationOpts);
  if (drew) registerZoom(constellationCanvas, t => drawConstellation(t, constellationOpts));
  constellationBlock.hidden = !drew;

  tbody.innerHTML = "";
  eventRows(events).forEach((row, i) => {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    td.innerHTML = `<button class="mini" data-correct="${i}" title="Tell the system what is really there">✎ Correct</button>`;
    tr.appendChild(td);
    tbody.appendChild(tr);
  });
  session.events = events;
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

/**
 * Classify one capture and draw it. `live` is set by the receiver loop: the
 * headline says LIVE, the result is NOT stored here (the loop decides which
 * dwells are worth keeping), and the manual buttons stay locked while the
 * stream runs. Returns the elapsed inference seconds, or null on failure.
 */
async function analyze(re, im, { source, caseNote = "", truth = null, snrDb = null,
                                  snrCapped = false, requestedSnrDb = null, live = false,
                                  signal = null }) {
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
      hop, signal,
      onProgress: (d, t) => { statusEl.textContent = `Running inference… ${d}/${t} windows`; },
    });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    session = { capture, result, source, caseNote, truth, snrDb, which,
                 snrCapped, requestedSnrDb, mode: live ? "LIVE" : "REPLAY" };
    render();
    statusEl.textContent = `${result.nWindows} windows classified in ${elapsed}s.`;
    if (live) return Number(elapsed);
    // Stored AFTER render: a storage backend that is slow, full or
    // misconfigured must not delay the picture the operator is waiting for,
    // and must not lose it either -- store() reports its own failures and
    // falls back to the local store.
    session.storing = store(pendingFile);
    pendingFile = null;
    return Number(elapsed);
  } catch (e) {
    if (e.name === "AbortError") { statusEl.textContent = "Stopped."; return null; }
    statusEl.textContent = `Error: ${e.message}`;
    console.error(e);
    return null;
  } finally {
    if (!rx) synthBtn.disabled = uploadBtn.disabled = false;
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

/**
 * Selected files -> {re, im, truth, name, meta, note, warnings}. A SigMF
 * recording (archive, or meta + data pair) or a raw interleaved-float32 file.
 * Shared by "Select & Analyze" and the receiver's File replay.
 */
async function readUpload(files) {
  if (files.some(f => isSigmfName(f.name))) {
    const rec = await readSigmf(files);
    if (rec.re.length < WINDOW_LEN) throw new Error(
      `Recording is ${rec.re.length} complex samples; at least ${WINDOW_LEN} are needed for one window.`);
    // An annotated recording carries its own ground truth (rec.truth).
    return { ...rec, note: describe(rec.meta) };
  }
  // Interleaved float32 I,Q,I,Q,... -- the same contract as
  // src/infer.py and src/ui/session.py:load_upload.
  const file = files[0];
  let raw = new Float32Array(await file.arrayBuffer());
  if (raw.length < 2) throw new Error(
    "File contains no complex samples. Expected interleaved float32 I,Q,I,Q,... with at least 2 values.");
  if (raw.length % 2) raw = raw.subarray(0, raw.length - 1);
  const n = raw.length / 2;
  if (n < WINDOW_LEN) throw new Error(
    `Capture is ${n} complex samples; at least ${WINDOW_LEN} are needed for one window.`);
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) { re[i] = raw[2 * i]; im[i] = raw[2 * i + 1]; }
  // No truth: never render a TRUTH overlay over data we do not actually have
  // ground truth for (session.py:analyze).
  return { re, im, truth: null, name: file.name, meta: null, note: "", warnings: [] };
}

fileInput.addEventListener("change", async () => {
  const files = [...(fileInput.files || [])];
  if (!files.length) return;
  statusEl.textContent = `Reading ${files.map(f => f.name).join(" + ")}…`;
  try {
    const rec = await readUpload(files);
    // Every selected file is kept (a SigMF pair is two); the record names the data.
    pendingFile = files;
    await analyze(rec.re, rec.im, {
      source: "upload", truth: rec.truth,
      caseNote: rec.warnings.length ? `${rec.note} · WARNING: ${rec.warnings.join(" ")}` : rec.note,
    });
    if (rec.warnings.length) statusEl.textContent += `  ⚠ ${rec.warnings.join(" ")}`;
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    console.error(e);
  } finally {
    fileInput.value = "";
  }
});

// ---------------------------------------------------------------------------
// Human correction: "the model said X here, it is really Y"
// ---------------------------------------------------------------------------

const corrDialog = el("corrDialog"), corrForm = el("corrForm"), corrTitle = el("corrTitle");
const corrModelSaid = el("corrModelSaid"), corrClasses = el("corrClasses");
const corrStart = el("corrStart"), corrEnd = el("corrEnd"), corrReason = el("corrReason");
const corrOperator = el("corrOperator"), corrError = el("corrError"), corrSubmit = el("corrSubmit");

const CLASS_GROUPS = [
  ["Civilian", ["BPSK", "QPSK", "16QAM", "64QAM"]],
  ["Military", ["LFM_RADAR", "FHSS"]],
  ["Hostile", ["JAMMING"]],
  ["Nothing", ["NOISE_FLOOR"]],
];
corrClasses.innerHTML = CLASS_GROUPS.map(([group, classes]) =>
  `<fieldset><legend>${group}</legend>` + classes.map(c =>
    `<label><input type="checkbox" value="${c}"> ${c === "NOISE_FLOOR" ? "Nothing here (noise only)" : c}</label>`).join("")
  + `</fieldset>`).join("");

// The session a correction is about: captured when the form opens, because a
// live stream replaces the on-screen session every dwell.
let corrCtx = null;
const eventsNote = el("eventsNote");

function openCorrection({ predicted, startS, endS, missed, suggested = null, reason = "", title = null }) {
  corrCtx = { sess: session, predicted };
  const durS = session.capture.re.length / FS;
  corrTitle.textContent = title || (missed ? "Report a signal the model missed" : "Correct this detection");
  corrModelSaid.textContent = predicted.length ? predicted.join(" + ") : "nothing (missed signal)";
  corrStart.value = (startS * 1000).toFixed(2);
  corrEnd.value = (Math.min(endS, durS) * 1000).toFixed(2);
  corrStart.max = corrEnd.max = (durS * 1000).toFixed(2);
  // Pre-ticked: the recommended answer when there is one, else what the
  // model said (so the operator only changes what is wrong).
  const ticks = suggested || predicted;
  for (const box of corrClasses.querySelectorAll("input")) box.checked = ticks.includes(box.value);
  corrReason.value = reason;
  corrOperator.value = operatorName();
  corrError.textContent = "";
  corrSubmit.disabled = false;
  corrDialog.showModal();
  corrReason.focus();
}

tbody.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-correct]");
  if (!btn || !session?.events) return;
  if (!(await supportsCorrections())) {
    eventsNote.textContent = "Corrections are stored in the local database: start the page with "
      + "scripts/serve_local.py (web/start_demo.bat).";
    return;
  }
  const e = session.events[Number(btn.dataset.correct)];
  openCorrection({ predicted: e.classes.filter(c => c !== "NOISE_FLOOR"),
                   startS: e.startUs / 1e6, endS: e.endUs / 1e6, missed: false });
});

el("corrMissed").addEventListener("click", async () => {
  if (!session) { eventsNote.textContent = "Analyse a capture first."; return; }
  if (!(await supportsCorrections())) {
    eventsNote.textContent = "Corrections need the local server (scripts/serve_local.py).";
    return;
  }
  openCorrection({ predicted: [], startS: 0, endS: session.capture.re.length / FS, missed: true });
});

el("corrCancel").addEventListener("click", () => corrDialog.close());

/** Make sure the capture being corrected is stored WITH its raw IQ: a live
 *  dwell that was not stored is stored now, and a synthesized scenario (saved
 *  without IQ) gets its samples attached. A correction nobody could retrain
 *  on or re-check would be an opinion, not evidence. */
async function ensureEvidence(sess) {
  const iqFile = () => new File([toInterleavedF32(sess.capture.re, sess.capture.im).buffer], "capture.f32");
  let rec = sess.record || (sess.storing && await sess.storing);
  if (!rec) rec = await store(iqFile(), sess);
  if (!rec || sess.recordBackend !== "server") throw new Error("the capture could not be saved to the local database");
  if (!rec.file_path && !sess.iqAttached) {
    await attachCapture(rec.id, iqFile());
    sess.iqAttached = true;
  }
  return rec;
}

corrForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const corrected = [...corrClasses.querySelectorAll("input:checked")].map(b => b.value);
  corrError.textContent = "";
  corrSubmit.disabled = true;
  try {
    setOperatorName(corrOperator.value);
    const rec = await ensureEvidence(corrCtx.sess);
    await submitCorrection({
      analysis_id: rec.id,
      start_s: Number(corrStart.value) / 1000, end_s: Number(corrEnd.value) / 1000,
      predicted_labels: corrCtx.predicted, corrected_labels: corrected,
      reason: corrReason.value, model: corrCtx.sess.which,
    });
    corrDialog.close();
    eventsNote.textContent = `Correction submitted by ${operatorName()}: `
      + `${corrCtx.predicted.join(" + ") || "nothing"} → ${corrected.join(" + ")}. `
      + "It waits in History ▸ Human corrections until another person approves it.";
    if (operatorInput) operatorInput.value = operatorName();
  } catch (e) {
    corrError.textContent = e.message.replace(/^Local database correction failed \(\d+\)\. /, "");
    corrSubmit.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Timeline: read the time under the cursor, click or drag to correct
// ---------------------------------------------------------------------------

let consoleGeom = null;
const tlCursor = el("tlCursor"), tlLabel = el("tlLabel"), tlSel = el("tlSel");
let tlDrag = null, enlargeRequested = false;

/** Mouse position -> {x (css px on the canvas), tMs, lane, inside, k}. */
function tlPos(ev) {
  const g = consoleGeom, r = consoleCanvas.getBoundingClientRect();
  const k = r.width / g.cssW;
  const x = (ev.clientX - r.left) / k, y = (ev.clientY - r.top) / k;
  const inside = x >= g.wfX && x <= g.wfX + g.wfW && y >= g.yTop && y <= g.yBot;
  const tMs = Math.min(Math.max((x - g.wfX) / g.wfW, 0), 1) * g.durationMs;
  const li = Math.floor((y - g.yLaneTop) / g.laneH);
  const lane = y >= g.yLaneTop && li >= 0 && li < g.lanes.length ? g.lanes[li] : null;
  return { x, y, tMs, lane, inside, k };
}

const msToCss = ms => (consoleGeom.wfX + ms / consoleGeom.durationMs * consoleGeom.wfW)
  * (consoleCanvas.getBoundingClientRect().width / consoleGeom.cssW);

/** Shade [aMs, bMs] over the plot (a drag in progress, or a hovered suggestion). */
function showSpan(aMs, bMs) {
  if (!consoleGeom) return;
  const k = consoleCanvas.getBoundingClientRect().width / consoleGeom.cssW;
  const x0 = msToCss(Math.min(aMs, bMs)), x1 = msToCss(Math.max(aMs, bMs));
  Object.assign(tlSel.style, { left: `${x0}px`, width: `${Math.max(x1 - x0, 2)}px`,
    top: `${consoleGeom.yTop * k}px`, height: `${(consoleGeom.yBot - consoleGeom.yTop) * k}px` });
  tlSel.hidden = false;
}

function hideCursor() { tlCursor.hidden = true; if (!tlDrag) tlSel.hidden = true; }

consoleCanvas.addEventListener("pointermove", (ev) => {
  if (!session || !consoleGeom) return;
  const p = tlPos(ev);
  if (!p.inside && !tlDrag) { hideCursor(); consoleCanvas.style.cursor = ""; return; }
  consoleCanvas.style.cursor = "crosshair";
  Object.assign(tlCursor.style, { left: `${p.x * p.k}px`, top: `${consoleGeom.yTop * p.k}px`,
    height: `${(consoleGeom.yBot - consoleGeom.yTop) * p.k}px` });
  tlLabel.textContent = tlDrag
    ? `${Math.min(tlDrag.tMs, p.tMs).toFixed(2)}–${Math.max(tlDrag.tMs, p.tMs).toFixed(2)} ms`
    : `${p.tMs.toFixed(2)} ms${p.lane ? ` · ${p.lane}` : ""} · click to correct`;
  // The label rides next to the pointer, so it is in view whichever panel
  // (waterfall or lanes) the operator is looking at; flips left near the edge.
  tlLabel.style.top = `${Math.max((p.y - consoleGeom.yTop) * p.k - 30, 0)}px`;
  const flip = p.x > consoleGeom.wfX + consoleGeom.wfW * 0.7;
  tlLabel.style.left = flip ? "auto" : "8px";
  tlLabel.style.right = flip ? "8px" : "auto";
  tlCursor.hidden = false;
  if (tlDrag) showSpan(tlDrag.tMs, p.tMs);
});

consoleCanvas.addEventListener("pointerleave", () => { if (!tlDrag) hideCursor(); });

consoleCanvas.addEventListener("pointerdown", (ev) => {
  if (!session || !consoleGeom || ev.button !== 0) return;
  const p = tlPos(ev);
  if (!p.inside) return;
  tlDrag = { x: p.x, tMs: p.tMs, lane: p.lane };
  consoleCanvas.setPointerCapture(ev.pointerId);
});

consoleCanvas.addEventListener("pointerup", (ev) => {
  if (!tlDrag) return;
  const start = tlDrag;
  tlDrag = null;
  tlSel.hidden = true;
  const p = tlPos(ev);
  const events = session.events || [];
  const suggestions = session.suggestions?.items || [];
  const durationS = session.capture.re.length / FS;
  if (Math.abs(p.x - start.x) > 4) {
    // A dragged stretch: whatever the model reported anywhere inside it.
    const a = Math.min(start.tMs, p.tMs) / 1000, b = Math.max(start.tMs, p.tMs) / 1000;
    startCorrection({ predicted: detectedIn(events, a, b), startS: a, endS: b,
                      missed: false, title: "Correct this stretch" });
    return;
  }
  const s = spanAt(start.tMs / 1000, { suggestions, events, durationS });
  let suggested = s.suggested;
  // Clicked inside an empty stretch of one class's lane: the operator is
  // pointing at a signal of THAT class the model did not report.
  if (!suggested && start.lane && start.lane !== "NOISE_FLOOR" && !s.predicted.includes(start.lane)) {
    suggested = [...s.predicted, start.lane];
  }
  startCorrection({ predicted: s.predicted, startS: s.startS, endS: s.endS, suggested,
                    reason: s.reason || "", missed: !s.predicted.length,
                    title: s.from === "suggestion" ? `Review: ${s.text}` : null });
});

// Clicks on the time plot are for correcting; the zoom view stays on the
// spectrum strip and the ⤢ Enlarge button.
consoleCanvas.addEventListener("click", (ev) => {
  if (enlargeRequested) { enlargeRequested = false; return; }
  if (consoleGeom && tlPos(ev).inside) ev.stopPropagation();
});

el("consoleEnlarge").addEventListener("click", () => {
  if (!consoleCanvas.classList.contains("zoomable")) return;
  enlargeRequested = true;
  consoleCanvas.dispatchEvent(new MouseEvent("click", { bubbles: true }));
});

async function startCorrection(opts) {
  if (!(await supportsCorrections())) {
    eventsNote.textContent = "Corrections are stored in the local database: start the page with "
      + "scripts/serve_local.py (web/start_demo.bat).";
    return;
  }
  openCorrection(opts);
}

// ---------------------------------------------------------------------------
// Recommended corrections
// ---------------------------------------------------------------------------

const suggestBlock = el("suggestBlock"), suggestNote = el("suggestNote"), suggestList = el("suggestList");
const SUGGEST_SHOWN = 12;
const KIND_LABEL = { missed: "MISSED", false_alarm: "FALSE ALARM", wrong_class: "MIXED UP", uncertain: "UNSURE" };

function renderSuggestions() {
  const { source, items } = session.suggestions;
  const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  suggestBlock.hidden = false;
  if (source === "truth") {
    suggestNote.textContent = items.length
      ? `${items.length} stretch${items.length === 1 ? "" : "es"} where the model disagrees with the ground truth `
        + "(the dashed boxes). Review opens the correction form pre-filled; you check it and submit. "
        + "A dashed box marks when an emitter was switched on: a pulsed radar is silent between pulses, "
        + "so a short \"miss\" inside a radar box can be correct."
      : "The model agrees with the ground truth everywhere in this capture. Nothing to correct.";
  } else {
    suggestNote.textContent = items.length
      ? "No ground truth for this capture, so nothing can say the model is wrong. These are the "
        + "detections it was least sure about (below 40%): check them against the waterfall."
      : "No ground truth for this capture, and every detection is confident (40% or more).";
  }
  suggestList.innerHTML = items.slice(0, SUGGEST_SHOWN).map((s, i) =>
    `<div class="sugg" data-sugg="${i}"><span class="sugg-kind ${s.kind}">${KIND_LABEL[s.kind]}</span>`
    + `<span class="sugg-time">${(s.startS * 1000).toFixed(2)}–${(s.endS * 1000).toFixed(2)} ms</span>`
    + `<span class="sugg-text">${esc(s.text)}</span>`
    + `<button class="mini" data-review-sugg="${i}">Review</button></div>`).join("")
    + (items.length > SUGGEST_SHOWN ? `<div class="note">+${items.length - SUGGEST_SHOWN} more, shorter or later in the capture</div>` : "");
}

suggestList.addEventListener("mouseover", (ev) => {
  const row = ev.target.closest("[data-sugg]");
  if (!row || !session?.suggestions) return;
  const s = session.suggestions.items[Number(row.dataset.sugg)];
  showSpan(s.startS * 1000, s.endS * 1000);
});
suggestList.addEventListener("mouseleave", () => { if (!tlDrag) tlSel.hidden = true; });

suggestList.addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-review-sugg]");
  if (!btn || !session?.suggestions) return;
  const s = session.suggestions.items[Number(btn.dataset.reviewSugg)];
  startCorrection({ predicted: s.predicted, startS: s.startS, endS: s.endS, suggested: s.suggested,
                    reason: s.reason, missed: !s.predicted.length, title: `Review: ${s.text}` });
});

// ---------------------------------------------------------------------------
// Receiver: live streaming from a simulated SDR or a replayed recording
// ---------------------------------------------------------------------------

const rxLight = el("rxLight"), rxState = el("rxState"), rxSourceSel = el("rxSource");
const rxDwellSel = el("rxDwell"), rxConnectBtn = el("rxConnect"), rxStopBtn = el("rxStop");
const rxReadouts = el("rxReadouts"), rxNote = el("rxNote"), rxFileInput = el("rxFileInput");
const rxLog = el("rxLog");

/** One line in the receiver's event log (newest last, last 8 kept). */
function rxLogLine(text) {
  const t = new Date().toLocaleTimeString([], { hour12: false });
  const line = document.createElement("div");
  line.textContent = `[${t}] ${text}`;
  rxLog.append(line);
  while (rxLog.childElementCount > 8) rxLog.firstElementChild.remove();
  rxLog.hidden = false;
}

// null when disconnected; otherwise the running stream's state.
let rx = null;

function rxSetState(state, text) {
  rxLight.className = `rx-light ${state}`;
  rxState.textContent = text;
}

const fmtHz = hz => hz == null ? "—" : hz >= 1e9 ? `${(hz / 1e9).toFixed(3)} GHz` : `${(hz / 1e6).toFixed(3)} MHz`;

const fmtDb = x => (x == null || !Number.isFinite(x) ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(1)} dB`);
const fmtClock = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function rxPaint() {
  if (!rx) return;
  const d = rx.info;
  const wallS = (performance.now() - rx.startedAt) / 1000;
  const sim = d.simulated ? " (sim)" : "";
  const cells = [
    ["Hardware", d.serial ? `${d.hardware} · ${d.serial}` : d.hardware],
    ["Link", d.link],
    ["Centre freq", d.centreHz == null ? "not recorded" : fmtHz(d.centreHz) + sim],
    ["Sample rate", `${(d.sampleRate / 1e6).toFixed(1)} MS/s`],
    ["Gain", d.gainDb == null ? "n/a" : `${d.gainDb} dB${sim}`],
    ["Uptime", rx.streaming ? fmtClock(wallS) : "—"],
    ["Samples in", rx.samples.toLocaleString()],
    ["Signal level", `${fmtDb(rx.level?.powerDb)} avg · ${fmtDb(rx.level?.peakDb)} peak`],
    ["Dwells", String(rx.dwells)],
    ["Stored", String(rx.stored)],
    ["Last dwell", rx.lastProcS == null ? "—" : `${rx.lastProcS.toFixed(2)} s to classify`],
  ];
  if (d.kind === "file") {
    const { doneS, totalS } = rx.source.progress;
    cells.push(["Progress", `${(doneS * 1000).toFixed(1)} / ${(totalS * 1000).toFixed(1)} ms`]);
  } else {
    // Share of air time actually examined: seconds of IQ classified per
    // second of wall clock. The rest went by unexamined between dwells.
    cells.push(["Coverage", wallS > 0 ? `${Math.min(100, rx.airS / wallS * 100).toFixed(1)}% of air time` : "—"]);
  }
  rxReadouts.innerHTML = cells.map(([k, v]) =>
    `<div><span class="rx-k">${k}</span><span class="rx-v">${v}</span></div>`).join("");
}

async function rxStart(source) {
  rx = {
    source, info: source.describe(), dwells: 0, stored: 0, airS: 0, samples: 0, level: null,
    lastProcS: null, prevClasses: null, startedAt: performance.now(), stopReason: null,
    streaming: false, timer: null, abort: new AbortController(),
  };
  rxLog.replaceChildren();
  synthBtn.disabled = uploadBtn.disabled = true;
  rxConnectBtn.disabled = true;
  rxStopBtn.disabled = false;
  rxSourceSel.disabled = rxDwellSel.disabled = true;
  rxSetState("connecting", "RECEIVER · CONNECTING");
  rxPaint();
  const me = rx;
  await source.connect(step => { if (rx === me) rxLogLine(step); });
  if (rx !== me || me.stopReason) return;       // Stop pressed mid-handshake
  me.streaming = true;
  me.startedAt = performance.now();
  // Ticks between dwells too, so the panel visibly runs while a dwell is
  // still being classified.
  me.timer = setInterval(rxPaint, 500);
  logEvent("receiver_connect", { source: rx.info.name, hardware: rx.info.hardware,
    dwell_ms: Number(rxDwellSel.value) * 1000, model: modelSel.value, hop: Number(hopSel.value) });
  rxSetState("on", `RECEIVER · STREAMING · ${rx.info.name}${rx.info.simulated ? " (SIMULATED)" : ""}`);
  rxNote.textContent = rx.info.kind === "sim"
    ? "Each dwell is a fresh slice of simulated air. Change the scenario case or SNR below at any "
      + "time and the next dwell follows. A dwell is stored only when what is detected changes."
    : "Streaming the recording dwell by dwell, every sample in order. A dwell is stored only when "
      + "what is detected changes.";
  rxPaint();
  rxLoop();
}

async function rxLoop() {
  const me = rx;
  while (rx === me && !me.stopReason) {
    let block;
    try {
      block = await me.source.next();
    } catch (e) {
      rxStop(`source error: ${e.message}`);
      return;
    }
    if (!block) { rxStop("end of recording"); return; }
    if (me.stopReason) break;

    const elapsed = await analyze(block.re, block.im, { ...block, live: true, signal: me.abort.signal });
    if (rx !== me || me.stopReason) return;
    if (elapsed == null) { rxStop("classification failed"); return; }
    me.dwells++;
    me.airS += block.re.length / FS;
    me.samples += block.re.length;
    me.level = dwellLevel(block.re, block.im);
    me.lastProcS = elapsed;

    const resolved = resolveSession(session.result, smoothingChoice === "Smoothed");
    const classes = [...new Set(resolved.emitterEvents.flatMap(e => e.classes))];
    if (shouldStore(me.prevClasses, classes)) {
      me.stored++;
      rxLogLine(`Dwell ${block.index + 1}: ${classes.join(" + ")} detected, stored`);
      const bytes = toInterleavedF32(block.re, block.im);
      session.storing = store(new File([bytes.buffer], `dwell-${String(block.index + 1).padStart(5, "0")}.f32`));
      await session.storing;
    }
    me.prevClasses = classes;
    rxPaint();
    // Let the page breathe between dwells (clicks, Stop, redraws).
    await new Promise(r => setTimeout(r, 30));
  }
}

function rxStop(reason = "stopped by operator") {
  if (!rx) return;
  const me = rx;
  me.stopReason = reason;
  me.abort.abort();          // cut the dwell being classified short
  clearInterval(me.timer);
  me.source.disconnect();
  rxLogLine(`Stream closed: ${reason}`);
  logEvent("receiver_disconnect", { source: me.info.name, reason, dwells: me.dwells,
    stored: me.stored, seconds: +((performance.now() - me.startedAt) / 1000).toFixed(1) });
  rxPaint();
  rx = null;
  const ended = reason === "end of recording";
  rxSetState(ended ? "done" : reason.startsWith("stopped") ? "off" : "error",
    ended ? "RECEIVER · END OF RECORDING" : reason.startsWith("stopped")
      ? "RECEIVER · DISCONNECTED" : `RECEIVER · ERROR (${reason})`);
  rxNote.textContent = `${me.dwells} dwells classified, ${me.stored} stored. ${reason[0].toUpperCase()}${reason.slice(1)}.`;
  rxConnectBtn.disabled = false;
  rxStopBtn.disabled = true;
  rxSourceSel.disabled = rxDwellSel.disabled = false;
  synthBtn.disabled = uploadBtn.disabled = false;
}

rxConnectBtn.addEventListener("click", () => {
  const dwellS = Number(rxDwellSel.value);
  if (rxSourceSel.value === "file") { rxFileInput.click(); return; }
  rxStart(new SimulatedReceiver({
    dwellS,
    getSettings: () => ({ caseName: caseSel.value, snrDb: Number(snrSel.value) }),
    getLibrary: () => loadCivilianLibrary("./data"),
  }));
});

rxFileInput.addEventListener("change", async () => {
  const files = [...(rxFileInput.files || [])];
  rxFileInput.value = "";
  if (!files.length) return;
  try {
    const rec = await readUpload(files);
    rec.note = rec.warnings.length ? `${rec.note} · WARNING: ${rec.warnings.join(" ")}` : rec.note;
    rxStart(new FileReplay({ rec, dwellS: Number(rxDwellSel.value) }));
  } catch (e) {
    rxSetState("error", "RECEIVER · COULD NOT OPEN FILE");
    rxNote.textContent = e.message;
  }
});

rxStopBtn.addEventListener("click", () => rxStop());

modelSel.addEventListener("change", async () => {
  if (!session || rx) return;
  await analyze(session.capture.re, session.capture.im, {
    source: session.source, caseNote: session.caseNote,
    truth: session.truth, snrDb: session.snrDb,
  });
});

hopSel.addEventListener("change", async () => {
  if (!session || rx) return;
  await analyze(session.capture.re, session.capture.im, {
    source: session.source, caseNote: session.caseNote,
    truth: session.truth, snrDb: session.snrDb,
  });
});

printBtn.addEventListener("click", async () => {
  // A real .pdf when the library is reachable; the print dialog (which also
  // produces a PDF, via the print stylesheet) when it is not, so an offline
  // demo still has a way to export.
  if (!lastRecord) { window.print(); return; }
  printBtn.disabled = true;
  try {
    await buildSingle(lastRecord, { classes: CLASSES });
    logEvent("report_export", { format: "pdf", scope: "single" },
      { target_type: "analysis", target_id: lastRecord.id });
    statusEl.textContent += "  Report downloaded.";
  } catch (e) {
    statusEl.textContent = `PDF library unavailable (${e.message}) — using the print dialog.`;
    window.print();
  } finally {
    printBtn.disabled = false;
  }
});

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
// Storing analyses
// ---------------------------------------------------------------------------

// The file that produced the session being analysed, so the record can name
// it (and upload it, when that is switched on). Cleared once consumed: a
// later scenario run must not inherit the previous upload's name.
let pendingFile = null;
let lastRecord = null;

/** Save the analysis of `sess` (the current session by default). Returns the
 *  saved record, or null if it could not be stored; remembers it on the
 *  session so a later correction knows which capture it is about. */
async function store(file, sess = session) {
  try {
    const resolved = resolveSession(sess.result, smoothingChoice === "Smoothed");
    // A SigMF pair arrives as [meta, data]: the record describes the data.
    const primary = Array.isArray(file)
      ? file.find(f => !/\.sigmf-meta$/i.test(f.name)) || file[0] : file;
    const record = buildRecord(sess, resolved, {
      classes: CLASSES,
      fileName: primary?.name || null,
      fileBytes: primary?.size ?? null,
    });
    // Which model produced it, down to the retrained version: a result that
    // cannot be traced to its model cannot be trusted or re-checked later.
    if (sess.which === "single" && activeSingleVersion) record.model = `single:${activeSingleVersion}`;
    const { record: saved, backend, warning } = await saveAnalysis(record, file);
    sess.record = saved;
    sess.recordBackend = backend;
    lastRecord = saved;
    printBtn.disabled = false;
    if (warning) statusEl.textContent += `  Storage: ${warning}`;
    else statusEl.textContent += {
      server: saved.file_path ? "  Saved to the local database, with its raw IQ." : "  Saved to the local database.",
      supabase: "  Saved to Supabase.",
    }[backend] || "  Saved to this browser.";
  } catch (e) {
    // Never throw into the analyse path: the analysis on screen is valid
    // whether or not it could be written down.
    statusEl.textContent += `  Not stored: ${e.message}`;
    console.error(e);
    return null;
  }
  return sess.record;
}

// ---------------------------------------------------------------------------
// Page switching + the other four pages
// ---------------------------------------------------------------------------

const winSlider = el("winSlider"), winReadout = el("winReadout");
const probsBox = el("probsBox"), winMetaBox = el("winMetaBox");
const attnCanvas = el("attnCanvas"), breakdownCanvas = el("breakdownCanvas");
let currentPage = "replay";
const chatLogPromise = openChatLog();
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
  if (page === "history") renderHistory();
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
  registerZoom(attnCanvas, t => drawAttention(t, session, idx));
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
  el("summaryBox").innerHTML = summaryHtml(perfData);
  box.innerHTML = scorecardHtml(perfData);
  drawPerClassRecall(el("recallBarCanvas"), perfData);
  registerZoom(el("recallBarCanvas"), t => drawPerClassRecall(t, perfData));
  animateBreakdown(breakdownCanvas, perfData);
  // The zoom redraws the finished chart rather than replaying the animation:
  // someone who clicked to look closely wants the figure, not the intro.
  registerZoom(breakdownCanvas, t => drawBreakdown(t, perfData));
  el("breakdownTable").innerHTML = breakdownTableHtml(perfData);
  el("denseQamBox").innerHTML = denseQamHtml(perfData);

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
  renderRetrain();
}

// ---------------------------------------------------------------------------
// Model page: retraining from human feedback
// ---------------------------------------------------------------------------

const rtReason = el("rtReason"), rtOverride = el("rtOverride"), rtOperator = el("rtOperator");
const rtMsg = el("rtMsg"), rtJob = el("rtJob"), rtJobHead = el("rtJobHead"), rtLog = el("rtLog");
const rtModels = el("rtModels"), rtActive = el("rtActive");
let rtPolling = null;
const escH = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/** The "Single" option names the model it will actually run. */
function labelSingleOption(active) {
  activeSingleVersion = active ? active.version : null;
  const opt = modelSel.querySelector('option[value="single"]');
  if (opt) opt.textContent = active ? `Single, retrained ${active.version}` : "Single, best_model.pt";
}

function recallCell(m, cls) {
  const g = m.metrics?.gate;
  const b = g?.before?.recall?.[cls], a = g?.after?.recall?.[cls];
  if (b == null || a == null) return "—";
  const d = (a - b) * 100;
  return `${(b * 100).toFixed(1)} → <strong>${(a * 100).toFixed(1)}</strong> <span class="${d < -0.05 ? "down" : d > 0.05 ? "up" : ""}">(${d >= 0 ? "+" : ""}${d.toFixed(1)})</span>`;
}

async function renderRetrain() {
  const block = el("retrainBlock");
  await detectServer();
  if (readConfig().backend !== "server") { block.hidden = true; return; }
  block.hidden = false;
  rtOperator.value = operatorName();
  const st = await renderTrigger(el("modelTrigger"));
  el("rtOverrideRow").hidden = !!st?.recommended;
  try {
    const [models, jobs] = await Promise.all([listModels(), listJobs()]);
    const active = models.find(m => m.status === "active") || null;
    labelSingleOption(active);
    rtActive.innerHTML = active
      ? `Single model in use: <strong>${escH(active.version)}</strong> (retrained, approved by ${escH(active.approved_by)}). `
        + `<button class="mini" id="rtRollback">↺ Roll back to shipped model</button>`
      : "Single model in use: <strong>shipped best_model.pt</strong>. The 5-model ensemble (the competition submission) is never changed by retraining.";
    rtModels.innerHTML = models.length ? `<table><thead><tr><th>Version</th><th>Trained on</th><th>Exam</th>`
      + `<th>LFM_RADAR</th><th>FHSS</th><th>JAMMING</th><th>Noise false alarms</th><th>Gate</th><th>Status</th><th></th></tr></thead><tbody>`
      + models.map(m => {
        const g = m.metrics?.gate || {}, tr = m.metrics?.train || {};
        const fa = g.before?.noise_false_alarm != null
          ? `${(g.before.noise_false_alarm * 100).toFixed(1)} → <strong>${(g.after.noise_false_alarm * 100).toFixed(1)}</strong>` : "—";
        const rules = (g.rules || []).map(r => `<li class="${r.met ? "met" : "unmet"}"><span>${r.met ? "✔" : "✖"}</span> ${escH(r.text)}</li>`).join("");
        const actions = m.status === "candidate"
          ? (g.passed ? `<button class="mini" data-model="approve" data-v="${escH(m.version)}">✔ Approve &amp; activate</button> ` : "")
            + `<button class="mini" data-model="reject" data-v="${escH(m.version)}">✖ Reject</button>` : "";
        return `<tr><td><strong>${escH(m.version)}</strong><div class="note">${escH(new Date(m.created_at).toLocaleString())}</div></td>`
          + `<td>${(tr.corrections_used || []).length} corrections (${tr.correction_windows} windows) + ${tr.replay_windows} replay`
          + `<div class="note">${tr.epochs} epoch(s), ${tr.seconds} s</div></td>`
          + `<td><span class="pill ${g.kind === "dataset" ? "approved" : "pending"}">${g.kind === "dataset" ? "real test split" : "synthetic"}</span></td>`
          + `<td>${recallCell(m, "LFM_RADAR")}</td><td>${recallCell(m, "FHSS")}</td><td>${recallCell(m, "JAMMING")}</td><td>${fa}</td>`
          + `<td><details><summary><span class="pill ${g.passed ? "approved" : "rejected"}">${g.passed ? "PASSED" : "FAILED"}</span></summary>`
          + `<ul class="checklist">${rules}</ul><div class="note">${escH(g.set || "")}</div></details></td>`
          + `<td><span class="pill ${m.status === "active" ? "approved" : m.status === "candidate" ? "pending" : "rejected"}">${m.status}</span>`
          + (m.approved_by ? `<div class="note">by ${escH(m.approved_by)}</div>` : "") + `</td>`
          + `<td class="row-actions">${actions}</td></tr>`;
      }).join("") + `</tbody></table>`
      : `<div class="note">No retrained versions yet.</div>`;
    const running = jobs.find(j => j.status === "queued" || j.status === "running");
    if (running && !rtPolling) pollJob(running.id);
    else if (!running && jobs[0] && !rtPolling) showJob(await getJob(jobs[0].id));
  } catch (e) {
    rtModels.textContent = `Could not read model versions: ${e.message}`;
  }
}

function showJob(job) {
  rtJob.hidden = false;
  const state = { queued: "QUEUED", running: "RUNNING", succeeded: "FINISHED", failed: "FAILED" }[job.status];
  rtJobHead.innerHTML = `<span class="pill ${job.status === "failed" ? "rejected" : job.status === "succeeded" ? "approved" : "pending"}">${state}</span> `
    + `Retrain started by <strong>${escH(job.started_by)}</strong>${job.override ? " (override)" : ""}: “${escH(job.reason)}”`
    + (job.model_version ? ` → candidate <strong>${escH(job.model_version)}</strong>` : "");
  // Library chatter (export progress, warnings) hidden; the run's own lines stay.
  rtLog.textContent = (job.log || "").replace(
    /^.*(torchvision|UserWarning|Warning:|torch\.onnx|\[torch\.onnx\]|return cls|rename_mapping|warnings\.warn).*$\n?/gm, "");
  rtLog.scrollTop = rtLog.scrollHeight;
}

function pollJob(id) {
  clearInterval(rtPolling);
  const tick = async () => {
    try {
      const job = await getJob(id);
      showJob(job);
      if (job.status === "succeeded" || job.status === "failed") {
        clearInterval(rtPolling);
        rtPolling = null;
        el("rtStart").disabled = false;
        await renderRetrain();
        renderAudit();
      }
    } catch (e) { rtMsg.textContent = e.message; }
  };
  rtPolling = setInterval(tick, 2000);
  tick();
}

el("rtStart").addEventListener("click", async () => {
  setOperatorName(rtOperator.value);
  rtMsg.textContent = "Checking the training packages and starting…";
  el("rtStart").disabled = true;
  try {
    const job = await startRetrain(rtReason.value, rtOverride.checked);
    rtMsg.textContent = "Retraining started. It runs in the background; this panel follows it.";
    rtReason.value = "";
    rtOverride.checked = false;
    pollJob(job.id);
  } catch (e) {
    rtMsg.textContent = e.message.replace(/^Local database retrain start failed \(\d+\)\. /, "");
    el("rtStart").disabled = false;
  }
});

el("retrainBlock").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-model], #rtRollback");
  if (!btn) return;
  setOperatorName(rtOperator.value);
  rtMsg.textContent = "";
  try {
    if (btn.id === "rtRollback") {
      const note = window.prompt(`Roll back to the shipped model as ${operatorName()}? Reason:`, "");
      if (note === null) return;
      await rollbackModel(note);
      rtMsg.textContent = "Rolled back: the shipped best_model.pt is in use again.";
    } else {
      const decision = btn.dataset.model;
      const note = window.prompt(decision === "approve"
        ? `Approve and activate ${btn.dataset.v} as ${operatorName()}? Optional note:`
        : `Reject ${btn.dataset.v} as ${operatorName()}. Why? (required)`, "");
      if (note === null) return;
      await reviewModel(btn.dataset.v, decision, note);
      rtMsg.textContent = decision === "approve"
        ? `${btn.dataset.v} is now the single model. Choose Model: Single on RF Replay to use it.`
        : `${btn.dataset.v} rejected.`;
    }
    modelCache.delete("single");          // the next "Single" analysis loads the newly served file
    await renderRetrain();
    renderAudit();
  } catch (e) {
    rtMsg.textContent = e.message.replace(/^Local database [a-z ]+ failed \(\d+\)\. /, "");
  }
});

// Checked ONCE at startup, not per question: a slow/absent Ollama should not
// add its own timeout to every message once we already know the answer. If
// it starts up mid-session it is picked up on the next page load.
const ollamaCheck = ollamaAvailable().catch(() => false);

Promise.all([chatLogPromise, ollamaCheck]).then(([chatLog, hasOllama]) => initAssistant({
  chatLog, listRecords: listAnalyses,
  logEl: el("chatLog"), formEl: el("chatForm"), inputEl: el("chatInput"),
  suggestEl: el("chatSuggest"), clearBtn: el("chatClear"), noteEl: el("chatNote"),
  rewriteFn: hasOllama ? ollamaRewrite : null,
})).catch(e => console.error("Assistant failed to start:", e));

// Pays Ollama's cold-load cost (seconds, sometimes tens of seconds) during
// page load instead of on whoever's first question -- see ollama.js:warmUp.
ollamaCheck.then(has => { if (has) ollamaWarmUp(); });

init();


// ---------------------------------------------------------------------------
// History page: stored analyses, the dashboard over them, and where they live
// ---------------------------------------------------------------------------

const histStatus = el("histStatus"), histSummary = el("histSummary");
const histClassBars = el("histClassBars"), histSnrBars = el("histSnrBars");
const histDayBars = el("histDayBars"), histTable = el("histTable");
const histTier = el("histTier"), histClassSel = el("histClass");
const histSource = el("histSource"), histQuery = el("histQuery");
const histFrom = el("histFrom"), histTo = el("histTo");
const histBackendNote = el("histBackendNote");

let histRecords = [];

/** Filter dropdowns are built from what is actually stored, not from the full
 *  class list: offering a filter that can only ever return nothing reads as a
 *  broken page. */
function fillFilters(records) {
  const keep = (sel, values) => {
    const chosen = sel.value;
    const first = sel.options[0];
    sel.innerHTML = "";
    sel.appendChild(first);
    for (const v of values) {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      sel.appendChild(o);
    }
    sel.value = values.includes(chosen) ? chosen : "";
  };
  keep(histTier, [...new Set(records.map(r => r.verdict))].sort());
  keep(histClassSel, [...new Set(records.flatMap(r => r.classes_detected || []))].sort());
}

function paintHistory() {
  const filtered = applyFilters(histRecords, {
    tier: histTier.value, cls: histClassSel.value,
    source: histSource.value, query: histQuery.value,
    from: histFrom.value, to: histTo.value,
  });
  const s = summarise(filtered);
  histSummary.innerHTML = histSummaryHtml(s);
  histClassBars.innerHTML = barsHtml(Object.entries(s.byClass),
    { empty: "No class has been detected in a stored capture yet." });
  histSnrBars.innerHTML = barsHtml(s.snr.map(b => [b.label, b.count]),
    { empty: "No capture carries a known SNR.", sort: false });
  histDayBars.innerHTML = barsHtml(s.byDay, { empty: "Nothing stored yet.", sort: false });
  histTable.innerHTML = tableHtml(filtered, { canDelete: canDelete() });
  paintReportBuilder(filtered.length);
  histStatus.textContent = filtered.length === histRecords.length
    ? `${histRecords.length} stored`
    : `${filtered.length} of ${histRecords.length} stored`;
}

async function renderHistory() {
  await detectServer();
  const cfg = readConfig();
  histBackendNote.textContent = cfg.backend === "server"
    ? `Stored in the local NEXA database (${cfg.server.db}) on this machine, with uploaded raw IQ `
      + "beside it. Nothing leaves this machine, and every change is written to the audit trail below."
    : usingSupabase(cfg)
      ? `Every analysis is stored in ${cfg.url} and read back from it.`
      : "No local server and no Supabase key, so analyses are kept in this browser only. "
        + "Start the page with scripts/serve_local.py (web/start_demo.bat) to use the local database.";
  renderAudit();
  renderCorrections();
  renderTrigger(el("histTrigger"));
  histStatus.textContent = "Loading…";
  try {
    const { records, warning } = await listAnalyses();
    histRecords = records;
    fillFilters(histRecords);
    paintHistory();
    if (warning) histBackendNote.textContent = warning;
  } catch (e) {
    histStatus.textContent = `Could not read stored analyses: ${e.message}`;
  }
}

for (const node of [histTier, histClassSel, histSource, histFrom, histTo]) {
  node.addEventListener("change", paintHistory);
}
histQuery.addEventListener("input", paintHistory);
el("histRefresh").addEventListener("click", renderHistory);

// ---------------------------------------------------------------------------
// Report builder: filters (above) -> sections -> format -> file
// ---------------------------------------------------------------------------

const rbMsg = el("rbMsg"), rbBanner = el("rbBanner"), rbCount = el("rbCount");

function currentFilters() {
  return { tier: histTier.value, cls: histClassSel.value, source: histSource.value,
           query: histQuery.value, from: histFrom.value, to: histTo.value };
}

/** "tier Hostile · from 2026-09-25" -- written into the report, so a reader
 *  knows it is a selection and not everything. */
function describeFilters(f) {
  const parts = [];
  if (f.tier) parts.push(`verdict ${f.tier}`);
  if (f.cls) parts.push(`class ${f.cls}`);
  if (f.source) parts.push(`source ${f.source}`);
  if (f.query) parts.push(`search "${f.query}"`);
  if (f.from) parts.push(`from ${f.from}`);
  if (f.to) parts.push(`to ${f.to}`);
  return parts.join(" · ");
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

el("rbBuild").addEventListener("click", async () => {
  const filters = currentFilters();
  const records = applyFilters(histRecords, filters);
  const sections = [...document.querySelectorAll("#rbSections input:checked")].map(b => b.value);
  const format = document.querySelector("input[name=rbFormat]:checked").value;
  if (!records.length) { rbMsg.textContent = "No captures match the filters above."; return; }
  if (["pdf", "csv", "xlsx", "json"].includes(format) && !sections.length) {
    rbMsg.textContent = "Tick at least one section."; return;
  }
  rbMsg.textContent = "Building…";
  el("rbBuild").disabled = true;
  try {
    const banner = rbBanner.value.trim();
    if (format === "pdf") {
      const ids = new Set(records.map(r => r.id));
      const server = readConfig().backend === "server";
      const corrections = server ? (await listCorrections()).filter(c => ids.has(c.analysis_id)) : [];
      const targets = new Set([...ids, ...corrections.map(c => c.id)]);
      const audit = server ? (await listAudit(1000)).filter(e => targets.has(e.target_id)).reverse() : [];
      const reportId = await buildReport({
        records, summary: summarise(records), corrections, audit, modelCard, sections, banner,
        generatedBy: operatorName(), filters: describeFilters(filters),
      });
      logEvent("report_export", { format: "pdf", scope: "report builder", captures: records.length,
        sections, classification: banner, report_id: reportId });
      rbMsg.textContent = `${reportId}.pdf downloaded (${records.length} captures).`;
    } else {
      const { blob, filename } = await exportReport({
        ids: records.map(r => r.id), sections, format, banner, filters: describeFilters(filters),
      });
      saveBlob(blob, filename);
      rbMsg.textContent = `${filename} downloaded (${(blob.size / 1024).toFixed(0)} KB, ${records.length} captures). `
        + "The export is recorded in the audit trail.";
    }
    await renderAudit();
  } catch (e) {
    rbMsg.textContent = `Report failed: ${e.message}`;
  } finally {
    el("rbBuild").disabled = false;
  }
});

// Formats the server builds need the local server; say so instead of failing.
function paintReportBuilder(nFiltered) {
  const server = readConfig().backend === "server";
  for (const r of document.querySelectorAll("input[name=rbFormat]")) {
    const needsServer = r.value !== "pdf";
    r.disabled = needsServer && !server;
    if (r.disabled && r.checked) document.querySelector("input[name=rbFormat][value=pdf]").checked = true;
  }
  el("rbServerNote").hidden = server;
  rbCount.textContent = `${nFiltered} capture${nFiltered === 1 ? "" : "s"}`;
}

// ---------------------------------------------------------------------------
// Audit trail (History page): who did what, and proof nobody edited it since
// ---------------------------------------------------------------------------

const auditBlock = el("auditBlock"), auditTable = el("auditTable");
const auditStatus = el("auditStatus"), operatorInput = el("operatorInput");
operatorInput.value = operatorName();
operatorInput.addEventListener("change", () => {
  setOperatorName(operatorInput.value);
  operatorInput.value = operatorName();
});

const AUDIT_LABEL = {
  "db.create": "Database created",
  "analysis.create": "Capture analysed",
  "analysis.delete": "Capture deleted",
  "analysis.clear": "History cleared",
  "iq.store": "Raw IQ stored",
  "client.report_export": "Report exported",
  "correction.create": "Correction submitted",
  "correction.approve": "Correction approved",
  "correction.reject": "Correction rejected",
  "retrain.start": "Retraining started",
  "retrain.finish": "Retraining finished",
  "retrain.fail": "Retraining failed",
  "model.approve": "Model approved & activated",
  "model.reject": "Model rejected",
  "model.rollback": "Model rolled back",
  "report.export": "Report exported",
  "client.receiver_connect": "Receiver connected",
  "client.receiver_disconnect": "Receiver disconnected",
};

function auditSummary(e) {
  const d = e.details || {};
  switch (e.action) {
    case "analysis.create":
      return `${d.verdict || "?"} · ${(d.classes_detected || []).join(", ") || "nothing detected"}`
        + (d.file_name ? ` · ${d.file_name}` : "");
    case "analysis.delete": return d.deleted?.file_name || d.deleted?.case_note || "";
    case "analysis.clear": return `${(d.deleted_ids || []).length} deleted, ${d.kept_with_corrections || 0} kept`;
    case "iq.store": return `${d.name} · ${(d.bytes / 1024).toFixed(0)} KB · sha256 ${String(d.sha256).slice(0, 12)}…`;
    case "client.report_export": return `${(d.format || "").toUpperCase()} · ${d.scope || ""}`
      + (d.captures ? ` · ${d.captures} captures` : "");
    case "correction.create":
      return `${(d.predicted || []).join(" + ") || "nothing"} → ${(d.corrected || []).join(" + ")} · “${d.reason}”`;
    case "correction.approve":
    case "correction.reject":
      return `submitted by ${d.submitted_by} · ${(d.corrected || []).join(" + ")}${d.note ? ` · “${d.note}”` : ""}`;
    case "retrain.start":
      return `“${d.reason}”${d.override ? " · OVERRIDE (rules not all met)" : ""} · correction rate ${(d.correction_rate * 100).toFixed(1)}%`;
    case "retrain.finish":
      return `candidate ${d.candidate} · gate ${d.gate_passed ? "passed" : "FAILED"}`;
    case "retrain.fail":
      return d.error || "";
    case "model.approve":
    case "model.reject":
    case "model.rollback":
      return `${e.target_id}${d.note ? ` · “${d.note}”` : ""}`;
    case "report.export":
      return `${(d.format || "").toUpperCase()} · ${d.captures} captures · ${d.classification || ""} · sha256 ${String(d.sha256 || "").slice(0, 12)}…`;
    case "client.receiver_connect":
      return `${d.source} · dwell ${d.dwell_ms} ms · ${d.model} model · hop ${d.hop}`;
    case "client.receiver_disconnect":
      return `${d.source} · ${d.reason} · ${d.dwells} dwells, ${d.stored} stored · ${d.seconds} s`;
    default: return "";
  }
}

async function renderAudit() {
  const cfg = readConfig();
  auditBlock.hidden = cfg.backend !== "server";
  if (auditBlock.hidden) return;
  try {
    const rows = await listAudit(100);
    const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    auditTable.innerHTML = `<table><thead><tr><th>#</th><th>Time (local)</th><th>Who</th><th>Action</th>`
      + `<th>Details</th><th>Hash</th></tr></thead><tbody>`
      + rows.map(e => `<tr><td>${e.seq}</td><td>${esc(new Date(e.ts).toLocaleString())}</td>`
        + `<td>${esc(e.actor)}<div class="note">${esc(e.client || "")}</div></td>`
        + `<td>${esc(AUDIT_LABEL[e.action] || e.action)}</td><td style="min-width:260px;">${esc(auditSummary(e))}</td>`
        + `<td><code title="${esc(e.hash)}">${esc(e.hash.slice(0, 10))}…</code></td></tr>`).join("")
      + `</tbody></table>`;
  } catch (e) {
    auditTable.textContent = `Could not read the audit trail: ${e.message}`;
  }
}

// ---------------------------------------------------------------------------
// Retraining trigger: the gauge and the checklist (History and Model pages)
// ---------------------------------------------------------------------------

/** Draws the trigger into `box`. Returns the status (or null without the
 *  local server, when the box is hidden). */
async function renderTrigger(box) {
  const block = box.closest(".block");
  if (readConfig().backend !== "server") { block.hidden = true; return null; }
  block.hidden = false;
  try {
    const st = await retrainStatus();
    const scaleMax = Math.max(st.rules.max_rate * 2, st.rate * 1.1);
    const pct = x => `${Math.min(100, x / scaleMax * 100).toFixed(1)}%`;
    const state = st.recommended ? "go" : st.triggered ? "warn" : "ok";
    box.innerHTML =
      `<div class="gauge-head"><span class="gauge-state ${state}">${st.recommended ? "RETRAIN RECOMMENDED"
        : st.triggered ? "TRIGGERED" : "NOT TRIGGERED"}</span><span>${st.summary}</span></div>`
      + `<div class="gauge" role="img" aria-label="Correction rate ${(st.rate * 100).toFixed(1)} percent, threshold ${(st.rules.max_rate * 100).toFixed(0)} percent">`
      + `<div class="gauge-fill ${st.rate > st.rules.max_rate ? "over" : ""}" style="width:${pct(st.rate)}"></div>`
      + `<div class="gauge-mark" style="left:${pct(st.rules.max_rate)}"><span>${(st.rules.max_rate * 100).toFixed(0)}% trigger</span></div></div>`
      + `<div class="note">Correction rate, last ${st.rules.window_days} days: <strong>${(st.rate * 100).toFixed(1)}%</strong> `
      + `(${st.corrected} of ${st.analysed} analysed captures had a correction that was not rejected).</div>`
      + `<ul class="checklist">${st.conditions.map(c =>
        `<li class="${c.met ? "met" : "unmet"}"><span aria-hidden="true">${c.met ? "✔" : "✖"}</span> ${c.text}</li>`).join("")}</ul>`;
    return st;
  } catch (e) {
    box.textContent = `Could not read the retraining trigger: ${e.message}`;
    return null;
  }
}

const corrBlock = el("corrBlock"), corrTable = el("corrTable"), corrStats = el("corrStats");
const corrStatusSel = el("corrStatusSel"), corrMsg = el("corrMsg");
let corrRows = [];

async function renderCorrections() {
  corrBlock.hidden = readConfig().backend !== "server";
  if (corrBlock.hidden) return;
  const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  try {
    const [rows, s] = await Promise.all([listCorrections({ status: corrStatusSel.value }), correctionStats()]);
    corrRows = rows;
    const perClass = Object.entries(s.approved_per_class).filter(([, n]) => n)
      .map(([c, n]) => `${c} ${n}`).join(" · ") || "none yet";
    corrStats.textContent = `${s.pending} waiting for review · ${s.approved} approved · ${s.rejected} rejected. `
      + `Approved labels ready for retraining: ${perClass}.`;
    const byId = new Map(histRecords.map(r => [r.id, r]));
    corrTable.innerHTML = rows.length ? `<table><thead><tr><th>Submitted</th><th>Capture · span</th>`
      + `<th>Model said → human says</th><th>Reason</th><th>By</th><th>Status</th><th></th></tr></thead><tbody>`
      + rows.map(c => {
        const cap = byId.get(c.analysis_id);
        const span = `${(c.start_s * 1000).toFixed(1)}–${(c.end_s * 1000).toFixed(1)} ms`;
        const status = c.status === "pending" ? `<span class="pill pending">pending</span>`
          : `<span class="pill ${c.status}">${c.status}</span><div class="note">by ${esc(c.reviewed_by)}${c.review_note ? `: “${esc(c.review_note)}”` : ""}</div>`;
        const actions = c.status === "pending"
          ? `<button class="mini" data-review="approve" data-id="${esc(c.id)}">✔ Approve</button> `
            + `<button class="mini" data-review="reject" data-id="${esc(c.id)}">✖ Reject</button>` : "";
        return `<tr><td>${esc(new Date(c.created_at).toLocaleString())}</td>`
          + `<td>${esc(cap?.file_name || cap?.case_note?.slice(0, 40) || c.analysis_id.slice(0, 8))}<div class="note">${span}</div></td>`
          + `<td><s>${esc(c.predicted_labels.join(" + ") || "nothing")}</s> → <strong>${esc(c.corrected_labels.join(" + "))}</strong></td>`
          + `<td>${esc(c.reason)}</td><td>${esc(c.operator)}</td><td>${status}</td><td class="row-actions">${actions}</td></tr>`;
      }).join("") + `</tbody></table>`
      : `<div class="note">No ${corrStatusSel.value || ""} corrections.</div>`;
  } catch (e) {
    corrTable.textContent = `Could not read corrections: ${e.message}`;
  }
}

corrStatusSel.addEventListener("change", renderCorrections);

corrTable.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-review]");
  if (!btn) return;
  const decision = btn.dataset.review;
  const note = window.prompt(decision === "approve"
    ? `Approve as ${operatorName()}? Optional note:`
    : `Reject as ${operatorName()}. Why? (required)`, "");
  if (note === null) return;
  corrMsg.textContent = "";
  try {
    await reviewCorrection(btn.dataset.id, decision, note);
    corrMsg.textContent = `Correction ${decision === "approve" ? "approved" : "rejected"} by ${operatorName()}.`;
    await renderCorrections();
    await renderAudit();
    await renderTrigger(el("histTrigger"));
  } catch (e) {
    corrMsg.textContent = e.message.replace(/^Local database review failed \(\d+\)\. /, "");
  }
});

el("auditVerify").addEventListener("click", async () => {
  auditStatus.textContent = "Checking every entry…";
  auditStatus.className = "note";
  try {
    const v = await verifyAudit();
    auditStatus.textContent = v.ok
      ? `✔ Intact: all ${v.entries} entries match their hashes; nothing has been edited or removed.`
      : `✖ TAMPERED: entry #${v.broken_at} ${v.reason}. Treat everything from #${v.broken_at} on as untrusted.`;
    auditStatus.className = v.ok ? "note audit-ok" : "note audit-bad";
  } catch (e) {
    auditStatus.textContent = `Verify failed: ${e.message}`;
  }
});

// One listener on the table rather than per row: the table is rebuilt on
// every filter keystroke, and per-row listeners would leak with it.
histTable.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const record = histRecords.find(r => r.id === btn.dataset.id);
  if (!record) return;
  if (btn.dataset.act === "pdf") {
    histStatus.textContent = "Building PDF…";
    try {
      await buildSingle(record, { classes: CLASSES });
      logEvent("report_export", { format: "pdf", scope: "single" },
        { target_type: "analysis", target_id: record.id });
      histStatus.textContent = "PDF downloaded.";
    } catch (e) {
      histStatus.textContent = `PDF failed: ${e.message}`;
    }
    return;
  }
  if (btn.dataset.act === "delete") {
    if (!confirm("Delete this stored analysis? This cannot be undone.")) return;
    try {
      await deleteAnalysis(record.id);
      await renderHistory();   // also redraws the audit trail
    } catch (e) {
      histStatus.textContent = `Delete failed: ${e.message}`;
    }
  }
});
