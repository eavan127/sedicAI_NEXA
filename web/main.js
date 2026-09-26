import { CASES, buildScenario, caseNeedsLibrary, loadCivilianLibrary } from "./generators.js";
import { classifyCapture, loadModel, CLASSES, FS, WINDOW_LEN } from "./model.js";
import {
  noiseFloorPower, occupancy, powerSpectrumDb, resolveSession, scipyStft,
} from "./analysis.js";
import { drawConsole } from "./console.js";
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
  buildRecord, deleteAnalysis, listAnalyses, readConfig, saveAnalysis, usingSupabase,
} from "./storage.js";
import { applyFilters, barsHtml, summarise, summaryHtml as histSummaryHtml, tableHtml } from "./history.js";
import { buildCombined, buildSingle } from "./report.js";

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
  return which === "ensemble" ? "5-model ensemble average" : "single checkpoint, best_model.pt";
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
    // Stored AFTER render: a storage backend that is slow, full or
    // misconfigured must not delay the picture the operator is waiting for,
    // and must not lose it either -- store() reports its own failures and
    // falls back to the local store.
    store(pendingFile);
    pendingFile = null;
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
      "File contains no complex samples. Expected interleaved float32 I,Q,I,Q,... with at least 2 values.");
    if (raw.length % 2) raw = raw.subarray(0, raw.length - 1);
    const n = raw.length / 2;
    if (n < WINDOW_LEN) throw new Error(
      `Capture is ${n} complex samples; at least ${WINDOW_LEN} are needed for one window.`);
    const re = new Float64Array(n), im = new Float64Array(n);
    for (let i = 0; i < n; i++) { re[i] = raw[2 * i]; im[i] = raw[2 * i + 1]; }
    // truth is scenario-only: never render a TRUTH overlay over data we do
    // not actually have ground truth for (session.py:analyze).
    pendingFile = file;
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

printBtn.addEventListener("click", async () => {
  // A real .pdf when the library is reachable; the print dialog (which also
  // produces a PDF, via the print stylesheet) when it is not, so an offline
  // demo still has a way to export.
  if (!lastRecord) { window.print(); return; }
  printBtn.disabled = true;
  try {
    await buildSingle(lastRecord, { classes: CLASSES });
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

async function store(file) {
  try {
    const resolved = resolveSession(session.result, smoothingChoice === "Smoothed");
    const record = buildRecord(session, resolved, {
      classes: CLASSES,
      fileName: file?.name || null,
      fileBytes: file?.size ?? null,
    });
    const { record: saved, backend, warning } = await saveAnalysis(record, file);
    lastRecord = saved;
    printBtn.disabled = false;
    if (warning) statusEl.textContent += `  Storage: ${warning}`;
    else statusEl.textContent += backend === "supabase"
      ? "  Saved to Supabase." : "  Saved to this browser.";
  } catch (e) {
    // Never throw into the analyse path: the analysis on screen is valid
    // whether or not it could be written down.
    statusEl.textContent += `  Not stored: ${e.message}`;
    console.error(e);
  }
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
  animateBreakdown(breakdownCanvas, perfData);
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
}

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
  });
  const s = summarise(filtered);
  histSummary.innerHTML = histSummaryHtml(s);
  histClassBars.innerHTML = barsHtml(Object.entries(s.byClass),
    { empty: "No class has been detected in a stored capture yet." });
  histSnrBars.innerHTML = barsHtml(s.snr.map(b => [b.label, b.count]),
    { empty: "No capture carries a known SNR.", sort: false });
  histDayBars.innerHTML = barsHtml(s.byDay, { empty: "Nothing stored yet.", sort: false });
  histTable.innerHTML = tableHtml(filtered, { canDelete: !usingSupabase() });
  histStatus.textContent = filtered.length === histRecords.length
    ? `${histRecords.length} stored`
    : `${filtered.length} of ${histRecords.length} stored`;
}

async function renderHistory() {
  const cfg = readConfig();
  histBackendNote.textContent = usingSupabase(cfg)
    ? `Every analysis is stored in ${cfg.url} and read back from it.`
    : "No Supabase key reached this page, so analyses are kept in this browser only. "
      + "See web/supabase/schema.sql for how the key is supplied.";
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

for (const node of [histTier, histClassSel, histSource]) {
  node.addEventListener("change", paintHistory);
}
histQuery.addEventListener("input", paintHistory);
el("histRefresh").addEventListener("click", renderHistory);

el("histPdfAll").addEventListener("click", async () => {
  const filtered = applyFilters(histRecords, {
    tier: histTier.value, cls: histClassSel.value,
    source: histSource.value, query: histQuery.value,
  });
  if (!filtered.length) { histStatus.textContent = "Nothing to report."; return; }
  histStatus.textContent = "Building PDF…";
  try {
    await buildCombined(filtered, summarise(filtered), {
      title: filtered.length === histRecords.length
        ? "All analysed signals" : "Analysed signals (filtered)",
    });
    histStatus.textContent = `${filtered.length} captures exported.`;
  } catch (e) {
    histStatus.textContent = `PDF failed: ${e.message}`;
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
      await renderHistory();
    } catch (e) {
      histStatus.textContent = `Delete failed: ${e.message}`;
    }
  }
});
