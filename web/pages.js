// Signal Analysis, Model and Performance, ported from
// src/ui/pages/{signal_analysis,model_page,performance}.py.
//
// Each keeps the provenance rule its Python counterpart enforces: MODEL
// values wear tier colours, MEASURED values wear INSTRUMENT/TEXT_DIM, and
// the caveats that separate the two are copied across rather than trimmed.

import { TIER_COLOR, estimateSnrDb, tierOfClasses, THRESHOLDS } from "./analysis.js";
import { CLASSES, FS, WINDOW_LEN } from "./model.js";

const PANEL = "#FFFFFF";
const BG = "#F7F8F5";
const GRID = "#DFE3D9";
const TEXT = "#121C27";
const TEXT_DIM = "#5F6B72";
const INSTRUMENT = "#42505C";
const MONO = '"JetBrains Mono", "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace';
const FONT = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif';

// src/ui/palette.py:CLASS_COLOR -- per-class hues, for charts that draw
// every class at once (tier colours give 8 classes only 4 colours).
export const CLASS_COLOR = {
  BPSK: "#1F6FB2", QPSK: "#2AA3A3", "16QAM": "#7A5BC0", "64QAM": "#C2569A",
  LFM_RADAR: "#627143", FHSS: "#B07D2B", JAMMING: "#C1121F", NOISE_FLOOR: "#6B7280",
};

/** src/ui/palette.py:lighten -- mix toward white, so a secondary series
 * separates from its primary without spending a second hue on it. */
export function lighten(hex, amount = 0.45) {
  const n = parseInt(hex.slice(1), 16);
  const mix = c => Math.round(c + (255 - c) * amount);
  return `#${[(n >> 16) & 255, (n >> 8) & 255, n & 255]
    .map(c => mix(c).toString(16).padStart(2, "0")).join("")}`;
}

// ---------------------------------------------------------------------------
// Signal Analysis (signal_analysis.py)
// ---------------------------------------------------------------------------

/** All 8 classes as bars, each marked against its OWN threshold.
 *
 * Bars, not a ranked list: the length carries the magnitude at a glance and
 * makes it visible that the values do NOT sum to 100%. This model is
 * multi-label sigmoid, so QPSK + JAMMING together is a legitimate answer,
 * and a softmax-style ranked list would quietly imply otherwise. */
export function probabilityHtml(result, windowIndex) {
  const base = windowIndex * result.nClasses;

  const row = cls => {
    const p = result.probs[base + CLASSES.indexOf(cls)];
    const hit = p > THRESHOLDS[cls];
    const colour = hit ? TIER_COLOR[tierOfClasses([cls])] : GRID;
    const text = hit ? TEXT : TEXT_DIM;
    return `<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">` +
      `<span style="width:14px;color:${text};font-weight:700;">${hit ? "&#10003;" : "&#9675;"}</span>` +
      `<span style="width:96px;font-family:${MONO};font-size:12px;color:${text};">${cls}</span>` +
      `<span style="flex:1;background:${BG};height:13px;border-radius:2px;overflow:hidden;">` +
      `<span style="display:block;width:${(p * 100).toFixed(1)}%;height:100%;background:${colour};"></span></span>` +
      `<span style="width:42px;text-align:right;font-family:${MONO};font-size:12px;color:${text};">${p.toFixed(2)}</span></div>`;
  };

  const bars = CLASSES.filter(c => c !== "NOISE_FLOOR").map(row).join("");
  const noiseP = result.probs[base + CLASSES.indexOf("NOISE_FLOOR")];
  const quiet = noiseP > THRESHOLDS.NOISE_FLOOR;
  // NOISE_FLOOR separated below the rule as a channel STATE, not an eighth
  // threat class.
  const noiseBlock = `<div style="margin-top:12px;padding-top:10px;border-top:1px solid ${GRID};">` +
    row("NOISE_FLOOR") +
    `<div style="color:${TEXT_DIM};font-size:11px;margin-left:22px;">` +
    `Signal state: ${quiet ? "QUIET / NO SIGNAL" : "ACTIVE"}</div></div>`;

  return `<div style="background:${PANEL};padding:16px;border-radius:6px;">` +
    `<div style="color:${TEXT_DIM};font-size:11px;margin-bottom:10px;">` +
    `independent probabilities &middot; multi-label &mdash; these do not sum to 100%</div>` +
    bars + noiseBlock + `</div>`;
}

export function windowMetadataHtml(session, windowIndex) {
  const { result, capture, source, snrDb } = session;
  const start = result.starts[windowIndex];
  let snr;
  if (source === "scenario" && snrDb !== null && snrDb !== undefined) {
    snr = `${snrDb.toFixed(1)} dB <span style="color:${TEXT_DIM};">KNOWN</span>`;
  } else {
    // MEASURED, and always prefixed `est.` -- the classifier does not
    // produce SNR and this is not a calibrated receiver measurement.
    const v = estimateSnrDb(capture.re, capture.im, start,
                             Math.min(start + WINDOW_LEN, capture.re.length),
                             capture.noisePower);
    snr = `est. ${v.toFixed(1)} dB`;
  }
  return `<div style="font-family:${MONO};color:${TEXT};background:${PANEL};padding:14px;border-radius:6px;margin-top:10px;">` +
    `WINDOW   #${windowIndex + 1} / ${result.nWindows}<br>` +
    `OFFSET   ${(start / FS * 1000).toFixed(3)} ms<br>` +
    `SAMPLES  ${WINDOW_LEN}<br>` +
    `DURATION ${(WINDOW_LEN / FS * 1e6).toFixed(0)} µs<br>` +
    `SNR      ${snr}</div>`;
}

/** plots.py:attention_figure -- I and Q (measured) with the model's
 * attention over them.
 *
 * I and Q separately, not |IQ|: the model's input is a (2, 512) real array
 * of exactly these two traces, so magnitude would display something the
 * classifier never sees, and would discard the phase that distinguishes
 * BPSK from QPSK from QAM.
 *
 * Attention is a per-window softmax, so heights are NOT comparable across
 * windows -- the axis label says so, because the plot cannot. */
export function drawAttention(canvas, session, windowIndex) {
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 700;
  const cssH = 260;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, cssW, cssH);
  ctx.font = `9px ${FONT}`;
  ctx.textBaseline = "middle";

  const { result, capture } = session;
  const start = result.starts[windowIndex];
  const n = Math.min(WINDOW_LEN, capture.re.length - start);
  if (n <= 0) return;

  const L = 52, R = 56, T = 14, B = 34;
  const w = cssW - L - R, h = cssH - T - B;

  let amp = 0;
  for (let i = 0; i < n; i++) {
    amp = Math.max(amp, Math.abs(capture.re[start + i]), Math.abs(capture.im[start + i]));
  }
  amp = amp || 1;
  const xOf = i => L + (i / (n - 1)) * w;
  const yOf = v => T + h / 2 - (v / amp) * (h / 2) * 0.92;

  // attention fill first, so the traces read on top of it
  let aMax = 0;
  for (let i = 0; i < n; i++) aMax = Math.max(aMax, result.attn[windowIndex * WINDOW_LEN + i]);
  aMax = aMax || 1;
  ctx.beginPath();
  ctx.moveTo(L, T + h);
  for (let i = 0; i < n; i++) {
    ctx.lineTo(xOf(i), T + h - (result.attn[windowIndex * WINDOW_LEN + i] / aMax) * h * 0.95);
  }
  ctx.lineTo(L + w, T + h);
  ctx.closePath();
  ctx.fillStyle = "rgba(180, 83, 9, 0.35)";     // tier_color("Military")
  ctx.fill();

  const trace = (get, colour, alpha) => {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xOf(i), y = yOf(get(start + i));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  };
  trace(i => capture.re[i], INSTRUMENT, 1);
  trace(i => capture.im[i], TIER_COLOR.Civilian, 0.85);

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.strokeRect(L, T, w, h);

  ctx.fillStyle = TEXT_DIM;
  ctx.textAlign = "center";
  const durUs = n / FS * 1e6;
  for (let k = 0; k <= 4; k++) {
    const us = (durUs * k) / 4;
    ctx.fillText(us.toFixed(0), L + (w * k) / 4, T + h + 12);
  }
  ctx.fillText("time within window (µs)", L + w / 2, T + h + 26);

  ctx.save();
  ctx.translate(12, T + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("I / Q (measured)", 0, 0);
  ctx.restore();
  ctx.save();
  ctx.translate(cssW - 12, T + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("attention · sums to 1 per window", 0, 0);
  ctx.restore();

  // legend
  ctx.textAlign = "left";
  let lx = L + 8;
  for (const [label, colour] of [["I", INSTRUMENT], ["Q", TIER_COLOR.Civilian]]) {
    ctx.strokeStyle = colour;
    ctx.beginPath(); ctx.moveTo(lx, T + 8); ctx.lineTo(lx + 14, T + 8); ctx.stroke();
    ctx.fillStyle = TEXT_DIM;
    ctx.fillText(label, lx + 18, T + 8);
    lx += 40;
  }
}

// ---------------------------------------------------------------------------
// Model page (model_page.py)
// ---------------------------------------------------------------------------

/** Every number comes from web/data/model_card.json, which web/build.py
 * introspects from the checkpoint actually in results/ -- nothing here is
 * hardcoded, so the page cannot go stale after someone swaps a checkpoint
 * and rebuilds. The page states what is running; it does not claim this
 * architecture is the best-performing one. */
export function modelCardHtml(card, which) {
  const pad = (s, n) => String(s).padEnd(n);
  const branches = Object.entries(card.branches)
    .map(([name, n]) => `           ${pad(name, 14)} ${n.toLocaleString()}`).join("<br>");
  const thresholds = Object.entries(card.thresholds)
    .map(([c, v]) => `           ${pad(c, 14)} ${v}`).join("<br>");
  const arch = which === "ensemble" ? card.architecture : card.member_architecture;
  const total = which === "ensemble" ? card.parameters * 5 : card.parameters;

  return `<div style="font-family:${MONO};background:${PANEL};padding:18px;border-radius:6px;color:${TEXT};line-height:1.8;">` +
    `ARCHITECTURE   ${arch}<br>` +
    `PARAMETERS     ${total.toLocaleString()}` +
    (which === "ensemble" ? ` <span style="color:${TEXT_DIM};">(5 × ${card.parameters.toLocaleString()})</span>` : "") + `<br>` +
    branches + `<br>` +
    `CLASSES        ${card.classes.length}<br>` +
    `           ${card.classes.join(", ")}<br>` +
    `INPUT          (2, ${card.window_len})<br>` +
    `WINDOW         ${(card.window_len / card.fs * 1e6).toFixed(0)} µs @ ${(card.fs / 1e6).toFixed(1)} MHz<br>` +
    `OUTPUT         sigmoid — multi-label, independent per class<br>` +
    `POOLING        energy-gated attention<br>` +
    `SAMPLING       SNR-weighted, 10^(-SNR/20)<br>` +
    `RUNTIME        onnxruntime-web (WASM) — exported from the .pt checkpoint<br>` +
    `THRESHOLDS     per class<br>${thresholds}<br><br>` +
    `<span style="color:${TEXT_DIM};">Read from the checkpoint at build time, not hardcoded. ` +
    `Describes what is running — not a claim that this architecture is the best performing.</span></div>`;
}

// ---------------------------------------------------------------------------
// Performance (performance.py)
// ---------------------------------------------------------------------------

/** The scorecard table. DISPLAYS what the Python evaluation produced -- the
 * numbers come from web/data/performance.json, written by web/build.py from
 * evals/scorecard.json. Nothing is recomputed here, by design: a page that
 * derived its own recall could show a judge numbers that disagree with the
 * official scorecard. */
export function scorecardHtml(perf) {
  if (!perf.scorecard) {
    return `<div style="background:${PANEL};padding:16px;border-radius:6px;color:${TEXT_DIM};">` +
      `No scorecard found. Run <code>python -m src.evaluate</code>, then <code>python web/build.py</code>.</div>`;
  }
  const judged = new Set(perf.judged_classes);
  const bar = perf.benchmark_recall * 100;

  const rows = Object.entries(perf.scorecard.per_class).map(([cls, m]) => {
    const isJudged = judged.has(cls);
    const recall = m.recall * 100;
    const pass = isJudged ? recall >= bar : null;
    const colour = pass === null ? TEXT_DIM : (pass ? "#0F766E" : "#C1121F");
    return `<tr>` +
      `<td style="font-family:${MONO};color:${isJudged ? TEXT : TEXT_DIM};font-weight:${isJudged ? 600 : 400};">` +
      `${cls}${isJudged ? ' <span style="font-size:10px;color:' + TEXT_DIM + ';">judged</span>' : ""}</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${colour};font-weight:${isJudged ? 600 : 400};">${recall.toFixed(1)}%</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${(m.precision * 100).toFixed(1)}%</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${(m["f1-score"] * 100).toFixed(1)}%</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${m.support}</td></tr>`;
  }).join("");

  return `<table style="width:100%;border-collapse:collapse;font-size:12px;">` +
    `<thead><tr><th style="text-align:left;">Class</th><th style="text-align:right;">Recall</th>` +
    `<th style="text-align:right;">Precision</th><th style="text-align:right;">F1</th>` +
    `<th style="text-align:right;">Support</th></tr></thead><tbody>${rows}</tbody></table>` +
    `<div style="color:${TEXT_DIM};font-size:11px;margin-top:8px;">` +
    `Per-window, ungated and unsmoothed. The RF Replay smoothing toggle, the NOISE_FLOOR ` +
    `gate and the event hold never reach this page. Pass mark is ${bar.toFixed(0)}% recall on the ` +
    `judged classes.</div>`;
}

/** Provenance banner. data/processed holds either a smoke run or the real
 * dataset and they look identical otherwise, so the page says which it
 * measured -- the confusion this project has already been bitten by, where
 * committed smoke scorecards read as real results. */
export function provenanceHtml(perf) {
  const ds = perf.dataset;
  const smoke = ds.total_windows < 5000;
  const colour = smoke ? "#B45309" : TEXT_DIM;
  return `<div style="background:${smoke ? "#FDF6EC" : PANEL};border:1px solid ${smoke ? "#B45309" : GRID};` +
    `padding:12px 14px;border-radius:6px;color:${colour};font-size:12px;line-height:1.6;">` +
    (smoke ? `<strong>These numbers come from a ${ds.total_windows}-window dataset — a smoke run, not the full dataset.</strong><br>` : "") +
    `Measured by the Python evaluation at build time on the held-out test split ` +
    `(${ds.test_windows} of ${ds.total_windows} windows, test_frac ${ds.test_frac}, seed ${ds.seed}) ` +
    `using ${perf.model_label}. Generated ${perf.generated}. ` +
    `Rebuild with <code>python -m src.evaluate</code> then <code>python web/build.py</code> after any retrain.</div>`;
}

/** plots the single- vs multi-signal recall sweep (performance.py's
 * _build_breakdown chart).
 *
 * Colour carries CLASS, lightness carries SINGLE vs MULTI: eight classes
 * drawn twice is sixteen lines, and colouring by tier gave those only four
 * colours -- the four civilian classes became indistinguishable and a
 * class's own two curves shared a colour too. */
export function drawBreakdown(canvas, perf) {
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 800;
  const cssH = 380;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, cssW, cssH);
  ctx.font = `9px ${FONT}`;
  ctx.textBaseline = "middle";

  const L = 46, R = 150, T = 14, B = 40;
  const w = cssW - L - R, h = cssH - T - B;
  const bins = perf.snr_bins;
  const xOf = snr => L + ((snr - bins[0]) / (bins[bins.length - 1] - bins[0])) * w;
  const yOf = pct => T + h - (pct / 102) * h;

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 0.6;
  ctx.fillStyle = TEXT_DIM;
  ctx.textAlign = "right";
  for (let pct = 0; pct <= 100; pct += 20) {
    const y = yOf(pct);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + w, y); ctx.stroke();
    ctx.fillText(String(pct), L - 6, y);
  }
  ctx.textAlign = "center";
  for (const snr of bins) ctx.fillText(String(snr), xOf(snr), T + h + 12);
  ctx.fillText("SNR (dB)", L + w / 2, T + h + 28);
  ctx.save();
  ctx.translate(12, T + h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("recall (%)", 0, 0);
  ctx.restore();

  // benchmark line
  const by = yOf(perf.benchmark_recall * 100);
  ctx.save();
  ctx.setLineDash([2, 3]);
  ctx.strokeStyle = "#e5484d";
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(L, by); ctx.lineTo(L + w, by); ctx.stroke();
  ctx.restore();

  let legendY = T + 4;
  ctx.textAlign = "left";
  for (const cls of perf.classes) {
    const base = CLASS_COLOR[cls] ?? TEXT_DIM;
    for (const group of ["single", "multi"]) {
      const colour = group === "single" ? base : lighten(base, 0.45);
      const series = perf.breakdown.recall[group]?.[cls] ?? {};
      const pts = bins.filter(s => series[s] !== null && series[s] !== undefined)
                       .map(s => [xOf(s), yOf(series[s])]);
      if (!pts.length) continue;
      ctx.save();
      if (group === "multi") ctx.setLineDash([4, 3]);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = colour;
      for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); }

      ctx.strokeStyle = colour;
      ctx.save();
      if (group === "multi") ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(L + w + 10, legendY); ctx.lineTo(L + w + 24, legendY); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = TEXT_DIM;
      ctx.fillText(`${cls} — ${group}`, L + w + 28, legendY);
      legendY += 11;
    }
  }

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.strokeRect(L, T, w, h);
}
