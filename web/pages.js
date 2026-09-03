// Signal Analysis, Model and Performance, ported from
// src/ui/pages/{signal_analysis,model_page,performance}.py.
//
// Each keeps the provenance rule its Python counterpart enforces: MODEL
// values wear tier colours, MEASURED values wear INSTRUMENT/TEXT_DIM, and
// the caveats that separate the two are copied across rather than trimmed.

import { TIER_COLOR, TIER_OF, estimateSnrDb, tierOfClasses, THRESHOLDS } from "./analysis.js";
import { CLASSES, FS, WINDOW_LEN } from "./model.js";

const PANEL = "#FFFFFF";
const BG = "#F7F8F5";
const GRID = "#DFE3D9";
// Provenance sits on grey so it reads as apparatus rather than as a result:
// every other block on the page is a white PANEL carrying numbers.
const PANEL_MUTED = "#EFF1EC";
const TEXT = "#121C27";
const TEXT_DIM = "#5F6B72";
const INSTRUMENT = "#42505C";
// src/ui/palette.py:MONO_STACK, but quoted with SINGLE quotes.
//
// The family names must be quoted in CSS, and this string is interpolated
// into style="..." attributes. With double quotes the first one terminated
// the attribute, so the browser parsed the remainder as junk attributes
// (jetbrains="", mono",="") and dropped the whole declaration -- every
// monospace element on the site silently lost its font, colour, background
// and padding. Single quotes are equally valid in CSS and survive the
// attribute.
const MONO = "'JetBrains Mono','Cascadia Mono',Consolas,'DejaVu Sans Mono',monospace";
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
      // 112px, not the Gradio original's 96px: NOISE_FLOOR renders 100px
      // wide in this stack and overflowed its box, colliding with the bar
      // beside it. flex-shrink:0 stops the flex row squeezing it back.
      `<span style="width:112px;flex:0 0 auto;font-family:${MONO};font-size:12px;color:${text};">${cls}</span>` +
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
    `independent probabilities &middot; multi-label &middot; these do not sum to 100%</div>` +
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
    `OUTPUT         sigmoid, multi-label, independent per class<br>` +
    `POOLING        energy-gated attention<br>` +
    `SAMPLING       SNR-weighted, 10^(-SNR/20)<br>` +
    `RUNTIME        onnxruntime-web (WASM), exported from the .pt checkpoint<br>` +
    `THRESHOLDS     per class<br>${thresholds}<br><br>` +
    `<span style="color:${TEXT_DIM};">Read from the checkpoint at build time, not hardcoded. ` +
    `Describes what is running. Not a claim that this architecture is the best performing.</span></div>`;
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
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">` +
      `${m.balanced_accuracy === undefined ? "n/a" : (m.balanced_accuracy * 100).toFixed(1) + "%"}</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${(m.precision * 100).toFixed(1)}%</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${(m["f1-score"] * 100).toFixed(1)}%</td>` +
      `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${m.support}</td></tr>`;
  }).join("");

  // The ensemble's own judged-class figures, shown SEPARATELY rather than
  // merged into the table above -- the two come from different models. The
  // table is src.evaluate's output, which defaults to the single checkpoint;
  // this is scripts/train_ensemble.py's. Putting both in one column with no
  // way to tell which was which is the provenance error this page exists to
  // avoid.
  let ensembleBlock = "";
  const ens = perf.ensemble_scorecard;
  if (ens && ens.ensemble) {
    const cells = Object.entries(ens.ensemble).map(([cls, r]) => {
      const prec = ens.ensemble_precision?.[cls];
      const pass = r * 100 >= bar;
      return `<tr><td style="font-family:${MONO};font-weight:600;">${cls}</td>` +
        `<td style="text-align:right;font-family:${MONO};font-weight:600;color:${pass ? "#0F766E" : "#C1121F"};">${(r * 100).toFixed(1)}%</td>` +
        `<td style="text-align:right;font-family:${MONO};color:${TEXT_DIM};">${prec === undefined ? "n/a" : (prec * 100).toFixed(1) + "%"}</td></tr>`;
    }).join("");
    ensembleBlock =
      `<div style="margin-top:16px;padding-top:12px;border-top:1px solid ${GRID};">` +
      `<div style="color:${TEXT_DIM};font-size:11px;margin-bottom:6px;">` +
      `${ens.n_models}-MODEL ENSEMBLE, judged classes only, from evals/ensemble_scorecard.json. ` +
      `This is what the team submits; the table above is a different model.</div>` +
      `<table style="width:100%;border-collapse:collapse;font-size:12px;">` +
      `<thead><tr><th style="text-align:left;">Class</th><th style="text-align:right;">Recall</th>` +
      `<th style="text-align:right;">Precision</th></tr></thead><tbody>${cells}</tbody></table></div>`;
  }

  return `<div style="color:${TEXT_DIM};font-size:11px;line-height:1.5;margin-bottom:8px;">` +
    `Recall first, then balanced accuracy, then precision and F1 for transparency. Judged ` +
    `classes are marked; the rest are mandatory to classify but are not measured against the ` +
    `pass mark. Every figure is per window, ungated and unsmoothed, so the RF Replay smoothing ` +
    `toggle, the NOISE_FLOOR gate and the event hold never reach this page.</div>` +
    `<table style="width:100%;border-collapse:collapse;font-size:12px;">` +
    `<thead><tr><th style="text-align:left;">Class</th><th style="text-align:right;">Recall</th>` +
    `<th style="text-align:right;">Bal. acc.</th>` +
    `<th style="text-align:right;">Precision</th><th style="text-align:right;">F1</th>` +
    `<th style="text-align:right;">Support</th></tr></thead><tbody>${rows}</tbody></table>` +
    ensembleBlock +
    `<div style="color:${TEXT_DIM};font-size:11px;margin-top:8px;border-top:1px solid ${GRID};padding-top:6px;">` +
    `Source: ${perf.scorecard_source ?? "evals/scorecard.json"}. ` +
    `Pass mark is ${bar.toFixed(0)}% recall on the judged classes.</div>`;
}

/** Dense-QAM order resolution: the ONE place on this page where a figure is
 * event-level rather than per-window, so it is fenced off and labelled.
 *
 * Why it exists: the scorecard's 16QAM and 64QAM rows are each measuring
 * "dense QAM detected, split by a coin flip". A single 512-sample window
 * carries ~56 symbols, and the |C42| separation between the two
 * constellations (0.680 vs 0.619) is smaller than the estimator's own spread
 * at that count -- so the split is not something a better model fixes. The
 * resolver pools |C42| ACROSS windows instead, which is why accuracy climbs
 * with the window count below.
 *
 * DISPLAYS what src/measure.py measured (C42_POOLED_ACCURACY), the same way
 * the rest of this page displays src.evaluate's output. Nothing recomputed. */
export function denseQamHtml(perf) {
  const dq = perf.dense_qam;
  if (!dq || !dq.pooled_accuracy) return "";
  const perWindow = perf.scorecard?.dense_qam_recall;

  const counts = Object.keys(dq.pooled_accuracy).map(Number).sort((a, b) => a - b);
  const rows = counts.map((n) => {
    const acc = dq.pooled_accuracy[String(n)] * 100;
    const usable = n >= dq.min_windows;
    return `<tr>` +
      `<td style="font-family:${MONO};color:${usable ? TEXT : TEXT_DIM};">${n} window${n === 1 ? "" : "s"}` +
      `${usable ? "" : ' <span style="font-size:10px;">below minimum, refused</span>'}</td>` +
      `<td style="text-align:right;font-family:${MONO};font-weight:${usable ? 600 : 400};` +
      `color:${usable ? TEXT : TEXT_DIM};">${acc.toFixed(1)}%</td></tr>`;
  }).join("");

  const combined = perWindow
    ? `<div style="color:${TEXT_DIM};font-size:11px;margin-bottom:8px;">` +
      `Per-window, the model's combined dense-QAM recall, meaning did it notice some dense QAM was ` +
      `present regardless of which it named, is ` +
      `<span style="font-family:${MONO};color:${TEXT};">${(perWindow.recall * 100).toFixed(1)}%</span> ` +
      `over ${perWindow.n_evaluated.toLocaleString()} windows. Naming which of the two is the part ` +
      `that needs pooling.</div>`
    : "";

  return `<div style="color:${TEXT_DIM};font-size:11px;line-height:1.5;margin-bottom:8px;">` +
    `Event level, not per window, and not part of the judged benchmark. A 512-sample window ` +
    `carries about 56 symbols, and the separation between the two constellations is smaller ` +
    `than the estimator's own spread at that count, so the split is not something a better ` +
    `model fixes. Pooling across windows is what resolves it.</div>` +
    combined +
    `<table style="width:100%;border-collapse:collapse;font-size:12px;">` +
    `<thead><tr><th style="text-align:left;">Windows pooled</th>` +
    `<th style="text-align:right;">16QAM vs 64QAM accuracy</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>` +
    `<div style="color:${TEXT_DIM};font-size:11px;margin-top:8px;border-top:1px solid ${GRID};padding-top:6px;">` +
    `Source: src/measure.py C42_POOLED_ACCURACY. ` +
    `Measured at SNR &ge; ${dq.min_snr_db} dB, the regime the ${dq.c42_boundary} boundary was ` +
    `calibrated for. Below that the channel pulls |C42| toward zero and the resolver ` +
    `<strong>refuses to decide rather than guessing</strong>, so on low-SNR captures this ` +
    `table does not apply and no order is reported. Fewer than ${dq.min_windows} windows is ` +
    `also refused. Pooling fixes the C42 estimator only; averaging the model's own 16QAM/64QAM ` +
    `probabilities stays at chance however many windows are used, because that error is a bias ` +
    `rather than noise.</div>`;
}

/** The dashboard summary from performance.py:_build_dashboard -- benchmark
 * verdict, the by-category view, and the CEMA criterion.
 *
 * The category table exists because a page showing only the three judged
 * classes reads as though the model knows three things. Civilian is not
 * judged, but "can it tell traffic from interference" is exactly what the
 * CEMA criterion turns on, so it belongs on screen. */
export function summaryHtml(perf) {
  const sc = perf.scorecard;
  if (!sc) return "";
  const perClass = sc.per_class ?? {};
  const bench = sc.benchmark;
  const coarse = sc.coarse_tier;
  const cvj = sc.comms_vs_jamming;
  const pc = v => `${(v * 100).toFixed(1)}%`;

  let out = "";

  if (bench) {
    const ok = bench.passed;
    // Verdict centred over the three class cards it summarises.
    out += `<div style="text-align:center;margin-bottom:14px;">` +
      `<div style="font-size:17px;font-weight:700;color:${ok ? "#0F766E" : "#C1121F"};">` +
      `Benchmark: ${ok ? "PASS" : "FAIL"}</div>` +
      `<div style="font-size:12px;color:${TEXT_DIM};margin-top:2px;">` +
      `Greater than ${(bench.benchmark_recall * 100).toFixed(0)}% recall on all three judged classes` +
      `</div></div>`;

    // One card per judged class, left to right. The metric order inside each
    // card is deliberate and is the same in all three:
    //   recall            PRIMARY. What the rule is written against and what
    //                     the thresholds were calibrated for.
    //   balanced accuracy SUPPORTING. The defensible reading of the
    //                     organiser's word "accuracy": it corrects for the
    //                     class imbalance and cannot be gamed by silence.
    //   precision, F1     TRANSPARENCY. Stated rather than omitted. Low
    //                     precision on the two military classes is the direct
    //                     cost of buying recall margin, and a reader finding
    //                     it unannounced is worse than us naming it.
    //   accuracy          DE-EMPHASISED, with its trivial baseline beside it.
    //                     Included for completeness only; a model predicting
    //                     nothing already scores the baseline.
    const cards = Object.entries(bench.judged_classes).map(([cls, r]) => {
      const m = perClass[cls] ?? {};
      const line = (label, value, opts = {}) =>
        `<div style="display:flex;justify-content:space-between;align-items:baseline;` +
        `padding:3px 0;${opts.rule ? `border-top:1px solid ${GRID};margin-top:5px;padding-top:6px;` : ""}">` +
        `<span style="font-size:11px;color:${TEXT_DIM};">${label}</span>` +
        `<span style="font-family:${MONO};font-size:${opts.size ?? 12}px;` +
        `font-weight:${opts.weight ?? 600};color:${opts.colour ?? TEXT};">${value}</span></div>`;

      return `<div style="flex:1 1 0;min-width:0;background:${PANEL};border:1px solid ${GRID};` +
        `border-radius:6px;padding:12px 14px;">` +
        `<div style="font-family:${MONO};font-size:14px;font-weight:700;color:${TEXT};` +
        `margin-bottom:8px;overflow-wrap:anywhere;">${cls}</div>` +
        line("Recall (primary)", pc(r.recall), { size: 20, colour: r.passed ? "#0F766E" : "#C1121F" }) +
        line("Balanced accuracy", m.balanced_accuracy === undefined ? "n/a" : pc(m.balanced_accuracy)) +
        line("Precision", m.precision === undefined ? "n/a" : pc(m.precision), { rule: true, weight: 400 }) +
        line("F1", m["f1-score"] === undefined ? "n/a" : pc(m["f1-score"]), { weight: 400 }) +
        line("Accuracy", m.accuracy === undefined ? "n/a" : pc(m.accuracy),
              { rule: true, weight: 400, colour: TEXT_DIM }) +
        (m.trivial_accuracy === undefined ? "" :
          `<div style="font-size:10px;color:${TEXT_DIM};line-height:1.4;margin-top:2px;">` +
          `predicting nothing already scores ${pc(m.trivial_accuracy)}</div>`) +
        `</div>`;
    }).join("");

    out += `<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;">${cards}</div>` +
      `<div style="font-size:11px;color:${TEXT_DIM};line-height:1.5;margin-bottom:16px;">` +
      `Recall is the primary figure: it is the metric the rule names and the one the per-class ` +
      `thresholds were calibrated against. Balanced accuracy supports it and is the reading of ` +
      `"accuracy" that survives the class imbalance. Precision and F1 are shown for transparency, ` +
      `not because they are judged. Plain accuracy is included for completeness and should not be ` +
      `leaned on, for the reason printed under each value.</div>`;
  }

  // src/config.py:TIERS, in its declared order.
  const TIERS = {
    Civilian: ["BPSK", "QPSK", "16QAM", "64QAM"],
    Military: ["LFM_RADAR", "FHSS"],
    Hostile: ["JAMMING"],
    Empty: ["NOISE_FLOOR"],
  };
  const rows = [];
  for (const [tier, members] of Object.entries(TIERS)) {
    const present = members.filter(c => (perClass[c]?.support ?? 0) > 0);
    if (!present.length) continue;
    const rec = coarse?.per_tier_recall?.[tier];
    rows.push(
      `<tr><td style="font-weight:600;color:${TIER_COLOR[tier]};padding:4px 12px 4px 0;">${tier}</td>` +
      `<td style="color:${TEXT_DIM};font-family:${MONO};font-size:11px;padding:4px 12px 4px 0;">` +
      present.map(c => `${c} ${(perClass[c].recall * 100).toFixed(0)}%`).join(", ") + `</td>` +
      `<td style="text-align:right;font-family:${MONO};font-weight:600;">` +
      (rec === undefined || rec === null ? "n/a" : pc(rec)) + `</td></tr>`);
  }

  if (cvj) {
    // The competition's "Competitive Advantage" criterion, in the same table
    // performance.py puts it in -- it is a tier-level discrimination result,
    // not a per-class one.
    rows.push(
      `<tr><td style="font-weight:600;padding:4px 12px 4px 0;">CEMA</td>` +
      `<td style="color:${TEXT_DIM};font-family:${MONO};font-size:11px;padding:4px 12px 4px 0;">` +
      `comms vs hostile, jamming recall ${pc(cvj.jamming_recall)}, ` +
      `false alarm ${(cvj.false_alarm_rate * 100).toFixed(2)}%</td>` +
      `<td style="text-align:right;font-family:${MONO};font-weight:700;color:#0F766E;">${pc(cvj.accuracy)}</td></tr>`);
  }

  // Analysis first, then the table, then the source. Same order in every
  // section on this page, so a reader always knows where to look.
  out += `<div style="font-size:14px;font-weight:700;color:${TEXT};margin-bottom:6px;">By category</div>` +
    `<div style="font-size:11px;color:${TEXT_DIM};line-height:1.5;margin-bottom:8px;">` +
    `The two secondary scorecard metrics live here. Coarse-tier accuracy asks whether a window ` +
    `was placed in the right category, so a radar window called FHSS is still correct at this ` +
    `level. Comms versus jamming accuracy is the discrimination the competition weighs most ` +
    `heavily, and its false-alarm rate is the share of civilian traffic wrongly flagged hostile.` +
    `</div>` +
    `<table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:auto;">` +
    `<thead><tr><th style="text-align:left;">Category</th><th style="text-align:left;">Classes</th>` +
    `<th style="text-align:right;">Tier recall</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;

  if (coarse) {
    out += `<div style="font-size:12px;color:${TEXT};margin-top:8px;">` +
      `Coarse-tier accuracy <strong>${pc(coarse.accuracy)}</strong>` +
      (cvj ? ` &nbsp;·&nbsp; comms versus jamming accuracy <strong>${pc(cvj.accuracy)}</strong>` +
              ` over ${cvj.n_evaluated.toLocaleString()} windows` : "") +
      `</div>`;
  }
  out += `<div style="font-size:11px;color:${TEXT_DIM};margin-top:8px;border-top:1px solid ${GRID};padding-top:6px;">` +
    `Source: evals/scorecard.json, written by <code>python -m src.evaluate</code>. ` +
    `Tiers as declared in src/config.py TIERS.</div>`;
  return out;
}

/** performance.py's per-class recall bar chart. Colour carries CLASS (the
 * per-class hues), with the benchmark drawn across. */
export function drawPerClassRecall(canvas, perf) {
  const sc = perf.scorecard;
  if (!sc?.per_class) return false;
  const names = CLASSES.filter(c => (sc.per_class[c]?.support ?? 0) > 0);
  if (!names.length) return false;

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

  const L = 40, R = 12, T = 12, B = 56;
  const w = cssW - L - R, h = cssH - T - B;
  const yOf = v => T + h - v * h;                 // recall is 0..1

  ctx.strokeStyle = GRID; ctx.lineWidth = 0.6; ctx.fillStyle = TEXT_DIM;
  ctx.textAlign = "right";
  for (let p = 0; p <= 100; p += 20) {
    const y = yOf(p / 100);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + w, y); ctx.stroke();
    ctx.fillText(`${p}%`, L - 5, y);
  }

  const slot = w / names.length, bw = Math.min(slot * 0.62, 46);
  names.forEach((cls, i) => {
    const cx = L + slot * (i + 0.5);
    const r = sc.per_class[cls].recall;
    ctx.fillStyle = CLASS_COLOR[cls] ?? TEXT_DIM;
    ctx.fillRect(cx - bw / 2, yOf(r), bw, h - (yOf(r) - T));
    ctx.fillStyle = TEXT_DIM;
    ctx.save();
    ctx.translate(cx, T + h + 8); ctx.rotate(-Math.PI / 6);
    ctx.textAlign = "right";
    ctx.fillText(cls, 0, 0);
    ctx.restore();
  });

  const bench = sc.benchmark?.benchmark_recall;
  if (bench) {
    ctx.save();
    ctx.setLineDash([2, 3]); ctx.strokeStyle = "#e5484d"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(L, yOf(bench)); ctx.lineTo(L + w, yOf(bench)); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#e5484d"; ctx.textAlign = "left";
    ctx.fillText(`${(bench * 100).toFixed(0)}% benchmark`, L + 4, yOf(bench) - 7);
  }
  ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.strokeRect(L, T, w, h);
  return true;
}

/** The numeric table behind the single- vs multi-signal chart. The chart
 * shows the shape; a judge checking a specific figure needs the number. */
export function breakdownTableHtml(perf) {
  const b = perf.breakdown;
  if (!b?.recall) return "";
  const bins = perf.snr_bins;
  const cell = v => v === null || v === undefined ? "n/a" : `${v.toFixed(0)}%`;

  // performance.py's bd_summary opens by stating the model and how the test
  // split divides, so a reader knows how much each half of the table rests
  // on before reading any figure in it.
  const nw = b.n_windows ?? {};
  const head =
    `<div style="font-size:11px;color:${TEXT_DIM};margin-top:10px;">` +
    `<strong style="color:${TEXT};">${perf.breakdown_model ?? perf.model_label}</strong>` +
    (nw.single !== undefined
      ? ` &nbsp;·&nbsp; ${nw.single.toLocaleString()} single-signal / ` +
        `${(nw.multi ?? 0).toLocaleString()} multi-signal windows in the test split`
      : "") + `</div>`;

  let rows = "";
  for (const group of ["single", "multi"]) {
    for (const cls of perf.classes) {
      const series = b.recall[group]?.[cls] ?? {};
      // NOISE_FLOOR never co-occurs, so it has no multi-signal rows; an
      // all-em-dash row reads like a failure rather than an absence.
      if (bins.every(s => series[s] === null || series[s] === undefined)) continue;
      const tot = b.totals[group]?.[cls];
      rows += `<tr><td style="color:${TIER_COLOR[TIER_OF[cls]]};font-weight:600;padding:3px 10px 3px 0;">` +
        `${cls}</td><td style="color:${TEXT_DIM};padding:3px 10px 3px 0;">${group}</td>` +
        bins.map(s => `<td style="text-align:right;font-family:${MONO};padding:3px 8px;">${cell(series[s])}</td>`).join("") +
        `<td style="text-align:right;font-family:${MONO};font-weight:700;padding:3px 0 3px 8px;">${cell(tot)}</td></tr>`;
    }
  }
  return head +
    `<table style="width:100%;border-collapse:collapse;font-size:11px;table-layout:auto;margin-top:6px;">` +
    `<thead><tr><th style="text-align:left;">Class</th><th style="text-align:left;">Group</th>` +
    bins.map(s => `<th style="text-align:right;">${s >= 0 ? "+" : ""}${s} dB</th>`).join("") +
    `<th style="text-align:right;">All</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/** Provenance banner. data/processed holds either a smoke run or the real
 * dataset and they look identical otherwise, so the page says which it
 * measured -- the confusion this project has already been bitten by, where
 * committed smoke scorecards read as real results. */
export function provenanceHtml(perf) {
  const ds = perf.dataset;
  const smoke = ds.total_windows < 5000;
  const colour = smoke ? "#B45309" : TEXT_DIM;
  return `<div style="background:${smoke ? "#FDF6EC" : PANEL_MUTED};border:1px solid ${smoke ? "#B45309" : GRID};` +
    `padding:12px 14px;border-radius:6px;color:${colour};font-size:12px;line-height:1.6;">` +
    `<div style="font-size:14px;font-weight:700;color:${smoke ? "#B45309" : TEXT};margin-bottom:4px;">` +
    `How these numbers were produced</div>` +
    (smoke ? `<strong>These numbers come from a ${ds.total_windows}-window dataset: a smoke run, not the full dataset.</strong><br>` : "") +
    `Measured by the Python evaluation at build time on the held-out test split ` +
    `(${ds.test_windows} of ${ds.total_windows} windows, test_frac ${ds.test_frac}, seed ${ds.seed}). ` +
    `The recall-vs-SNR breakdown below uses ${perf.breakdown_model ?? perf.model_label}; ` +
    `the scorecard is whatever src.evaluate last wrote; see its own source line. ` +
    `Generated ${perf.generated}. ` +
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
      ctx.fillText(`${cls}, ${group}`, L + w + 28, legendY);
      legendY += 11;
    }
  }

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.strokeRect(L, T, w, h);
}
