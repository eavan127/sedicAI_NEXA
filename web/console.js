// The console figure, ported from src/ui/plots.py:console_figure.
//
// Layout, top to bottom, all sharing time on X:
//
//     spectrum | WATERFALL      x = time, y = frequency
//              | DETECTIONS     one lane per class, model output
//              | TIER
//
// The spectrum is the only panel that is a function of frequency, so it
// rotates to the left and shares the waterfall's Y -- the same way a real
// spectrum analyser is laid out, and the same reason the Python version
// gives.
//
// Provenance styling is preserved from src/ui/palette.py: waterfall and
// spectrum are MEASURED (INSTRUMENT grey), detection boxes/lanes/ribbon are
// MODEL (tier colours), truth is dashed outline and never filled.

import { TIER_COLOR, tierOfClasses } from "./analysis.js";
import { CLASSES } from "./model.js";

const PANEL = "#FFFFFF";
const BG = "#F7F8F5";
const GRID = "#DFE3D9";
const TEXT = "#121C27";
const TEXT_DIM = "#5F6B72";
const INSTRUMENT = "#42505C";
const TRUTH_COLOR = "#121C27";
const FONT = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif';

/** Google's turbo colormap (the polynomial approximation) -- palette.py
 * picks turbo for the waterfall as "perceptually better than jet, reads
 * almost identically". */
function turbo(x) {
  x = Math.min(Math.max(x, 0), 1);
  const x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
  const r = 0.13572138 + 4.61539260 * x - 42.66032258 * x2 + 132.13108234 * x3 - 152.94239396 * x4 + 59.28637943 * x5;
  const g = 0.09140261 + 2.19418839 * x + 4.84296658 * x2 - 14.18503333 * x3 + 4.27729857 * x4 + 2.82956604 * x5;
  const b = 0.10667330 + 12.64194608 * x - 60.58204836 * x2 + 110.36276771 * x3 - 89.90310912 * x4 + 27.34824973 * x5;
  return [
    Math.round(255 * Math.min(Math.max(r, 0), 1)),
    Math.round(255 * Math.min(Math.max(g, 0), 1)),
    Math.round(255 * Math.min(Math.max(b, 0), 1)),
  ];
}

function percentile(sortedArr, p) {
  const pos = (p / 100) * (sortedArr.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (pos - lo);
}

/** Nice round tick values covering [lo, hi], roughly `target` of them. */
function ticks(lo, hi, target = 6) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

function dashedRect(ctx, x, y, w, h, color, lw) {
  ctx.save();
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

/**
 * Draws the console figure into `canvas`.
 *
 * spectro: { power, freqs, times, nFreq, nFrames } from analysis.scipyStft
 * spectrum: { freqs, spectrumDb } from analysis.powerSpectrumDb
 * events / tiers: resolved MODEL output
 * truth: [{ className, startS, endS }] or null (scenario captures only)
 */
export function drawConsole(canvas, {
  spectro, spectrum, events, tiers, truth, durationMs, starts, hop, fs,
}) {
  const dpr = window.devicePixelRatio || 1;
  // A hidden or not-yet-laid-out canvas reports zero width (and its parent
  // does too). Fall back to a default rather than skipping the draw, so the
  // figure always exists; main.js re-renders once a real width appears, and
  // the redraw is cheap next to the inference that produced the data.
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 1100;

  const truthClasses = new Set((truth || []).map(s => s.className));
  const lanes = CLASSES.filter(c =>
    c !== "NOISE_FLOOR" && (events.some(e => e.classes.includes(c)) || truthClasses.has(c)));

  // Mirrors the Python gridspec: height_ratios [6.0, lanes*0.52, 0.5] on a
  // figure declared figsize=(13, sum(heights) + 1.2). Deriving the unit from
  // the canvas width (13 units wide) reproduces matplotlib's aspect ratio at
  // any width, instead of a fixed pixel size that squashes the waterfall.
  const UNIT = Math.min(Math.max(cssW / 13, 40), 100);
  const hWaterfall = 6.0 * UNIT;
  const hLanes = Math.max(lanes.length * 0.52, 0.6) * UNIT;
  const hTier = 0.5 * UNIT;
  const HSPACE = 10;
  const padTop = 8, padBottom = 34, padLeft = 62, padRight = 14;

  const cssH = padTop + hWaterfall + HSPACE + hLanes + HSPACE + hTier + padBottom;

  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, cssW, cssH);
  ctx.font = `8px ${FONT}`;
  ctx.textBaseline = "middle";

  // width_ratios [1, 9], wspace 0.02
  const innerW = cssW - padLeft - padRight;
  const WSPACE = 6;
  const specW = (innerW - WSPACE) * 0.1;
  const wfX = padLeft + specW + WSPACE;
  const wfW = innerW - specW - WSPACE;

  const yWfTop = padTop, yWfBot = padTop + hWaterfall;
  const yLaneTop = yWfBot + HSPACE, yLaneBot = yLaneTop + hLanes;
  const yTierTop = yLaneBot + HSPACE, yTierBot = yTierTop + hTier;

  const fMin = spectro.freqs[0] / 1e6, fMax = spectro.freqs[spectro.nFreq - 1] / 1e6;
  const tToX = ms => wfX + (ms / durationMs) * wfW;
  const fToY = mhz => yWfBot - ((mhz - fMin) / (fMax - fMin)) * hWaterfall;

  // ---- waterfall: MEASURED ------------------------------------------------
  const powerDb = new Float64Array(spectro.power.length);
  for (let i = 0; i < spectro.power.length; i++) {
    powerDb[i] = 10 * Math.log10(spectro.power[i] + 1e-20);
  }
  const sorted = Float64Array.from(powerDb).sort();
  const vmin = percentile(sorted, 60), vmax = percentile(sorted, 99.5);
  const vrange = Math.max(vmax - vmin, 1e-9);

  // Rendered at the spectrogram's OWN resolution into an offscreen canvas,
  // then scaled in with drawImage. putImageData was wrong here: it ignores
  // the context transform, so on a devicePixelRatio != 1 display the
  // waterfall landed at device-pixel coordinates while every vector element
  // around it used CSS pixels -- the image ended up both mis-sized and
  // misaligned against the detection lanes below it. drawImage respects the
  // transform, and scaling from native resolution also matches pcolormesh's
  // interpolation more closely than nearest-neighbour sampling did.
  const off = document.createElement("canvas");
  off.width = spectro.nFrames;
  off.height = spectro.nFreq;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(spectro.nFrames, spectro.nFreq);
  for (let k = 0; k < spectro.nFreq; k++) {
    // row 0 of the image is the TOP of the display = highest frequency
    const row = spectro.nFreq - 1 - k;
    for (let t = 0; t < spectro.nFrames; t++) {
      const [r, g, b] = turbo((powerDb[k * spectro.nFrames + t] - vmin) / vrange);
      const o = (row * spectro.nFrames + t) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  ctx.drawImage(off, wfX, yWfTop, wfW, hWaterfall);

  // ---- spectrum, rotated: MEASURED ---------------------------------------
  let sMin = Infinity, sMax = -Infinity;
  for (const v of spectrum.spectrumDb) { if (v < sMin) sMin = v; if (v > sMax) sMax = v; }
  const sRange = Math.max(sMax - sMin, 1e-9);
  // x inverted (invert_xaxis in the Python): high dB toward the waterfall
  const dbToX = db => padLeft + specW - ((db - sMin) / sRange) * specW;

  ctx.beginPath();
  ctx.moveTo(padLeft + specW, fToY(spectrum.freqs[0] / 1e6));
  for (let i = 0; i < spectrum.freqs.length; i++) {
    ctx.lineTo(dbToX(spectrum.spectrumDb[i]), fToY(spectrum.freqs[i] / 1e6));
  }
  ctx.lineTo(padLeft + specW, fToY(spectrum.freqs[spectrum.freqs.length - 1] / 1e6));
  ctx.closePath();
  ctx.fillStyle = "rgba(66, 80, 92, 0.30)";
  ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < spectrum.freqs.length; i++) {
    const x = dbToX(spectrum.spectrumDb[i]), y = fToY(spectrum.freqs[i] / 1e6);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = INSTRUMENT;
  ctx.lineWidth = 1;
  ctx.stroke();

  // frequency axis (shared by spectrum + waterfall)
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.strokeRect(padLeft, yWfTop, specW, hWaterfall);
  ctx.strokeRect(wfX, yWfTop, wfW, hWaterfall);
  ctx.fillStyle = TEXT_DIM;
  ctx.textAlign = "right";
  for (const mhz of ticks(fMin, fMax, 7)) {
    const y = fToY(mhz);
    if (y < yWfTop - 1 || y > yWfBot + 1) continue;
    ctx.fillText(mhz.toFixed(1), padLeft - 5, y);
  }
  ctx.save();
  ctx.translate(13, (yWfTop + yWfBot) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.font = `9px ${FONT}`;
  ctx.fillText("frequency (MHz), baseband", 0, 0);
  ctx.restore();
  ctx.font = `8px ${FONT}`;

  // ---- MODEL overlays on the waterfall: full height, time-bounded --------
  // The classifier has no frequency axis (STFTBranch collapses it), so a box
  // bounded in frequency would assert something the model never computed.
  for (const e of events) {
    const x0 = tToX(e.startUs / 1000), x1 = tToX(e.endUs / 1000);
    ctx.strokeStyle = TIER_COLOR[tierOfClasses(e.classes)];
    ctx.lineWidth = 1.6;
    ctx.globalAlpha = 0.9;
    ctx.strokeRect(x0, yWfTop, Math.max(x1 - x0, 1), hWaterfall);
    ctx.globalAlpha = 1;
  }

  // ---- detection lanes: MODEL --------------------------------------------
  const laneH = lanes.length ? hLanes / lanes.length : hLanes;
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.strokeRect(wfX, yLaneTop, wfW, hLanes);

  for (let i = 0; i < lanes.length; i++) {
    const cls = lanes[i];
    const color = TIER_COLOR[tierOfClasses([cls])];
    const laneTop = yLaneTop + i * laneH;

    for (const e of events) {
      if (!e.classes.includes(cls)) continue;
      const x0 = tToX(e.startUs / 1000), x1 = tToX(e.endUs / 1000);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x0, laneTop + laneH * 0.12, Math.max(x1 - x0, 1), laneH * 0.76);
      ctx.globalAlpha = 1;
      if (e.durationUs / 1000 > durationMs * 0.06) {
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.font = `bold 8px ${FONT}`;
        ctx.fillText(`${Math.round(e.peak[cls] * 100)}%`, (x0 + x1) / 2, laneTop + laneH / 2);
        ctx.font = `8px ${FONT}`;
      }
    }
    // TRUTH into the SAME lane, dashed outline over the filled bar
    for (const seg of (truth || [])) {
      if (seg.className !== cls) continue;
      const x0 = tToX(seg.startS * 1000), x1 = tToX(seg.endS * 1000);
      dashedRect(ctx, x0, laneTop + laneH * 0.04, Math.max(x1 - x0, 1), laneH * 0.92,
                 TRUTH_COLOR, 1.3);
    }

    ctx.fillStyle = TEXT_DIM;
    ctx.textAlign = "right";
    ctx.fillText(cls, wfX - 5, laneTop + laneH / 2);
  }
  ctx.save();
  ctx.translate(padLeft - 46, (yLaneTop + yLaneBot) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillStyle = TEXT_DIM;
  ctx.fillText("detections (model)", 0, 0);
  ctx.restore();

  // ---- tier ribbon: MODEL -------------------------------------------------
  const stepMs = (hop / fs) * 1e3;
  for (let i = 0; i < tiers.length; i++) {
    const tm = (starts[i] / fs) * 1e3;
    ctx.fillStyle = TIER_COLOR[tiers[i]];
    ctx.fillRect(tToX(tm), yTierTop, Math.max(tToX(tm + stepMs) - tToX(tm), 1), hTier);
  }
  ctx.strokeStyle = GRID;
  ctx.strokeRect(wfX, yTierTop, wfW, hTier);
  ctx.fillStyle = TEXT_DIM;
  ctx.textAlign = "right";
  ctx.fillText("tier", wfX - 5, (yTierTop + yTierBot) / 2);

  // ---- time axis ----------------------------------------------------------
  ctx.textAlign = "center";
  ctx.fillStyle = TEXT_DIM;
  for (const ms of ticks(0, durationMs, 8)) {
    const x = tToX(ms);
    if (x < wfX - 1 || x > wfX + wfW + 1) continue;
    ctx.beginPath();
    ctx.moveTo(x, yTierBot);
    ctx.lineTo(x, yTierBot + 4);
    ctx.strokeStyle = GRID;
    ctx.stroke();
    ctx.fillText(ms.toFixed(ms < 10 ? 1 : 0), x, yTierBot + 12);
  }
  ctx.font = `9px ${FONT}`;
  ctx.fillStyle = TEXT_DIM;
  ctx.fillText("time (ms)", wfX + wfW / 2, yTierBot + 25);
}
