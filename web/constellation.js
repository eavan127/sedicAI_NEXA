// The civilian constellation panel, ported from src/ui/plots.py
// (rrc_taps, carrier_offset, recover_symbols, cluster_score,
// constellation_figure) and src/measure.py (constellation_order).
//
// Why this panel exists: the waterfall cannot tell civilian modulations
// apart -- BPSK, QPSK, 16QAM and 64QAM are the same flat wideband smear on
// it at every SNR. Cluster count IS the modulation order (2, 4, 16, 64), so
// this is the one display that carries the distinction.
//
// Everything drawn here is MEASURED -- computed from the capture's own
// samples with no model involvement -- so nothing on it takes a tier colour
// except each column's class-probability text, which is the one MODEL
// element.
//
// PARITY NOTE: every function here is deterministic and matches Python
// exactly EXCEPT cluster_score, which draws from numpy's PCG64 in Python
// (k-means++ seeding and the null resamples). Reproducing that bit stream in
// JS is not practical, so cluster_score uses a local PRNG and agrees
// statistically rather than exactly -- web/test/constellation_check.mjs
// measures that agreement instead of asserting equality. The deterministic
// chain (taps -> matched filter -> carrier offset -> decimation -> |C42|)
// is checked to float precision.

import { fft, makeRng } from "./dsp.js";
import { estimateSnrDb } from "./analysis.js";
import { CLASSES } from "./model.js";

// RadioML 2018.01A is stored at 8 samples per symbol. Named rather than
// inlined because it is the one constant a capture at another rate would
// invalidate: the decimation below would then sample the pulse shape instead
// of the symbol instants, and the constellation would be wrong without
// looking wrong.
export const SAMPLES_PER_SYMBOL = 8;
const RRC_ROLLOFF = 0.35;
const RRC_SPAN_SYMBOLS = 8;

export const CONSTELLATION_ORDER = { BPSK: 2, QPSK: 4, "16QAM": 16, "64QAM": 64 };
export const CIVILIAN = ["BPSK", "QPSK", "16QAM", "64QAM"];

// cluster_score's null resamples the k-means fit, so the statistic needs
// enough points per cluster to say anything at all. Below this the gap
// statistic measures sampling noise, not structure.
const MIN_POINTS_PER_CLUSTER = 8;
const CLUSTER_SCORE_WEAK_FLOOR = 0.07;
const CLUSTER_SCORE_CLEAR_FLOOR = 0.20;

/** Root-raised-cosine taps, unit energy, odd length so they add no delay.
 * The two singular points (t = 0 and t = 1/(4*beta)) are written out
 * separately because the general expression divides by zero there; both
 * branches are the limit of the closed form. */
export function rrcTaps(sps = SAMPLES_PER_SYMBOL, beta = RRC_ROLLOFF, span = RRC_SPAN_SYMBOLS) {
  if ((span * sps) % 2) {
    throw new Error(`rrcTaps needs an odd tap count for a centre tap (no delay): ` +
                     `span (${span}) and sps (${sps}) cannot both be odd`);
  }
  const n = span * sps + 1;
  const taps = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ti = (-span * sps / 2 + i) / sps;
    let v;
    if (Math.abs(ti) < 1e-8) {
      v = 1 - beta + (4 * beta) / Math.PI;
    } else if (Math.abs(Math.abs(ti) - 1 / (4 * beta)) < 1e-8) {
      v = (beta / Math.SQRT2) *
        ((1 + 2 / Math.PI) * Math.sin(Math.PI / (4 * beta)) +
         (1 - 2 / Math.PI) * Math.cos(Math.PI / (4 * beta)));
    } else {
      v = (Math.sin(Math.PI * ti * (1 - beta)) +
           4 * beta * ti * Math.cos(Math.PI * ti * (1 + beta))) /
          (Math.PI * ti * (1 - Math.pow(4 * beta * ti, 2)));
    }
    taps[i] = v;
  }
  let energy = 0;
  for (const v of taps) energy += v * v;
  const norm = Math.sqrt(energy);
  for (let i = 0; i < n; i++) taps[i] /= norm;
  return taps;
}

/** np.convolve(z, taps, mode="same") for a complex z and real taps. */
function convolveSame(re, im, taps) {
  const n = re.length, m = taps.length;
  const offset = Math.floor((n + m - 1 - n) / 2);   // == (m - 1) / 2 for odd m
  const outRe = new Float64Array(n), outIm = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sr = 0, si = 0;
    const k = i + offset;
    const jLo = Math.max(0, k - m + 1), jHi = Math.min(n - 1, k);
    for (let j = jLo; j <= jHi; j++) {
      const t = taps[k - j];
      sr += re[j] * t; si += im[j] * t;
    }
    outRe[i] = sr; outIm[i] = si;
  }
  return { re: outRe, im: outIm };
}

/** Blind estimate of residual carrier offset, in cycles per sample.
 *
 * Raising the signal to the 4th power collapses a QPSK or QAM constellation
 * onto a single tone at 4x the offset, which then shows as an FFT peak. Used
 * for BPSK too: it locks there as well, at the cost of a 90-degree phase
 * ambiguity, harmless because the panel only de-rotates and never labels an
 * axis with an absolute phase.
 *
 * MEASURED -- reads the capture's own samples and fits nothing to an
 * expected constellation, so it cannot manufacture clusters the samples do
 * not contain. */
export function carrierOffset(re, im, order = 4) {
  const n = re.length;
  // An FFT over fewer than a couple of cycles has no meaningful peak to
  // find, so the honest answer is no estimate rather than a spurious one.
  if (n < order * 2) return 0.0;

  // z ** order, by repeated complex multiplication
  const pr = new Float64Array(n), pi = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let ar = re[i], ai = im[i], br = 1, bi = 0;
    for (let k = 0; k < order; k++) {
      const nr = br * ar - bi * ai;
      bi = br * ai + bi * ar;
      br = nr;
    }
    pr[i] = br; pi[i] = bi;
  }
  // n is 512 here (a power of 2), so the radix-2 fft applies directly.
  fft(pr, pi, false);
  let best = -Infinity, k = 0;
  for (let i = 0; i < n; i++) {
    const mag = Math.hypot(pr[i], pi[i]);
    if (mag > best) { best = mag; k = i; }
  }
  if (k >= n / 2) k -= n;             // negative frequencies live upper half
  return k / n / order;
}

/** Symbol points from one raw IQ window.
 *
 * Four operations, none model-derived: unit-power scaling, matched filtering
 * with the RRC receive filter, de-rotation by the estimated carrier offset,
 * and decimation to one sample per symbol at the timing phase whose points
 * have the tightest amplitude spread.
 *
 * Any symbol whose filter support extends past the window edge is DROPPED:
 * mode="same" convolves those against implicit zero padding, and letting
 * them pull toward the origin would misrepresent the constellation. On a
 * 512-sample window at sps=8 with 65 taps this drops 4 per edge, leaving 56
 * of 64.
 *
 * Degenerate windows -- shorter than one symbol, or carrying no power --
 * come back unchanged rather than raising: this feeds a display, and a
 * capture with a silent stretch must render, not crash the page. */
export function recoverSymbols(re, im, sps = SAMPLES_PER_SYMBOL) {
  const n = re.length;
  let power = 0;
  for (let i = 0; i < n; i++) power += re[i] * re[i] + im[i] * im[i];
  power = n ? power / n : 0;
  if (n < sps || power <= 0) {
    return { re: Float64Array.from(re), im: Float64Array.from(im),
             offset: 0, phase: 0, recovered: false };
  }

  const s = 1 / Math.sqrt(power);
  const zr = new Float64Array(n), zi = new Float64Array(n);
  for (let i = 0; i < n; i++) { zr[i] = re[i] * s; zi[i] = im[i] * s; }

  // Matched filter FIRST: the carrier estimate is a 4th-power FFT peak and
  // finds it more reliably once the out-of-band noise is gone.
  const taps = rrcTaps(sps);
  const f = convolveSame(zr, zi, taps);
  const offset = carrierOffset(f.re, f.im);

  const dr = new Float64Array(n), di = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ang = -2 * Math.PI * offset * i;
    const c = Math.cos(ang), sn = Math.sin(ang);
    dr[i] = f.re[i] * c - f.im[i] * sn;
    di[i] = f.re[i] * sn + f.im[i] * c;
  }

  const margin = Math.floor(taps.length / 2);
  const lo = margin, hi = n - 1 - margin;

  let bestPhase = 0, bestScore = -Infinity, bestRe = null, bestIm = null;
  for (let phase = 0; phase < sps; phase++) {
    const pr = [], pi = [];
    for (let i = phase; i < n; i += sps) {
      if (i >= lo && i <= hi) { pr.push(dr[i]); pi.push(di[i]); }
    }
    if (!pr.length) continue;
    // Power over amplitude spread. At the symbol instant the amplitudes take
    // the constellation's own discrete levels; between symbols they smear
    // across the pulse shape, which widens the spread.
    let sumSq = 0, sumAbs = 0;
    const abs = pr.map((v, j) => Math.hypot(v, pi[j]));
    for (let j = 0; j < abs.length; j++) { sumSq += abs[j] * abs[j]; sumAbs += abs[j]; }
    const meanAbs = sumAbs / abs.length;
    let varAbs = 0;
    for (const a of abs) varAbs += (a - meanAbs) * (a - meanAbs);
    varAbs /= abs.length;
    const score = (sumSq / abs.length) / (varAbs + 1e-9);
    if (score > bestScore) {
      bestPhase = phase; bestScore = score;
      bestRe = Float64Array.from(pr); bestIm = Float64Array.from(pi);
    }
  }
  return { re: bestRe ?? new Float64Array(0), im: bestIm ?? new Float64Array(0),
           offset, phase: bestPhase, recovered: true };
}

/** How many decimated symbol points recoverSymbols returns for a
 * non-degenerate window -- NOT window_len / sps, because edge symbols are
 * dropped. Used by the caption so the printed count stays true. */
export function symbolsPerWindow(windowLen, sps = SAMPLES_PER_SYMBOL) {
  const margin = Math.floor(rrcTaps(sps).length / 2);
  let count = 0;
  for (let i = 0; i < windowLen; i += sps) {
    if (i >= margin && i <= windowLen - 1 - margin) count++;
  }
  return count;
}

// --- k-means + gap statistic ------------------------------------------------

function weightedPick(rng, probs) {
  const u = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  return probs.length - 1;
}

/** Deterministic-given-rng Lloyd's algorithm with k-means++ seeding.
 * Multiple restarts guard against one unlucky seeding landing in a bad local
 * optimum; the lowest-inertia restart wins. */
function kmeans(xs, ys, k, rng, nInit, nIter) {
  const n = xs.length;
  let bestInertia = Infinity, bestCx = null, bestCy = null, bestLabels = null;

  for (let init = 0; init < nInit; init++) {
    const cx = new Float64Array(k), cy = new Float64Array(k);
    let idx = Math.floor(rng() * n);
    cx[0] = xs[idx]; cy[0] = ys[idx];
    const d2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      d2[i] = (xs[i] - cx[0]) ** 2 + (ys[i] - cy[0]) ** 2;
    }
    for (let c = 1; c < k; c++) {
      let total = 0;
      for (const v of d2) total += v;
      const probs = new Float64Array(n);
      for (let i = 0; i < n; i++) probs[i] = total > 0 ? d2[i] / total : 1 / n;
      idx = weightedPick(rng, probs);
      cx[c] = xs[idx]; cy[c] = ys[idx];
      for (let i = 0; i < n; i++) {
        d2[i] = Math.min(d2[i], (xs[i] - cx[c]) ** 2 + (ys[i] - cy[c]) ** 2);
      }
    }

    const labels = new Int32Array(n);
    const assign = () => {
      for (let i = 0; i < n; i++) {
        let best = Infinity, bl = 0;
        for (let c = 0; c < k; c++) {
          const dd = (xs[i] - cx[c]) ** 2 + (ys[i] - cy[c]) ** 2;
          if (dd < best) { best = dd; bl = c; }
        }
        labels[i] = bl;
      }
    };
    for (let it = 0; it < nIter; it++) {
      assign();
      const sx = new Float64Array(k), sy = new Float64Array(k), cnt = new Int32Array(k);
      for (let i = 0; i < n; i++) { sx[labels[i]] += xs[i]; sy[labels[i]] += ys[i]; cnt[labels[i]]++; }
      let moved = false;
      for (let c = 0; c < k; c++) {
        if (!cnt[c]) continue;
        const nx = sx[c] / cnt[c], ny = sy[c] / cnt[c];
        if (Math.abs(nx - cx[c]) > 1e-8 || Math.abs(ny - cy[c]) > 1e-8) moved = true;
        cx[c] = nx; cy[c] = ny;
      }
      if (!moved) break;
    }
    assign();
    let inertia = 0;
    for (let i = 0; i < n; i++) {
      inertia += (xs[i] - cx[labels[i]]) ** 2 + (ys[i] - cy[labels[i]]) ** 2;
    }
    if (inertia < bestInertia) {
      bestInertia = inertia;
      bestCx = Float64Array.from(cx); bestCy = Float64Array.from(cy);
      bestLabels = Int32Array.from(labels);
    }
  }
  return { inertia: bestInertia, cx: bestCx, cy: bestCy, labels: bestLabels };
}

/** Are there `order` distinct, roughly equally occupied clusters?
 *
 * MEASURED, never MODEL. Normalise for scale, partition into `order` groups
 * with k-means, then compare the within-cluster tightness against a NULL:
 * the same clustering run on the points' own distance from the data's
 * centroid paired with a uniformly random phase about that centroid. The
 * null matters because k-means always "finds" some separation when asked to
 * cut a continuous blob into `order` pieces -- that is what k-means does to
 * anything, and is not evidence of real clusters.
 *
 * Centring on the data's own centroid (rather than the origin) is what makes
 * this correct for BOTH a ring and an off-origin blob -- see the Python
 * docstring for the two nulls that were tried and rejected.
 *
 * Multiplied by a balance term so `order` clusters holding wildly unequal
 * point counts cannot pass; an empty cluster forces 0. Squashed into [0, 1]
 * at the end. */
export function clusterScore(pointsRe, pointsIm, order = 4, seed = 0, nRef = 15) {
  const n = pointsRe.length;
  if (n < order) return 0.0;
  let m2 = 0;
  for (let i = 0; i < n; i++) m2 += pointsRe[i] ** 2 + pointsIm[i] ** 2;
  const scale = Math.sqrt(m2 / n);
  if (scale === 0) return 0.0;

  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) { xs[i] = pointsRe[i] / scale; ys[i] = pointsIm[i] / scale; }

  const rng = makeRng(seed);
  const actual = kmeans(xs, ys, order, rng, 5, 30);
  const counts = new Int32Array(order);
  for (const l of actual.labels) counts[l]++;
  let minC = Infinity, maxC = 0;
  for (const c of counts) { if (c < minC) minC = c; if (c > maxC) maxC = c; }
  if (minC === 0) return 0.0;

  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  const radii = new Float64Array(n);
  for (let i = 0; i < n; i++) radii[i] = Math.hypot(xs[i] - mx, ys[i] - my);

  let refSum = 0;
  const rx = new Float64Array(n), ry = new Float64Array(n);
  for (let r = 0; r < nRef; r++) {
    for (let i = 0; i < n; i++) {
      const a = rng() * 2 * Math.PI;
      rx[i] = mx + radii[i] * Math.cos(a);
      ry[i] = my + radii[i] * Math.sin(a);
    }
    refSum += kmeans(rx, ry, order, rng, 2, 20).inertia;
  }
  const refWcss = refSum / nRef;

  const gap = Math.log(refWcss + 1e-9) - Math.log(actual.inertia + 1e-9);
  const balance = minC / maxC;
  const score = Math.max(gap, 0) * balance;
  return score / (score + 1);
}

/** Qualitative label for a clusterScore value. Boundaries are inclusive on
 * their lower edge, matching the measurements the floors were drawn from. */
export function clusterScoreBand(score) {
  if (score < CLUSTER_SCORE_WEAK_FLOOR) return "no structure";
  if (score < CLUSTER_SCORE_CLEAR_FLOOR) return "weak";
  return "clear";
}

// --- fourth-order cumulant: 16QAM vs 64QAM ---------------------------------

/** |C42| of a set of recovered points, normalised to unit average power so
 * absolute amplitude (an AGC artefact, not information about the
 * constellation) cannot move the result. Returns null for an empty or
 * zero-power set -- both mean there is nothing here to measure, not a C42
 * of 0. */
export function normalizedC42(re, im) {
  const n = re.length;
  if (n === 0) return null;
  let m2 = 0;
  for (let i = 0; i < n; i++) m2 += re[i] ** 2 + im[i] ** 2;
  m2 /= n;
  if (m2 <= 0) return null;
  const s = 1 / Math.sqrt(m2);
  let m2n = 0, m4n = 0;
  for (let i = 0; i < n; i++) {
    const p2 = (re[i] * s) ** 2 + (im[i] * s) ** 2;
    m2n += p2; m4n += p2 * p2;
  }
  m2n /= n; m4n /= n;
  return Math.abs(m4n - 2.0 * m2n * m2n);
}

/** Resolve 16QAM vs 64QAM by pooling |C42| across windows -- a distinction
 * the classifier cannot make at all (51.4% single-window accuracy, 49.7%
 * even pooled over 64 windows, and BIASED rather than noisy, so averaging
 * the model's own output cannot fix it).
 *
 * Refuses rather than guesses, two independent ways: below `minWindows`
 * there is not enough pooling for the estimator's variance to have shrunk,
 * and below `minSnrDb` the channel is outside the regime the boundary was
 * calibrated for. Either way `decision` is null while the measured values
 * are still returned, so the caller can show what WAS measured and why it
 * refused. */
export function constellationOrder(windows, noisePower, c42cfg) {
  const per = [];
  for (const w of windows) {
    const rec = recoverSymbols(w.re, w.im);
    const c = normalizedC42(rec.re, rec.im);
    if (c !== null) per.push(c);
  }
  if (!per.length) {
    return { decision: null, meanC42: NaN, nWindows: 0, margin: NaN, accuracy: null, snrDb: null };
  }
  const meanC42 = per.reduce((a, b) => a + b, 0) / per.length;
  const margin = Math.abs(meanC42 - c42cfg.boundary);

  const measured = Object.keys(c42cfg.pooled_accuracy).map(Number)
    .filter(k => k <= per.length);
  const accuracy = measured.length
    ? c42cfg.pooled_accuracy[String(Math.max(...measured))] : null;

  // estimate_snr_db over the concatenation of every pooled window
  let total = 0, count = 0;
  for (const w of windows) {
    for (let i = 0; i < w.re.length; i++) { total += w.re[i] ** 2 + w.im[i] ** 2; count++; }
  }
  const flatRe = new Float64Array(1), flatIm = new Float64Array(1);
  const snrDb = count
    ? (() => {
        const signal = Math.max(total / count - noisePower, 1e-20);
        return 10 * Math.log10(signal / Math.max(noisePower, 1e-20));
      })()
    : null;

  const decision = (per.length >= c42cfg.min_windows && snrDb !== null && snrDb >= c42cfg.min_snr_db)
    ? (meanC42 >= c42cfg.boundary ? "16QAM" : "64QAM")
    : null;

  return { decision, meanC42, nWindows: per.length, margin, accuracy, snrDb };
}

// --- window selection (session.py:civilian_windows) -------------------------

/** `count` windows spread evenly across the strongest civilian class's span.
 *
 * Two stages, not to be conflated. WHICH CLASS: of the four civilian
 * classes, whichever's PEAK probability across the capture is highest, and
 * only if that peak clears its own threshold -- the model's own answer, not
 * a quality judgement. WHICH WINDOWS: every window where that class clears
 * threshold, then `count` at evenly spaced POSITIONS in that list.
 *
 * The spacing is even rather than best-first on purpose: showing the
 * tightest-clustering windows would be choosing the picture that most looks
 * like the answer being displayed. Even spacing carries no opinion about how
 * a window looks, so what reaches the screen is the real spread, seams and
 * all. */
export function civilianWindows(probs, nWindows, nClasses, thresholds, count = 4) {
  if (!nWindows) return [];
  let bestClass = null, bestPeak = -Infinity;
  for (const cls of CIVILIAN) {
    const j = CLASSES.indexOf(cls);
    let peak = -Infinity;
    for (let w = 0; w < nWindows; w++) peak = Math.max(peak, probs[w * nClasses + j]);
    if (peak < thresholds[cls]) continue;
    if (peak > bestPeak) { bestClass = cls; bestPeak = peak; }
  }
  if (bestClass === null) return [];

  const j = CLASSES.indexOf(bestClass);
  const qualifying = [];
  for (let w = 0; w < nWindows; w++) {
    if (probs[w * nClasses + j] >= thresholds[bestClass]) qualifying.push(w);
  }
  const n = qualifying.length;
  let chosen;
  if (n <= count) {
    chosen = qualifying;
  } else {
    const positions = [];
    for (let i = 0; i < count; i++) {
      positions.push(Math.round((i * (n - 1)) / (count - 1)));
    }
    const uniq = [...new Set(positions)];
    if (uniq.length < count) {
      for (let p = 0; p < n && uniq.length < count; p++) if (!uniq.includes(p)) uniq.push(p);
    }
    uniq.sort((a, b) => a - b);
    chosen = uniq.map(p => qualifying[p]);
  }
  return chosen.map(i => ({ index: i, cls: bestClass, prob: probs[i * nClasses + j] }));
}

// --- the figure (plots.py:constellation_figure) -----------------------------

const PANEL = "#FFFFFF";
const GRID = "#DFE3D9";
const TEXT_DIM = "#5F6B72";
const INSTRUMENT = "#42505C";
const CIVILIAN_TIER = "#0F766E";
const FONT = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif';

/** Draws the panel. Returns false when there is nothing to draw, so the
 * caller can HIDE the component rather than render an empty figure --
 * military-only captures then look exactly as they did before this panel
 * existed.
 *
 * Four SEPARATE windows, not one pooled scatter: the 4th-power carrier
 * estimate leaves a 90-degree ambiguity per window, so pooling would render
 * a BPSK capture as four clusters instead of two -- asserting the wrong
 * modulation order. */
export function drawConstellation(canvas, { picks, capture, starts, windowLen, fs, noisePower, c42cfg }) {
  if (!picks || !picks.length) return false;

  const cols = picks.length;
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 900;
  const cellW = cssW / cols;
  const cellH = Math.min(cellW, 190);
  const capLines = 5;                       // set below; sized for the max
  const capH = 12 + capLines * 13;
  const cssH = 26 + cellH + 26 + cellH + 24 + capH;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, 0, cssW, cssH);
  ctx.textBaseline = "middle";

  const className = picks[0].cls;
  const order = CONSTELLATION_ORDER[className];
  const padL = 34, padR = 10;
  const plotW = cellW - padL - padR;

  const scatter = (re, im, x0, y0, w, h, size, alpha) => {
    let amp = 0;
    for (let i = 0; i < re.length; i++) amp = Math.max(amp, Math.abs(re[i]), Math.abs(im[i]));
    amp = (amp || 1) * 1.08;
    // equal aspect, or a QPSK square renders as a rectangle and the eye
    // reads a constellation that is not there
    const s = Math.min(w, h) / (2 * amp);
    const cx = x0 + w / 2, cy = y0 + h / 2;
    ctx.fillStyle = INSTRUMENT;
    ctx.globalAlpha = alpha;
    for (let i = 0; i < re.length; i++) {
      ctx.beginPath();
      ctx.arc(cx + re[i] * s, cy - im[i] * s, size, 0, 7);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, w, h);
  };

  picks.forEach((pick, col) => {
    const x0 = col * cellW + padL;
    const start = starts[pick.index];
    const n = Math.min(windowLen, capture.re.length - start);
    const wRe = capture.re.subarray(start, start + n);
    const wIm = capture.im.subarray(start, start + n);

    // --- top: the exact samples the model is fed, unit-power scaled -------
    let power = 0;
    for (let i = 0; i < n; i++) power += wRe[i] * wRe[i] + wIm[i] * wIm[i];
    const inv = 1 / Math.sqrt(power / n + 1e-20);
    const rawRe = new Float64Array(n), rawIm = new Float64Array(n);
    for (let i = 0; i < n; i++) { rawRe[i] = wRe[i] * inv; rawIm[i] = wIm[i] * inv; }

    ctx.font = `bold 8px ${FONT}`;
    ctx.textAlign = "center";
    ctx.fillStyle = CIVILIAN_TIER;          // the one MODEL element here
    ctx.fillText(`${pick.cls} ${Math.round(pick.prob * 100)}%`, x0 + plotW / 2, 8);
    ctx.font = `8px ${FONT}`;
    ctx.fillStyle = TEXT_DIM;
    ctx.fillText(`win ${pick.index} @ ${(start / fs * 1e3).toFixed(2)} ms`,
                  x0 + plotW / 2, 20);
    scatter(rawRe, rawIm, x0, 26, plotW, cellH, 1.1, 0.45);

    // --- bottom: the same samples through recoverSymbols ------------------
    const rec = recoverSymbols(wRe, wIm);
    const y1 = 26 + cellH + 26;
    scatter(rec.re, rec.im, x0, y1, plotW, cellH, 2.2, 0.85);

    // "points is shorter than window" is the honest test for whether
    // recovery actually ran -- a degenerate window comes back UNCHANGED, and
    // this display must never dress up 512 raw samples as symbol points.
    let title;
    if (!rec.recovered || rec.re.length >= n) {
      title = "no power in this window";
    } else if (rec.re.length >= order * MIN_POINTS_PER_CLUSTER) {
      const score = clusterScore(rec.re, rec.im, order);
      title = `${rec.re.length} symbol points · clusters ${score.toFixed(2)} ${clusterScoreBand(score)}`;
    } else {
      // Honest refusal, not a number: too few points per cluster for the gap
      // statistic to measure anything but sampling noise.
      title = `${rec.re.length} symbol points · too few symbols to score at order ${order}`;
    }
    ctx.fillStyle = TEXT_DIM;
    ctx.font = `7px ${FONT}`;
    ctx.textAlign = "center";
    ctx.fillText(title, x0 + plotW / 2, y1 + cellH + 9);
    ctx.font = `8px ${FONT}`;
  });

  // axis labels: leftmost column only for Q, bottom row only for I
  ctx.save();
  ctx.fillStyle = TEXT_DIM;
  ctx.textAlign = "center";
  for (const y of [26 + cellH / 2, 26 + cellH + 26 + cellH / 2]) {
    ctx.save();
    ctx.translate(11, y); ctx.rotate(-Math.PI / 2);
    ctx.fillText("Q (measured)", 0, 0);
    ctx.restore();
  }
  ctx.fillText("I (measured)", padL + plotW / 2, 26 + cellH + 26 + cellH + 20);
  ctx.restore();

  // --- captions ------------------------------------------------------------
  const lines = [];
  if (className === "16QAM" || className === "64QAM") {
    // MEASURED, printed FIRST because it is the caption most likely to
    // DISAGREE with the class-probability text above: 16QAM vs 64QAM is a
    // distinction the classifier cannot make at all.
    const pooled = picks.map(p => {
      const s = starts[p.index];
      return { re: capture.re.subarray(s, s + windowLen), im: capture.im.subarray(s, s + windowLen) };
    });
    const est = constellationOrder(pooled, noisePower, c42cfg);
    if (est.decision !== null) {
      lines.push(`measured constellation order (|C42|, ${est.nWindows} windows pooled, ` +
                  `${Math.round(est.accuracy * 100)}% accuracy at this pooling): ${est.decision}. ` +
                  `The classifier called this span ${className}.`);
    } else if (est.nWindows < c42cfg.min_windows) {
      lines.push(`measured constellation order: only ${est.nWindows} qualifying window(s) pooled, ` +
                  `below the ${c42cfg.min_windows} needed for a reliable 16QAM-vs-64QAM call. ` +
                  `Refusing rather than guessing. The classifier called this span ${className}.`);
    } else {
      lines.push(`measured constellation order: ${est.nWindows} windows pooled at an estimated ` +
                  `${est.snrDb.toFixed(1)} dB, below the ${c42cfg.min_snr_db.toFixed(0)} dB this ` +
                  `measurement is calibrated for. Refusing rather than guessing. ` +
                  `The classifier called this span ${className}.`);
    }
  }
  lines.push(
    `"clusters" is measured from this window's own recovered symbols, not the classifier. ` +
    `0 means no cluster structure; clean QPSK at +10 dB reads ~0.3, below 0.07 means nothing ` +
    `is there, and 1.0 is never reached on a real capture`,
    `${className}: unit-power scale → matched filter → de-rotate → decimate 1-in-${SAMPLES_PER_SYMBOL}`,
    `cluster count is the modulation order. ${symbolsPerWindow(windowLen)} symbols separates ` +
    `2 clusters from 4, not enough to resolve 64QAM`,
    `four windows spaced evenly across the civilian span, not chosen for how they look. A ` +
    `synthesized scene splices independent recordings, so some windows straddle a seam and will not cluster`,
  );

  ctx.font = `7px ${FONT}`;
  ctx.textAlign = "left";
  ctx.fillStyle = TEXT_DIM;
  let cy = 26 + cellH + 26 + cellH + 34;
  for (const line of lines) {
    // wrap to the canvas width rather than clipping
    const words = line.split(" ");
    let buf = "";
    for (const word of words) {
      const test = buf ? `${buf} ${word}` : word;
      if (ctx.measureText(test).width > cssW - 8 && buf) {
        ctx.fillText(buf, 4, cy); cy += 9; buf = word;
      } else buf = test;
    }
    if (buf) { ctx.fillText(buf, 4, cy); cy += 11; }
  }
  return true;
}
