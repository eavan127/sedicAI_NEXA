// Display-layer analysis, ported from src/timeline.py, src/ui/session.py and
// src/measure.py.
//
// Everything here mirrors the Python behaviour, including the split that
// module docstrings insist on: MODEL-derived values (probabilities, events,
// tiers) come from the classifier; MEASURED values (occupancy, noise floor,
// SNR, spectrum) are computed from the samples with no model involved. The UI
// colours them differently and must never mix them up.

import { fft, hannWindow } from "./dsp.js";
import { CLASSES } from "./model.js";

// configs/default.yaml:multilabel_thresholds_per_class. Calibrated for this
// specific 5-checkpoint ensemble -- re-run scripts/calibrate_thresholds.py
// --ensemble after any retrain and update this table with it.
export const THRESHOLDS = {
  BPSK: 0.33, QPSK: 0.265, "16QAM": 0.265, "64QAM": 0.275,
  LFM_RADAR: 0.26, FHSS: 0.27, JAMMING: 0.77, NOISE_FLOOR: 0.265,
};

// src/config.py:TIERS
export const TIER_OF = {
  BPSK: "Civilian", QPSK: "Civilian", "16QAM": "Civilian", "64QAM": "Civilian",
  LFM_RADAR: "Military", FHSS: "Military",
  JAMMING: "Hostile",
  NOISE_FLOOR: "Empty",
};
// src/timeline.py:TIER_PRIORITY -- worst first.
export const TIER_PRIORITY = ["Hostile", "Military", "Civilian", "Empty"];

// src/ui/palette.py:TIER_COLOR
export const TIER_COLOR = {
  Civilian: "#0F766E", Military: "#B45309", Hostile: "#C1121F", Empty: "#6B7280",
};

// src/ui/session.py display-rule defaults.
export const DEFAULT_NOISE_GATE = 0.5;
export const DEFAULT_HOLD_US = 3000.0;
export const DEFAULT_ALPHA = 0.3;

const NOISE_IDX = CLASSES.indexOf("NOISE_FLOOR");

export function tierOfClasses(classNames) {
  const tiers = new Set(classNames.map(c => TIER_OF[c]));
  return TIER_PRIORITY.find(t => tiers.has(t)) ?? "Empty";
}

/** src/timeline.py:smooth -- exponential moving average per class,
 * independently. DISPLAY ONLY. */
export function smoothProbs(probs, nWindows, nClasses, alpha = DEFAULT_ALPHA) {
  const out = new Float32Array(probs.length);
  const acc = new Float64Array(nClasses);
  for (let c = 0; c < nClasses; c++) { acc[c] = probs[c]; out[c] = probs[c]; }
  for (let w = 1; w < nWindows; w++) {
    for (let c = 0; c < nClasses; c++) {
      acc[c] = alpha * probs[w * nClasses + c] + (1 - alpha) * acc[c];
      out[w * nClasses + c] = acc[c];
    }
  }
  return out;
}

function overThreshold(probs, nWindows, nClasses) {
  const over = new Uint8Array(probs.length);
  for (let w = 0; w < nWindows; w++) {
    for (let c = 0; c < nClasses; c++) {
      over[w * nClasses + c] = probs[w * nClasses + c] > THRESHOLDS[CLASSES[c]] ? 1 : 0;
    }
  }
  return over;
}

/** src/timeline.py:apply_noise_gate -- where NOISE_FLOOR dominates, the
 * window is empty, so drop every other class. Without this a 3-emitter
 * scenario reported hundreds of phantom-radar events on quiet spectrum. */
function applyNoiseGate(probs, over, nWindows, nClasses, gate) {
  for (let w = 0; w < nWindows; w++) {
    if (probs[w * nClasses + NOISE_IDX] > gate) {
      for (let c = 0; c < nClasses; c++) over[w * nClasses + c] = 0;
      over[w * nClasses + NOISE_IDX] = 1;
    }
  }
  return over;
}

/** src/timeline.py:apply_hold -- bridge short gaps in each class's presence,
 * independently per class. A pulsed radar is genuinely absent between
 * pulses; without this one emitter fragments into dozens of events.
 * NOISE_FLOOR is excluded: it is a state, not a pulsed emitter. */
function applyHold(over, nWindows, nClasses, holdWindows) {
  if (holdWindows <= 0) return over;
  for (let c = 0; c < nClasses; c++) {
    if (c === NOISE_IDX) continue;
    const idx = [];
    for (let w = 0; w < nWindows; w++) if (over[w * nClasses + c]) idx.push(w);
    for (let k = 0; k + 1 < idx.length; k++) {
      const a = idx[k], b = idx[k + 1];
      if (b - a - 1 <= holdWindows) {
        for (let w = a + 1; w < b; w++) over[w * nClasses + c] = 1;
      }
    }
  }
  return over;
}

function resolvedMatrix(probs, nWindows, nClasses, { noiseGate, holdUs, hop, fs }) {
  let over = overThreshold(probs, nWindows, nClasses);
  if (noiseGate !== null && noiseGate !== undefined) {
    over = applyNoiseGate(probs, over, nWindows, nClasses, noiseGate);
  }
  if (holdUs > 0) {
    const holdWindows = Math.round(holdUs * fs / 1e6 / hop);
    over = applyHold(over, nWindows, nClasses, holdWindows);
    // Re-assert mutual exclusion AFTER hold: holding an emitter across a gap
    // can extend it over windows where NOISE_FLOOR fired, and "an emitter is
    // present AND the channel is empty" is not a state the dataset contains.
    for (let w = 0; w < nWindows; w++) {
      let others = false;
      for (let c = 0; c < nClasses; c++) {
        if (c !== NOISE_IDX && over[w * nClasses + c]) { others = true; break; }
      }
      if (others) over[w * nClasses + NOISE_IDX] = 0;
    }
  }
  return over;
}

function setsFromMatrix(over, nWindows, nClasses) {
  const sets = [];
  for (let w = 0; w < nWindows; w++) {
    const s = [];
    for (let c = 0; c < nClasses; c++) if (over[w * nClasses + c]) s.push(CLASSES[c]);
    sets.push(s);
  }
  return sets;
}

/** src/timeline.py:detections -- group consecutive windows that reported the
 * SAME set of classes into one event, keyed on the whole set (not per
 * class, which would emit overlapping rows describing one situation). */
export function computeDetections(result, opts) {
  const { probs, nWindows, nClasses, starts, hop, windowLen, fs } = result;
  const over = resolvedMatrix(probs, nWindows, nClasses, { ...opts, hop, fs });
  const sets = setsFromMatrix(over, nWindows, nClasses);

  const events = [];
  let i = 0;
  while (i < sets.length) {
    const current = sets[i];
    if (!current.length) { i++; continue; }
    const key = current.join(",");
    let j = i;
    while (j + 1 < sets.length && sets[j + 1].join(",") === key) j++;

    const peak = {};
    for (const c of current) {
      const ci = CLASSES.indexOf(c);
      let m = 0;
      for (let w = i; w <= j; w++) m = Math.max(m, probs[w * nClasses + ci]);
      peak[c] = m;
    }
    const startUs = starts[i] / fs * 1e6;
    const endUs = (starts[j] + windowLen) / fs * 1e6;
    events.push({
      startUs, endUs, durationUs: endUs - startUs,
      classes: current, peak, startWindow: i, endWindow: j,
      label: current.join(" + "),
    });
    i = j + 1;
  }
  return events;
}

/** src/timeline.py:tier_track -- one tier name per window, for the ribbon. */
export function computeTiers(result, opts) {
  const { probs, nWindows, nClasses, hop, fs } = result;
  const over = resolvedMatrix(probs, nWindows, nClasses, { ...opts, hop, fs });
  return setsFromMatrix(over, nWindows, nClasses).map(tierOfClasses);
}

/** src/ui/session.py:_rules -- the display rules apply only in smoothed
 * mode. Raw mode shows what the model actually did, window by window. */
export function rulesFor(smoothed) {
  return smoothed
    ? { noiseGate: DEFAULT_NOISE_GATE, holdUs: DEFAULT_HOLD_US }
    : { noiseGate: null, holdUs: 0 };
}

/** Resolve a raw classifier result into what the page shows: smoothing,
 * then the gate/hold rules, then events and tiers. Mirrors CaptureSession's
 * events()/tiers() so both come from ONE resolved view -- the Python class
 * exists partly to stop the page reporting different event counts in
 * different panels. */
export function resolveSession(result, smoothed) {
  const probs = smoothed
    ? smoothProbs(result.probs, result.nWindows, result.nClasses)
    : result.probs;
  const resolved = { ...result, probs };
  const opts = rulesFor(smoothed);
  const events = computeDetections(resolved, opts);
  return {
    probs,
    events,
    // "Empty channel is not an event" -- session.py:emitter_events
    emitterEvents: events.filter(e => !(e.classes.length === 1 && e.classes[0] === "NOISE_FLOOR")),
    tiers: computeTiers(resolved, opts),
  };
}

// ---------------------------------------------------------------------------
// MEASURED (src/measure.py) -- no model involvement.
// ---------------------------------------------------------------------------

/** scipy.signal.stft-compatible framing: boundary='zeros' (extend by
 * nperseg//2 each side), padded=True, noverlap=nperseg//2, periodic Hann,
 * scaled by 1/win.sum(). Matching the framing matters because measure.py and
 * plots.py both call scipy's stft, and an off-by-a-frame spectrogram would
 * misalign the waterfall against the detection lanes below it. */
export function scipyStft(iqRe, iqIm, nperseg, fs) {
  const noverlap = Math.floor(nperseg / 2);
  const step = nperseg - noverlap;
  const half = Math.floor(nperseg / 2);
  const n = iqRe.length;

  const extLen = n + 2 * half;
  let nFrames = Math.floor((extLen - nperseg) / step) + 1;
  if ((extLen - nperseg) % step !== 0) nFrames += 1;   // padded=True
  const paddedLen = (nFrames - 1) * step + nperseg;

  const xRe = new Float64Array(paddedLen), xIm = new Float64Array(paddedLen);
  for (let i = 0; i < n; i++) { xRe[half + i] = iqRe[i]; xIm[half + i] = iqIm[i]; }

  const win = hannWindow(nperseg);
  let winSum = 0;
  for (let i = 0; i < nperseg; i++) winSum += win[i];

  const mag2 = new Float64Array(nperseg * nFrames);   // [freq][frame] power
  const fre = new Float64Array(nperseg), fim = new Float64Array(nperseg);
  for (let t = 0; t < nFrames; t++) {
    const s = t * step;
    for (let k = 0; k < nperseg; k++) {
      fre[k] = xRe[s + k] * win[k];
      fim[k] = xIm[s + k] * win[k];
    }
    fft(fre, fim, false);
    for (let k = 0; k < nperseg; k++) {
      const re = fre[k] / winSum, im = fim[k] / winSum;
      mag2[k * nFrames + t] = re * re + im * im;
    }
  }

  // fftshifted frequency axis, -fs/2 .. +fs/2
  const freqs = new Float64Array(nperseg);
  for (let k = 0; k < nperseg; k++) {
    freqs[k] = (k < nperseg / 2 ? k : k - nperseg) * (fs / nperseg);
  }
  const order = new Int32Array(nperseg);
  const shift = Math.ceil(nperseg / 2);
  for (let k = 0; k < nperseg; k++) order[k] = (k + shift) % nperseg;

  const freqsShifted = new Float64Array(nperseg);
  const powerShifted = new Float64Array(nperseg * nFrames);
  for (let k = 0; k < nperseg; k++) {
    freqsShifted[k] = freqs[order[k]];
    for (let t = 0; t < nFrames; t++) powerShifted[k * nFrames + t] = mag2[order[k] * nFrames + t];
  }

  const times = new Float64Array(nFrames);
  for (let t = 0; t < nFrames; t++) times[t] = (t * step) / fs;

  return { power: powerShifted, freqs: freqsShifted, times, nFreq: nperseg, nFrames };
}

/** src/measure.py:occupancy -- fraction of time-frequency cells above the
 * noise floor. MEASURED, deliberately not "channel load" (which would be
 * model output wearing a measurement's name). */
export function occupancy(iqRe, iqIm, fs, nperseg = 256, marginDb = 6.0) {
  const { power } = scipyStft(iqRe, iqIm, nperseg, fs);
  const sorted = Float64Array.from(power).sort();
  const median = Math.max(sorted[Math.floor(sorted.length / 2)], 1e-20);
  const thresh = median * Math.pow(10, marginDb / 10);
  let count = 0;
  for (let i = 0; i < power.length; i++) if (power[i] > thresh) count++;
  return count / power.length;
}

/** src/measure.py:noise_floor_power -- from the quietest frames. */
export function noiseFloorPower(iqRe, iqIm, percentile = 10.0, frameLen = 512) {
  const nFrames = Math.max(Math.floor(iqRe.length / frameLen), 1);
  const powers = new Float64Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    let s = 0;
    for (let i = 0; i < frameLen; i++) {
      const j = f * frameLen + i;
      s += iqRe[j] * iqRe[j] + iqIm[j] * iqIm[j];
    }
    powers[f] = s / frameLen;
  }
  const sorted = Float64Array.from(powers).sort();
  // numpy's linear-interpolation percentile
  const pos = (percentile / 100) * (sorted.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  const val = sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  return Math.max(val, 1e-20);
}

/** src/measure.py:estimate_snr_db -- subtracts the noise floor before the
 * ratio, so a 0 dB window doesn't read +3 dB. Always an estimate; the UI
 * must print it with an `est.` prefix. */
export function estimateSnrDb(iqRe, iqIm, lo, hi, noisePower) {
  let total = 0;
  const n = hi - lo;
  for (let i = lo; i < hi; i++) total += iqRe[i] * iqRe[i] + iqIm[i] * iqIm[i];
  total /= n;
  const signal = Math.max(total - noisePower, 1e-20);
  return 10 * Math.log10(signal / Math.max(noisePower, 1e-20));
}

/** src/measure.py:power_spectrum_db -- average power spectrum over the full
 * complex band, fftshifted. */
export function powerSpectrumDb(iqRe, iqIm, fs, nperseg = 1024) {
  const { power, freqs, nFreq, nFrames } = scipyStft(iqRe, iqIm, nperseg, fs);
  const spec = new Float64Array(nFreq);
  for (let k = 0; k < nFreq; k++) {
    let s = 0;
    for (let t = 0; t < nFrames; t++) s += power[k * nFrames + t];
    spec[k] = 10 * Math.log10(s / nFrames + 1e-20);
  }
  return { freqs, spectrumDb: spec };
}

// ---------------------------------------------------------------------------
// Headline selection (src/ui/pages/rf_replay.py)
// ---------------------------------------------------------------------------

export const MIN_HEADLINE_CONFIDENCE = 0.5;
export const MAX_LISTED_DETECTIONS = 8;

function tierConfidence(event) {
  const tier = tierOfClasses(event.classes);
  return Math.max(...event.classes
    .filter(c => tierOfClasses([c]) === tier)
    .map(c => event.peak[c]));
}

/** Worst tier first, then longest, then most confident. */
export function byPriority(events) {
  return [...events].sort((a, b) => {
    const ta = TIER_PRIORITY.indexOf(tierOfClasses(a.classes));
    const tb = TIER_PRIORITY.indexOf(tierOfClasses(b.classes));
    if (ta !== tb) return ta - tb;
    if (a.durationUs !== b.durationUs) return b.durationUs - a.durationUs;
    return Math.max(...Object.values(b.peak)) - Math.max(...Object.values(a.peak));
  });
}

export function headlineEvent(events) {
  if (!events.length) return null;
  const confident = events.filter(e => tierConfidence(e) >= MIN_HEADLINE_CONFIDENCE);
  return byPriority(confident.length ? confident : events)[0];
}

export function headlineIsConfident(events) {
  return events.some(e => tierConfidence(e) >= MIN_HEADLINE_CONFIDENCE);
}
