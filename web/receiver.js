// Receiver sources: where live IQ comes from.
//
// A real electronic-support receiver does not hand the classifier one tidy
// file. It streams IQ continuously, and the processor takes a DWELL -- a
// fixed slice of it -- classifies that, then takes the next. When processing
// is slower than the air (it always is here: a browser cannot classify
// 3.2 million samples a second), the receiver keeps running and the gap
// between dwells is simply not examined. That fraction is reported as
// COVERAGE rather than hidden.
//
// Two sources, one interface -- connect(), next() -> block | null, describe():
//
//   SimulatedReceiver  a virtual SDR: every dwell is a fresh scenario from
//                      generators.js, using whatever case and SNR the
//                      operator has selected AT THAT MOMENT (changing them
//                      mid-stream changes what the "air" contains).
//   FileReplay         a recording (.bin or SigMF) streamed dwell by dwell,
//                      every sample in order, as if it were arriving live.
//
// A real SDR would be a third source with the same interface (the local
// server would read the device and hand blocks to the page).
//
// connect(onStep) reports each stage of bringing the source up (the
// "device found -> tuned -> gain set -> streaming" sequence an operator sees
// on real equipment). For the simulated receiver every stage is acted out and
// every value is labelled SIMULATED: there is no hardware, and the console
// never pretends otherwise.
//
// No DOM here, so web/test/receiver_check.mjs runs it in node.

const wait = ms => new Promise(r => setTimeout(r, ms));

/** Received power of one dwell, in dB relative to unit power (the
 *  generators' and recordings' own scale -- NOT dBm, which would need a
 *  calibrated front end). Measured from the samples, so it moves with what
 *  is on the air: louder emitters, jamming, or just noise. */
export function dwellLevel(re, im) {
  let sum = 0, peak = 0;
  for (let i = 0; i < re.length; i++) {
    const p = re[i] * re[i] + im[i] * im[i];
    sum += p;
    if (p > peak) peak = p;
  }
  const db = x => (x > 0 ? 10 * Math.log10(x) : -Infinity);
  return { powerDb: db(sum / Math.max(re.length, 1)), peakDb: db(peak) };
}

import { CASES, buildScenario, caseNeedsLibrary } from "./generators.js";
import { FS, WINDOW_LEN } from "./model.js";

/** Truth segments that overlap [startS, endS), re-based to the block's own
 *  clock and clipped to it -- so a recording's annotations line up with the
 *  dwell being displayed. */
export function sliceTruth(truth, startS, endS) {
  if (!truth) return null;
  const out = [];
  for (const s of truth) {
    const a = Math.max(s.startS, startS), b = Math.min(s.endS, endS);
    if (b > a) out.push({ className: s.className, startS: a - startS, endS: b - startS });
  }
  return out;
}

export class SimulatedReceiver {
  /**
   * getSettings() -> {caseName, snrDb}: read on every dwell, so the
   * operator's dropdowns steer the live stream.
   * getLibrary() -> civilian window library (only called when a case needs it).
   */
  constructor({ dwellS, getSettings, getLibrary, seed = Date.now(), stepDelayMs = 350 }) {
    this.dwellS = dwellS;
    this.stepDelayMs = stepDelayMs;
    this.getSettings = getSettings;
    this.getLibrary = getLibrary;
    this.seed = seed >>> 0;
    this.index = 0;
  }

  describe() {
    return {
      kind: "sim", simulated: true, name: "Simulated receiver",
      hardware: "NEXA-SIM virtual SDR", serial: "SIM-0001", link: "software (no physical device)",
      // The simulation is generated at baseband: this tuning is the value a
      // real front end would be set to, shown as simulated, not a claim
      // about any real frequency on the air.
      centreHz: 2.4e9, sampleRate: FS, gainDb: 30, totalS: null,
    };
  }

  async connect(onStep = () => {}) {
    const d = this.describe();
    const steps = [
      "Searching for receivers…",
      `Found ${d.hardware} · serial ${d.serial} · ${d.link}`,
      `Sample rate set: ${(d.sampleRate / 1e6).toFixed(1)} MS/s`,
      `Tuned: ${(d.centreHz / 1e9).toFixed(3)} GHz (simulated)`,
      `Gain set: ${d.gainDb} dB (simulated)`,
      `IQ stream open · ${(this.dwellS * 1000).toFixed(0)} ms dwells`,
    ];
    for (const s of steps) {
      onStep(s);
      await wait(this.stepDelayMs);
    }
    this.index = 0;
  }

  async next() {
    const { caseName, snrDb } = this.getSettings();
    const script = CASES[caseName];
    if (!script) throw new Error(`Unknown scenario case "${caseName}".`);
    const library = caseNeedsLibrary(script) ? await this.getLibrary() : null;
    const sc = buildScenario({
      totalDuration: this.dwellS, snrDb, seed: (this.seed + this.index * 7919) >>> 0,
      script, library, librarySnrDb: library?.snrDb ?? null,
    });
    const i = this.index++;
    return {
      index: i, re: sc.re, im: sc.im, truth: sc.segments,
      source: "scenario", caseNote: `LIVE · simulated receiver · dwell ${i + 1} · case \`${caseName}\``,
      snrDb: sc.trueSnrDb, snrCapped: sc.snrCapped, requestedSnrDb: sc.requestedSnrDb,
      startS: null, endS: null,
    };
  }

  disconnect() {}
}

export class FileReplay {
  /** rec: {re, im, truth, name, meta?} as produced by the upload readers. */
  constructor({ rec, dwellS, stepDelayMs = 250 }) {
    this.rec = rec;
    this.stepDelayMs = stepDelayMs;
    this.dwellN = Math.max(WINDOW_LEN, Math.round(dwellS * FS));
    this.pos = 0;
    this.index = 0;
  }

  describe() {
    const m = this.rec.meta;
    return {
      kind: "file", simulated: false, name: `File replay: ${this.rec.name}`,
      hardware: m?.hw || (m ? "SigMF recording" : "raw IQ file (no metadata)"),
      serial: null, link: "file on this machine",
      centreHz: m?.frequency ?? null, sampleRate: m?.sampleRate ?? FS, gainDb: null,
      totalS: this.rec.re.length / FS,
    };
  }

  async connect(onStep = () => {}) {
    const d = this.describe(), m = this.rec.meta;
    const steps = [
      `Opening ${this.rec.name}…`,
      m ? `Format: SigMF ${m.fmt.datatype} · recorded by ${d.hardware}` : "Format: raw interleaved float32 (no metadata)",
      `Length: ${(d.totalS * 1000).toFixed(1)} ms · ${(d.sampleRate / 1e6).toFixed(3)} MS/s`
        + (d.centreHz != null ? ` · centre ${(d.centreHz / 1e6).toFixed(3)} MHz` : ""),
      this.rec.truth ? `Annotations: ${this.rec.truth.length} labelled segment(s) shown as TRUTH` : "Annotations: none",
      `Replay started · ${(this.dwellN / FS * 1000).toFixed(0)} ms dwells`,
    ];
    for (const s of steps) {
      onStep(s);
      await wait(this.stepDelayMs);
    }
    this.pos = 0;
    this.index = 0;
  }

  get progress() {
    return { doneS: this.pos / FS, totalS: this.rec.re.length / FS };
  }

  async next() {
    const n = this.rec.re.length;
    // A tail shorter than one window cannot be classified; the replay ends
    // there rather than padding samples that were never recorded.
    if (n - this.pos < WINDOW_LEN) return null;
    const start = this.pos, end = Math.min(n, start + this.dwellN);
    this.pos = end;
    const i = this.index++;
    return {
      index: i,
      re: this.rec.re.subarray(start, end), im: this.rec.im.subarray(start, end),
      truth: sliceTruth(this.rec.truth, start / FS, end / FS),
      source: "upload",
      caseNote: `LIVE · replay of ${this.rec.name} · dwell ${i + 1} · `
        + `${(start / FS * 1000).toFixed(1)}–${(end / FS * 1000).toFixed(1)} ms`
        + (this.rec.note ? ` · ${this.rec.note}` : ""),
      snrDb: null, snrCapped: false, requestedSnrDb: null,
      startS: start / FS, endS: end / FS,
    };
  }

  disconnect() {}
}

/**
 * Which dwells are worth writing down. Storing every dwell would bury the
 * History page (and the disk) in near-identical rows; an electronic-support
 * operator logs CHANGES -- a new emitter appears, one goes away. So a dwell
 * is stored when the set of detected classes differs from the previous
 * dwell's and is not empty.
 */
export function shouldStore(prevClasses, classes) {
  if (!classes.length) return false;
  const key = c => [...c].sort().join("|");
  return prevClasses === null || key(prevClasses) !== key(classes);
}

/** Interleaved float32 I,Q,I,Q,... -- the exact bytes a dwell arrived as,
 *  so a stored dwell can be re-uploaded or used for retraining. */
export function toInterleavedF32(re, im) {
  const out = new Float32Array(re.length * 2);
  for (let i = 0; i < re.length; i++) { out[2 * i] = re[i]; out[2 * i + 1] = im[i]; }
  return out;
}
