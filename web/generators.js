// Ports of src/generators/{radar,fhss,jamming}.py and the scenario-building
// logic in src/scenarios.py, including the civilian cases.
//
// Civilian classes (BPSK/QPSK/16QAM/64QAM) have no generator in the Python
// version either -- they are real RadioML recordings. src/ui/session.py
// reads them from data/processed at request time; here the same fixed slice
// is exported by web/build.py to web/data/civilian_*.bin and fetched once
// (see loadCivilianLibrary), which is why the civilian cases work on a
// static host.
//
// RNG note: uses a local seedable PRNG (dsp.js:makeRng), NOT numpy's PCG64.
// Every call site in the Python UI also draws a fresh random seed per click
// (src/ui/session.py:load_scenario), so there was never a cross-run
// reproducibility guarantee to preserve -- only the same generative
// distributions, which this reproduces.

import { fft, fftPadded, gaussian, makeRng, uniform, randint, choice } from "./dsp.js";

// Mirrors configs/default.yaml's radar/fhss/jamming/dataset sections.
// Keep these in sync if that file changes -- see also model.js's THRESHOLDS.
export const CFG = {
  fs: 3200000,
  radar: {
    pulse_width_s: [0.00001, 0.0001],
    bandwidth_hz: [50000, 1500000],
    pri_s: [0.000017, 0.01],
    time_delay_s: [0.000001, 0.00001],
    max_duty_cycle: 0.15,
    n_pulses: [2, 7],
    burst_fraction: 0.5,
  },
  fhss: {
    hop_rate_hz: [25000, 150000],
    n_channels: [8, 64],
    channel_spacing_hz: [10000, 48000],
  },
  jamming: {
    jsr_db: [0, 20],
    max_tones: 1,
    sweep_bandwidth_hz: [100000, 500000],
    barrage_bandwidth_hz: [200000, 1200000],
  },
  dataset: {
    mixture_sir_db: [-6, 6],
  },
};

// ---- radar.py ----

function generateLfmChirpIq(fs, duration, bandwidth, fStart) {
  const n = Math.round(duration * fs);
  const re = new Float64Array(n), im = new Float64Array(n);
  const k = bandwidth / duration;
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const phase = 2 * Math.PI * (fStart * t + 0.5 * k * t * t);
    re[i] = Math.cos(phase); im[i] = Math.sin(phase);
  }
  return { re, im };
}

function embedPulseTrain(pulse, pri, fs, totalDuration, timeDelay, nPulses) {
  const totalSamples = Math.round(totalDuration * fs);
  const priSamples = Math.max(Math.round(pri * fs), 1);
  const offset = Math.round(timeDelay * fs);
  const out = { re: new Float64Array(totalSamples), im: new Float64Array(totalSamples) };
  let emitted = 0;
  const lastStart = Math.max(totalSamples - pulse.re.length, 1);
  for (let start = offset; start < lastStart; start += priSamples) {
    if (nPulses !== null && emitted >= nPulses) break;
    for (let i = 0; i < pulse.re.length && start + i < totalSamples; i++) {
      out.re[start + i] += pulse.re[i];
      out.im[start + i] += pulse.im[i];
    }
    emitted++;
  }
  return out;
}

export function randomRadarExample(rng, fs, totalDuration) {
  const cfg = CFG.radar;
  const pulseWidth = uniform(rng, ...cfg.pulse_width_s);
  let bandwidth = uniform(rng, ...cfg.bandwidth_hz);
  const timeDelay = uniform(rng, ...cfg.time_delay_s);

  const priLo = Math.max(cfg.pri_s[0], pulseWidth / cfg.max_duty_cycle);
  const priHi = Math.max(priLo, cfg.pri_s[1]);
  const pri = Math.exp(uniform(rng, Math.log(priLo), Math.log(priHi)));

  let fStart = rng() > 0.5 ? -bandwidth / 2 : bandwidth / 2;
  bandwidth = fStart < 0 ? bandwidth : -bandwidth;

  const nPulses = rng() < cfg.burst_fraction ? randint(rng, cfg.n_pulses[0], cfg.n_pulses[1]) : null;

  const pulse = generateLfmChirpIq(fs, pulseWidth, bandwidth, fStart);
  return embedPulseTrain(pulse, pri, fs, totalDuration, timeDelay, nPulses);
}

// ---- fhss.py ----

export function randomFhssExample(rng, fs, totalDuration) {
  const cfg = CFG.fhss;
  const hopRate = uniform(rng, ...cfg.hop_rate_hz);
  const hopDuration = 1 / hopRate;
  const nChannels = randint(rng, cfg.n_channels[0], cfg.n_channels[1]);
  const spacing = uniform(rng, ...cfg.channel_spacing_hz);
  const hopFreqs = [];
  for (let i = 0; i < nChannels; i++) hopFreqs.push((i - nChannels / 2) * spacing);

  const samplesPerHop = Math.max(Math.round(hopDuration * fs), 1);
  const nHops = Math.max(Math.floor(totalDuration / hopDuration), 1);
  const re = new Float64Array(nHops * samplesPerHop), im = new Float64Array(nHops * samplesPerHop);
  for (let h = 0; h < nHops; h++) {
    const f = choice(rng, hopFreqs);
    for (let i = 0; i < samplesPerHop; i++) {
      const t = i / fs;
      const phase = 2 * Math.PI * f * t;
      re[h * samplesPerHop + i] = Math.cos(phase);
      im[h * samplesPerHop + i] = Math.sin(phase);
    }
  }
  return { re, im };
}

// ---- jamming.py ----

function generateBarrageJamming(nSamples, rng, fs) {
  const whiteRe = new Float64Array(nSamples), whiteIm = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) { whiteRe[i] = gaussian(rng); whiteIm[i] = gaussian(rng); }

  const cfg = CFG.jamming;
  const bandwidth = uniform(rng, ...cfg.barrage_bandwidth_hz);
  const nyquist = fs / 2;
  const margin = Math.max(nyquist - bandwidth / 2, 0);
  const center = uniform(rng, -margin, margin);

  const { re: Xre, im: Xim, n } = fftPadded(whiteRe, whiteIm);
  // numpy.fft.fftfreq(n, 1/fs): bins [0, n/2-1] are positive k*(fs/n);
  // bins [n/2, n-1] (including the Nyquist bin itself) are (k-n)*(fs/n).
  let any = false;
  for (let k = 0; k < n; k++) {
    const freq = (k < n / 2 ? k : k - n) * (fs / n);
    if (Math.abs(freq - center) > bandwidth / 2) { Xre[k] = 0; Xim[k] = 0; }
    else any = true;
  }
  if (!any) return { re: whiteRe, im: whiteIm };

  fft(Xre, Xim, true);   // in-place inverse FFT
  const fre = Xre, fim = Xim;

  let sumSq = 0;
  for (let i = 0; i < nSamples; i++) sumSq += fre[i] * fre[i] + fim[i] * fim[i];
  const rms = Math.sqrt(sumSq / nSamples);
  if (rms <= 0) return { re: whiteRe, im: whiteIm };

  const outRe = new Float64Array(nSamples), outIm = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) { outRe[i] = fre[i] / rms; outIm[i] = fim[i] / rms; }
  return { re: outRe, im: outIm };
}

function generateToneJamming(fs, nSamples, freqs, rng) {
  const re = new Float64Array(nSamples), im = new Float64Array(nSamples);
  for (const f of freqs) {
    const phase0 = uniform(rng, 0, 2 * Math.PI);
    const amp = uniform(rng, 0.5, 1.0);
    for (let i = 0; i < nSamples; i++) {
      const t = i / fs;
      const phase = 2 * Math.PI * f * t + phase0;
      re[i] += amp * Math.cos(phase);
      im[i] += amp * Math.sin(phase);
    }
  }
  return { re, im };
}

export function randomJammingExample(rng, fs, totalDuration) {
  const nSamples = Math.round(fs * totalDuration);
  const kind = choice(rng, ["barrage", "tone", "sweep"]);
  if (kind === "barrage") return generateBarrageJamming(nSamples, rng, fs);
  if (kind === "tone") {
    const nTones = randint(rng, 1, CFG.jamming.max_tones + 1);
    const freqs = [];
    for (let i = 0; i < nTones; i++) freqs.push(uniform(rng, -fs / 4, fs / 4));
    return generateToneJamming(fs, nSamples, freqs, rng);
  }
  const bandwidth = uniform(rng, ...CFG.jamming.sweep_bandwidth_hz);
  return generateLfmChirpIq(fs, totalDuration, bandwidth, -bandwidth / 2);
}

export const GENERATORS = {
  LFM_RADAR: randomRadarExample,
  FHSS: randomFhssExample,
  JAMMING: randomJammingExample,
};

// ---- scenarios.py (generator-backed cases only) ----

export const CIVILIAN = ["BPSK", "QPSK", "16QAM", "64QAM"];

export const CASES = {
  "Radar only": [["LFM_RADAR", 0.25, 0.75]],
  "FHSS only": [["FHSS", 0.25, 0.75]],
  "Jamming only": [["JAMMING", 0.25, 0.75]],
  "Radar + FHSS": [["LFM_RADAR", 0.15, 0.70], ["FHSS", 0.40, 0.85]],
  "FHSS + Jamming": [["FHSS", 0.15, 0.70], ["JAMMING", 0.40, 0.85]],
  "All three": [["LFM_RADAR", 0.10, 0.45], ["FHSS", 0.30, 0.70], ["JAMMING", 0.55, 0.85]],
  // Civilian cases draw real RadioML captures from the exported library
  // rather than a generator -- see loadCivilianLibrary.
  "Civilian only": [["QPSK", 0.25, 0.75]],
  "Civilian + Jamming": [["QPSK", 0.15, 0.70], ["JAMMING", 0.40, 0.85]],
  "Civilian + Radar": [["BPSK", 0.15, 0.70], ["LFM_RADAR", 0.35, 0.85]],
  "Contested band": [["QPSK", 0.05, 0.60], ["LFM_RADAR", 0.20, 0.55],
                      ["FHSS", 0.35, 0.80], ["JAMMING", 0.55, 0.95]],
};

export function caseNeedsLibrary(script) {
  return script.some(([cls]) => CIVILIAN.includes(cls));
}

let _library = null;

/** Fetches the civilian window library web/build.py exported from
 * src/ui/session.py:civilian_library() -- the same fixed slice: standalone
 * windows of each civilian class, drawn from the TRAIN split at the
 * cleanest SNR bin. Cached after the first call.
 *
 * Returns { snrDb, classes: { BPSK: [{re, im}, ...], ... } }. */
export async function loadCivilianLibrary(baseUrl = "./data") {
  if (_library) return _library;
  const manifest = await (await fetch(`${baseUrl}/civilian_library.json`)).json();
  const classes = {};
  for (const [cls, meta] of Object.entries(manifest.classes)) {
    const buf = await (await fetch(`${baseUrl}/${meta.file}`)).arrayBuffer();
    const raw = new Float32Array(buf);
    const wl = manifest.window_len;
    // stored as (n, 2, window_len): channel 0 is I, channel 1 is Q
    const windows = [];
    for (let i = 0; i < meta.n; i++) {
      const base = i * 2 * wl;
      windows.push({
        re: Float64Array.from(raw.subarray(base, base + wl)),
        im: Float64Array.from(raw.subarray(base + wl, base + 2 * wl)),
      });
    }
    classes[cls] = windows;
  }
  _library = { snrDb: manifest.snr_db, classes };
  return _library;
}

/** src/scenarios.py:_from_library -- assemble one emitter of `length`
 * samples by concatenating real captured windows.
 *
 * The dataset stores independent 512-sample captures, so a longer stretch
 * has to be built by concatenating several, and consecutive captures are
 * unrelated -- every join is a phase discontinuity. Joins are crossfaded
 * over a raised-cosine ramp so that discontinuity does not radiate
 * broadband splatter across the display. The result is still a
 * concatenation of separate recordings, honest for DEMONSTRATING civilian
 * traffic in a scene and NOT a basis for measuring civilian detection
 * performance. */
function fromLibrary(className, length, library, rng) {
  const pool = library?.classes?.[className];
  if (!pool || !pool.length) {
    throw new Error(
      `${className} has no generator and no library entry — civilian classes ` +
      `must be supplied from the dataset`);
  }
  const win = pool[0].re.length;
  const fade = Math.max(Math.floor(win / 16), 8);
  const outRe = new Float64Array(length + win), outIm = new Float64Array(length + win);
  const ramp = new Float64Array(fade);
  for (let i = 0; i < fade; i++) ramp[i] = 0.5 * (1 - Math.cos((Math.PI * i) / (fade - 1)));

  let pos = 0;
  while (pos < length) {
    const w = pool[Math.floor(rng() * pool.length)];
    const segRe = Float64Array.from(w.re), segIm = Float64Array.from(w.im);
    if (pos) {                       // crossfade into whatever is already there
      for (let i = 0; i < fade; i++) {
        segRe[i] *= ramp[i]; segIm[i] *= ramp[i];
        outRe[pos + i] *= ramp[fade - 1 - i];
        outIm[pos + i] *= ramp[fade - 1 - i];
      }
    }
    for (let i = 0; i < win; i++) { outRe[pos + i] += segRe[i]; outIm[pos + i] += segIm[i]; }
    pos += win - fade;
  }
  return { re: outRe.subarray(0, length), im: outIm.subarray(0, length) };
}

function raisedCosineRamp(nSamples, rampLen = 256) {
  const env = new Float64Array(nSamples).fill(1);
  rampLen = Math.min(rampLen, Math.floor(nSamples / 2));
  if (rampLen < 1) return env;
  const rise = new Float64Array(rampLen);
  for (let i = 0; i < rampLen; i++) rise[i] = 0.5 * (1 - Math.cos((Math.PI * i) / (rampLen - 1)));
  for (let i = 0; i < rampLen; i++) { env[i] = rise[i]; env[nSamples - 1 - i] = rise[i]; }
  return env;
}

// composite.py: active_power / unit_power
function activePower(re, im) {
  const n = re.length;
  const magSq = new Float64Array(n);
  let peak = 0;
  for (let i = 0; i < n; i++) { magSq[i] = re[i] * re[i] + im[i] * im[i]; if (magSq[i] > peak) peak = magSq[i]; }
  if (peak === 0) return 0;
  const thresh = 0.01 * peak;
  let sum = 0, count = 0;
  for (let i = 0; i < n; i++) if (magSq[i] > thresh) { sum += magSq[i]; count++; }
  return count ? sum / count : 0;
}
function unitPower(re, im) {
  const p = activePower(re, im);
  if (p === 0) return { re, im };
  const scale = 1 / Math.sqrt(p);
  const outRe = new Float64Array(re.length), outIm = new Float64Array(im.length);
  for (let i = 0; i < re.length; i++) { outRe[i] = re[i] * scale; outIm[i] = im[i] * scale; }
  return { re: outRe, im: outIm };
}

/** Builds one scenario capture. Returns { re, im, segments } where segments
 * is [{ className, startS, endS }] -- start/end only (no radiating_spans /
 * duty-cycle ground truth; that's display-only detail the Python UI uses
 * that isn't needed to prove the model pipeline works end to end). */
export function buildScenario({ fs = CFG.fs, totalDuration = 0.05, snrDb = 0, seed = 0, script,
                                 library = null, librarySnrDb = null }) {
  const rng = makeRng(seed);
  const nTotal = Math.round(totalDuration * fs);
  const re = new Float64Array(nTotal), im = new Float64Array(nTotal);
  const segments = [];
  const emitterPowers = [];
  const civilianSpans = [];

  for (const [className, startFrac, endFrac] of script) {
    const start = Math.round(startFrac * nTotal);
    const end = Math.min(Math.round(endFrac * nTotal), nTotal);
    if (end - start < 2) continue;
    const length = end - start;

    const isCivilian = !(className in GENERATORS);
    let emitter = isCivilian
      ? fromLibrary(className, length, library, rng)
      : GENERATORS[className](rng, fs, length / fs);
    if (emitter.re.length < length) {
      const padRe = new Float64Array(length), padIm = new Float64Array(length);
      padRe.set(emitter.re); padIm.set(emitter.im);
      emitter = { re: padRe, im: padIm };
    }
    const ramp = raisedCosineRamp(length);
    const eRe = new Float64Array(length), eIm = new Float64Array(length);
    for (let i = 0; i < length; i++) { eRe[i] = emitter.re[i] * ramp[i]; eIm[i] = emitter.im[i] * ramp[i]; }

    let { re: uRe, im: uIm } = unitPower(eRe, eIm);
    if (className === "JAMMING") {
      const jsr = uniform(rng, ...CFG.jamming.jsr_db);
      const g = Math.pow(10, jsr / 20);
      for (let i = 0; i < length; i++) { uRe[i] *= g; uIm[i] *= g; }
    } else if (emitterPowers.length) {
      const sir = uniform(rng, ...CFG.dataset.mixture_sir_db);
      const g = Math.pow(10, sir / 20);
      for (let i = 0; i < length; i++) { uRe[i] *= g; uIm[i] *= g; }
    }

    for (let i = 0; i < length; i++) { re[start + i] += uRe[i]; im[start + i] += uIm[i]; }
    let power = 0;
    for (let i = 0; i < length; i++) power += uRe[i] * uRe[i] + uIm[i] * uIm[i];
    power /= length;
    emitterPowers.push([className, power]);
    // Recorded AFTER the unit-power/SIR scaling, so this is the span's own
    // ACTUAL placed power -- the noise section below needs each civilian
    // span's own power to compute that span's own carried noise, not the
    // first non-jamming emitter's.
    if (isCivilian) civilianSpans.push({ start, end, power });
    segments.push({ className, startS: start / fs, endS: end / fs });
  }

  // Noise references the FIRST NON-JAMMING emitter, exactly as
  // mix_components does. Averaging over all emitters instead would let a
  // jammer -- deliberately 0-20 dB hot -- drag the reference up and raise
  // the noise floor, so adding a jammer would quietly make the victim
  // harder to see.
  const nonJam = emitterPowers.filter(([n]) => n !== "JAMMING");
  const referencePower = nonJam.length ? nonJam[0][1] : (emitterPowers.length ? emitterPowers[0][1] : 1.0);
  const target = referencePower / Math.pow(10, snrDb / 10);

  const noiseRe = new Float64Array(nTotal), noiseIm = new Float64Array(nTotal);
  for (let i = 0; i < nTotal; i++) { noiseRe[i] = gaussian(rng); noiseIm[i] = gaussian(rng); }

  if (librarySnrDb === null || !civilianSpans.length) {
    const amp = Math.sqrt(target / 2);
    for (let i = 0; i < nTotal; i++) { re[i] += noiseRe[i] * amp; im[i] += noiseIm[i] * amp; }
  } else {
    // A civilian recording already carries noise at librarySnrDb, so a
    // target SNR better than that bin is not achievable -- you can add
    // noise to a recording but never remove it. Noising it again on top of
    // the scenario noise would double-count.
    const carried = civilianSpans.map(s => ({
      ...s, carried: s.power / Math.pow(10, librarySnrDb / 10),
    }));
    const floor = Math.max(target, ...carried.map(c => c.carried));
    // Everywhere gets `floor`, except inside a civilian span, which already
    // has its own `carried` baked into the recording and only needs
    // `floor - carried` on top. Noise POWERS add, so the added component's
    // amplitude is sqrt(floor - carried), not sqrt(floor) - sqrt(carried).
    const addedPower = new Float64Array(nTotal).fill(floor);
    for (const c of carried) {
      addedPower.fill(Math.max(floor - c.carried, 0), c.start, c.end);
    }
    for (let i = 0; i < nTotal; i++) {
      const amp = Math.sqrt(addedPower[i] / 2);
      re[i] += noiseRe[i] * amp; im[i] += noiseIm[i] * amp;
    }
  }

  // The achieved SNR is whichever is worse (lower) once a civilian
  // recording's own carried noise is accounted for -- load_scenario's
  // snr_capped / requested_snr_db, which the header reports.
  const needsLibrary = civilianSpans.length > 0;
  const trueSnrDb = (needsLibrary && librarySnrDb !== null)
    ? Math.min(snrDb, librarySnrDb) : snrDb;

  return {
    re, im, segments, fs, totalDuration, trueSnrDb,
    snrCapped: needsLibrary && librarySnrDb !== null && snrDb > librarySnrDb,
    requestedSnrDb: snrDb,
  };
}
