// Ports of src/generators/{radar,fhss,jamming}.py and the scenario-building
// logic in src/scenarios.py. Only the three GENERATOR classes are supported
// here (LFM_RADAR, FHSS, JAMMING) -- civilian classes (BPSK/QPSK/16QAM/64QAM)
// have no generator in the Python version either; they're drawn from real
// RadioML recordings on disk, which a static site has no server to serve
// from, so "Civilian only" / "Civilian + Jamming" / "Civilian + Radar" /
// "Contested band" are intentionally not offered here.
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

export const CASES = {
  "Radar only": [["LFM_RADAR", 0.25, 0.75]],
  "FHSS only": [["FHSS", 0.25, 0.75]],
  "Jamming only": [["JAMMING", 0.25, 0.75]],
  "Radar + FHSS": [["LFM_RADAR", 0.15, 0.70], ["FHSS", 0.40, 0.85]],
  "FHSS + Jamming": [["FHSS", 0.15, 0.70], ["JAMMING", 0.40, 0.85]],
  "All three": [["LFM_RADAR", 0.10, 0.45], ["FHSS", 0.30, 0.70], ["JAMMING", 0.55, 0.85]],
};

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
export function buildScenario({ fs = CFG.fs, totalDuration = 0.05, snrDb = 0, seed = 0, script }) {
  const rng = makeRng(seed);
  const nTotal = Math.round(totalDuration * fs);
  const re = new Float64Array(nTotal), im = new Float64Array(nTotal);
  const segments = [];
  const emitterPowers = [];

  for (const [className, startFrac, endFrac] of script) {
    const start = Math.round(startFrac * nTotal);
    const end = Math.min(Math.round(endFrac * nTotal), nTotal);
    if (end - start < 2) continue;
    const length = end - start;

    let emitter = GENERATORS[className](rng, fs, length / fs);
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
    segments.push({ className, startS: start / fs, endS: end / fs });
  }

  const nonJam = emitterPowers.filter(([n]) => n !== "JAMMING");
  const referencePower = nonJam.length ? nonJam[0][1] : (emitterPowers.length ? emitterPowers[0][1] : 1.0);
  const target = referencePower / Math.pow(10, snrDb / 10);

  const noiseAmp = Math.sqrt(target / 2);
  for (let i = 0; i < nTotal; i++) {
    re[i] += gaussian(rng) * noiseAmp;
    im[i] += gaussian(rng) * noiseAmp;
  }

  return { re, im, segments, fs, totalDuration };
}
