// End-to-end parity: JS pipeline vs the real Python pipeline, on the same
// capture. Run pipeline_reference.py first.
//
// Node has no `ort` global and onnxruntime-node isn't a dependency here, so
// this reimplements the ONNX call via onnxruntime-node if available; when it
// isn't, it still checks the PREPROCESSING stage (the part that was broken),
// which is the stage a browser test can't isolate.
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { hannWindow, stftMag } from "../dsp.js";
import { preprocessWindow } from "../model.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "pipeline_reference.json"), "utf8"));

const iqRe = Float64Array.from(ref.iq_re);
const iqIm = Float64Array.from(ref.iq_im);
const WINDOW_LEN = 512, N_FFT = 16, STFT_HOP = 4;
const stftWin = hannWindow(N_FFT);

let ort = null;
try { ort = (await import("onnxruntime-node")).default; }
catch { console.log("onnxruntime-node not installed — running preprocessing-only check\n"); }

const normRe = new Float64Array(WINDOW_LEN), normIm = new Float64Array(WINDOW_LEN);

// --- preprocessing sanity: normalized windows must be zero-mean/unit-std ---
let maxMeanErr = 0, maxStdErr = 0;
for (let w = 0; w < ref.n_windows; w++) {
  const s = ref.starts[w];
  preprocessWindow(iqRe.subarray(s, s + WINDOW_LEN), iqIm.subarray(s, s + WINDOW_LEN), normRe, normIm);
  let sum = 0;
  for (let i = 0; i < WINDOW_LEN; i++) sum += normRe[i] + normIm[i];
  const mean = sum / (2 * WINDOW_LEN);
  let v = 0;
  for (let i = 0; i < WINDOW_LEN; i++) { v += (normRe[i] - mean) ** 2 + (normIm[i] - mean) ** 2; }
  const std = Math.sqrt(v / (2 * WINDOW_LEN));
  maxMeanErr = Math.max(maxMeanErr, Math.abs(mean));
  maxStdErr = Math.max(maxStdErr, Math.abs(std - 1));
}
console.log(`preprocess: max|mean|=${maxMeanErr.toExponential(2)}  max|std-1|=${maxStdErr.toExponential(2)}`);
if (maxMeanErr > 1e-6 || maxStdErr > 1e-6) {
  console.error("FAIL: preprocessWindow does not produce zero-mean/unit-std windows");
  process.exit(1);
}

if (!ort) {
  console.log("PASS (preprocessing only — install onnxruntime-node for full parity)");
  process.exit(0);
}

// --- full pipeline parity ---
const sessions = [];
for (let i = 0; i < 5; i++) {
  sessions.push(await ort.InferenceSession.create(join(here, "..", "models", `ensemble_${i}.onnx`)));
}
const sigmoid = x => 1 / (1 + Math.exp(-x));
const nClasses = ref.classes.length;
const got = new Float64Array(ref.n_windows * nClasses);

const BATCH = 32;
for (let b0 = 0; b0 < ref.n_windows; b0 += BATCH) {
  const b1 = Math.min(b0 + BATCH, ref.n_windows);
  const bn = b1 - b0;
  const nFrames = 1 + Math.floor((WINDOW_LEN - N_FFT) / STFT_HOP);
  const iqData = new Float32Array(bn * 2 * WINDOW_LEN);
  const magData = new Float32Array(bn * N_FFT * nFrames);
  for (let bi = 0; bi < bn; bi++) {
    const s = ref.starts[b0 + bi];
    preprocessWindow(iqRe.subarray(s, s + WINDOW_LEN), iqIm.subarray(s, s + WINDOW_LEN), normRe, normIm);
    iqData.set(normRe, bi * 2 * WINDOW_LEN);
    iqData.set(normIm, bi * 2 * WINDOW_LEN + WINDOW_LEN);
    const { mag } = stftMag(normRe, normIm, N_FFT, STFT_HOP, stftWin);
    magData.set(mag, bi * N_FFT * nFrames);
  }
  const iqT = new ort.Tensor("float32", iqData, [bn, 2, WINDOW_LEN]);
  const magT = new ort.Tensor("float32", magData, [bn, 1, N_FFT, nFrames]);
  const members = [];
  for (const sess of sessions) {
    const out = await sess.run({ iq: iqT, stft_mag: magT });
    members.push(out.logits.data);
  }
  for (let bi = 0; bi < bn; bi++) {
    for (let c = 0; c < nClasses; c++) {
      let sum = 0;
      for (const m of members) sum += sigmoid(m[bi * nClasses + c]);
      got[(b0 + bi) * nClasses + c] = sum / members.length;
    }
  }
}

let maxDiff = 0, argmaxAgree = 0;
for (let w = 0; w < ref.n_windows; w++) {
  for (let c = 0; c < nClasses; c++) {
    maxDiff = Math.max(maxDiff, Math.abs(got[w * nClasses + c] - ref.probs[w][c]));
  }
  const gi = [...Array(nClasses).keys()].reduce((a, b) => got[w * nClasses + b] > got[w * nClasses + a] ? b : a, 0);
  const ri = ref.probs[w].indexOf(Math.max(...ref.probs[w]));
  if (gi === ri) argmaxAgree++;
}
console.log(`full pipeline: max|prob diff|=${maxDiff.toExponential(3)}`);
console.log(`top-class agreement: ${argmaxAgree}/${ref.n_windows}`);
if (maxDiff > 1e-4 || argmaxAgree !== ref.n_windows) {
  console.error("FAIL: JS pipeline does not match Python pipeline");
  process.exit(1);
}
console.log("PASS: JS pipeline matches Python pipeline end to end");
