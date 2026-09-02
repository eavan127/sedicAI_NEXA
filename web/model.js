// onnxruntime-web wiring: loads a checkpoint (single or the 5-member
// ensemble), runs batched sliding-window inference, and averages sigmoid
// probabilities -- the same averaging EnsembleModel.forward
// (src/ui/app_models.py) does, on probabilities rather than logits.
//
// Requires the global `ort` (onnxruntime-web UMD build) to be loaded before
// this module runs -- see index.html.

import { hannWindow, stftMag } from "./dsp.js";

export const CLASSES = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING", "NOISE_FLOOR"];

export const WINDOW_LEN = 512;
export const FS = 3200000;
const N_FFT = 16, STFT_HOP = 4;
const STFT_WINDOW = hannWindow(N_FFT);
const N_STFT_FRAMES = 1 + Math.floor((WINDOW_LEN - N_FFT) / STFT_HOP);

export async function loadModel(which = "ensemble", baseUrl = "./models", onProgress = null) {
  const names = which === "ensemble"
    ? ["ensemble_0", "ensemble_1", "ensemble_2", "ensemble_3", "ensemble_4"]
    : ["best_model"];
  const sessions = [];
  for (let i = 0; i < names.length; i++) {
    sessions.push(await ort.InferenceSession.create(
      `${baseUrl}/${names[i]}.onnx`, { executionProviders: ["wasm"] }));
    if (onProgress) onProgress(i + 1, names.length);
  }
  return sessions;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

/** Port of src/data/preprocess.py:preprocess_window.
 *
 * CRITICAL: the model was trained on windows normalized this way, and
 * src/timeline.py:sliding_windows applies it to every window before
 * inference. Feeding raw samples instead puts the input wildly out of
 * distribution -- absolute amplitude varies by ~100x between a quiet
 * capture and a jammed one, which the normalization is what removes.
 *
 * mean and std are SCALARS over the combined (2, window_len) array (both I
 * and Q together), not per-channel -- matching numpy's arr.mean()/arr.std()
 * on the stacked array, with population std (ddof=0).
 *
 * Writes into outRe/outIm, which the STFT is then computed from -- in
 * PyTorch, STFTBranch receives the model input x, i.e. the ALREADY
 * normalized window, so the spectrogram must come from the normalized
 * samples too.
 */
export function preprocessWindow(wRe, wIm, outRe, outIm) {
  const n = wRe.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += wRe[i] + wIm[i];
  const mean = sum / (2 * n);
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const a = wRe[i] - mean, b = wIm[i] - mean;
    varSum += a * a + b * b;
  }
  const std = Math.sqrt(varSum / (2 * n));
  const scale = 1 / (std + 1e-8);
  for (let i = 0; i < n; i++) {
    outRe[i] = (wRe[i] - mean) * scale;
    outIm[i] = (wIm[i] - mean) * scale;
  }
}

/** Runs the model over every sliding window of a capture.
 *
 * Returns { starts, probs, nWindows, nClasses } with probs a Float32Array
 * laid out [window][class], already sigmoided and (for the ensemble)
 * already averaged. This is the RAW, unsmoothed, ungated result -- the
 * display-layer rules live in analysis.js, exactly as src/timeline.py keeps
 * them out of classify_capture.
 */
export async function classifyCapture(sessions, iqRe, iqIm, { hop = WINDOW_LEN, batchSize = 64, onProgress = null } = {}) {
  const n = iqRe.length;
  const nWindows = 1 + Math.floor(Math.max(n - WINDOW_LEN, 0) / hop);
  const starts = new Int32Array(nWindows);
  for (let i = 0; i < nWindows; i++) starts[i] = i * hop;

  const nClasses = CLASSES.length;
  const probs = new Float32Array(nWindows * nClasses);
  // Member 0's attention, NOT an average across the ensemble: attention
  // weights are a per-model internal, and averaging five models' attention
  // would produce a curve no model actually computed. EnsembleModel does the
  // same (src/ui/app_models.py) and the UI labels it as member 0's.
  const attn = new Float32Array(nWindows * WINDOW_LEN);

  const normRe = new Float64Array(WINDOW_LEN), normIm = new Float64Array(WINDOW_LEN);

  for (let b0 = 0; b0 < nWindows; b0 += batchSize) {
    const b1 = Math.min(b0 + batchSize, nWindows);
    const batchN = b1 - b0;

    const iqData = new Float32Array(batchN * 2 * WINDOW_LEN);
    const magData = new Float32Array(batchN * N_FFT * N_STFT_FRAMES);

    for (let bi = 0; bi < batchN; bi++) {
      const start = starts[b0 + bi];
      preprocessWindow(iqRe.subarray(start, start + WINDOW_LEN),
                        iqIm.subarray(start, start + WINDOW_LEN),
                        normRe, normIm);

      const iqBase = bi * 2 * WINDOW_LEN;
      iqData.set(normRe, iqBase);
      iqData.set(normIm, iqBase + WINDOW_LEN);

      const { mag } = stftMag(normRe, normIm, N_FFT, STFT_HOP, STFT_WINDOW);
      magData.set(mag, bi * N_FFT * N_STFT_FRAMES);
    }

    const iqTensor = new ort.Tensor("float32", iqData, [batchN, 2, WINDOW_LEN]);
    const magTensor = new ort.Tensor("float32", magData, [batchN, 1, N_FFT, N_STFT_FRAMES]);

    const memberProbs = [];
    for (let m = 0; m < sessions.length; m++) {
      const out = await sessions[m].run({ iq: iqTensor, stft_mag: magTensor });
      const logits = out.logits.data;
      const p = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) p[i] = sigmoid(logits[i]);
      memberProbs.push(p);
      if (m === 0) attn.set(out.attention.data, b0 * WINDOW_LEN);
    }
    for (let bi = 0; bi < batchN; bi++) {
      for (let c = 0; c < nClasses; c++) {
        let sum = 0;
        for (const p of memberProbs) sum += p[bi * nClasses + c];
        probs[(b0 + bi) * nClasses + c] = sum / memberProbs.length;
      }
    }
    if (onProgress) onProgress(b1, nWindows);
  }

  return { starts, probs, attn, nWindows, nClasses, hop, windowLen: WINDOW_LEN, fs: FS };
}
