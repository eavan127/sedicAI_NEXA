// onnxruntime-web wiring: loads the classifier, runs batched sliding-window
// inference and returns sigmoid probabilities -- the same numbers
// src/pipeline/classify.py:Classifier returns in Python.
//
// The model is the 60-class one the Python Signal Pipeline runs
// (configs/default.yaml browser_model, exported by scripts/export_onnx.py and
// copied here by web/build.py). Its class list, window length and STFT size
// come from model_meta.js, which web/build.py generates from the same
// checkpoint, so they cannot drift from the file. The 8-class model and the
// 5-member ensemble (26 Sept 2026) are in git history.
//
// Requires the global `ort` (onnxruntime-web UMD build) to be loaded before
// this module runs -- see index.html.

import { hannWindow, stftMag } from "./dsp.js";
import { CLASSES, FS, MODEL, STFT_HOP, STFT_N_FFT, WINDOW_LEN } from "./model_meta.js";

export { CLASSES, FS, MODEL, WINDOW_LEN };

const N_FFT = STFT_N_FFT;
const STFT_WINDOW = hannWindow(N_FFT);
const N_STFT_FRAMES = 1 + Math.floor((WINDOW_LEN - N_FFT) / STFT_HOP);

/** One session in a list (classifyCapture averages over the list, so a
 * future ensemble needs no change there). `which` is kept for callers that
 * still pass it; there is only one model now. */
export async function loadModel(which = "single", baseUrl = "./models", onProgress = null) {
  const session = await ort.InferenceSession.create(
    `${baseUrl}/best_model.onnx`, { executionProviders: ["wasm"] });
  const nOut = session.outputMetadata?.[0]?.shape?.[1] ?? null;
  if (typeof nOut === "number" && nOut !== CLASSES.length) {
    // e.g. an 8-class retrained version from before the switch, served by
    // serve_local as the active model: its scores would be read against the
    // wrong class names.
    throw new Error(`models/best_model.onnx has ${nOut} outputs but the page expects ${CLASSES.length} classes`);
  }
  if (onProgress) onProgress(1, 1);
  return [session];
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
 * Windows are cut from the capture as it is: a signal stays at its own
 * frequency, never moved to 0 Hz. That is how the 60-class training windows
 * were made (since f993d8ebc417 every one sits somewhere in the band, none at
 * exactly 0 Hz), and the Python pipeline does the same since KEEP_POSITION
 * (src/pipeline/extract.py): centring a tone turns it into a DC offset the
 * model never saw (RADAR_CW / JAM_TONE went from 80% to 0% at 15 dB).
 *
 * Returns { starts, probs, nWindows, nClasses } with probs a Float32Array
 * laid out [window][class], already sigmoided and (for the ensemble)
 * already averaged. This is the RAW, unsmoothed, ungated result -- the
 * display-layer rules live in analysis.js, exactly as src/timeline.py keeps
 * them out of classify_capture.
 */
export async function classifyCapture(sessions, iqRe, iqIm,
                                       { hop = WINDOW_LEN, batchSize = 64, onProgress = null, signal = null } = {}) {
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
    // Checked between batches: Stop takes effect within one batch instead of
    // after the whole capture.
    if (signal?.aborted) throw new DOMException("Inference cancelled", "AbortError");
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

    const memberProbs = [];
    for (let m = 0; m < sessions.length; m++) {
      // Fresh tensors per member: with ort.env.wasm.proxy on, run() hands
      // the input buffers to the inference worker (they arrive detached on
      // this side), so a tensor cannot be fed to a second ensemble member.
      // Copy for every member but the last, which may take the originals.
      const last = m === sessions.length - 1;
      const iqTensor = new ort.Tensor("float32", last ? iqData : iqData.slice(), [batchN, 2, WINDOW_LEN]);
      const magTensor = new ort.Tensor("float32", last ? magData : magData.slice(), [batchN, 1, N_FFT, N_STFT_FRAMES]);
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
    // Hand the event loop a turn between batches: clicks, the receiver's
    // clock and redraws get through even when inference is not proxied to
    // a worker (e.g. a browser without cross-origin isolation).
    await new Promise(r => setTimeout(r, 0));
  }

  return { starts, probs, attn, nWindows, nClasses, hop, windowLen: WINDOW_LEN, fs: FS };
}
