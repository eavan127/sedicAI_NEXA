// Compares dsp.js's stftMag() against the Python/torch reference computed
// by stft_reference.py. Run stft_reference.py first.
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { hannWindow, stftMag } from "../dsp.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "stft_reference.json"), "utf8"));

const iqRe = Float64Array.from(ref.iq_re);
const iqIm = Float64Array.from(ref.iq_im);
const window = hannWindow(ref.n_fft);

const { mag, nFreq, nFrames } = stftMag(iqRe, iqIm, ref.n_fft, ref.hop, window);

if (nFreq !== ref.n_freq || nFrames !== ref.n_frames) {
  console.error(`SHAPE MISMATCH: got (${nFreq},${nFrames}) expected (${ref.n_freq},${ref.n_frames})`);
  process.exit(1);
}

let maxDiff = 0;
for (let f = 0; f < nFreq; f++) {
  for (let t = 0; t < nFrames; t++) {
    const got = mag[f * nFrames + t];
    const want = ref.mag[f][t];
    maxDiff = Math.max(maxDiff, Math.abs(got - want));
  }
}

console.log(`shape: (${nFreq}, ${nFrames})`);
console.log(`max|diff| vs Python reference: ${maxDiff.toExponential(3)}`);
// Python side computes in float32 (matching the model's actual input dtype);
// JS computes in float64. ~1e-6 on magnitudes of order 1-10 is exactly what
// float32 rounding looks like, not an algorithm mismatch -- same 1e-4 bar
// scripts/export_onnx.py used for the ONNX-vs-PyTorch check.
if (maxDiff > 1e-4) {
  console.error("FAIL: JS stftMag does not match Python compute_stft_mag");
  process.exit(1);
}
console.log("PASS: JS stftMag matches Python compute_stft_mag");
