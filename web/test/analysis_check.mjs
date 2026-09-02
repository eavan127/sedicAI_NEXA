// Parity for the display layer: smoothing, noise gate, hold, event grouping,
// tier track, and the MEASURED figures -- against the same Python session
// pipeline_reference.py built. Run pipeline_reference.py first.
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { noiseFloorPower, occupancy, resolveSession } from "../analysis.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "pipeline_reference.json"), "utf8"));

const iqRe = Float64Array.from(ref.iq_re);
const iqIm = Float64Array.from(ref.iq_im);
const FS = 3200000;

// Feed the PYTHON probabilities in, so this isolates the display layer from
// the inference layer (pipeline_check.mjs already proved inference matches).
const nClasses = ref.classes.length;
const probs = new Float32Array(ref.n_windows * nClasses);
for (let w = 0; w < ref.n_windows; w++) {
  for (let c = 0; c < nClasses; c++) probs[w * nClasses + c] = ref.probs[w][c];
}
const result = {
  probs, nWindows: ref.n_windows, nClasses,
  starts: Int32Array.from(ref.starts),
  hop: ref.hop, windowLen: 512, fs: FS,
};

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`  ${ok ? "OK  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
}

for (const mode of ["smoothed", "raw"]) {
  console.log(`\n== ${mode} ==`);
  const got = resolveSession(result, mode === "smoothed");
  const want = ref[mode];

  check("event count", got.emitterEvents.length === want.events.length,
        `got ${got.emitterEvents.length}, want ${want.events.length}`);

  let maxTimeErr = 0, classMismatch = 0, maxPeakErr = 0;
  const n = Math.min(got.emitterEvents.length, want.events.length);
  for (let i = 0; i < n; i++) {
    const g = got.emitterEvents[i], w = want.events[i];
    maxTimeErr = Math.max(maxTimeErr, Math.abs(g.startUs - w.startUs), Math.abs(g.endUs - w.endUs));
    if (g.classes.join(",") !== w.classes.join(",")) classMismatch++;
    for (const c of w.classes) {
      if (g.peak[c] !== undefined) maxPeakErr = Math.max(maxPeakErr, Math.abs(g.peak[c] - w.peak[c]));
    }
  }
  check("event class sets", classMismatch === 0, `${classMismatch} mismatched`);
  check("event times", maxTimeErr < 1e-6, `max|diff|=${maxTimeErr.toExponential(2)} us`);
  check("event peak confidences", maxPeakErr < 1e-5, `max|diff|=${maxPeakErr.toExponential(2)}`);

  const tierMismatch = got.tiers.filter((t, i) => t !== want.tiers[i]).length;
  check("tier track", got.tiers.length === want.tiers.length && tierMismatch === 0,
        `${tierMismatch}/${want.tiers.length} differ`);
}

console.log("\n== measured ==");
const occ = occupancy(iqRe, iqIm, FS);
const nf = noiseFloorPower(iqRe, iqIm);
check("occupancy", Math.abs(occ - ref.occupancy) < 2e-3,
      `got ${occ.toFixed(5)}, want ${ref.occupancy.toFixed(5)}`);
check("noise floor power", Math.abs(nf - ref.noise_floor_power) / ref.noise_floor_power < 1e-6,
      `got ${nf.toExponential(4)}, want ${ref.noise_floor_power.toExponential(4)}`);

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} FAILURES`}`);
process.exit(failures === 0 ? 0 : 1);
