// Parity for the constellation panel. Run constellation_reference.py first.
//
// Two bars, deliberately different (see the reference script's docstring):
//   deterministic chain -> float precision
//   cluster_score       -> statistical agreement, since numpy's PCG64 bit
//                          stream cannot be reproduced in JS
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import {
  carrierOffset, civilianWindows, clusterScore, clusterScoreBand,
  constellationOrder, normalizedC42, recoverSymbols, rrcTaps, symbolsPerWindow,
} from "../constellation.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "constellation_reference.json"), "utf8"));
const card = JSON.parse(readFileSync(join(here, "..", "data", "model_card.json"), "utf8"));

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`  ${ok ? "OK  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
};
const maxAbsDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

console.log("== deterministic chain ==");

const taps = rrcTaps(8);
check("rrc_taps length", taps.length === ref.rrc_taps.length,
      `${taps.length}/${ref.rrc_taps.length}`);
check("rrc_taps values", maxAbsDiff(taps, ref.rrc_taps) < 1e-12,
      `max|diff|=${maxAbsDiff(taps, ref.rrc_taps).toExponential(2)}`);
check("symbols_per_window", symbolsPerWindow(ref.window_len) === ref.symbols_per_window,
      `${symbolsPerWindow(ref.window_len)}/${ref.symbols_per_window}`);

let maxOffsetErr = 0, maxRawOffsetErr = 0, maxPointErr = 0, maxC42Err = 0;
let phaseMism = 0, countMism = 0, recoveredMism = 0;
for (let k = 0; k < ref.per_window.length; k++) {
  const w = ref.windows[k], want = ref.per_window[k];
  const re = Float64Array.from(w.re), im = Float64Array.from(w.im);

  maxRawOffsetErr = Math.max(maxRawOffsetErr,
    Math.abs(carrierOffset(re, im) - want.carrier_offset_raw));

  const got = recoverSymbols(re, im);
  maxOffsetErr = Math.max(maxOffsetErr, Math.abs(got.offset - want.offset));
  if (got.phase !== want.phase) phaseMism++;
  if (got.re.length !== want.n_points) countMism++;
  if (got.recovered !== want.recovered) recoveredMism++;
  if (got.re.length === want.n_points) {
    maxPointErr = Math.max(maxPointErr,
      maxAbsDiff(got.re, want.points_re), maxAbsDiff(got.im, want.points_im));
  }
  const c42 = normalizedC42(got.re, got.im);
  const wantC42 = want.n_points ? null : null;   // compared via pooled estimate below
  if (c42 !== null && Number.isFinite(c42)) maxC42Err = Math.max(maxC42Err, 0);
}
check("carrier_offset (raw window)", maxRawOffsetErr === 0,
      `max|diff|=${maxRawOffsetErr.toExponential(2)}`);
check("recover_symbols offset", maxOffsetErr === 0,
      `max|diff|=${maxOffsetErr.toExponential(2)}`);
check("recover_symbols timing phase", phaseMism === 0, `${phaseMism} differ`);
check("recover_symbols point count", countMism === 0, `${countMism} differ`);
check("recover_symbols recovered flag", recoveredMism === 0, `${recoveredMism} differ`);
check("recover_symbols point values", maxPointErr < 1e-9,
      `max|diff|=${maxPointErr.toExponential(2)}`);

console.log("\n== civilian window selection ==");
const nClasses = ref.n_classes;
const probs = new Float32Array(ref.n_windows * nClasses);
for (let w = 0; w < ref.n_windows; w++) {
  for (let c = 0; c < nClasses; c++) probs[w * nClasses + c] = ref.probs[w][c];
}
const picks = civilianWindows(probs, ref.n_windows, nClasses, ref.thresholds, 4);
check("pick count", picks.length === ref.civilian_picks.length,
      `${picks.length}/${ref.civilian_picks.length}`);
let pickMism = 0;
for (let i = 0; i < Math.min(picks.length, ref.civilian_picks.length); i++) {
  const g = picks[i], w = ref.civilian_picks[i];
  if (g.index !== w.index || g.cls !== w.cls || Math.abs(g.prob - w.prob) > 1e-6) pickMism++;
}
check("picks match (index, class, prob)", pickMism === 0, `${pickMism} differ`);

const allQ = civilianWindows(probs, ref.n_windows, nClasses, ref.thresholds, ref.n_windows);
check("qualifying window count", allQ.length === ref.n_qualifying,
      `${allQ.length}/${ref.n_qualifying}`);

console.log("\n== constellation_order (|C42| pooling) ==");
if (ref.constellation_order) {
  const pooled = allQ.map(p => {
    const w = ref.windows[p.index];
    return { re: Float64Array.from(w.re), im: Float64Array.from(w.im) };
  });
  const got = constellationOrder(pooled, ref.noise_power, card.c42);
  const want = ref.constellation_order;
  check("decision", got.decision === want.decision,
        `got ${got.decision}, want ${want.decision}`);
  check("n_windows", got.nWindows === want.n_windows, `${got.nWindows}/${want.n_windows}`);
  check("mean |C42|", Math.abs(got.meanC42 - want.mean_c42) < 1e-9,
        `${got.meanC42.toFixed(9)} / ${want.mean_c42.toFixed(9)}`);
  check("margin", Math.abs(got.margin - want.margin) < 1e-9);
  check("pooled accuracy", got.accuracy === want.accuracy,
        `got ${got.accuracy}, want ${want.accuracy}`);
  check("estimated SNR", Math.abs(got.snrDb - want.snr_db) < 1e-6,
        `${got.snrDb?.toFixed(6)} / ${want.snr_db?.toFixed(6)}`);
} else {
  console.log("  (no qualifying windows in the reference capture — skipped)");
}

console.log("\n== cluster_score (statistical — see module note) ==");
let n = 0, bandAgree = 0, sumAbs = 0, maxDiff = 0;
for (let k = 0; k < ref.per_window.length; k++) {
  const w = ref.windows[k], want = ref.per_window[k];
  const got = recoverSymbols(Float64Array.from(w.re), Float64Array.from(w.im));
  for (const [orderStr, wantScore] of Object.entries(want.cluster_score)) {
    const order = Number(orderStr);
    const gotScore = clusterScore(got.re, got.im, order);
    const d = Math.abs(gotScore - wantScore);
    sumAbs += d; maxDiff = Math.max(maxDiff, d); n++;
    if (clusterScoreBand(gotScore) === want.cluster_band[orderStr]) bandAgree++;
  }
}
const meanAbs = n ? sumAbs / n : 0;
console.log(`  ${n} comparisons: mean|diff|=${meanAbs.toFixed(4)} max|diff|=${maxDiff.toFixed(4)}`);
console.log(`  band agreement: ${bandAgree}/${n} (${(bandAgree / n * 100).toFixed(0)}%)`);
// The bar: the qualitative band is what the panel prints, so that must agree
// almost always; the raw value is allowed to drift a little.
check("band agreement >= 90%", bandAgree / n >= 0.9);
check("mean |diff| < 0.05", meanAbs < 0.05);

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} FAILURES`}`);
process.exit(failures === 0 ? 0 : 1);
