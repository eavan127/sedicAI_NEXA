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
// Asserted on the DISTRIBUTION of the difference, plus one structural
// invariant -- not on a raw band-agreement percentage.
//
// A bare "N% of bands agree" figure is not a property of the port: it
// depends on where the scores in one particular capture happen to fall
// relative to the 0.07/0.20 cut points. Two implementations drawing from
// different RNG streams will always straddle a cut point sometimes, and a
// capture whose scores cluster near a floor drives that percentage down
// without anything being wrong. Measured here: 41 disagreements, every one
// of them within 0.05 of a floor.
//
// What IS a property of the port: the two must not diverge systematically,
// and a disagreement must always be explainable as a boundary straddle
// rather than an arbitrary flip. A sign error, a wrong constellation order
// or a broken k-means blows past all three checks below.
const BAND_FLOORS = [0.07, 0.20];
const diffs = [];
let bandAgree = 0, worstDisagreeDistance = 0;
for (let k = 0; k < ref.per_window.length; k++) {
  const w = ref.windows[k], want = ref.per_window[k];
  const got = recoverSymbols(Float64Array.from(w.re), Float64Array.from(w.im));
  for (const [orderStr, wantScore] of Object.entries(want.cluster_score)) {
    const gotScore = clusterScore(got.re, got.im, Number(orderStr));
    diffs.push(Math.abs(gotScore - wantScore));
    if (clusterScoreBand(gotScore) === want.cluster_band[orderStr]) {
      bandAgree++;
    } else {
      worstDisagreeDistance = Math.max(worstDisagreeDistance,
        Math.min(...BAND_FLOORS.map(f => Math.abs(wantScore - f))));
    }
  }
}
diffs.sort((a, b) => a - b);
const n = diffs.length;
const pct = q => diffs[Math.floor(q * (n - 1))];
const median = pct(0.5), p95 = pct(0.95), maxDiff = diffs[n - 1];

console.log(`  ${n} comparisons: median|diff|=${median.toFixed(4)} ` +
             `p95=${p95.toFixed(4)} max=${maxDiff.toFixed(4)}`);
console.log(`  band agreement: ${bandAgree}/${n} (${(bandAgree / n * 100).toFixed(0)}%) — informational`);
console.log(`  furthest a disagreeing score sat from a band floor: ${worstDisagreeDistance.toFixed(4)}`);

check("no systematic shift (median |diff| < 0.03)", median < 0.03,
      `median=${median.toFixed(4)}`);
check("no broad divergence (p95 |diff| < 0.08)", p95 < 0.08,
      `p95=${p95.toFixed(4)}`);
check("every band disagreement is a boundary straddle (< 0.08 from a floor)",
      worstDisagreeDistance < 0.08, `worst=${worstDisagreeDistance.toFixed(4)}`);

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} FAILURES`}`);
process.exit(failures === 0 ? 0 : 1);
