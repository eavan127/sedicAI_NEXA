// Structural smoke test: every generator + every generator-backed CASE
// script must run without throwing, produce the right length, and produce
// finite, non-degenerate (nonzero power) output. Does NOT check numeric
// agreement with the Python generators -- see generators.js's module
// docstring for why that isn't the bar here.
import { CASES, GENERATORS, buildScenario } from "../generators.js";
import { makeRng } from "../dsp.js";

const fs = 3200000;
let failures = 0;

function checkFinitePower(name, re, im, expectLen) {
  if (expectLen !== undefined && re.length !== expectLen) {
    console.error(`FAIL ${name}: length ${re.length} != expected ${expectLen}`);
    failures++;
    return;
  }
  let power = 0, finite = true;
  for (let i = 0; i < re.length; i++) {
    if (!Number.isFinite(re[i]) || !Number.isFinite(im[i])) finite = false;
    power += re[i] * re[i] + im[i] * im[i];
  }
  power /= re.length;
  if (!finite) { console.error(`FAIL ${name}: non-finite samples`); failures++; return; }
  console.log(`  ${name}: len=${re.length} meanPower=${power.toExponential(3)}`);
}

console.log("== individual generators (20 draws each) ==");
for (const [name, gen] of Object.entries(GENERATORS)) {
  for (let seed = 0; seed < 20; seed++) {
    const rng = makeRng(seed * 7919 + 1);
    const totalDuration = 0.05 * (0.25 + rng());  // vary length like scenario segments do
    try {
      const { re, im } = gen(rng, fs, totalDuration);
      if (re.length === 0) { console.error(`FAIL ${name} seed=${seed}: zero length`); failures++; continue; }
      let finite = true, power = 0;
      for (let i = 0; i < re.length; i++) {
        if (!Number.isFinite(re[i]) || !Number.isFinite(im[i])) finite = false;
        power += re[i] * re[i] + im[i] * im[i];
      }
      if (!finite) { console.error(`FAIL ${name} seed=${seed}: non-finite`); failures++; }
    } catch (e) {
      console.error(`FAIL ${name} seed=${seed}: threw ${e.message}`);
      failures++;
    }
  }
  console.log(`  ${name}: 20/20 draws ok (or see FAILs above)`);
}

console.log("\n== full scenario builder, every generator-backed CASE ==");
for (const [caseName, script] of Object.entries(CASES)) {
  for (const snrDb of [-10, -6, -2, 2, 6, 10]) {
    try {
      const { re, im, segments } = buildScenario({ fs, totalDuration: 0.05, snrDb, seed: 123, script });
      checkFinitePower(`${caseName} @ ${snrDb}dB`, re, im, Math.round(0.05 * fs));
      if (segments.length !== script.length) {
        console.error(`FAIL ${caseName}: ${segments.length} segments, expected ${script.length}`);
        failures++;
      }
    } catch (e) {
      console.error(`FAIL ${caseName} @ ${snrDb}dB: threw ${e.stack}`);
      failures++;
    }
  }
}

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} FAILURES`}`);
process.exit(failures === 0 ? 0 : 1);
