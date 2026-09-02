// Parity for the civilian path: the exported library must match
// civilian_library() window-for-window, the case list must match CASES, and
// the SNR-capping arithmetic must match load_scenario. Run
// civilian_reference.py and web/build.py first.
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { CASES, caseNeedsLibrary } from "../generators.js";

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(here, "civilian_reference.json"), "utf8"));
const dataDir = join(here, "..", "data");
const manifest = JSON.parse(readFileSync(join(dataDir, "civilian_library.json"), "utf8"));

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`  ${ok ? "OK  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
  if (!ok) failures++;
};

console.log("== exported library vs civilian_library() ==");
check("library SNR bin", manifest.snr_db === ref.library_snr_db,
      `got ${manifest.snr_db}, want ${ref.library_snr_db}`);
for (const [cls, want] of Object.entries(ref.library)) {
  const meta = manifest.classes[cls];
  if (!meta) { check(`${cls} present`, false); continue; }
  const buf = readFileSync(join(dataDir, meta.file));
  const raw = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  let sum = 0, absmax = 0;
  for (const v of raw) { sum += v; absmax = Math.max(absmax, Math.abs(v)); }
  const okN = meta.n === want.n;
  const okLen = raw.length === want.n * 2 * manifest.window_len;
  // float32 summed over ~22k values: relative tolerance, not exact
  const okSum = Math.abs(sum - want.sum) <= Math.max(1e-3, Math.abs(want.sum) * 1e-5);
  const okMax = Math.abs(absmax - want.absmax) < 1e-5;
  check(`${cls}`, okN && okLen && okSum && okMax,
        `n=${meta.n}/${want.n} sum=${sum.toFixed(4)}/${want.sum.toFixed(4)} absmax=${absmax.toFixed(5)}/${want.absmax.toFixed(5)}`);
}

console.log("\n== case list ==");
const jsNames = Object.keys(CASES);
check("all 10 cases present", jsNames.length === ref.case_names.length,
      `got ${jsNames.length}, want ${ref.case_names.length}`);
const missing = ref.case_names.filter(n => !jsNames.includes(n));
check("names match Python CASES", missing.length === 0,
      missing.length ? `missing: ${missing.join(", ")}` : "");

console.log("\n== needs-library classification ==");
let mism = 0;
for (const [name, want] of Object.entries(ref.cases)) {
  if (!CASES[name]) continue;
  if (caseNeedsLibrary(CASES[name]) !== want.needs_library) {
    console.log(`    ${name}: got ${caseNeedsLibrary(CASES[name])}, want ${want.needs_library}`);
    mism++;
  }
}
check("every case", mism === 0, `${mism} mismatched`);

console.log("\n== SNR capping (load_scenario arithmetic) ==");
let capMism = 0;
for (const [name, want] of Object.entries(ref.cases)) {
  if (!CASES[name]) continue;
  const needs = caseNeedsLibrary(CASES[name]);
  const librarySnrDb = needs ? manifest.snr_db : null;
  for (const [snrStr, w] of Object.entries(want.per_snr)) {
    const snrDb = Number(snrStr);
    const trueSnr = (needs && librarySnrDb !== null) ? Math.min(snrDb, librarySnrDb) : snrDb;
    const capped = needs && librarySnrDb !== null && snrDb > librarySnrDb;
    if (trueSnr !== w.true_snr_db || capped !== w.snr_capped) {
      console.log(`    ${name} @ ${snrDb}: got (${trueSnr}, ${capped}), want (${w.true_snr_db}, ${w.snr_capped})`);
      capMism++;
    }
  }
}
check("every case x SNR bin", capMism === 0, `${capMism} mismatched`);

console.log(`\n${failures === 0 ? "ALL PASS" : `${failures} FAILURES`}`);
process.exit(failures === 0 ? 0 : 1);
