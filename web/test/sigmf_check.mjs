// SigMF reading (web/sigmf.js): datatypes, byte order, annotations -> TRUTH,
// pairing rules, and the archive form. Pure functions, so this runs in node:
//     node web/test/sigmf_check.mjs
//
// The last block reads a real pack from scripts/export_verification_pack.py
// (--format both) when one exists, and checks the .sigmf archive, the
// .sigmf-meta/.sigmf-data pair and the plain .f32 all decode to the same IQ.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { annotationsToTruth, decodeSamples, parseDatatype, parseMeta, readSigmf } from "../sigmf.js";

let failures = 0;
function check(name, ok, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}
function throws(fn) { try { fn(); return false; } catch { return true; } }
async function rejects(p) { try { await p; return false; } catch { return true; } }

/** A stand-in for a browser File. */
function file(name, bytes) {
  const u8 = typeof bytes === "string" ? new TextEncoder().encode(bytes) : new Uint8Array(bytes);
  return { name, arrayBuffer: async () => u8.slice().buffer };
}
function meta(over = {}, annotations = []) {
  return JSON.stringify({
    global: { "core:datatype": "cf32_le", "core:sample_rate": 3200000, "core:version": "1.2.0", ...over },
    captures: [{ "core:sample_start": 0, "core:frequency": 2.4e9 }],
    annotations,
  });
}

// --- datatypes ---------------------------------------------------------------

check("cf32_le parses", parseDatatype("cf32_le").bytes === 4);
check("ci8 needs no byte order", parseDatatype("ci8").bytes === 1);
check("real datatype rejected", throws(() => parseDatatype("rf32_le")));
check("multi-byte without byte order rejected", throws(() => parseDatatype("ci16")));
check("nonsense rejected", throws(() => parseDatatype("complex64")));

{
  const f32 = new Float32Array([0.5, -0.25, 1, 2]);
  const { re, im } = decodeSamples(f32.buffer, parseDatatype("cf32_le"));
  check("cf32_le decodes I,Q interleaved", re[0] === 0.5 && im[0] === -0.25 && re[1] === 1 && im[1] === 2);
}
{
  const buf = new ArrayBuffer(8), v = new DataView(buf);
  v.setFloat32(0, 0.5, false); v.setFloat32(4, -1.5, false);
  const { re, im } = decodeSamples(buf, parseDatatype("cf32_be"));
  check("cf32_be honours big-endian", re[0] === 0.5 && im[0] === -1.5);
}
{
  const { re, im } = decodeSamples(new Int16Array([16384, -32768]).buffer, parseDatatype("ci16_le"));
  check("ci16_le scaled to ±1", re[0] === 0.5 && im[0] === -1);
}
{
  // An RTL-SDR's idle output is 127/128: unsigned samples must be re-centred.
  const { re, im } = decodeSamples(new Uint8Array([127, 128, 255, 0]).buffer, parseDatatype("cu8"));
  check("cu8 re-centred on zero", Math.abs(re[0]) < 0.01 && Math.abs(im[0]) < 0.01 && re[1] > 0.99 && im[1] < -0.99);
}

// --- meta --------------------------------------------------------------------

check("meta without sample rate rejected",
  throws(() => parseMeta(JSON.stringify({ global: { "core:datatype": "cf32_le" } }))));
check("meta that is not JSON rejected", throws(() => parseMeta("{nope")));
check("frequency read from first capture", parseMeta(meta()).frequency === 2.4e9);

// --- annotations -> TRUTH ----------------------------------------------------

{
  const t = annotationsToTruth([
    { "core:sample_start": 0, "core:sample_count": 3200, "core:label": "lfm radar" },
    { "core:sample_start": 3200, "core:sample_count": 1600, "core:label": "JAMMING" },
    { "core:sample_start": 0, "core:sample_count": 10, "core:label": "operator note" },
    { "core:sample_start": 0, "core:label": "FHSS" },   // no count: skipped
  ], 3200000);
  check("known labels become truth segments, loosely matched",
    t.length === 2 && t[0].className === "LFM_RADAR" && t[1].className === "JAMMING");
  check("segment times in seconds", t[0].endS === 0.001 && t[1].startS === 0.001 && t[1].endS === 0.0015);
  check("no usable annotations -> no truth", annotationsToTruth([{ "core:label": "x" }], 1) === null);
}

// --- pairing -----------------------------------------------------------------

{
  const data = new Float32Array(2 * 600).fill(0.1);
  const rec = await readSigmf([file("a.sigmf-meta", meta()), file("a.sigmf-data", data.buffer)]);
  check("meta+data pair reads", rec.re.length === 600 && rec.warnings.length === 0);

  check("data without meta rejected", await rejects(readSigmf([file("a.sigmf-data", data.buffer)])));
  check("mismatched base names rejected",
    await rejects(readSigmf([file("a.sigmf-meta", meta()), file("b.sigmf-data", data.buffer)])));

  const slow = await readSigmf([file("s.sigmf-meta", meta({ "core:sample_rate": 2.4e6 })),
                                file("s.sigmf-data", data.buffer)]);
  check("wrong sample rate warns", slow.warnings.some(w => /2\.400 MS\/s/.test(w)));

  const ragged = new Uint8Array(data.buffer.byteLength + 3);
  ragged.set(new Uint8Array(data.buffer));
  const r = await readSigmf([file("r.sigmf-meta", meta()), file("r.sigmf-data", ragged)]);
  check("trailing partial sample warns", r.re.length === 600 && r.warnings.some(w => /trailing/.test(w)));
}

// --- a real exported pack, when present ------------------------------------

{
  const here = dirname(fileURLToPath(import.meta.url));
  const pack = join(here, "..", "..", "verification_pack");
  const base = join(pack, "mixed_sequence");
  if (existsSync(base + ".sigmf") && existsSync(base + ".sigmf-meta") && existsSync(base + ".f32")) {
    const archive = await readSigmf([file("mixed_sequence.sigmf", readFileSync(base + ".sigmf"))]);
    const pair = await readSigmf([
      file("mixed_sequence.sigmf-meta", readFileSync(base + ".sigmf-meta")),
      file("mixed_sequence.sigmf-data", readFileSync(base + ".sigmf-data")),
    ]);
    const buf = readFileSync(base + ".f32");
    const raw = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const same = archive.re.length === raw.length / 2
      && archive.re.every((x, i) => x === raw[2 * i] && archive.im[i] === raw[2 * i + 1])
      && pair.re.every((x, i) => x === archive.re[i]);
    check("exported archive, pair and .f32 decode identically", same, `${archive.re.length} samples`);
    check("exported annotations become 5 truth segments",
      archive.truth?.length === 5 && archive.truth.map(t => t.className).join() === "NOISE_FLOOR,LFM_RADAR,FHSS,JAMMING,QPSK");
  } else {
    console.log("skip  no verification_pack/ (python scripts/export_verification_pack.py --format both)");
  }
}

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
