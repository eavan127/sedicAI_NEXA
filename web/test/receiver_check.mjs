// Receiver sources (web/receiver.js): the simulated SDR and file replay.
// Pure logic, no DOM or model, so this runs in node:
//     node web/test/receiver_check.mjs
import { FileReplay, SimulatedReceiver, dwellLevel, shouldStore, sliceTruth, toInterleavedF32 } from "../receiver.js";
import { FS, WINDOW_LEN } from "../model.js";

let failures = 0;
function check(name, ok, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}

// --- sliceTruth ----------------------------------------------------------------

{
  const truth = [{ className: "FHSS", startS: 0.001, endS: 0.004 }, { className: "JAMMING", startS: 0.006, endS: 0.009 }];
  const t = sliceTruth(truth, 0.003, 0.007);
  check("truth clipped to the dwell and re-based to its clock",
    t.length === 2 && t[0].className === "FHSS" && Math.abs(t[0].startS) < 1e-12
      && Math.abs(t[0].endS - 0.001) < 1e-12 && Math.abs(t[1].startS - 0.003) < 1e-12
      && Math.abs(t[1].endS - 0.004) < 1e-12);
  check("segments outside the dwell dropped", sliceTruth(truth, 0.0045, 0.0055).length === 0);
  check("no truth stays no truth", sliceTruth(null, 0, 1) === null);
}

// --- shouldStore: log changes, not every dwell ------------------------------------

check("first detection stored", shouldStore(null, ["JAMMING"]));
check("same classes (any order) not stored again", !shouldStore(["FHSS", "JAMMING"], ["JAMMING", "FHSS"]));
check("a new emitter stored", shouldStore(["FHSS"], ["FHSS", "JAMMING"]));
check("empty dwell never stored", !shouldStore(["FHSS"], []));

// --- dwellLevel ---------------------------------------------------------------------

{
  const lv = dwellLevel(new Float64Array([1, 1, 0, 0]), new Float64Array([0, 0, 1, 1]));
  check("unit-power dwell is 0 dB", Math.abs(lv.powerDb) < 1e-9 && Math.abs(lv.peakDb) < 1e-9);
  const loud = dwellLevel(new Float64Array([10, 0]), new Float64Array([0, 0]));
  check("peak and average differ for a burst", Math.abs(loud.peakDb - 20) < 1e-9 && Math.abs(loud.powerDb - 16.9897) < 1e-3);
}

// --- toInterleavedF32 ----------------------------------------------------------------

{
  const f = toInterleavedF32(new Float64Array([1, 2]), new Float64Array([-1, -2]));
  check("dwell bytes are I,Q,I,Q float32", f.length === 4 && f[0] === 1 && f[1] === -1 && f[3] === -2);
}

// --- FileReplay --------------------------------------------------------------------

async function drain(n, truth = null) {
  const re = new Float64Array(n).map((_, i) => i), im = new Float64Array(n);
  const rep = new FileReplay({ rec: { re, im, name: "t.f32", truth }, dwellS: 0.02, stepDelayMs: 0 });
  await rep.connect();
  const blocks = [];
  for (let b; (b = await rep.next());) blocks.push(b);
  return { rep, blocks };
}
const dwellN = Math.round(0.02 * FS);

{
  // 2.5 dwells: the last one is a half dwell, still long enough to classify.
  const n = Math.round(2.5 * dwellN);
  const { rep, blocks } = await drain(n, [{ className: "LFM_RADAR", startS: 0.015, endS: 0.025 }]);
  check("replay covers every sample in order, in dwell-sized blocks",
    blocks.length === 3 && blocks[0].re.length === dwellN && blocks[1].re[0] === dwellN
      && blocks[2].re[blocks[2].re.length - 1] === n - 1,
    blocks.map(b => b.re.length).join(","));
  check("annotation split across the two dwells it spans",
    blocks[0].truth.length === 1 && blocks[1].truth.length === 1 && blocks[2].truth.length === 0);
  check("progress reaches the end", Math.abs(rep.progress.doneS - n / FS) < 1e-9);
}
{
  // A tail shorter than one model window cannot be classified.
  const { blocks } = await drain(2 * dwellN + WINDOW_LEN - 1);
  check("sub-window tail is not padded into a fake dwell", blocks.length === 2);
}

// --- SimulatedReceiver ------------------------------------------------------------------

{
  let caseName = "Radar only";
  const sim = new SimulatedReceiver({
    dwellS: 0.02, seed: 1, stepDelayMs: 0,
    getSettings: () => ({ caseName, snrDb: 6 }),
    getLibrary: async () => { throw new Error("library should not be needed"); },
  });
  const steps = [];
  await sim.connect(t => steps.push(t));
  check("connect acts out the device handshake, labelled simulated",
    steps.length >= 5 && /Found NEXA-SIM/.test(steps[1]) && steps.some(t => /Tuned: 2\.400 GHz \(simulated\)/.test(t)),
    steps.join(" / "));
  check("simulated device reports tuning and gain", sim.describe().centreHz === 2.4e9 && sim.describe().gainDb === 30);
  const a = await sim.next();
  caseName = "Jamming only";                 // operator changes the dropdown mid-stream
  const b = await sim.next();
  check("dwell length matches the setting", a.re.length === Math.round(0.02 * FS));
  check("each dwell reads the current case", a.truth[0].className === "LFM_RADAR" && b.truth[0].className === "JAMMING");
  check("dwells are numbered", a.index === 0 && b.index === 1 && /dwell 2/.test(b.caseNote));
  check("simulated SNR is known", a.snrDb !== null && a.source === "scenario");
}

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
