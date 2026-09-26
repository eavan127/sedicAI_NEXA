// Recommended corrections (web/review.js): comparing detections with truth,
// low-confidence picks, and what a click on the timeline opens.
//     node web/test/review_check.mjs
import { compareWithTruth, detectedIn, lowConfidence, spanAt, suggestCorrections } from "../review.js";

let failures = 0;
function check(name, ok, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}
const near = (a, b) => Math.abs(a - b) < 1e-9;
const ev = (ms0, ms1, classes, peak = {}) => ({
  startUs: ms0 * 1000, endUs: ms1 * 1000, classes,
  peak: Object.fromEntries(classes.map(c => [c, peak[c] ?? 0.9])),
});
const seg = (cls, ms0, ms1) => ({ className: cls, startS: ms0 / 1000, endS: ms1 / 1000 });

// The screenshot's case, simplified: FHSS truth 17.5–40 ms, model only saw it to 28 ms;
// jamming detected 28.5–47.5 inside a 27.5–47.5 truth box; a BPSK false alarm.
const events = [ev(17.5, 28, ["FHSS"]), ev(28.5, 47.5, ["JAMMING"]), ev(30, 32, ["BPSK"], { BPSK: 0.22 })];
const truth = [seg("FHSS", 17.5, 40), seg("JAMMING", 27.5, 47.5)];
const got = compareWithTruth({ events, truth, durationS: 0.05 });
const text = got.map(s => `${(s.startS * 1000).toFixed(1)}-${(s.endS * 1000).toFixed(1)} ${s.text}`);

check("missed FHSS after the detection ends is found",
  got.some(s => s.kind === "missed" && s.suggested.includes("FHSS") && near(s.endS, 0.040)), text.join(" | "));
check("the gap before the jamming bar is a miss of JAMMING",
  got.some(s => near(s.startS, 0.028) && near(s.endS, 0.0285) && s.suggested.join() === "FHSS,JAMMING"));
check("a stretch with both a miss and a false alarm says both",
  got.some(s => s.kind === "wrong_class" && near(s.startS, 0.030) && near(s.endS, 0.032)
    && /missed FHSS and falsely reported BPSK/.test(s.text) && s.suggested.join() === "FHSS,JAMMING"));
check("a pure false alarm names the truth that was there",
  compareWithTruth({ events: [ev(0, 10, ["FHSS", "BPSK"])], truth: [seg("FHSS", 0, 10)], durationS: 0.01 })
    .some(s => s.kind === "false_alarm" && /False alarm: BPSK \(truth: FHSS only\)/.test(s.text)));
check("agreeing stretches produce nothing",
  !got.some(s => s.startS < 0.0275 && s.endS > 0.0175));
check("each suggestion carries a reason with its ms range",
  got.every(s => /from \d+\.\d\d to \d+\.\d\d ms/.test(s.reason)));
check("detection with nothing true there suggests NOISE_FLOOR",
  compareWithTruth({ events: [ev(1, 3, ["LFM_RADAR"])], truth: [seg("FHSS", 10, 20)], durationS: 0.05 })
    .some(s => s.kind === "false_alarm" && s.suggested.join() === "NOISE_FLOOR"));
check("slivers shorter than minS are dropped",
  compareWithTruth({ events: [ev(0, 10.0001, ["FHSS"])], truth: [seg("FHSS", 0, 10)], durationS: 0.05 }).length === 0);

// No truth: low confidence
{
  const lc = lowConfidence([ev(0, 5, ["QPSK"], { QPSK: 0.18 }), ev(5, 9, ["JAMMING"], { JAMMING: 0.99 }),
                            ev(9, 12, ["BPSK"], { BPSK: 0.3 })]);
  check("low-confidence detections listed, least sure first",
    lc.length === 2 && lc[0].predicted[0] === "QPSK" && /18%/.test(lc[0].text) && lc[1].predicted[0] === "BPSK");
  check("no truth means no suggested answer", lc.every(s => s.suggested === null));
  check("suggestCorrections picks the source",
    suggestCorrections({ events, truth, durationS: 0.05 }).source === "truth"
      && suggestCorrections({ events, truth: null, durationS: 0.05 }).source === "confidence");
}

// Ranking: judged-class problems first, misses before false alarms, longer first
{
  const ranked = suggestCorrections({ events, truth, durationS: 0.05 }).items;
  check("judged-class miss ranked first",
    ranked[0].kind === "missed" && ranked[0].suggested.includes("FHSS") && Math.abs(ranked[0].endS - 0.040) < 1e-9,
    ranked.map(s => s.text).join(" | "));
  const civ = suggestCorrections({ events: [ev(0, 9, ["BPSK"]), ev(20, 21, ["JAMMING"])], truth: [seg("QPSK", 0, 30)], durationS: 0.05 }).items;
  check("a short judged-class problem outranks a long civilian one", civ[0].predicted.includes("JAMMING"));
}

// Clicking the timeline
{
  const s = spanAt(0.035, { suggestions: got, events, durationS: 0.05 });
  check("click inside a mismatch opens that suggestion", s.from === "suggestion" && s.suggested.includes("FHSS"));
  const e = spanAt(0.020, { suggestions: got, events, durationS: 0.05 });
  check("click on an agreeing detection opens that event", e.from === "event" && e.predicted.join() === "FHSS");
  const n = spanAt(0.0495, { suggestions: [], events: [], durationS: 0.05 });
  check("click on empty time opens a 1 ms span, clipped to the capture",
    n.from === "empty" && near(n.startS, 0.049) && near(n.endS, 0.05) && n.predicted.length === 0);
  check("detectedIn unions classes over a dragged span", detectedIn(events, 0.025, 0.031).join() === "BPSK,FHSS,JAMMING");
}

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
