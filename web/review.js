// Recommended corrections: where should a human look?
//
// With ground truth (a synthesized scenario, or a SigMF recording whose
// annotations name the classes), the model's detections are compared with
// the truth over time, and every stretch where they disagree becomes a
// suggestion: "27.2–40.1 ms: model missed FHSS". Without truth (a real
// recording), nothing can say the model is WRONG -- so the suggestions are
// the detections it was least sure about, for a human to check.
//
// A suggestion is exactly what the correction form needs: a span, what the
// model said there, and (when truth exists) what is really there. The human
// still decides, writes the reason and submits; nothing is corrected
// automatically.
//
// Pure functions, no DOM: web/test/review_check.mjs runs them in node.

const NOTHING = "NOISE_FLOOR";
const ORDER = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING", NOTHING];
const sortClasses = cs => [...new Set(cs)].sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));

/** Classes the model reported at time t (seconds), NOISE_FLOOR excluded. */
export function detectedAt(events, t) {
  const out = [];
  for (const e of events) {
    if (e.startUs / 1e6 <= t && t < e.endUs / 1e6) out.push(...e.classes.filter(c => c !== NOTHING));
  }
  return sortClasses(out);
}

/** Classes the truth says were active at time t. */
export function truthAt(truth, t) {
  return sortClasses((truth || []).filter(s => s.startS <= t && t < s.endS).map(s => s.className));
}

/** Classes the model reported anywhere in [a, b). */
export function detectedIn(events, a, b) {
  const out = [];
  for (const e of events) {
    if (e.startUs / 1e6 < b && a < e.endUs / 1e6) out.push(...e.classes.filter(c => c !== NOTHING));
  }
  return sortClasses(out);
}

function describe(pred, tru) {
  const missed = tru.filter(c => !pred.includes(c));
  const extra = pred.filter(c => !tru.includes(c));
  if (missed.length && !extra.length) return { kind: "missed", text: `Model missed ${missed.join(" + ")}` };
  if (extra.length && !missed.length) {
    return {
      kind: "false_alarm",
      text: `False alarm: ${extra.join(" + ")}` + (tru.length ? ` (truth: ${tru.join(" + ")} only)` : " (truth: nothing there)"),
    };
  }
  return { kind: "wrong_class", text: `Model missed ${missed.join(" + ")} and falsely reported ${extra.join(" + ")}` };
}

/**
 * Every stretch of at least `minS` where the model and the truth disagree.
 * The timeline is cut at every event and truth edge; each piece compares
 * "what the model said" with "what was there"; neighbouring pieces with the
 * same disagreement are merged.
 */
export function compareWithTruth({ events, truth, durationS, minS = 0.0003 }) {
  const cuts = new Set([0, durationS]);
  for (const e of events) { cuts.add(e.startUs / 1e6); cuts.add(e.endUs / 1e6); }
  for (const s of truth) { cuts.add(s.startS); cuts.add(s.endS); }
  const edges = [...cuts].filter(t => t >= 0 && t <= durationS).sort((a, b) => a - b);

  const runs = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i], b = edges[i + 1];
    if (b - a <= 1e-12) continue;
    const mid = (a + b) / 2;
    const pred = detectedAt(events, mid), tru = truthAt(truth, mid);
    const key = `${pred.join(",")}|${tru.join(",")}`;
    if (pred.join() === tru.join()) continue;
    const last = runs[runs.length - 1];
    if (last && last.key === key && Math.abs(last.endS - a) < 1e-9) last.endS = b;
    else runs.push({ key, startS: a, endS: b, predicted: pred, truth: tru });
  }

  return runs.filter(r => r.endS - r.startS >= minS).map(r => {
    const { kind, text } = describe(r.predicted, r.truth);
    return {
      kind, text, startS: r.startS, endS: r.endS, predicted: r.predicted,
      suggested: r.truth.length ? r.truth : [NOTHING],
      reason: `Ground truth says ${r.truth.length ? r.truth.join(" + ") + " active" : "nothing active"} `
        + `from ${(r.startS * 1000).toFixed(2)} to ${(r.endS * 1000).toFixed(2)} ms; `
        + `the model reported ${r.predicted.length ? r.predicted.join(" + ") : "nothing"}.`,
    };
  });
}

/** No truth: detections whose best class is below `below`, least sure first. */
export function lowConfidence(events, { below = 0.4 } = {}) {
  return events
    .map(e => {
      const classes = e.classes.filter(c => c !== NOTHING);
      const conf = Math.max(0, ...classes.map(c => e.peak[c] ?? 0));
      return { e, classes, conf };
    })
    .filter(x => x.classes.length && x.conf < below)
    .sort((a, b) => a.conf - b.conf)
    .map(({ e, classes, conf }) => ({
      kind: "uncertain",
      text: `Low confidence (${Math.round(conf * 100)}%): ${classes.join(" + ")}, please check`,
      startS: e.startUs / 1e6, endS: e.endUs / 1e6, predicted: classes,
      suggested: null, reason: "",
    }));
}

// The classes the competition judges. A disagreement about one of them is
// what an operator should look at first.
const JUDGED = ["LFM_RADAR", "FHSS", "JAMMING"];
const KIND_RANK = { missed: 0, wrong_class: 1, false_alarm: 2 };

/** Most important first: involves a judged class, then misses before false
 *  alarms (a missed jammer costs more than a spurious QPSK), then longest. */
export function rankSuggestions(items) {
  const judged = s => {
    const diff = [...s.predicted.filter(c => !s.suggested.includes(c)),
                  ...s.suggested.filter(c => !s.predicted.includes(c))];
    return diff.some(c => JUDGED.includes(c)) ? 0 : 1;
  };
  return [...items].sort((a, b) => judged(a) - judged(b)
    || KIND_RANK[a.kind] - KIND_RANK[b.kind]
    || (b.endS - b.startS) - (a.endS - a.startS));
}

/** Ground truth when there is some, low confidence when there is not. */
export function suggestCorrections({ events, truth, durationS }) {
  return truth && truth.length
    ? { source: "truth", items: rankSuggestions(compareWithTruth({ events, truth, durationS })) }
    : { source: "confidence", items: lowConfidence(events) };
}

/**
 * What a click at time t on the timeline should open: the suggestion under
 * the cursor if there is one, else the detection event under it, else a
 * 1 ms span around it (a signal the model did not report at all).
 */
export function spanAt(t, { suggestions = [], events = [], durationS }) {
  const s = suggestions.find(x => x.startS <= t && t < x.endS);
  if (s) return { ...s, from: "suggestion" };
  const e = events.find(x => x.startUs / 1e6 <= t && t < x.endUs / 1e6
    && x.classes.some(c => c !== NOTHING));
  if (e) {
    return { startS: e.startUs / 1e6, endS: e.endUs / 1e6, from: "event",
             predicted: sortClasses(e.classes.filter(c => c !== NOTHING)), suggested: null, reason: "" };
  }
  const a = Math.max(0, t - 0.0005), b = Math.min(durationS, t + 0.0005);
  return { startS: a, endS: b, from: "empty", predicted: [], suggested: null, reason: "" };
}
