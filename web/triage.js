// Triage: what the operator must be TOLD about a capture, decided from the
// model's own scores. Two independent questions, so a capture can raise both:
//
//   ALERT        is there a military or hostile emitter? (act on it)
//                "confirmed" when the model is clearly past its threshold,
//                "possible" when it only just is -- or, for JAMMING, when the
//                score is high but still under the strict 0.89 reporting
//                threshold: missing a jammer costs more than a false alarm.
//
//   REVIEW FLAG  did the model only just make up its mind? (a human checks)
//                A detection is a CLOSE CALL when its score is under
//                CLOSE_CALL x that class's own threshold. The scores are not
//                "percent sure": each class is reported once it crosses its
//                calibrated threshold (16-24% for most, 89% for JAMMING), so
//                closeness is measured against that line, not against 80%.
//
// Kept pure (no DOM) so the rule can be swapped for a measured one -- e.g.
// the score above which test-set detections are right 90% of the time.

import { THRESHOLDS } from "./thresholds.js";

export const CLOSE_CALL = 1.5;          // under 1.5x the class threshold = close call
export const POSSIBLE_JAMMING = 0.5;    // unreported jamming score that still alerts
export const THREAT_CLASSES = ["LFM_RADAR", "FHSS", "JAMMING"];

/** The score a detection must reach to be "clear", never above halfway
 *  between the threshold and 100% (JAMMING's 0.89 x 1.5 would exceed 1). */
export function clearLine(cls) {
  const t = THRESHOLDS[cls] ?? 0.5;
  return Math.min(t * CLOSE_CALL, (t + 1) / 2);
}

/**
 * events:      resolveSession(...).emitterEvents ({classes, peak, startUs, endUs})
 * jammingPeak: the highest raw JAMMING score anywhere in the capture
 * Returns { alertLevel: "confirmed"|"possible"|null, threats: [{cls, peak, sure}],
 *           needsReview, closeCalls: [{cls, peak, line, startUs, endUs}] }.
 */
export function triage(events, jammingPeak = 0) {
  const threats = new Map();
  const closeCalls = [];
  for (const e of events) {
    for (const cls of e.classes) {
      if (cls === "NOISE_FLOOR") continue;
      const peak = e.peak[cls] ?? 0, line = clearLine(cls), sure = peak >= line;
      if (!sure) closeCalls.push({ cls, peak, line, startUs: e.startUs, endUs: e.endUs });
      if (THREAT_CLASSES.includes(cls)) {
        const prev = threats.get(cls);
        if (!prev || peak > prev.peak) threats.set(cls, { cls, peak, sure });
      }
    }
  }
  if (!threats.has("JAMMING") && jammingPeak >= POSSIBLE_JAMMING) {
    threats.set("JAMMING", { cls: "JAMMING", peak: jammingPeak, sure: false, unreported: true });
  }
  const list = [...threats.values()].sort((a, b) => b.peak - a.peak);
  const alertLevel = !list.length ? null : list.some(t => t.sure) ? "confirmed" : "possible";
  return {
    alertLevel, threats: list, closeCalls,
    needsReview: closeCalls.length > 0 || list.some(t => t.unreported),
  };
}

/** Highest raw score of one class over every window of a classifier result. */
export function peakScore(result, classIndex) {
  let m = 0;
  for (let w = 0; w < result.nWindows; w++) m = Math.max(m, result.probs[w * result.nClasses + classIndex]);
  return m;
}
