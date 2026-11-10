// Class-name helpers for the 60-class model, on top of the generated
// model_meta.js (web/build.py).
//
// The model names 60 fine classes (JAM_BARRAGE, RADAR_FRANK, 8PSK, ...).
// Some inputs still speak in FAMILY names: the scenario generator
// (generators.js) labels its truth "JAMMING" without a jammer type, and a
// SigMF annotation may say "jamming". A family name in a truth is therefore
// matched by any class of that family -- a JAM_SWEPT detection over a
// "JAMMING" truth segment is right, not a miss plus a false alarm.

import { CLASSES, FAMILY_OF, GROUPS, TIER_OF as CLASS_TIER } from "./model_meta.js";

export { CLASSES, FAMILY_OF, GROUPS };

/** Family names a truth or annotation may use instead of a class. Only
 *  JAMMING is used today; every family is accepted. */
export const FAMILY_NAMES = Object.keys(GROUPS).filter(g => g !== "NOISE");

/** Tier of a class, or of a family name (all classes of a family share a tier
 *  except where web/build.py CLASS_TIER says otherwise; a family takes its
 *  first class's). */
export const TIER_OF = { ...CLASS_TIER };
for (const g of FAMILY_NAMES) if (!(g in TIER_OF)) TIER_OF[g] = CLASS_TIER[GROUPS[g][0]];

/** Classes of the hostile tier (every jammer type). */
export const JAMMERS = CLASSES.filter(c => CLASS_TIER[c] === "Hostile");

/** Military + hostile classes: what an operator is alerted about. */
export const THREAT_CLASSES = CLASSES.filter(c => CLASS_TIER[c] === "Hostile" || CLASS_TIER[c] === "Military");

/** True when a detected class answers a truth name: the same class, or a
 *  member of the family the truth names. */
export function answers(detected, truthName) {
  return detected === truthName || FAMILY_OF[detected] === truthName;
}

/** Detected classes rewritten in a truth's vocabulary: a class whose family
 *  the truth names becomes that family name, so the two lists compare. */
export function inTruthTerms(detected, truthNames) {
  const fams = new Set(truthNames.filter(n => FAMILY_NAMES.includes(n)));
  return [...new Set(detected.map(c => (fams.has(FAMILY_OF[c]) ? FAMILY_OF[c] : c)))];
}

/** Display order: the model's own class order, family names after their
 *  family's classes, NOISE_FLOOR last. */
export const ORDER = [...CLASSES.filter(c => c !== "NOISE_FLOOR"), ...FAMILY_NAMES, "NOISE_FLOOR"];
