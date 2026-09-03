// The HTML panels of the RF Replay page, ported from
// src/ui/pages/rf_replay.py: the status line, the measured/model status
// block, the primary-detection card with its tier chips and full detection
// list, and the events table.
//
// Strings and structure are kept as close to the Python as the two languages
// allow, because these panels are what a judge reads -- including the caveats
// (est. prefixes, "display-only", the headline-confidence note), which are
// load-bearing under the provenance rule, not decoration.

import {
  MAX_LISTED_DETECTIONS, TIER_COLOR, byPriority, estimateSnrDb,
  headlineEvent, headlineIsConfident, tierOfClasses,
} from "./analysis.js";

const PANEL = "#FFFFFF";
const GRID = "#DFE3D9";
const TEXT = "#121C27";
const TEXT_DIM = "#5F6B72";
const BRAND_OLIVE = "#627143";

const pct = v => `${Math.round(v * 100)}%`;

/** rf_replay.py:_channel_state */
export function channelState(tiers) {
  if (!tiers.length) return "";
  const empty = tiers.filter(t => t === "Empty").length / tiers.length * 100;
  if (empty >= 90) {
    return `<span style="color:${BRAND_OLIVE};font-weight:700;">CHANNEL EMPTY — ${empty.toFixed(0)}% of windows</span>`;
  }
  return `channel ${empty.toFixed(0)}% empty`;
}

/** rf_replay.py:_render's `head` markdown line. */
export function headerLine({ source, snrKnown, trueSnrDb, snrCapped, requestedSnrDb,
                              modelLabel, caseNote, durationMs, nWindows, hop, nEvents, tiers }) {
  // "capped from X dB" says what was requested AND what was delivered: a
  // civilian recording carries noise at its own bin, so a cleaner SNR than
  // that bin is not achievable and the dropdown's value was not honoured.
  const capNote = snrCapped && requestedSnrDb !== null
    ? ` (capped from ${requestedSnrDb.toFixed(0)} dB)` : "";
  const snrNote = (snrKnown && trueSnrDb !== null && trueSnrDb !== undefined)
    ? `SNR ${trueSnrDb.toFixed(1)} dB KNOWN${capNote} &nbsp;·&nbsp; ` : "";
  return (
    `<strong>● REPLAY</strong> &nbsp; source <code>${source}</code> &nbsp;·&nbsp; ` +
    `BASEBAND · fs 3.2 MHz &nbsp;·&nbsp; ${snrNote}` +
    `${modelLabel} &nbsp;·&nbsp; ` +
    (caseNote ? `${caseNote} &nbsp;·&nbsp; ` : "") +
    `${durationMs.toFixed(1)} ms &nbsp;·&nbsp; ` +
    `${nWindows} windows @ hop ${hop} &nbsp;·&nbsp; ` +
    `<strong>${nEvents} emitter events</strong> &nbsp;·&nbsp; ` +
    channelState(tiers)
  );
}

/** rf_replay.py:_render's status_html -- MEASURED and MODEL figures, each
 * labelled with which it is. */
export function statusBlock({ occupancyValue, nEvents, nWindows, hop, durationMs, emptyPct }) {
  const pad = (s, n) => String(s).padStart(n);
  return `<div style="font-family:monospace;background:${PANEL};padding:14px;border-radius:6px;color:${TEXT};line-height:1.8;">` +
    `Occupancy   ${pad((occupancyValue * 100).toFixed(1), 5)}%   ` +
    `<span style="color:${TEXT_DIM};">measured — fraction of the spectrogram above the noise floor</span><br>` +
    `Detections  ${pad(nEvents, 5)}   ` +
    `<span style="color:${TEXT_DIM};">model — grouped events, not windows</span><br>` +
    `Windows     ${pad(nWindows, 5)}   ` +
    `<span style="color:${TEXT_DIM};">hop ${hop} · ${durationMs.toFixed(1)} ms capture</span><br>` +
    `Channel     ${pad(emptyPct.toFixed(0), 5)}%   ` +
    `<span style="color:${TEXT_DIM};">model — windows reported as empty spectrum</span></div>`;
}

/** rf_replay.py:_detection_list_html */
function detectionList(events) {
  const ordered = byPriority(events);
  const rows = ordered.slice(0, MAX_LISTED_DETECTIONS).map(e => {
    const tier = tierOfClasses(e.classes);
    const conf = e.classes.map(c => `${c} ${pct(e.peak[c])}`).join(" · ");
    return `<div style="display:flex;align-items:baseline;gap:8px;padding:3px 0;border-bottom:1px solid ${GRID};">` +
      `<span style="width:8px;height:8px;border-radius:50%;background:${TIER_COLOR[tier]};display:inline-block;flex:0 0 auto;"></span>` +
      `<span style="color:${TIER_COLOR[tier]};font-weight:600;font-size:12px;min-width:120px;">${conf}</span>` +
      `<span style="color:${TEXT_DIM};font-family:monospace;font-size:11px;">` +
      `${(e.startUs / 1000).toFixed(2)} ms · ${(e.durationUs / 1000).toFixed(2)} ms</span></div>`;
  });
  const hidden = ordered.length - rows.length;
  if (hidden > 0) {
    rows.push(`<div style="color:${TEXT_DIM};font-size:11px;padding-top:6px;">+${hidden} more, lower priority</div>`);
  }
  return `<div style="margin-top:12px;padding-top:10px;border-top:1px solid ${GRID};">` +
    `<div style="color:${TEXT_DIM};font-size:11px;margin-bottom:4px;">ALL DETECTIONS · worst tier first</div>` +
    rows.join("") + `</div>`;
}

function tierChips(events) {
  const counts = {};
  for (const e of events) {
    const t = tierOfClasses(e.classes);
    counts[t] = (counts[t] ?? 0) + 1;
  }
  return Object.keys(counts).sort().map(t =>
    `<span style="display:inline-block;margin:4px 8px 0 0;padding:2px 10px;border-radius:9px;font-size:11px;font-weight:600;` +
    `background:${TIER_COLOR[t]}22;color:${TIER_COLOR[t]};">${t} ${counts[t]}</span>`).join("");
}

/** rf_replay.py:_render's latest_html -- the CHANNEL STATE / PRIMARY
 * DETECTION card. */
export function latestBlock(events, emptyPct, capture) {
  const channelEmpty = emptyPct >= 90;
  const chips = tierChips(events);

  if (channelEmpty) {
    let extra = "";
    const headline = headlineEvent(events);
    if (headline) {
      const color = TIER_COLOR[tierOfClasses(headline.classes)];
      extra = `<div style="margin-top:12px;padding-top:10px;border-top:1px solid ${GRID};color:${TEXT_DIM};font-size:12px;">` +
        `Isolated detection, not sustained: <span style="color:${color};font-weight:600;">${headline.label}</span> at ` +
        `${(headline.startUs / 1000).toFixed(2)} ms for ${(headline.durationUs / 1000).toFixed(2)} ms</div>`;
    }
    return `<div style="background:${PANEL};padding:16px;border-radius:6px;color:${TEXT};">` +
      `<div style="color:${TEXT_DIM};font-size:11px;">CHANNEL STATE</div>` +
      `<div style="font-size:22px;font-weight:700;color:${BRAND_OLIVE};margin:6px 0;">EMPTY</div>` +
      `<div style="color:${TEXT_DIM};font-family:monospace;font-size:12px;">${emptyPct.toFixed(0)}% of windows report no emitter</div>` +
      `${extra}</div>`;
  }

  if (!events.length) {
    return `<div style="background:${PANEL};padding:16px;border-radius:6px;color:${TEXT_DIM};">No emitter detected in this capture.</div>`;
  }

  const e = headlineEvent(events);
  const color = TIER_COLOR[tierOfClasses(e.classes)];
  const caveat = headlineIsConfident(events) ? "" :
    `<div style="color:${TEXT_DIM};font-size:11px;margin-top:6px;">nothing in this capture cleared 50% on the class setting its tier — showing the strongest available</div>`;

  return `<div style="background:${PANEL};padding:16px;border-radius:6px;color:${TEXT};">` +
    `<div style="color:${TEXT_DIM};font-size:11px;">PRIMARY DETECTION</div>` +
    `<div style="font-size:20px;font-weight:600;color:${color};margin:6px 0;">${e.label}</div>` +
    `<div style="color:${TEXT_DIM};font-family:monospace;font-size:12px;">` +
    `${(e.startUs / 1000).toFixed(2)} ms · ${(e.durationUs / 1000).toFixed(2)} ms long<br>` +
    e.classes.map(c => `${c} ${pct(e.peak[c])}`).join(" · ") + `</div>` +
    caveat + `<div>${chips}</div>` + detectionList(events) + `</div>`;
}

/** Cover block for the PDF export.
 *
 * Two jobs, and the second is the important one.
 *
 * It records the run's provenance -- case, SNR, hop, model, thresholds,
 * timestamp -- because a page of detections with no record of what produced
 * them cannot be checked or reproduced later.
 *
 * And it states plainly that this is a SYNTHESIZED capture. Every click
 * draws a fresh random seed, so what the document shows is a demonstration
 * of one scene with known ground truth, not a measurement of accuracy. The
 * measured figures live in the scorecard and on the Performance page; a
 * reader who mistook a good-looking synthetic run for benchmark evidence
 * would be misled by this document, which is the one thing it must not do.
 */
export function printHeaderHtml({ source, caseNote, snrDb, snrKnown, hop, nWindows,
                                   durationMs, modelLabel, nEvents, truth, thresholds }) {
  const now = new Date();
  const truthLine = (truth && truth.length)
    ? truth.map(s => `${s.className} ${(s.startS * 1000).toFixed(2)}–${(s.endS * 1000).toFixed(2)} ms`).join(" · ")
    : "none — uploaded capture, no ground truth exists for it";
  const thrLine = Object.entries(thresholds)
    .map(([c, v]) => `${c} ${v}`).join(" · ");

  const row = (k, v) => `<tr><td style="color:${TEXT_DIM};padding:2px 14px 2px 0;white-space:nowrap;">${k}</td>` +
    `<td style="font-family:monospace;">${v}</td></tr>`;

  return `<div style="border-bottom:2px solid ${BRAND_OLIVE};padding-bottom:10px;margin-bottom:12px;">` +
    `<div style="font-size:17px;font-weight:700;color:${TEXT};">OMNI — RF Replay capture report</div>` +
    `<div style="font-size:11px;color:${TEXT_DIM};margin-top:3px;">` +
    `Generated ${now.toISOString().slice(0, 19).replace("T", " ")}</div></div>` +

    `<div style="background:#FDF6EC;border:1px solid #B45309;color:#7a3c06;padding:9px 11px;` +
    `border-radius:5px;font-size:11px;line-height:1.55;margin-bottom:12px;">` +
    `<strong>This is a demonstration, not a measurement.</strong> The capture below was ` +
    `synthesized in the browser from a fresh random seed, so the detections show what the ` +
    `model does on one scene whose ground truth is known — they are not an accuracy result ` +
    `and no recall or precision figure should be read from them. Measured performance on the ` +
    `held-out test split is on the Performance page and in evals/scorecard.json.</div>` +

    `<table style="font-size:11px;border-collapse:collapse;margin-bottom:10px;">` +
    row("Source", source === "scenario" ? `synthesized scenario — ${caseNote.replace(/`/g, "")}` : "uploaded capture") +
    row("SNR", snrKnown && snrDb !== null ? `${snrDb.toFixed(1)} dB (known, per emitter)` : "unknown — not a generated scene") +
    row("Capture", `${durationMs.toFixed(1)} ms @ fs 3.2 MHz, baseband`) +
    row("Windows", `${nWindows} @ hop ${hop} (512-sample window, 160 µs)`) +
    row("Model", modelLabel) +
    row("Detections", `${nEvents} emitter events (smoothed, gated, held — display rules on)`) +
    row("Ground truth", truthLine) +
    row("Thresholds", thrLine) +
    `</table>`;
}

/** rf_replay.py:_rows -- one row per event; class names and peak
 * confidences share one column (splitting them doubled the table width). */
export function eventRows(events) {
  return events.map((e, i) => [
    i + 1,
    (e.startUs / 1000).toFixed(2),
    (e.durationUs / 1000).toFixed(2),
    e.classes.map(c => `${c} ${pct(e.peak[c])}`).join(" · "),
  ]);
}
