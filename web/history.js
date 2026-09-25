// The History page: every stored analysis, summarised and listed.
//
// The rest of the app answers "what is in THIS capture". This answers "what
// have we seen": how many captures, how they split by tier, which classes
// keep appearing, and how the SNR of the material we are fed is distributed.
//
// summarise() is pure and takes plain records, so web/test/history_check.mjs
// can check the arithmetic in node with no DOM. Everything that touches the
// document lives below it and is called only from main.js.

import { TIER_COLOR, TIER_PRIORITY } from "./analysis.js";

export const SNR_BINS = [
  { label: "< −8", lo: -Infinity, hi: -8 },
  { label: "−8…−4", lo: -8, hi: -4 },
  { label: "−4…0", lo: -4, hi: 0 },
  { label: "0…4", lo: 0, hi: 4 },
  { label: "4…8", lo: 4, hi: 8 },
  { label: "≥ 8", lo: 8, hi: Infinity },
];

/**
 * Aggregate stored records for the dashboard.
 *
 * Counts are per CAPTURE, not per window: "3 captures contained JAMMING" is
 * the operational fact, where "1,900 windows" mostly measures how long the
 * captures were. n_windows is still summed, reported separately as total
 * material processed.
 */
export function summarise(records) {
  const byTier = {};
  const byClass = {};
  const bySource = { scenario: 0, upload: 0 };
  const snr = SNR_BINS.map(b => ({ ...b, count: 0 }));
  const byDay = new Map();

  let windows = 0, seconds = 0, withSnr = 0, snrSum = 0;

  for (const r of records) {
    byTier[r.verdict] = (byTier[r.verdict] || 0) + 1;
    for (const c of r.classes_detected || []) byClass[c] = (byClass[c] || 0) + 1;
    if (r.source in bySource) bySource[r.source]++;
    windows += r.n_windows || 0;
    seconds += r.duration_s || 0;

    if (typeof r.snr_db === "number") {
      withSnr++; snrSum += r.snr_db;
      // hi is exclusive so a capture at exactly −8 dB lands in one bin only.
      const bin = snr.find(b => r.snr_db >= b.lo && r.snr_db < b.hi);
      if (bin) bin.count++;
    }
    const day = (r.created_at || "").slice(0, 10);
    if (day) byDay.set(day, (byDay.get(day) || 0) + 1);
  }

  return {
    total: records.length,
    windows,
    seconds: Math.round(seconds * 1000) / 1000,
    meanSnrDb: withSnr ? Math.round((snrSum / withSnr) * 10) / 10 : null,
    byTier,
    byClass,
    bySource,
    snr,
    // Ascending: a timeline reads left to right even though the table below
    // it is newest-first.
    byDay: [...byDay.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)),
  };
}

/** Records whose stored fields match every active filter. Empty/absent
 *  filters match everything, so the default view is "all". */
export function applyFilters(records, { tier = "", cls = "", source = "", query = "" } = {}) {
  const q = query.trim().toLowerCase();
  return records.filter(r => {
    if (tier && r.verdict !== tier) return false;
    if (cls && !(r.classes_detected || []).includes(cls)) return false;
    if (source && r.source !== source) return false;
    if (q) {
      const hay = [r.file_name, r.case_note, r.model, r.verdict,
                    ...(r.classes_detected || [])].filter(Boolean).join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

// ---------------------------------------------------------------------------
// Rendering (DOM; imported by main.js, never by the node tests)
// ---------------------------------------------------------------------------

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function fmtTime(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso ?? "") : d.toLocaleString();
}

export function summaryHtml(s) {
  const tiles = [
    ["Captures analysed", s.total],
    ["Windows classified", s.windows.toLocaleString()],
    ["Signal time", `${s.seconds.toFixed(2)} s`],
    ["Mean SNR", s.meanSnrDb == null ? "—" : `${s.meanSnrDb} dB`],
  ];
  const tierRow = TIER_PRIORITY.map(t => {
    const n = s.byTier[t] || 0;
    const pct = s.total ? Math.round((n / s.total) * 100) : 0;
    return `<span class="tier-chip" style="border-color:${TIER_COLOR[t] || "#888"}">
      <b style="color:${TIER_COLOR[t] || "#888"}">${t}</b> ${n} <span class="dim">(${pct}%)</span></span>`;
  }).join("");
  return `<div class="kpi-row">${tiles.map(([k, v]) =>
    `<div class="kpi"><div class="kpi-label">${esc(k)}</div><div class="kpi-value">${esc(v)}</div></div>`).join("")}
    </div><div class="tier-row">${tierRow}</div>`;
}

/** Horizontal bars, drawn as divs rather than on a canvas: they reflow with
 *  the panel, carry their own text labels, and survive the print stylesheet
 *  that produces the PDF. */
export function barsHtml(pairs, { empty = "Nothing recorded yet.", sort = true } = {}) {
  // sort=false matters for any axis that has its OWN order: SNR bands read
  // low-to-high and days read oldest-to-newest, and sorting those by count
  // turns the chart into a ranking of bins that no longer means anything.
  const rows = sort ? [...pairs].sort((a, b) => b[1] - a[1]) : [...pairs];
  const max = Math.max(1, ...rows.map(r => r[1]));
  if (!rows.length) return `<div class="note">${esc(empty)}</div>`;
  return `<div class="barlist">${rows.map(([label, n]) => `
    <div class="barrow">
      <div class="barlabel">${esc(label)}</div>
      <div class="bartrack"><div class="barfill" style="width:${(n / max) * 100}%"></div></div>
      <div class="barvalue">${n}</div>
    </div>`).join("")}</div>`;
}

export function tableHtml(records) {
  if (!records.length) {
    return `<div class="note">No stored analyses match. Analyse a capture on RF Replay,
             or clear the filters.</div>`;
  }
  const rows = records.map(r => `
    <tr>
      <td>${esc(fmtTime(r.created_at))}</td>
      <td>${esc(r.file_name || r.case_note || r.source)}</td>
      <td><span style="color:${TIER_COLOR[r.verdict] || "inherit"}"><b>${esc(r.verdict)}</b></span></td>
      <td>${esc((r.classes_detected || []).join(", ") || "—")}</td>
      <td class="num">${r.snr_db == null ? "—" : esc(r.snr_db)}</td>
      <td class="num">${esc(r.n_windows)}</td>
      <td>${esc(r.model)}</td>
      <td class="row-actions">
        <button class="mini" data-act="pdf" data-id="${esc(r.id)}">PDF</button>
        <button class="mini" data-act="delete" data-id="${esc(r.id)}">Delete</button>
      </td>
    </tr>`).join("");
  return `<table class="history">
    <thead><tr><th>When</th><th>Capture</th><th>Verdict</th><th>Classes</th>
      <th class="num">SNR dB</th><th class="num">Windows</th><th>Model</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}
