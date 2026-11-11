// Performance page, "What NEXA has seen": the stored captures as three
// pictures (moved here from History, which keeps the lists and reviews).
//
//   classMixHtml  tier donut + ranked lollipops: which classes keep turning up
//   snrLadderHtml the SNR bands as columns from weak to strong, mean marked
//   calendarHtml  captures per day as a calendar heatmap
//
// Input is history.js:summarise(records). Pure string builders with no DOM,
// so web/test/seen_check.mjs runs them in node.

import { TIER_COLOR } from "./analysis.js";

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const TIERS = ["Hostile", "Military", "Civilian"];
const pct = (n, d) => (d ? Math.round((n / d) * 100) : 0);

/** Class detections summed per tier (NOISE_FLOOR and unknown classes left out). */
export function tierTotals(byClass, tierOf) {
  const t = Object.fromEntries(TIERS.map(k => [k, 0]));
  for (const [c, n] of Object.entries(byClass || {})) if (tierOf[c] in t) t[tierOf[c]] += n;
  return t;
}

/** Donut arcs as [tier, dash, offset] on a circle of circumference `circ`. */
export function donutArcs(totals, circ) {
  const sum = TIERS.reduce((a, k) => a + totals[k], 0);
  let at = 0;
  return TIERS.filter(k => totals[k]).map(k => {
    const len = (totals[k] / sum) * circ;
    const arc = [k, len, -at];
    at += len;
    return arc;
  });
}

export function classMixHtml(summary, tierOf, { top = 10 } = {}) {
  const entries = Object.entries(summary.byClass || {}).filter(([c]) => tierOf[c] !== "Empty")
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  if (!entries.length) return `<div class="note">No class has been detected in a stored capture yet.</div>`;
  const totals = tierTotals(summary.byClass, tierOf);
  const sum = TIERS.reduce((a, k) => a + totals[k], 0);
  const R = 64, C = 2 * Math.PI * R;
  const arcs = donutArcs(totals, C).map(([k, len, off], i) =>
    `<circle class="donut-arc" cx="90" cy="90" r="${R}" stroke="${TIER_COLOR[k]}" stroke-dasharray="${len.toFixed(2)} ${C.toFixed(2)}"
      stroke-dashoffset="${off.toFixed(2)}" style="animation-delay:${i * 0.15}s"><title>${k}: ${totals[k]} detections (${pct(totals[k], sum)}%)</title></circle>`).join("");
  const max = entries[0][1];
  const shown = entries.slice(0, top);
  return `<div class="seen-mix">
    <figure class="seen-donut">
      <svg viewBox="0 0 180 180" role="img" aria-label="Class detections by tier">
        <circle cx="90" cy="90" r="${R}" fill="none" stroke="var(--grid)" stroke-width="24"/>
        <g transform="rotate(-90 90 90)" fill="none" stroke-width="24">${arcs}</g>
        <text x="90" y="86" text-anchor="middle" class="donut-num">${summary.total}</text>
        <text x="90" y="108" text-anchor="middle" class="donut-cap">captures</text>
      </svg>
      <figcaption>${TIERS.map(k => `<span class="tier-key${totals[k] ? "" : " zero"}"><i style="background:${TIER_COLOR[k]}"></i><b>${k}</b> ${pct(totals[k], sum)}%</span>`).join("")}
        <span class="note" style="flex-basis:100%;text-align:center;margin:0;">share of class detections</span></figcaption>
    </figure>
    <div class="lolli">${shown.map(([c, n], i) => `
      <div class="lolli-row" title="${esc(c)}: in ${n} of ${summary.total} captures">
        <span class="lolli-name">${esc(c)}</span>
        <span class="lolli-track"><span class="lolli-stem" style="--w:${((n / max) * 100).toFixed(1)}%;--c:${TIER_COLOR[tierOf[c]] || "var(--brand)"};animation-delay:${(i * 0.04).toFixed(2)}s"></span></span>
        <span class="lolli-n">${n}</span>
      </div>`).join("")}
      ${entries.length > top ? `<div class="note">+ ${entries.length - top} more classes seen less often</div>` : ""}
    </div>
  </div>`;
}

// The open-ended bands are drawn as 4 dB wide, like the others.
const LADDER_LO = -12, LADDER_HI = 12;

/** Where `db` sits along the ladder, 0..100 (%), for the mean marker. */
export function ladderPos(db) {
  return Math.max(0, Math.min(100, ((db - LADDER_LO) / (LADDER_HI - LADDER_LO)) * 100));
}

export function snrLadderHtml(summary) {
  const bins = summary.snr || [];
  const total = bins.reduce((a, b) => a + b.count, 0);
  if (!total) return `<div class="note">No capture carries a known SNR.</div>`;
  const max = Math.max(...bins.map(b => b.count));
  const mean = summary.meanSnrDb;
  return `<div class="ladder">
    <div class="ladder-cols">
      ${bins.map((b, i) => `<div class="ladder-col" title="${esc(b.label)} dB: ${b.count} captures (${pct(b.count, total)}%)">
        <span class="ladder-n">${b.count || ""}</span>
        <span class="ladder-bar" style="--h:${((b.count / max) * 100).toFixed(1)}%;--k:${(0.35 + 0.65 * i / Math.max(1, bins.length - 1)).toFixed(2)};animation-delay:${(i * 0.06).toFixed(2)}s"></span>
      </div>`).join("")}
      ${mean == null ? "" : `<div class="ladder-mean${ladderPos(mean) > 70 ? " flip" : ""}" style="left:${ladderPos(mean).toFixed(1)}%"><span>mean ${mean} dB</span></div>`}
    </div>
    <div class="ladder-axis">${bins.map(b => `<span>${esc(b.label)}</span>`).join("")}</div>
    <div class="ladder-ends"><span>◀ buried in noise</span><span>clear signal ▶</span></div>
  </div>`;
}

const DAY_MS = 86400000;
const iso = t => new Date(t).toISOString().slice(0, 10);

/**
 * Calendar cells, Monday-first weeks, from the week of the first capture (at
 * most `maxWeeks` back, at least `minWeeks` shown) to the week of `today`.
 * -> { weeks: [[{day, n, level, future}] x7], months: [{col, label}], clipped }
 */
export function calendarCells(byDay, today = iso(Date.now()), { maxWeeks = 26, minWeeks = 12 } = {}) {
  const counts = new Map(byDay);
  const end = Date.parse(`${today}T00:00:00Z`);
  const monday = t => t - ((new Date(t).getUTCDay() + 6) % 7) * DAY_MS;
  const first = byDay.length ? Date.parse(`${byDay[0][0]}T00:00:00Z`) : end;
  const lastWeek = monday(end);
  let start = monday(Math.min(first, end));
  const clipped = (lastWeek - start) / (7 * DAY_MS) + 1 > maxWeeks;
  start = Math.max(start, lastWeek - (maxWeeks - 1) * 7 * DAY_MS);
  start = Math.min(start, lastWeek - (minWeeks - 1) * 7 * DAY_MS);
  const max = Math.max(0, ...[...counts.values()]);
  const level = n => (!n ? 0 : Math.min(4, Math.ceil((n / max) * 4)));
  const weeks = [], months = [];
  let lastMonth = -1;
  for (let w = start, col = 0; w <= lastWeek; w += 7 * DAY_MS, col++) {
    const m = new Date(w + 3 * DAY_MS).getUTCMonth();       // the week's Thursday names its month
    if (m !== lastMonth) {
      months.push({ col, label: new Date(w + 3 * DAY_MS).toLocaleString("en", { month: "short", timeZone: "UTC" }) });
      lastMonth = m;
    }
    weeks.push(Array.from({ length: 7 }, (_, d) => {
      const day = iso(w + d * DAY_MS), n = counts.get(day) || 0;
      return { day, n, level: level(n), future: w + d * DAY_MS > end };
    }));
  }
  // A month that only just starts the calendar would print on top of the next one.
  const roomy = months.filter((m, i) => !months[i + 1] || months[i + 1].col - m.col >= 3);
  return { weeks, months: roomy, clipped };
}

export function calendarHtml(summary, today) {
  if (!summary.byDay?.length) return `<div class="note">Nothing stored yet.</div>`;
  const { weeks, months, clipped } = calendarCells(summary.byDay, today);
  const busiest = [...summary.byDay].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? 1 : -1))[0];
  const nice = d => new Date(`${d}T00:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  return `<div class="cal-stats">
      <div><b>${summary.byDay.length}</b><span>days with captures</span></div>
      <div><b>${busiest[1]}</b><span>on the busiest day, ${esc(nice(busiest[0]))}</span></div>
      <div><b>${(summary.total / summary.byDay.length).toFixed(1)}</b><span>captures per active day</span></div>
    </div>
    <div class="cal" style="--weeks:${weeks.length}">
      <div class="cal-months">${months.map(m => `<span style="grid-column:${m.col + 1}">${esc(m.label)}</span>`).join("")}</div>
      <div class="cal-days"><span>Mon</span><span></span><span>Wed</span><span></span><span>Fri</span><span></span><span></span></div>
      <div class="cal-grid">${weeks.map(w => w.map(c => c.future ? `<i class="cal-cell future"></i>`
        : `<i class="cal-cell l${c.level}" title="${esc(nice(c.day))}: ${c.n} capture${c.n === 1 ? "" : "s"}"></i>`).join("")).join("")}</div>
    </div>
    <div class="cal-key"><span>fewer</span>${[0, 1, 2, 3, 4].map(l => `<i class="cal-cell l${l}"></i>`).join("")}<span>more</span>
      ${clipped ? `<span class="note" style="margin-left:12px;">last 26 weeks shown</span>` : ""}</div>`;
}
