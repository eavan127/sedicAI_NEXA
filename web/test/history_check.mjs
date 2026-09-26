// The History page's arithmetic: the dashboard is only worth reading if its
// counts are right, and they are easy to get subtly wrong (per-capture vs
// per-window, SNR bin edges, filters that silently AND into nothing).
//
// Pure functions only -- no DOM, no IndexedDB -- so this runs in node:
//     node web/test/history_check.mjs
import { applyFilters, barsHtml, summarise, tableHtml, SNR_BINS } from "../history.js";
import { buildRecord, topTier } from "../storage.js";

let failures = 0;
function check(name, ok, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}

function rec(over = {}) {
  return {
    id: over.id || Math.random().toString(36).slice(2),
    created_at: "2026-09-25T10:00:00.000Z",
    source: "scenario", case_note: "", file_name: null,
    model: "ensemble", duration_s: 0.05, n_windows: 100, hop: 256,
    snr_db: 0, verdict: "Military", classes_detected: ["LFM_RADAR"],
    peak_probability: {}, n_events: 1, tier_counts: {},
    ...over,
  };
}

// --- summarise --------------------------------------------------------------

{
  const s = summarise([
    rec({ verdict: "Hostile", classes_detected: ["JAMMING", "FHSS"], snr_db: -10, n_windows: 50 }),
    rec({ verdict: "Military", classes_detected: ["FHSS"], snr_db: 6, n_windows: 150, source: "upload" }),
    rec({ verdict: "Military", classes_detected: ["FHSS"], snr_db: null, n_windows: 200 }),
  ]);
  check("counts captures, not windows", s.total === 3 && s.byClass.FHSS === 3,
    `total=${s.total} FHSS=${s.byClass.FHSS}`);
  check("sums windows separately", s.windows === 400, `windows=${s.windows}`);
  check("tiers counted per capture", s.byTier.Military === 2 && s.byTier.Hostile === 1);
  check("source split", s.bySource.scenario === 2 && s.bySource.upload === 1);
  check("mean SNR skips unknown SNR", s.meanSnrDb === -2,
    `mean=${s.meanSnrDb} (mean of -10 and 6, not of -10, 6 and 0)`);
  const band = Object.fromEntries(s.snr.map(b => [b.label, b.count]));
  check("SNR lands in one bin each", band["< −8"] === 1 && band["4…8"] === 1,
    JSON.stringify(band));
  check("unknown SNR is in no bin", s.snr.reduce((a, b) => a + b.count, 0) === 2);
}

{
  // Bin edges: lo inclusive, hi exclusive, so a value on an edge is counted once.
  for (const v of [-8, -4, 0, 4, 8]) {
    const s = summarise([rec({ snr_db: v })]);
    const hits = s.snr.filter(b => b.count === 1);
    check(`SNR ${v} dB falls in exactly one bin`, hits.length === 1,
      hits.map(h => h.label).join(", ") || "none");
  }
  check("bins are contiguous",
    SNR_BINS.every((b, i) => i === 0 || b.lo === SNR_BINS[i - 1].hi));
}

{
  const s = summarise([
    rec({ created_at: "2026-09-24T23:00:00Z" }),
    rec({ created_at: "2026-09-25T01:00:00Z" }),
    rec({ created_at: "2026-09-25T09:00:00Z" }),
  ]);
  check("per-day series is ascending and grouped",
    JSON.stringify(s.byDay) === JSON.stringify([["2026-09-24", 1], ["2026-09-25", 2]]),
    JSON.stringify(s.byDay));
}

check("empty input does not divide by zero", (() => {
  const s = summarise([]);
  return s.total === 0 && s.meanSnrDb === null && s.windows === 0;
})());

// --- filters ----------------------------------------------------------------

{
  const rows = [
    rec({ id: "a", verdict: "Hostile", classes_detected: ["JAMMING"], file_name: "alpha.f32" }),
    rec({ id: "b", verdict: "Military", classes_detected: ["FHSS", "LFM_RADAR"], source: "upload" }),
    rec({ id: "c", verdict: "Military", classes_detected: ["LFM_RADAR"], case_note: "case `radar_sweep`" }),
  ];
  const ids = f => applyFilters(rows, f).map(r => r.id).join("");
  check("no filter returns everything", ids({}) === "abc");
  check("tier filter", ids({ tier: "Military" }) === "bc");
  check("class filter matches any listed class", ids({ cls: "LFM_RADAR" }) === "bc");
  check("source filter", ids({ source: "upload" }) === "b");
  check("filters combine with AND", ids({ tier: "Military", source: "upload" }) === "b");
  check("query searches file name", ids({ query: "alpha" }) === "a");
  check("query searches the case note", ids({ query: "radar_sweep" }) === "c");
  check("query is case-insensitive", ids({ query: "ALPHA" }) === "a");
  check("a filter matching nothing returns nothing", ids({ tier: "Civilian" }) === "");
}

// --- the record itself ------------------------------------------------------

{
  const CLASSES = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING", "NOISE_FLOOR"];
  const nClasses = CLASSES.length, nWindows = 3;
  const probs = new Float32Array(nWindows * nClasses);
  probs[0 * nClasses + 4] = 0.2;       // LFM_RADAR, window 0
  probs[1 * nClasses + 4] = 0.9;       // LFM_RADAR, window 1 -- the peak
  probs[2 * nClasses + 6] = 0.4;       // JAMMING
  const session = {
    result: { probs, nWindows, nClasses, hop: 256, fs: 3200000 },
    source: "upload", caseNote: "", which: "ensemble", snrDb: 3.14159,
    snrCapped: false, requestedSnrDb: null,
  };
  const resolved = {
    emitterEvents: [{ classes: ["LFM_RADAR"] }, { classes: ["LFM_RADAR", "JAMMING"] }],
    tiers: ["Empty", "Military", "Hostile"],
  };
  const r = buildRecord(session, resolved, { classes: CLASSES, fileName: "x.f32", fileBytes: 40 });

  check("peak is the max over windows, not the last", r.peak_probability.LFM_RADAR === 0.9,
    String(r.peak_probability.LFM_RADAR));
  check("detected classes come from the events, de-duplicated and sorted",
    r.classes_detected.join(",") === "JAMMING,LFM_RADAR", r.classes_detected.join(","));
  check("tier counts", r.tier_counts.Hostile === 1 && r.tier_counts.Empty === 1);
  check("verdict is the most serious tier present", r.verdict === "Hostile", r.verdict);
  check("numbers are rounded for storage", r.snr_db === 3.142, String(r.snr_db));
  // 3 windows x hop 256 / 3.2 MHz = 0.00024 s. Three-decimal rounding stored
  // this as 0, which is what put r6() in storage.js.
  check("short captures keep their duration", r.duration_s === 0.00024, String(r.duration_s));
  check("file is recorded", r.file_name === "x.f32" && r.file_bytes === 40);
  check("every id is distinct", buildRecord(session, resolved, { classes: CLASSES }).id !== r.id);
}

// --- bar ordering -----------------------------------------------------------

{
  // Class bars rank by count; SNR bands and days must keep their own order,
  // or the chart silently becomes a ranking of bins (seen on screen: the SNR
  // panel listed "< -8, -8..-4, 0..4, 4..8, >= 8, -4..0").
  const labels = html => [...html.matchAll(/class="barlabel">([^<]*)</g)].map(m => m[1]);
  const pairs = [["a", 1], ["b", 3], ["c", 2]];
  check("sorted by default", labels(barsHtml(pairs)).join("") === "bca",
    labels(barsHtml(pairs)).join(""));
  check("sort:false keeps the given order",
    labels(barsHtml(pairs, { sort: false })).join("") === "abc",
    labels(barsHtml(pairs, { sort: false })).join(""));
  check("empty message when there is nothing", barsHtml([]).includes("Nothing recorded yet."));
}

// --- the Delete button follows what the backend allows ----------------------

{
  // The shared database gives the anon key insert and select but not delete
  // (web/supabase/schema.sql), so a Delete button on those rows could only
  // ever fail. Local rows keep theirs.
  const rows = [rec({ id: "a" })];
  check("delete offered by default", tableHtml(rows).includes('data-act="delete"'));
  check("delete hidden when the backend forbids it",
    !tableHtml(rows, { canDelete: false }).includes('data-act="delete"'));
  check("PDF stays either way", tableHtml(rows, { canDelete: false }).includes('data-act="pdf"'));
}

// --- date filter (local days, inclusive) ----------------------------------------
{
  const day = d => rec({ created_at: new Date(`${d}T12:00:00`).toISOString() });
  const rows = [day("2026-09-24"), day("2026-09-25"), day("2026-09-26")];
  check("from/to keep inclusive local days",
    applyFilters(rows, { from: "2026-09-25", to: "2026-09-26" }).length === 2);
  check("from alone", applyFilters(rows, { from: "2026-09-26" }).length === 1);
  check("no dates keeps everything", applyFilters(rows, {}).length === 3);
}

check("topTier prefers Hostile over Military", topTier({ Military: 9, Hostile: 1 }) === "Hostile");
check("topTier of nothing is Empty", topTier({}) === "Empty");

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
