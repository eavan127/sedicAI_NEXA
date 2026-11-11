// node web/test/seen_check.mjs -- the Performance page's "What NEXA has seen" charts.
import { tierTotals, donutArcs, classMixHtml, ladderPos, snrLadderHtml, calendarCells, calendarHtml } from "../seen.js";

let failed = 0;
const check = (name, ok) => { console.log(`${ok ? "ok  " : "FAIL"}  ${name}`); if (!ok) failed++; };
const tierOf = { QPSK: "Civilian", BPSK: "Civilian", LFM_RADAR: "Military", JAM_SPOT: "Hostile", NOISE_FLOOR: "Empty" };

const t = tierTotals({ QPSK: 4, BPSK: 2, LFM_RADAR: 3, JAM_SPOT: 1, NOISE_FLOOR: 9, UNKNOWN: 5 }, tierOf);
check("tier totals ignore noise floor and unknown classes", t.Civilian === 6 && t.Military === 3 && t.Hostile === 1);

const arcs = donutArcs(t, 100);
check("donut arcs add up to the whole circle", Math.abs(arcs.reduce((a, x) => a + x[1], 0) - 100) < 1e-9);
check("each arc starts where the last ended", arcs[1][2] === -arcs[0][1] && arcs[2][2] === -(arcs[0][1] + arcs[1][1]));
check("a tier with nothing gets no arc", donutArcs({ Hostile: 0, Military: 2, Civilian: 2 }, 10).length === 2);

const mix = classMixHtml({ total: 7, byClass: { QPSK: 4, LFM_RADAR: 3, NOISE_FLOOR: 2 } }, tierOf);
check("most common class listed first", mix.indexOf("QPSK") < mix.indexOf("LFM_RADAR"));
check("noise floor not listed as a class", !mix.includes(">NOISE_FLOOR<"));
check("empty class mix says so", classMixHtml({ total: 0, byClass: {} }, tierOf).includes("No class"));

check("ladder ends clamp", ladderPos(-40) === 0 && ladderPos(40) === 100 && ladderPos(0) === 50);
const lad = snrLadderHtml({ meanSnrDb: 2, snr: [{ label: "0…4", count: 3 }, { label: "4…8", count: 1 }] });
check("ladder marks the mean", lad.includes("mean 2 dB"));
check("ladder with no SNR says so", snrLadderHtml({ snr: [{ label: "x", count: 0 }] }).includes("No capture"));

const cal = calendarCells([["2026-09-30", 7], ["2026-10-02", 1]], "2026-11-11");
check("calendar weeks are 7 days, Monday first", cal.weeks.every(w => w.length === 7) && new Date(`${cal.weeks[0][0].day}T00:00:00Z`).getUTCDay() === 1);
const cells = cal.weeks.flat();
check("busiest day is the darkest level", cells.find(c => c.day === "2026-09-30").level === 4);
check("a quiet day is level 0", cells.find(c => c.day === "2026-10-01").level === 0);
check("days after today are marked future", cells.filter(c => c.day > "2026-11-11").every(c => c.future));
check("today is inside the calendar", cells.some(c => c.day === "2026-11-11" && !c.future));
check("at least 12 weeks shown", cal.weeks.length >= 12);
check("months labelled", cal.months.map(m => m.label).join(",").includes("Oct"));
const long = calendarCells([["2025-01-06", 1]], "2026-11-11");
check("long history clipped to 26 weeks", long.weeks.length === 26 && long.clipped);
check("calendar names the busiest day", /30 Sept? 2026/.test(calendarHtml({ total: 8, byDay: [["2026-09-30", 7], ["2026-10-02", 1]] }, "2026-11-11")));

check("month labels never crowd", cal.months.every((m, i) => !cal.months[i + 1] || cal.months[i + 1].col - m.col >= 3));

console.log(failed ? `${failed} FAILED` : "all checks passed");
process.exit(failed ? 1 : 0);
