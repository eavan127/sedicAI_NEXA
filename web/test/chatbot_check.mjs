// Checks web/chatbot.js: the answers, the saved history, and the record
// builder. Run: node web/test/chatbot_check.mjs
import assert from "node:assert/strict";
import { answer, ask, openStore, summarizeUpload } from "../chatbot.js";

const U = [
  { id: 1, name: "capture1.iq", time: "2026-09-25 10:00", windows: 400, counts: { FHSS: 120, JAMMING: 30 } },
  { id: 2, name: "harbour.iq", time: "2026-09-25 11:30", windows: 300, counts: { QPSK: 200, NOISE_FLOOR: 40 } },
  { id: 3, name: "radar_run.iq", time: "2026-09-25 14:15", windows: 500, counts: { LFM_RADAR: 210, JAMMING: 15 } },
];
const a = q => answer(q, U);
let n = 0;
const ok = (name, fn) => { fn(); n++; console.log("ok:", name); };

ok("last upload", () => assert.match(a("Show me my last upload"), /radar_run\.iq/));
ok("list uploads", () => { const r = a("list uploads"); for (const f of ["capture1.iq", "harbour.iq", "radar_run.iq"]) assert.match(r, new RegExp(f)); });
ok("which had jamming", () => { const r = a("Which uploads had jamming?"); assert.match(r, /capture1/); assert.match(r, /radar_run/); assert.doesNotMatch(r, /harbour/); });
ok("summary of upload 2", () => assert.match(a("summary of upload 2"), /harbour\.iq/));
ok("compare", () => { const r = a("compare upload 1 and 3"); assert.match(r, /capture1/); assert.match(r, /radar_run/); });
ok("missing upload", () => assert.match(a("summary of upload 99"), /could not find/));
ok("how many", () => assert.match(a("how many uploads do I have"), /3 uploads/));
ok("faq", () => assert.match(a("what is FHSS"), /frequency-hopping/));
ok("unknown", () => assert.match(a("what is the weather"), /Sorry/));
ok("empty history", () => assert.match(answer("last upload", []), /no uploads/));
ok("asked before", () => assert.match(answer("what did I ask before", U, [{ q: "list uploads", a: "" }]), /list uploads/));
ok("asked before, none", () => assert.match(answer("what did I ask before", U, []), /not asked/));

// summarizeUpload: counts are windows per class summed over events
const rec = summarizeUpload({
  name: "x.iq", source: "upload", nWindows: 10, hop: 512, durationMs: 1, model: "ensemble",
  events: [
    { classes: ["FHSS"], startWindow: 0, endWindow: 3 },
    { classes: ["FHSS", "JAMMING"], startWindow: 4, endWindow: 5 },
  ],
});
ok("summarizeUpload counts", () => { assert.equal(rec.counts.FHSS, 6); assert.equal(rec.counts.JAMMING, 2); assert.match(rec.time, /^\d{4}-\d\d-\d\d \d\d:\d\d$/); });

// Full loop through the store (memory fallback under node): save, ask, remember
const store = await openStore();
await store.addUpload(rec);
await store.addUpload({ ...rec, name: "y.iq" });
const r1 = await ask(store, "list uploads");
ok("store loop lists saved uploads", () => { assert.match(r1, /x\.iq/); assert.match(r1, /y\.iq/); });
const r2 = await ask(store, "what did I ask before");
ok("store loop remembers questions", () => assert.match(r2, /list uploads/));
await store.clearAll();
ok("clearAll", async () => {});
assert.equal((await store.listUploads()).length, 0);
assert.equal((await store.listChat()).length, 0);
console.log(`\n${n} checks passed`);
