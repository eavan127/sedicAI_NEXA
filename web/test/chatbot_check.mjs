// Checks web/chatbot.js against storage.js's own record shape (buildRecord's
// output), not a private one -- so this catches drift if storage.js's field
// names ever change. Run: node web/test/chatbot_check.mjs
import assert from "node:assert/strict";
import { answer, ask, openChatLog } from "../chatbot.js";

// Same shape buildRecord() in storage.js produces, newest first (as
// listAnalyses returns it) -- created_at strings a day apart so numbering by
// time is unambiguous regardless of which order they're passed in.
const R = [
  { id: "c3", created_at: "2026-09-26T14:15:00.000Z", source: "scenario", case_note: "case `Radar`",
    file_name: null, n_windows: 500, classes_detected: ["LFM_RADAR", "JAMMING"],
    peak_probability: { LFM_RADAR: 0.91, JAMMING: 0.4 }, verdict: "Hostile" },
  { id: "c1", created_at: "2026-09-24T10:00:00.000Z", source: "upload", case_note: "",
    file_name: "capture1.iq", n_windows: 400, classes_detected: ["FHSS", "JAMMING"],
    peak_probability: { FHSS: 0.7, JAMMING: 0.6 }, verdict: "Hostile" },
  { id: "c2", created_at: "2026-09-25T11:30:00.000Z", source: "upload", case_note: "",
    file_name: "harbour.iq", n_windows: 300, classes_detected: ["QPSK"],
    peak_probability: { QPSK: 0.8 }, verdict: "Civilian" },
];
const a = q => answer(q, R);
let n = 0;
const ok = (name, fn) => { fn(); n++; console.log("ok:", name); };

// Numbering is chronological (oldest = #1) regardless of the array's own order.
ok("last upload is the most recent by time, not array position", () => assert.match(a("Show me my last upload"), /case `Radar`/));
ok("list uploads shows all three, numbered oldest first", () => {
  const r = a("list uploads");
  assert.match(r, /#1 capture1\.iq/); assert.match(r, /#2 harbour\.iq/); assert.match(r, /#3.*case `Radar`/);
});
ok("which had jamming", () => { const r = a("Which uploads had jamming?"); assert.match(r, /capture1/); assert.match(r, /case `Radar`/); assert.doesNotMatch(r, /harbour/); });
ok("summary of upload 2 (by chronological number)", () => assert.match(a("summary of upload 2"), /harbour\.iq/));
ok("compare by chronological number", () => { const r = a("compare upload 1 and 3"); assert.match(r, /capture1/); assert.match(r, /case `Radar`/); });
ok("missing upload", () => assert.match(a("summary of upload 99"), /could not find/));
ok("how many", () => assert.match(a("how many uploads do I have"), /3 uploads/));
ok("faq", () => assert.match(a("what is FHSS"), /frequency-hopping/));
ok("unknown", () => assert.match(a("what is the weather"), /Sorry/));
ok("empty history", () => assert.match(answer("last upload", []), /no uploads/));
ok("asked before", () => assert.match(answer("what did I ask before", R, [{ q: "list uploads", a: "" }]), /list uploads/));
ok("asked before, none", () => assert.match(answer("what did I ask before", R, []), /not asked/));
ok("summary reports peak probability, not a window count", () => assert.match(a("summary of upload 1"), /FHSS \(peak 0\.70\)/));
ok("verdict is carried through", () => assert.match(a("summary of upload 2"), /verdict Civilian/));

// The chat log (this module's own store) plumbed against a fake listRecords,
// exactly the shape storage.js's listAnalyses() resolves to: {records, ...}.
const chatLog = await openChatLog();
const listRecords = async () => ({ records: R, backend: "local", warning: null });
const r1 = await ask(chatLog, listRecords, "list uploads");
ok("ask() reads records live via the injected listRecords", () => assert.match(r1, /capture1\.iq/));
const r2 = await ask(chatLog, listRecords, "what did I ask before");
ok("ask() remembers its own chat log", () => assert.match(r2, /list uploads/));
await chatLog.clearChat();
ok("clearChat empties the log", async () => assert.equal((await chatLog.listChat()).length, 0));
console.log(`\n${n} checks passed`);
