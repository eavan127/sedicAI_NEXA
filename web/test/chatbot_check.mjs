// Checks web/chatbot.js against storage.js's own record shape (buildRecord's
// output), not a private one -- so this catches drift if storage.js's field
// names ever change. Run: node web/test/chatbot_check.mjs
import assert from "node:assert/strict";
import { answer, ask, askRich, explain, openChatLog, suggestionsFor } from "../chatbot.js";

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
ok("faq covers the tiers, not just the 8 model classes", () => {
  assert.match(a("what is civilian class"), /Civilian is the tier/);
  assert.match(a("what is military"), /Military is the tier/);
  assert.match(a("what is hostile"), /Hostile is the tier/);
});
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

// ask() with a rewriteFn (the Ollama path), faked here -- ollama_check.mjs
// covers rewrite()'s own network/whitelist behaviour; this checks ask() wires
// a rewrite in correctly and, just as importantly, logs the ORIGINAL question
// the person typed, not the rewritten one, so the chat transcript stays
// honest about what was actually asked.
const r3 = await ask(chatLog, listRecords, "did anything jam that capture", async () => "which uploads had jamming");
ok("ask() uses the rewrite to match, when given one", () => assert.match(r3, /case `Radar`/));
const logged = await chatLog.listChat();
ok("ask() logs the ORIGINAL text, not the rewrite", () => assert.equal(logged.at(-1).q, "did anything jam that capture"));
const r4 = await ask(chatLog, listRecords, "list uploads", async () => null);
ok("ask() falls back to the original text when rewriteFn returns null", () => assert.match(r4, /capture1\.iq/));
const r5 = await ask(chatLog, listRecords, "list uploads", async () => { throw new Error("ollama down"); });
ok("ask() survives a throwing rewriteFn", () => assert.match(r5, /capture1\.iq/));

// explain(): the structure the page draws cards and follow-up chips from.
// Every follow-up and suggestion must be a question answer() understands,
// or clicking it would get "Sorry, I did not understand".
const x = q => explain(q, R, []);
ok("explain() text is exactly answer()", () => {
  for (const q of ["list uploads", "compare upload 1 and 3", "what is FHSS", "nonsense"]) assert.equal(x(q).text, a(q));
});
ok("explain() returns the uploads an answer is about", () => {
  assert.deepEqual(x("list uploads").rows.map(r => r.n), [1, 2, 3]);
  assert.deepEqual(x("Which uploads had jamming?").rows.map(r => r.rec.id), ["c1", "c3"]);
  assert.equal(x("Which uploads had jamming?").cls, "JAMMING");
  assert.deepEqual(x("compare upload 1 and 3").rows.map(r => r.n), [1, 3]);
  assert.equal(x("show my last upload").rows[0].rec.id, "c3");
});
ok("explain() offers re-asking earlier questions", () =>
  assert.deepEqual(explain("what did I ask before", R, [{ q: "list uploads", a: "" }]).reask, ["list uploads"]));
ok("every follow-up and suggestion is understood", () => {
  const offered = new Set([...suggestionsFor(R), ...suggestionsFor([])]);
  for (const q of ["list uploads", "how many uploads", "show my last upload", "summary of upload 2",
                   "compare upload 1 and 3", "Which uploads had jamming?", "what is FHSS", "what is snr", "nonsense"]) {
    for (const f of x(q).followUps) offered.add(f);
  }
  for (const q of offered) assert.doesNotMatch(a(q), /did not understand/, q);
});
ok("suggestions use the stored data", () => {
  const s = suggestionsFor(R);
  assert.ok(s.includes("Which uploads had JAMMING?") && s.includes("Compare upload 2 and 3"), s.join(" | "));
});
const rich = await askRich(chatLog, listRecords, "list uploads");
const richLogged = (await chatLog.listChat()).at(-1);
ok("askRich() returns rows and logs the plain text", () => {
  assert.equal(rich.rows.length, 3);
  assert.equal(richLogged.a, rich.text);
});
await chatLog.clearChat();

console.log(`\n${n} checks passed`);
