// Checks web/ollama.js's OWN logic (the whitelist, the timeout, the failure
// handling) with a mocked global.fetch -- so this runs with no Ollama
// installed, in CI or on any machine. It does NOT check whether the model
// answers well; that is a judgement call, tried by hand against a real
// Ollama in the session that built this (see the chat log for examples),
// not something a fixed assertion can pin for a non-deterministic model.
//
// The one property this file DOES pin, and the one that actually matters for
// safety: rewrite() never returns anything outside its fixed pattern list.
// Run: node web/test/ollama_check.mjs
import assert from "node:assert/strict";
import { isAvailable, rewrite, warmUp } from "../ollama.js";

let n = 0;
const ok = (name, fn) => { fn(); n++; console.log("ok:", name); };
const okA = async (name, fn) => { await fn(); n++; console.log("ok:", name); };

function mockFetch(handler) {
  const real = globalThis.fetch;
  globalThis.fetch = handler;
  return () => { globalThis.fetch = real; };
}

const respond = (obj, ok2 = true) => Promise.resolve({ ok: ok2, json: async () => obj });

await okA("isAvailable true when the server answers ok", async () => {
  const restore = mockFetch(async () => respond({}));
  try { assert.equal(await isAvailable(), true); } finally { restore(); }
});

await okA("isAvailable false when the server errors", async () => {
  const restore = mockFetch(async () => respond({}, false));
  try { assert.equal(await isAvailable(), false); } finally { restore(); }
});

await okA("isAvailable false when fetch throws (server not running)", async () => {
  const restore = mockFetch(async () => { throw new Error("ECONNREFUSED"); });
  try { assert.equal(await isAvailable(), false); } finally { restore(); }
});

await okA("rewrite accepts a whitelisted response", async () => {
  const restore = mockFetch(async () => respond({ response: "list uploads" }));
  try { assert.equal(await rewrite("show me everything"), "list uploads"); } finally { restore(); }
});

await okA("rewrite strips quotes/punctuation the model tends to add", async () => {
  const restore = mockFetch(async () => respond({ response: '"how many uploads".' }));
  try { assert.equal(await rewrite("count them"), "how many uploads"); } finally { restore(); }
});

await okA("rewrite accepts numbers in summary/compare patterns", async () => {
  const restore = mockFetch(async () => respond({ response: "compare upload 1 and 2" }));
  try { assert.equal(await rewrite("how do 1 and 2 differ"), "compare upload 1 and 2"); } finally { restore(); }
});

await okA("rewrite REJECTS anything off the fixed list -- this is the safety property", async () => {
  for (const bad of ["the weather is nice today", "SELECT * FROM uploads", "FHSS: 99 windows detected",
                      "which uploads had spaceships", "compare upload one and two"]) {
    const restore = mockFetch(async () => respond({ response: bad }));
    try { assert.equal(await rewrite("anything"), null, `should have rejected: ${bad}`); } finally { restore(); }
  }
});

await okA("rewrite returns null for the model's own 'unknown'", async () => {
  const restore = mockFetch(async () => respond({ response: "unknown" }));
  try { assert.equal(await rewrite("asdkjaskdj"), null); } finally { restore(); }
});

await okA("rewrite returns null, not a throw, when the server errors", async () => {
  const restore = mockFetch(async () => respond({}, false));
  try { assert.equal(await rewrite("anything"), null); } finally { restore(); }
});

await okA("rewrite returns null, not a throw, when fetch rejects", async () => {
  const restore = mockFetch(async () => { throw new Error("ECONNREFUSED"); });
  try { assert.equal(await rewrite("anything"), null); } finally { restore(); }
});

await okA("rewrite returns null on a timeout without hanging the caller", async () => {
  const restore = mockFetch((url, opts) => new Promise((resolve, reject) => {
    opts.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
  }));
  try { assert.equal(await rewrite("anything", { timeoutMs: 50 }), null); } finally { restore(); }
});

await okA("warmUp fires a request and does not throw even if the server is down", async () => {
  const restore = mockFetch(async () => { throw new Error("ECONNREFUSED"); });
  try { warmUp(); await wait(10); } finally { restore(); }   // fire-and-forget: must not reject into the caller
});

await okA("warmUp does not make the caller wait for the response", async () => {
  let resolved = false;
  const restore = mockFetch(() => new Promise(r => setTimeout(() => { resolved = true; r({ ok: true, json: async () => ({}) }); }, 200)));
  try {
    const t0 = Date.now();
    warmUp();
    assert.ok(Date.now() - t0 < 20, "warmUp() itself must return immediately");
    assert.equal(resolved, false, "the fake slow response should not have completed yet");
  } finally { restore(); }
});

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

ok("summary", () => {});
console.log(`\n${n} checks passed`);
