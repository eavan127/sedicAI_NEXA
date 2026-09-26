// Rule-based assistant for the OMNI/NEXA demo. No model, no network: it reads
// a question, spots keywords, looks the facts up and fills in a sentence.
// Mirrors src/chatbot/bot.py; web/test/chatbot_check.mjs pins the two
// together (that Python mirror still uses its own simplified upload shape --
// only this file talks to storage.js).
//
// Uploads/analyses are NOT stored by this module. They come from
// storage.js's listAnalyses(), the same records the History page reads --
// one source of truth, so the assistant answers from whatever backend the
// team has configured (this browser, or the shared Supabase project) with no
// extra wiring here. Only the CHAT LOG (the questions asked and answers
// given) is this module's own: storage.js has no concept of a conversation.
// That log lives in the browser's IndexedDB, so it survives closing the page
// and works with the Wi-Fi off. If IndexedDB is unavailable (private window,
// blocked storage) it falls back to memory: the assistant still works, but
// the log is lost on reload.

const ALIASES = [
  ["fhss", "FHSS"], ["hopping", "FHSS"], ["hop", "FHSS"],
  ["jamming", "JAMMING"], ["jammer", "JAMMING"], ["jam", "JAMMING"],
  ["radar", "LFM_RADAR"], ["lfm", "LFM_RADAR"], ["chirp", "LFM_RADAR"],
  ["noise", "NOISE_FLOOR"],
  ["bpsk", "BPSK"], ["qpsk", "QPSK"], ["16qam", "16QAM"], ["64qam", "64QAM"],
];
const EXTRA_TERMS = [["snr", "SNR"], ["window", "WINDOW"], ["threshold", "THRESHOLD"]];

const CIVILIAN = "BPSK, QPSK, 16QAM and 64QAM are ordinary civilian digital modulations.";
export const FAQ = {
  FHSS: "FHSS (frequency-hopping spread spectrum) is a signal that jumps between channels many times a second. It is one of the three judged classes.",
  LFM_RADAR: "LFM_RADAR is a radar pulse whose frequency slides up during the pulse (a chirp). It is one of the three judged classes.",
  JAMMING: "JAMMING is deliberate interference (barrage noise, a tone or a sweep) meant to drown out other signals. It is one of the three judged classes.",
  NOISE_FLOOR: "NOISE_FLOOR means no signal was found, only background noise. At -10 dB a weak signal is indistinguishable from noise, so this class is limited there.",
  BPSK: CIVILIAN, QPSK: CIVILIAN, "16QAM": CIVILIAN, "64QAM": CIVILIAN,
  SNR: "SNR (signal-to-noise ratio) is how far a signal sits above the background noise, in dB. Higher is easier to detect; at -10 dB the signal is below the noise.",
  WINDOW: "The model looks at one window at a time: 512 samples at 3.2 MHz, which is 160 microseconds.",
  THRESHOLD: "A class is reported when its score is above that class's threshold. Each class has its own calibrated threshold.",
};

export const HELP = "You can ask: 'show my last upload', 'list uploads', 'how many uploads', 'which uploads had jamming', "
  + "'summary of upload 2', 'compare upload 1 and 3', 'what did I ask before', or 'what is FHSS'.";

export const SUGGESTIONS = [
  "Show my last upload", "List uploads", "Which uploads had jamming?",
  "Compare upload 1 and 2", "What did I ask before?", "What is FHSS?",
];

function findClass(text) {
  for (const [word, cls] of ALIASES) {
    if (new RegExp(`\\b${word}`).test(text)) return cls;
  }
  return null;
}

/** storage.js's record shape has no per-class window COUNT (it stores peak
 * probability per class instead, to keep rows small -- see buildRecord's own
 * comment). "Detected" here means "the peak probability reached this class's
 * threshold at some point", which is exactly classes_detected. */
function fmtDetected(rec) {
  const classes = rec.classes_detected || [];
  if (!classes.length) return "nothing detected";
  return classes.map(c => {
    const p = rec.peak_probability?.[c];
    return p == null ? c : `${c} (peak ${p.toFixed(2)})`;
  }).join(", ");
}

function fmtName(rec) {
  return rec.file_name || rec.case_note || (rec.source === "upload" ? "uploaded capture" : "synthetic capture");
}

function fmtTime(rec) {
  return (rec.created_at || "").slice(0, 16).replace("T", " ") || "unknown time";
}

function summaryLine(n, rec) {
  return `#${n} ${fmtName(rec)} (${fmtTime(rec)}): ${rec.n_windows} windows, verdict ${rec.verdict}. `
    + `Detected ${fmtDetected(rec)}.`;
}

/** Chronological, oldest first, each tagged with the position number a user
 * would say ("upload 1", "my last upload"). storage.js returns newest first
 * (it's built for a dashboard reading top-down), so this is the one place
 * that reverses it for the assistant's own, conversational numbering. */
function numbered(records) {
  return [...records].sort((a, b) => (a.created_at < b.created_at ? -1 : 1))
    .map((rec, i) => ({ n: i + 1, rec }));
}

/** Pure function: no DOM, no storage. `records` are storage.js rows, any
 * order; `chat` is the list of earlier {q, a} pairs, not including this
 * question. */
export function answer(question, records, chat = []) {
  const q = question.trim().toLowerCase();
  if (!q || q === "help" || q === "?") return HELP;

  const rows = numbered(records);
  const nums = (q.match(/\b\d+\b/g) || []).map(Number);
  const byNumber = n => rows.find(r => r.n === n);

  if (/\b(what|which)\b.*\b(ask|asked|said)\b|\bchat history\b|\bearlier questions\b/.test(q)) {
    const prev = chat.slice(-5);
    return prev.length ? "Your recent questions: " + prev.map(c => `"${c.q}"`).join("; ")
                       : "You have not asked anything before.";
  }

  if (q.includes("compare")) {
    if (nums.length < 2) return "Tell me two upload numbers, for example 'compare upload 1 and 3'.";
    const a = byNumber(nums[0]), b = byNumber(nums[1]);
    if (!a || !b) return "I could not find one of those uploads. Try 'list uploads'.";
    return `A) ${summaryLine(a.n, a.rec)}\nB) ${summaryLine(b.n, b.rec)}`;
  }

  if (/\bhow many\b/.test(q) && q.includes("upload")) {
    return `There ${rows.length === 1 ? "is 1 upload" : `are ${rows.length} uploads`} saved.`;
  }

  if (/\b(last|latest|recent)\b/.test(q)) {
    if (!rows.length) return "There are no uploads yet.";
    const last = rows[rows.length - 1];
    return summaryLine(last.n, last.rec);
  }

  if (/\b(list|all|history)\b/.test(q) && q.includes("upload")) {
    return rows.length ? rows.map(r => summaryLine(r.n, r.rec)).join("\n") : "There are no uploads yet.";
  }

  if (q.includes("summary") || (nums.length && q.includes("upload"))) {
    if (!nums.length) return "Which upload number? Try 'summary of upload 2'.";
    const r = byNumber(nums[0]);
    return r ? summaryLine(r.n, r.rec) : `I could not find upload ${nums[0]}.`;
  }

  const cls = findClass(q);
  const asksHistory = /\b(which|had|has|with|contain|contained|found|show)\b/.test(q) && q.includes("upload");
  if (cls && asksHistory) {
    const hits = rows.filter(r => (r.rec.classes_detected || []).includes(cls));
    if (!hits.length) return `No upload has ${cls} detected.`;
    return `Uploads with ${cls}: ` + hits.map(r => `#${r.n} ${fmtName(r.rec)}`).join("; ");
  }

  if (cls && FAQ[cls]) return FAQ[cls];
  for (const [word, key] of EXTRA_TERMS) if (q.includes(word)) return FAQ[key];

  return "Sorry, I did not understand that. " + HELP;
}

// ---------------------------------------------------------------------------
// Chat log storage: IndexedDB with a memory fallback. The uploads themselves
// are NOT stored here -- see the file header. Same async interface either
// way, so callers never need to know which one is live.
// ---------------------------------------------------------------------------

function memoryChatLog() {
  const chat = [];
  let cid = 0;
  return {
    persistent: false,
    async addChat(c) { chat.push({ ...c, id: ++cid }); },
    async listChat() { return chat.slice(); },
    async clearChat() { chat.length = 0; },
  };
}

function idbChatLog(db) {
  const tx = (mode, fn) => new Promise((resolve, reject) => {
    const t = db.transaction("chat", mode);
    const req = fn(t.objectStore("chat"));
    t.oncomplete = () => resolve(req && req.result);
    t.onerror = t.onabort = () => reject(t.error);
  });
  return {
    persistent: true,
    addChat: c => tx("readwrite", s => s.add(c)),
    listChat: () => tx("readonly", s => s.getAll()),
    clearChat: () => tx("readwrite", s => s.clear()),
  };
}

/** Bumped to v2 and renamed from the assistant's first version, which also
 * kept its own "uploads" object store -- dropped now that storage.js is the
 * one source of truth for analyses. Anyone with the old omni-assistant
 * database keeps it (harmless, just unused); this is a fresh database. */
export async function openChatLog() {
  try {
    if (typeof indexedDB === "undefined") return memoryChatLog();
    const db = await new Promise((resolve, reject) => {
      const req = indexedDB.open("omni-assistant-chat", 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore("chat", { keyPath: "id", autoIncrement: true });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      req.onblocked = () => reject(new Error("blocked"));
    });
    return idbChatLog(db);
  } catch (e) {
    console.warn("Assistant chat log unavailable, using memory:", e);
    return memoryChatLog();
  }
}

/** One question through the whole loop: read the current uploads (fresh,
 * every time -- an analysis saved a moment ago must be answerable straight
 * away) and this session's chat log, answer, then save the pair. `listRecords`
 * is storage.js's listAnalyses, injected rather than imported directly so
 * this module (and its tests) do not need a DOM or a Supabase project to run.
 *
 * `rewriteFn`, when given, is ollama.js's rewrite(): an OPTIONAL local model
 * that only ever picks one of the fixed phrasings answer() already handles
 * (see ollama.js's own header for why). It runs first; if it returns a
 * canonical line, that replaces `question` for matching purposes only -- the
 * ORIGINAL text is still what gets shown in the chat log and in "what did I
 * ask before", so a garbled rewrite never surfaces to the person reading the
 * transcript. A null return (not running, timed out, or no confident match)
 * falls straight through to today's plain keyword matching on the original
 * text, unchanged. */
export async function ask(chatLog, listRecords, question, rewriteFn = null) {
  const [{ records }, chat] = await Promise.all([listRecords(), chatLog.listChat()]);
  const matchOn = (rewriteFn && await rewriteFn(question).catch(() => null)) || question;
  const a = answer(matchOn, records, chat);
  await chatLog.addChat({ q: question, a, t: Date.now() });
  return a;
}

// ---------------------------------------------------------------------------
// Page wiring
// ---------------------------------------------------------------------------

export async function initAssistant({ chatLog, listRecords, logEl, formEl, inputEl, suggestEl, clearBtn, noteEl, rewriteFn = null }) {
  const add = (text, who) => {
    const div = document.createElement("div");
    div.className = `chat-msg ${who}`;
    div.textContent = text;          // textContent, never innerHTML: file names and case notes are user/data-derived
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
  };

  async function reload() {
    logEl.textContent = "";
    const chat = await chatLog.listChat();
    if (!chat.length) add("Hello. I can look up your saved uploads and explain the terms. " + HELP, "bot");
    for (const c of chat) { add(c.q, "user"); add(c.a, "bot"); }
  }

  async function send(text) {
    if (!text.trim()) return;
    add(text, "user");
    add(await ask(chatLog, listRecords, text, rewriteFn), "bot");
  }

  formEl.addEventListener("submit", async e => {
    e.preventDefault();
    const text = inputEl.value;
    inputEl.value = "";
    await send(text);
  });
  for (const s of SUGGESTIONS) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s;
    b.addEventListener("click", () => send(s));
    suggestEl.appendChild(b);
  }
  clearBtn.addEventListener("click", async () => {
    if (!confirm("Delete this assistant's saved chat log on this computer? "
      + "Stored analyses are not affected -- delete those from the History page.")) return;
    await chatLog.clearChat();
    await reload();
  });
  if (noteEl) {
    const storageNote = chatLog.persistent
      ? "Your questions are saved in this browser and work offline."
      : "Chat storage is unavailable here, so questions are kept only until the page is closed.";
    const ollamaNote = rewriteFn
      ? " A local Ollama model is helping understand differently-worded questions; it never invents facts, only picks a known question type."
      : " (Ollama not detected -- exact-phrase matching only. See the Assistant page's suggested questions.)";
    noteEl.textContent = storageNote + " Uploads are read from wherever the History page stores them." + ollamaNote;
  }
  await reload();
}
