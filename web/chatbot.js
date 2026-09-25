// Rule-based assistant for the OMNI demo. No model, no network: it reads a
// question, spots keywords, looks the facts up in the saved upload history and
// fills in a sentence. Mirrors src/chatbot/bot.py; web/test/chatbot_check.mjs
// pins the two together.
//
// History persists in the browser's IndexedDB, so it survives closing the page
// and works with the Wi-Fi off. If IndexedDB is unavailable (private window,
// blocked storage) it falls back to memory: the assistant still works, but
// history is lost on reload.

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

function fmtCounts(counts) {
  const items = Object.entries(counts || {}).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  return items.length ? items.map(([c, n]) => `${c}: ${n}`).join(", ") : "nothing detected";
}

function summaryLine(u) {
  return `#${u.id} ${u.name} (${u.time}): ${u.windows} windows. Detected ${fmtCounts(u.counts)}.`;
}

/** Pure function: no DOM, no storage. `uploads` oldest first; `chat` is the
 * list of earlier {q, a} pairs, not including this question. */
export function answer(question, uploads, chat = []) {
  const q = question.trim().toLowerCase();
  if (!q || q === "help" || q === "?") return HELP;

  const nums = (q.match(/\b\d+\b/g) || []).map(Number);
  const byId = id => uploads.find(u => u.id === id);

  if (/\b(what|which)\b.*\b(ask|asked|said)\b|\bchat history\b|\bearlier questions\b/.test(q)) {
    const prev = chat.slice(-5);
    return prev.length ? "Your recent questions: " + prev.map(c => `"${c.q}"`).join("; ")
                       : "You have not asked anything before.";
  }

  if (q.includes("compare")) {
    if (nums.length < 2) return "Tell me two upload numbers, for example 'compare upload 1 and 3'.";
    const a = byId(nums[0]), b = byId(nums[1]);
    if (!a || !b) return "I could not find one of those uploads. Try 'list uploads'.";
    return `A) ${summaryLine(a)}\nB) ${summaryLine(b)}`;
  }

  if (/\bhow many\b/.test(q) && q.includes("upload")) {
    return `There ${uploads.length === 1 ? "is 1 upload" : `are ${uploads.length} uploads`} saved.`;
  }

  if (/\b(last|latest|recent)\b/.test(q)) {
    return uploads.length ? summaryLine(uploads[uploads.length - 1]) : "There are no uploads yet.";
  }

  if (/\b(list|all|history)\b/.test(q) && q.includes("upload")) {
    return uploads.length ? uploads.map(summaryLine).join("\n") : "There are no uploads yet.";
  }

  if (q.includes("summary") || (nums.length && q.includes("upload"))) {
    if (!nums.length) return "Which upload number? Try 'summary of upload 2'.";
    const u = byId(nums[0]);
    return u ? summaryLine(u) : `I could not find upload ${nums[0]}.`;
  }

  const cls = findClass(q);
  const asksHistory = /\b(which|had|has|with|contain|contained|found|show)\b/.test(q) && q.includes("upload");
  if (cls && asksHistory) {
    const hits = uploads.filter(u => (u.counts?.[cls] || 0) > 0);
    if (!hits.length) return `No upload has ${cls} detected.`;
    return `Uploads with ${cls}: ` + hits.map(u => `#${u.id} ${u.name} (${u.counts[cls]} windows)`).join("; ");
  }

  if (cls && FAQ[cls]) return FAQ[cls];
  for (const [word, key] of EXTRA_TERMS) if (q.includes(word)) return FAQ[key];

  return "Sorry, I did not understand that. " + HELP;
}

/** One upload record from a finished analysis. `events` are the resolved
 * detections (analysis.js:resolveSession(...).events); counts are windows per
 * class, the same figures the RF Replay page shows. */
export function summarizeUpload({ name, source, caseNote = "", snrDb = null, nWindows, hop, durationMs, events, model }) {
  const counts = {};
  for (const e of events) {
    const n = e.endWindow - e.startWindow + 1;
    for (const c of e.classes) counts[c] = (counts[c] || 0) + n;
  }
  const d = new Date();
  const pad = n => String(n).padStart(2, "0");
  return {
    name, source, caseNote, snrDb, windows: nWindows, hop, durationMs, model, counts,
    time: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`,
  };
}

// ---------------------------------------------------------------------------
// Storage: IndexedDB with a memory fallback. Same async interface for both.
// ---------------------------------------------------------------------------

function memoryStore() {
  const uploads = [], chat = [];
  let uid = 0, cid = 0;
  return {
    persistent: false,
    async addUpload(u) { u = { ...u, id: ++uid }; uploads.push(u); return u; },
    async listUploads() { return uploads.slice(); },
    async addChat(c) { chat.push({ ...c, id: ++cid }); },
    async listChat() { return chat.slice(); },
    async clearAll() { uploads.length = 0; chat.length = 0; },
  };
}

function idbStore(db) {
  const tx = (store, mode, fn) => new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const req = fn(t.objectStore(store));
    t.oncomplete = () => resolve(req && req.result);
    t.onerror = t.onabort = () => reject(t.error);
  });
  return {
    persistent: true,
    async addUpload(u) { const id = await tx("uploads", "readwrite", s => s.add(u)); return { ...u, id }; },
    listUploads: () => tx("uploads", "readonly", s => s.getAll()),
    addChat: c => tx("chat", "readwrite", s => s.add(c)),
    listChat: () => tx("chat", "readonly", s => s.getAll()),
    async clearAll() {
      await tx("uploads", "readwrite", s => s.clear());
      await tx("chat", "readwrite", s => s.clear());
    },
  };
}

export async function openStore() {
  try {
    if (typeof indexedDB === "undefined") return memoryStore();
    const db = await new Promise((resolve, reject) => {
      const req = indexedDB.open("omni-assistant", 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore("uploads", { keyPath: "id", autoIncrement: true });
        req.result.createObjectStore("chat", { keyPath: "id", autoIncrement: true });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      req.onblocked = () => reject(new Error("blocked"));
    });
    return idbStore(db);
  } catch (e) {
    console.warn("Assistant history storage unavailable, using memory:", e);
    return memoryStore();
  }
}

/** One question through the whole loop: read history, answer, save the pair. */
export async function ask(store, question) {
  const [uploads, chat] = await Promise.all([store.listUploads(), store.listChat()]);
  const a = answer(question, uploads, chat);
  await store.addChat({ q: question, a, t: Date.now() });
  return a;
}

// ---------------------------------------------------------------------------
// Page wiring
// ---------------------------------------------------------------------------

export async function initAssistant({ store, logEl, formEl, inputEl, suggestEl, clearBtn, noteEl }) {
  const add = (text, who) => {
    const div = document.createElement("div");
    div.className = `chat-msg ${who}`;
    div.textContent = text;          // textContent, never innerHTML: uploaded file names are user input
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
  };

  async function reload() {
    logEl.textContent = "";
    const chat = await store.listChat();
    if (!chat.length) add("Hello. I can look up your saved uploads and explain the terms. " + HELP, "bot");
    for (const c of chat) { add(c.q, "user"); add(c.a, "bot"); }
  }

  async function send(text) {
    if (!text.trim()) return;
    add(text, "user");
    add(await ask(store, text), "bot");
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
    if (!confirm("Delete all saved uploads and chat history on this computer?")) return;
    await store.clearAll();
    await reload();
  });
  if (noteEl) {
    noteEl.textContent = store.persistent
      ? "History is saved in this browser on this computer and works offline."
      : "Storage is unavailable here, so history is kept only until the page is closed.";
  }
  await reload();
}
