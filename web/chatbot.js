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
const EXTRA_TERMS = [["snr", "SNR"], ["window", "WINDOW"], ["threshold", "THRESHOLD"],
  ["civilian", "TIER_CIVILIAN"], ["military", "TIER_MILITARY"], ["hostile", "TIER_HOSTILE"], ["tier", "TIER_INFO"]];

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
  // Tiers group the 8 classes into what a window's verdict is displayed as
  // (analysis.js:TIER_OF/TIER_PRIORITY) -- these are the words used on
  // screen, not classes the model itself outputs.
  TIER_CIVILIAN: `Civilian is the tier for ordinary traffic: ${CIVILIAN.replace(" are ordinary civilian digital modulations.", "")}.`,
  TIER_MILITARY: "Military is the tier for LFM_RADAR and FHSS: emitters that are not hostile by themselves, but are not ordinary civilian traffic either.",
  TIER_HOSTILE: "Hostile is the tier for JAMMING, and it always wins: a window with jamming plus anything else is still reported as Hostile.",
  TIER_INFO: "A window's tier is the most serious thing found in it: Hostile beats Military, which beats Civilian, which beats Empty (no signal, NOISE_FLOOR).",
};

export const HELP = "You can ask: 'show my last upload', 'list uploads', 'how many uploads', 'which uploads had jamming', "
  + "'summary of upload 2', 'compare upload 1 and 3', 'what did I ask before', or 'what is FHSS'.";

export const SUGGESTIONS = [
  "Show my last upload", "List uploads", "Which uploads had jamming?",
  "Compare upload 1 and 2", "What did I ask before?",
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
  return explain(question, records, chat).text;
}

/** answer(), plus what the page needs to make the reply interactive:
 *   text      the plain answer (what answer() returns and the log stores)
 *   kind      which question type matched
 *   lead      a short line shown ABOVE the upload cards, in place of the long text
 *   rows      the uploads the answer is about, as {n, rec}
 *   cls       the class the question was about, if any
 *   followUps questions worth offering next (all phrased so answer() knows them)
 * Still pure: no DOM, no storage. */
export function explain(question, records, chat = []) {
  const q = question.trim().toLowerCase();
  const rows = numbered(records);
  const out = (kind, text, extra = {}) => ({
    kind, text, lead: null, rows: [], cls: null, ...extra,
    followUps: followUpsFor(kind, rows, extra),
  });
  if (!q || q === "help" || q === "?") return out("help", HELP);

  const nums = (q.match(/\b\d+\b/g) || []).map(Number);
  const byNumber = n => rows.find(r => r.n === n);

  if (/\b(what|which)\b.*\b(ask|asked|said)\b|\bchat history\b|\bearlier questions\b/.test(q)) {
    const prev = chat.slice(-5);
    return prev.length
      ? out("asked", "Your recent questions: " + prev.map(c => `"${c.q}"`).join("; "),
            { lead: "Your recent questions (click one to ask it again):", reask: prev.map(c => c.q) })
      : out("asked", "You have not asked anything before.");
  }

  if (q.includes("compare")) {
    if (nums.length < 2) return out("compare_help", "Tell me two upload numbers, for example 'compare upload 1 and 3'.");
    const a = byNumber(nums[0]), b = byNumber(nums[1]);
    if (!a || !b) return out("not_found", "I could not find one of those uploads. Try 'list uploads'.");
    return out("compare", `A) ${summaryLine(a.n, a.rec)}\nB) ${summaryLine(b.n, b.rec)}`,
               { lead: `Upload #${a.n} vs upload #${b.n} (differences highlighted):`, rows: [a, b] });
  }

  if (/\bhow many\b/.test(q) && q.includes("upload")) {
    return out("count", `There ${rows.length === 1 ? "is 1 upload" : `are ${rows.length} uploads`} saved.`);
  }

  if (/\b(last|latest|recent)\b/.test(q)) {
    if (!rows.length) return out("empty", "There are no uploads yet.");
    const last = rows[rows.length - 1];
    return out("last", summaryLine(last.n, last.rec), { lead: "Your most recent upload:", rows: [last] });
  }

  if (/\b(list|all|history)\b/.test(q) && q.includes("upload")) {
    return rows.length
      ? out("list", rows.map(r => summaryLine(r.n, r.rec)).join("\n"),
            { lead: `${rows.length} upload${rows.length === 1 ? "" : "s"} saved, oldest first:`, rows })
      : out("empty", "There are no uploads yet.");
  }

  if (q.includes("summary") || (nums.length && q.includes("upload"))) {
    if (!nums.length) return out("summary_help", "Which upload number? Try 'summary of upload 2'.");
    const r = byNumber(nums[0]);
    return r ? out("summary", summaryLine(r.n, r.rec), { lead: `Upload #${r.n}:`, rows: [r] })
             : out("not_found", `I could not find upload ${nums[0]}.`);
  }

  const cls = findClass(q);
  const asksHistory = /\b(which|had|has|with|contain|contained|found|show)\b/.test(q) && q.includes("upload");
  if (cls && asksHistory) {
    const hits = rows.filter(r => (r.rec.classes_detected || []).includes(cls));
    if (!hits.length) return out("with_class", `No upload has ${cls} detected.`, { cls });
    return out("with_class", `Uploads with ${cls}: ` + hits.map(r => `#${r.n} ${fmtName(r.rec)}`).join("; "),
               { lead: `${hits.length} of ${rows.length} upload${rows.length === 1 ? "" : "s"} had ${cls}:`,
                 rows: hits, cls });
  }

  if (cls && FAQ[cls]) return out("term", FAQ[cls], { cls });
  for (const [word, key] of EXTRA_TERMS) if (q.includes(word)) return out("term", FAQ[key]);

  return out("unknown", "Sorry, I did not understand that. " + HELP);
}

/** What to offer after an answer. Only questions answer() understands. */
function followUpsFor(kind, rows, { cls = null, rows: hit = [] } = {}) {
  const n = rows.length;
  const lastTwo = n >= 2 ? [`Compare upload ${n - 1} and ${n}`] : [];
  switch (kind) {
    case "term":
      return cls ? [`Which uploads had ${cls}?`, "What is a tier?"] : ["What is SNR?", "What is a window?"];
    case "with_class":
      return [...(cls ? [`What is ${cls}?`] : []), "List uploads", ...lastTwo];
    case "summary": case "last": {
      const m = hit[0]?.n;
      return [...(m > 1 ? [`Compare upload ${m - 1} and ${m}`] : []),
              ...(hit[0]?.rec.classes_detected || []).filter(c => FAQ[c]).slice(0, 2).map(c => `What is ${c}?`)];
    }
    case "compare": return ["List uploads", "Which uploads had jamming?"];
    case "list": case "count": return ["Show my last upload", ...lastTwo, "Which uploads had jamming?"];
    case "empty": return ["What is FHSS?", "What is a tier?"];
    case "asked": return [];
    default: return SUGGESTIONS.slice(0, 3);
  }
}

/** Suggestions built from what is actually stored: the judged classes that
 * occur, real upload numbers. Falls back to explaining terms with no data. */
export function suggestionsFor(records) {
  const n = records.length;
  if (!n) return ["What is FHSS?", "What is JAMMING?", "What is a tier?", "Help"];
  const seen = new Set(records.flatMap(r => r.classes_detected || []));
  const judged = ["JAMMING", "LFM_RADAR", "FHSS"].filter(c => seen.has(c)).slice(0, 2);
  return ["Show my last upload", "List uploads", ...judged.map(c => `Which uploads had ${c}?`),
          ...(n >= 2 ? [`Compare upload ${n - 1} and ${n}`] : []), "What did I ask before?"];
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
  return (await askRich(chatLog, listRecords, question, rewriteFn)).text;
}

/** ask(), returning explain()'s whole result (plus the records it read), so
 * the page can draw cards and follow-ups. Logs exactly what ask() logs. */
export async function askRich(chatLog, listRecords, question, rewriteFn = null) {
  const [{ records }, chat] = await Promise.all([listRecords(), chatLog.listChat()]);
  const matchOn = (rewriteFn && await rewriteFn(question).catch(() => null)) || question;
  const res = explain(matchOn, records, chat);
  await chatLog.addChat({ q: question, a: res.text, t: Date.now() });
  return { ...res, records };
}

// ---------------------------------------------------------------------------
// Page wiring
// ---------------------------------------------------------------------------

const CARDS_SHOWN = 6;

/** A tiny DOM builder. Text always goes in through textContent, never
 * innerHTML: file names and case notes are user/data-derived. */
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid);
  return e;
}

export async function initAssistant({ chatLog, listRecords, logEl, formEl, inputEl, suggestEl, clearBtn, noteEl,
                                      rewriteFn = null, tierColor = {}, classColor = {}, onOpenRecord = null }) {
  let comparePick = null;          // upload number picked first for "Compare"
  let busy = false;
  const asked = [];                // this session's questions, for the ↑ / ↓ keys
  let recall = -1;

  const scroll = () => { logEl.scrollTop = logEl.scrollHeight; };
  const add = (text, who) => {
    const div = h("div", { class: `chat-msg ${who}` }, text);
    logEl.appendChild(div);
    scroll();
    return div;
  };

  const chip = (text, onclick, cls = "chat-chip") => h("button", { type: "button", class: cls, onclick }, text);

  const classChip = (c, rec, highlight) => {
    const p = rec.peak_probability?.[c];
    const color = classColor[c] || "#6B7280";
    return h("button", {
      type: "button", class: `chat-class${highlight ? " hl" : ""}`, title: `What is ${c}? (click to ask)`,
      style: { borderColor: color }, onclick: () => send(`What is ${c}?`),
    }, h("span", { class: "dot", style: { background: color } }), c,
    p == null ? null : h("span", { class: "bar" }, h("span", { style: { width: `${Math.round(p * 100)}%`, background: color } })),
    p == null ? null : h("span", { class: "pct" }, `${Math.round(p * 100)}%`));
  };

  const verdictPill = v => h("span", { class: "chat-verdict", style: { background: tierColor[v] || "#6B7280" } }, v || "?");

  function card({ n, rec }, cls) {
    const snr = rec.snr_db == null ? null : `SNR ${Number(rec.snr_db).toFixed(1)} dB`;
    const picked = comparePick === n;
    return h("div", { class: `chat-card${picked ? " picked" : ""}`, "data-n": n },
      h("div", { class: "chat-card-head" },
        h("span", { class: "chat-num" }, `#${n}`),
        h("span", { class: "chat-name", title: fmtName(rec) }, fmtName(rec)),
        verdictPill(rec.verdict)),
      h("div", { class: "chat-meta" },
        [fmtTime(rec), `${rec.n_windows ?? "?"} windows`, snr, rec.model].filter(Boolean).join(" · ")),
      h("div", { class: "chat-classes" },
        (rec.classes_detected || []).length
          ? rec.classes_detected.map(c => classChip(c, rec, c === cls))
          : h("span", { class: "chat-meta" }, "nothing detected")),
      h("div", { class: "chat-actions" },
        chip("Details", () => send(`Summary of upload ${n}`), "mini"),
        chip(picked ? "✓ Picked: now choose another" : comparePick ? `Compare with #${comparePick}` : "Compare",
             () => pickCompare(n), "mini"),
        onOpenRecord ? chip("Open in History ↗", () => onOpenRecord(rec), "mini") : null));
  }

  function pickCompare(n) {
    if (comparePick === n) comparePick = null;
    else if (comparePick) { const a = comparePick; comparePick = null; refreshCards(); send(`Compare upload ${a} and ${n}`); return; }
    else comparePick = n;
    refreshCards();
  }

  /** Redraw every card still on screen, so the compare state shows on all. */
  function refreshCards() {
    for (const grid of logEl.querySelectorAll(".chat-cards")) {
      const items = grid._items;
      if (!items) continue;
      const shown = grid.querySelectorAll(".chat-card").length;
      grid.replaceChildren(...items.slice(0, shown).map(it => card(it, grid._cls)));
    }
  }

  function cards(rows, cls) {
    const grid = h("div", { class: "chat-cards" });
    grid._items = rows;
    grid._cls = cls;
    grid.append(...rows.slice(0, CARDS_SHOWN).map(r => card(r, cls)));
    if (rows.length <= CARDS_SHOWN) return grid;
    const more = chip(`Show all ${rows.length}`, () => {
      grid.replaceChildren(...rows.map(r => card(r, cls)));
      more.remove();
    }, "mini");
    return h("div", {}, grid, more);
  }

  function compareTable([a, b]) {
    const pk = (rec, c) => rec.peak_probability?.[c];
    const has = (rec, c) => (rec.classes_detected || []).includes(c);
    const classes = [...new Set([...(a.rec.classes_detected || []), ...(b.rec.classes_detected || [])])];
    const cell = (rec, c) => has(rec, c)
      ? h("td", {}, "✓", pk(rec, c) == null ? "" : ` ${Math.round(pk(rec, c) * 100)}%`)
      : h("td", { class: "dim" }, "—");
    const row = (label, va, vb) => h("tr", { class: String(va) !== String(vb) ? "diff" : null },
      h("th", {}, label), h("td", {}, va ?? "—"), h("td", {}, vb ?? "—"));
    return h("table", { class: "chat-compare" },
      h("thead", {}, h("tr", {}, h("th", {}, ""),
        h("th", {}, `#${a.n} ${fmtName(a.rec)}`), h("th", {}, `#${b.n} ${fmtName(b.rec)}`))),
      h("tbody", {},
        h("tr", { class: a.rec.verdict !== b.rec.verdict ? "diff" : null }, h("th", {}, "Verdict"),
          h("td", {}, verdictPill(a.rec.verdict)), h("td", {}, verdictPill(b.rec.verdict))),
        row("When", fmtTime(a.rec), fmtTime(b.rec)),
        row("Windows", a.rec.n_windows, b.rec.n_windows),
        row("SNR (dB)", a.rec.snr_db == null ? null : Number(a.rec.snr_db).toFixed(1),
                        b.rec.snr_db == null ? null : Number(b.rec.snr_db).toFixed(1)),
        row("Model", a.rec.model, b.rec.model),
        classes.map(c => h("tr", { class: has(a.rec, c) !== has(b.rec, c) ? "diff" : null },
          h("th", {}, h("button", { type: "button", class: "chat-link", onclick: () => send(`What is ${c}?`) }, c)),
          cell(a.rec, c), cell(b.rec, c)))));
  }

  function addBot(res) {
    const div = h("div", { class: "chat-msg bot rich" });
    if (res.kind === "compare") div.append(h("div", {}, res.lead), compareTable(res.rows));
    else if (res.rows.length) div.append(h("div", {}, res.lead), cards(res.rows, res.cls));
    else if (res.reask?.length) {
      div.append(h("div", {}, res.lead),
                 h("div", { class: "chat-follow" }, res.reask.map(q => chip(q, () => send(q)))));
    } else div.append(res.text);
    logEl.appendChild(div);
    // Follow-ups belong to the newest answer only.
    for (const old of logEl.querySelectorAll(".chat-next")) old.remove();
    if (res.followUps.length) {
      logEl.appendChild(h("div", { class: "chat-next" },
        h("span", { class: "chat-next-label" }, "Ask next:"), res.followUps.map(q => chip(q, () => send(q)))));
    }
    scroll();
  }

  function renderSuggestions(records) {
    suggestEl.replaceChildren(...suggestionsFor(records).map(s => chip(s, () => send(s), "")));
  }

  async function reload() {
    logEl.textContent = "";
    comparePick = null;
    const chat = await chatLog.listChat();
    if (!chat.length) {
      add("Hello. I can look up your saved uploads and explain the terms. Click a suggestion below, "
        + "or type a question. In my answers, click a class to have it explained, or Compare on two uploads.", "bot");
    }
    for (const c of chat) { add(c.q, "user"); add(c.a, "bot"); }
    try { renderSuggestions((await listRecords()).records); } catch { renderSuggestions([]); }
  }

  async function send(text) {
    if (!text.trim() || busy) return;
    busy = true;
    asked.push(text);
    recall = -1;
    add(text, "user");
    const typing = add("", "bot typing");
    typing.append(h("span"), h("span"), h("span"));
    typing.setAttribute("aria-label", "Assistant is thinking");
    try {
      const res = await askRich(chatLog, listRecords, text, rewriteFn);
      typing.remove();
      addBot(res);
      renderSuggestions(res.records);
    } catch (e) {
      typing.remove();
      add(`Something went wrong reading the saved uploads: ${e.message}`, "bot");
    } finally {
      busy = false;
      inputEl.focus();
    }
  }

  formEl.addEventListener("submit", async e => {
    e.preventDefault();
    const text = inputEl.value;
    inputEl.value = "";
    await send(text);
  });
  // ↑ / ↓ walk back through this session's questions, like a terminal.
  inputEl.addEventListener("keydown", e => {
    if ((e.key !== "ArrowUp" && e.key !== "ArrowDown") || !asked.length) return;
    e.preventDefault();
    recall = e.key === "ArrowUp" ? (recall < 0 ? asked.length - 1 : Math.max(recall - 1, 0))
                                 : (recall < 0 ? -1 : recall + 1);
    if (recall >= asked.length) recall = -1;
    inputEl.value = recall < 0 ? "" : asked[recall];
  });
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
    noteEl.textContent = storageNote + " Uploads are read from wherever the History page stores them."
      + ollamaNote + " Tip: ↑ recalls your previous question.";
  }
  await reload();
}
