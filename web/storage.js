// Persistent record of every capture the app analyses.
//
// Until now a session lived only in memory: reload the page and the analysis
// was gone, and nothing could compare two captures or show what an operator
// looked at last week. This module stores one row per analysed capture and,
// optionally, the uploaded file that produced it.
//
// Two backends behind one interface, chosen by what is configured:
//
//   local     IndexedDB in the browser. Always available, nothing to set up,
//             never leaves the machine. The default.
//   supabase  A Supabase project (Postgres + Storage) over its REST API,
//             used only when a URL and key have been entered on the History
//             page. Records are then shared across machines and operators.
//
// The Supabase path is deliberately plain `fetch` against the REST endpoints
// rather than @supabase/supabase-js: the site is a static folder with no
// build step (web/build.py only inlines the logo and copies the models), so
// an npm dependency would mean adding a bundler for three HTTP calls.
//
// PRIVACY: the local backend keeps everything on the machine. Configuring
// Supabase sends every analysed record -- and, if file upload is switched on,
// the raw IQ of uploaded captures -- to that project. Nothing is sent
// anywhere until someone enters a project URL, and the History page says so
// beside the fields.

// The team's project (SEDIC26). A project URL is not a secret -- it is in
// every request the browser makes -- so it ships here.
//
// The KEY does not ship. This repository is public, and the schema's policies
// let the anon key insert, select and delete, so a committed key would let
// anyone who finds the repo empty the table. It is supplied at runtime
// instead, by whichever of these exists:
//
//   web/supabase-config.js   a one-line file, gitignored, for a local machine
//   build-time injection      web/build.py writes the SUPABASE_ANON_KEY
//                             environment variable into index.html, which is
//                             how the deployed site gets it
//
// With a key present every analysis goes to Supabase automatically. With no
// key the app stores in this browser and says so -- it never silently drops a
// record, and it never asks anyone to paste anything.
const DEFAULT_SUPABASE_URL = "https://yoirhstytgrhvfunxvlc.supabase.co";

function injected() {
  const c = globalThis.NEXA_SUPABASE || {};
  // A build that ran without the variable set leaves the placeholder behind;
  // treat that as "no key" rather than sending it as one.
  const key = typeof c.anonKey === "string" && !c.anonKey.startsWith("__") ? c.anonKey.trim() : "";
  return { url: (c.url || DEFAULT_SUPABASE_URL).trim(), key };
}

const DB_NAME = "omni-analyses";
const DB_VERSION = 1;
const STORE = "analyses";

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

/** Round for storage: full float precision on a probability is noise, and a
 *  record is read by people as often as by code. */
function r3(x) {
  return x == null || Number.isNaN(x) ? null : Math.round(x * 1000) / 1000;
}

/** Seconds need more than three decimals: a 3-window capture at hop 256 and
 *  3.2 MHz lasts 0.00024 s, which r3 stores as 0 -- the dashboard then reports
 *  "0.00 s of signal" for a real capture. Caught by web/test/history_check.mjs. */
function r6(x) {
  return x == null || Number.isNaN(x) ? null : Math.round(x * 1e6) / 1e6;
}

/**
 * One stored row, built from a finished session.
 *
 * `session` is main.js's session object ({capture, result, source, caseNote,
 * truth, snrDb, which, ...}) and `resolved` is analysis.js's resolveSession
 * output, so the record carries the SAME detections the operator saw rather
 * than a second, possibly differently-parameterised, pass over the data.
 *
 * Deliberately stores summaries, not the raw probability matrix: a 400-window
 * capture is 3,200 floats per model, and the dashboard reads counts and peaks.
 * The uploaded file (when kept) is what allows an exact re-run.
 */
export function buildRecord(session, resolved, extra = {}) {
  const { result } = session;
  const classes = extra.classes || [];
  const nClasses = result.nClasses;

  // Peak probability per class across the capture: what the operator judges
  // "was this class present anywhere" on.
  const peak = new Array(nClasses).fill(0);
  for (let w = 0; w < result.nWindows; w++) {
    for (let c = 0; c < nClasses; c++) {
      const p = result.probs[w * nClasses + c];
      if (p > peak[c]) peak[c] = p;
    }
  }

  const events = resolved?.emitterEvents || [];
  const detected = [...new Set(events.flatMap(e => e.classes))].sort();
  const tiers = resolved?.tiers || [];
  const tierCounts = {};
  for (const t of tiers) tierCounts[t] = (tierCounts[t] || 0) + 1;

  return {
    id: extra.id || cryptoId(),
    created_at: new Date().toISOString(),
    source: session.source,                       // "scenario" | "upload"
    case_note: session.caseNote || "",
    file_name: extra.fileName || null,
    file_bytes: extra.fileBytes ?? null,
    file_path: null,                              // set when the IQ is stored
    model: session.which,
    duration_s: r6(result.nWindows * result.hop / result.fs),
    n_windows: result.nWindows,
    hop: result.hop,
    snr_db: r3(session.snrDb),
    requested_snr_db: r3(session.requestedSnrDb),
    snr_capped: !!session.snrCapped,
    classes_detected: detected,
    peak_probability: Object.fromEntries(classes.map((c, i) => [c, r3(peak[i])])),
    n_events: events.length,
    tier_counts: tierCounts,
    verdict: topTier(tierCounts),
    app_version: extra.appVersion || null,
  };
}

const TIER_ORDER = ["Hostile", "Military", "Civilian", "Empty"];

/** The most serious tier seen anywhere in the capture -- the headline an
 *  operator scanning a list of past captures needs first. */
export function topTier(tierCounts) {
  for (const t of TIER_ORDER) if (tierCounts[t]) return t;
  return "Empty";
}

function cryptoId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return "id-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

// ---------------------------------------------------------------------------
// Configuration (kept in localStorage, entered by the operator)
// ---------------------------------------------------------------------------

/** What the app is configured to do, with no operator input: Supabase when a
 *  key reached the page, this browser otherwise. `storeFiles` stays off --
 *  raw IQ is ~25 MB per second of capture, which is a deliberate decision per
 *  deployment, not a default. */
export function readConfig() {
  const { url, key } = injected();
  return { backend: key ? "supabase" : "local", url, key, storeFiles: false };
}

export function usingSupabase(cfg = readConfig()) {
  return cfg.backend === "supabase" && !!cfg.url && !!cfg.key;
}

// ---------------------------------------------------------------------------
// Local backend: IndexedDB
// ---------------------------------------------------------------------------

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("created_at", "created_at");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(out?.result ?? out);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

const localBackend = {
  name: "local",
  async save(record) {
    const db = await openDb();
    await tx(db, "readwrite", s => s.put(record));
    db.close();
    return record;
  },
  async list() {
    const db = await openDb();
    const rows = await new Promise((resolve, reject) => {
      const req = db.transaction(STORE, "readonly").objectStore(STORE).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
    db.close();
    return rows.sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
  },
  async remove(id) {
    const db = await openDb();
    await tx(db, "readwrite", s => s.delete(id));
    db.close();
  },
  async clear() {
    const db = await openDb();
    await tx(db, "readwrite", s => s.clear());
    db.close();
  },
  // Raw IQ is not copied into IndexedDB: a one-minute capture is ~500 MB and
  // the browser quota is not the operator's to manage. Files go to Supabase
  // Storage or nowhere.
  async putFile() { return null; },
};

// ---------------------------------------------------------------------------
// Supabase backend: PostgREST + Storage over fetch
// ---------------------------------------------------------------------------

function headers(cfg, extra = {}) {
  return {
    apikey: cfg.key,
    Authorization: `Bearer ${cfg.key}`,
    "Content-Type": "application/json",
    ...extra,
  };
}

function base(cfg) {
  return cfg.url.replace(/\/+$/, "");
}

async function check(res, what) {
  if (res.ok) return res;
  let detail = "";
  try { detail = (await res.text()).slice(0, 200); } catch { /* body already consumed */ }
  throw new Error(`Supabase ${what} failed (${res.status}). ${detail}`);
}

function supabaseBackend(cfg) {
  const table = `${base(cfg)}/rest/v1/analyses`;
  return {
    name: "supabase",
    async save(record) {
      const res = await fetch(table, {
        method: "POST",
        headers: headers(cfg, { Prefer: "return=representation" }),
        body: JSON.stringify(record),
      });
      await check(res, "insert");
      const [row] = await res.json();
      return row || record;
    },
    async list(limit = 500) {
      const res = await fetch(
        `${table}?select=*&order=created_at.desc&limit=${limit}`,
        { headers: headers(cfg) });
      await check(res, "select");
      return res.json();
    },
    async remove(id) {
      const res = await fetch(`${table}?id=eq.${encodeURIComponent(id)}`,
        { method: "DELETE", headers: headers(cfg) });
      await check(res, "delete");
    },
    async clear() {
      // `id=neq.` with an impossible value matches every row; PostgREST
      // refuses an unfiltered DELETE, which is a good default to keep.
      const res = await fetch(`${table}?id=neq.__none__`,
        { method: "DELETE", headers: headers(cfg) });
      await check(res, "delete all");
    },
    async putFile(file, id) {
      const path = `${id}/${file.name.replace(/[^\w.\-]/g, "_")}`;
      const res = await fetch(`${base(cfg)}/storage/v1/object/captures/${path}`, {
        method: "POST",
        headers: { apikey: cfg.key, Authorization: `Bearer ${cfg.key}` },
        body: file,
      });
      await check(res, "file upload");
      return path;
    },
  };
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export function backendFor(cfg = readConfig()) {
  return usingSupabase(cfg) ? supabaseBackend(cfg) : localBackend;
}

/**
 * Store one analysis. Returns {record, backend, warning}.
 *
 * A storage failure must never lose the analysis the operator is looking at,
 * so a failing Supabase write falls back to the local store and reports the
 * reason instead of throwing into the analyse path.
 */
export async function saveAnalysis(record, file = null) {
  const cfg = readConfig();
  const backend = backendFor(cfg);
  try {
    if (file && cfg.storeFiles && backend.name === "supabase") {
      record = { ...record, file_path: await backend.putFile(file, record.id) };
    }
    return { record: await backend.save(record), backend: backend.name, warning: null };
  } catch (e) {
    if (backend.name === "supabase") {
      const saved = await localBackend.save(record);
      return { record: saved, backend: "local", warning: `${e.message} Saved locally instead.` };
    }
    throw e;
  }
}

/**
 * Every stored analysis, newest first: {records, backend, warning}.
 *
 * Reads fall back exactly like writes do. Without this, a bad or expired key
 * made the History page empty -- while saveAnalysis was quietly writing those
 * same captures to the local store, so the records existed and the page
 * claimed they did not.
 */
export async function listAnalyses() {
  const backend = backendFor();
  try {
    return { records: await backend.list(), backend: backend.name, warning: null };
  } catch (e) {
    if (backend.name !== "supabase") throw e;
    return {
      records: await localBackend.list(),
      backend: "local",
      warning: `${e.message} Showing what is stored in this browser instead.`,
    };
  }
}

export async function deleteAnalysis(id) {
  return backendFor().remove(id);
}

export async function clearAnalyses() {
  return backendFor().clear();
}

/** Round-trip check for the settings form: proves the URL and key work
 *  before the operator trusts the next capture to them. */
export async function testConnection(cfg) {
  if (!cfg.url || !cfg.key) throw new Error("Both a project URL and a key are required.");
  const res = await fetch(`${base(cfg)}/rest/v1/analyses?select=id&limit=1`,
    { headers: headers(cfg) });
  await check(res, "connection test");
  return true;
}
