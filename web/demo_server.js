// Phone demo: the local server's API, answered inside the browser.
//
// The public site (Vercel, opened from the brochure on a judge's phone) has no
// scripts/serve_local.py behind it. Rather than a second copy of the page's
// logic, this answers the same /api/* requests the page already makes, with
// the same rules as src/localdb.py and scripts/serve_local.py: two roles,
// four-eyes on corrections, retention, the retraining trigger, model
// versions and a hash-chained audit trail. So every button behaves as in the
// local version.
//
// What differs, honestly labelled on the page:
//   * data lives in THIS browser only (IndexedDB) -- nothing is shared or sent;
//   * retraining is SIMULATED: a phone cannot fine-tune the model, so the run
//     plays out its log (and so the progress bar) and registers a candidate
//     whose exam numbers are the shipped ensemble's test recall with small
//     illustrative changes. The weights in use never change here.
//
// Installed by storage.js:detectServer() only when no real server answers.

const CLASSES = ["BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING", "NOISE_FLOOR"];
const JUDGED = ["LFM_RADAR", "FHSS", "JAMMING"];
const ROLES = ["operator", "analyst"];
const PEOPLE = { operator: "operator", analyst: "analyst" };
const RULES = { window_days: 7, min_reviewed: 100, max_rate: 0.15, min_per_class: 30, cooldown_days: 7 };
const ROLLING_IQ_BYTES = 100 * 1024 ** 2;        // a phone, not a laptop: 100 MB
const THREAT_VERDICTS = ["Hostile", "Military"];
const SHIPPED = { single: "shipped best_model.pt", ensemble: "shipped 5-model submission" };
const GENESIS_HASH = "0".repeat(64);
// Shipped 5-model ensemble, held-out test split (evals/ensemble_scorecard.json).
const SHIPPED_RECALL = { LFM_RADAR: 0.8415, FHSS: 0.8291, JAMMING: 0.8399 };
const SIM_SECONDS = { single: 45, both: 90 };
const PERSON_KEY = "nexa-demo-as";

// ---------------------------------------------------------------------------
// Storage: one JSON state object plus raw IQ buffers, in IndexedDB
// ---------------------------------------------------------------------------

let dbp = null;
function idb() {
  dbp ||= new Promise((resolve, reject) => {
    const req = indexedDB.open("nexa-phone-demo", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("kv");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}
async function kvGet(key) {
  const db = await idb();
  return new Promise((resolve, reject) => {
    const r = db.transaction("kv").objectStore("kv").get(key);
    r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error);
  });
}
async function kvSet(key, value) {
  const db = await idb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("kv", "readwrite");
    if (value === undefined) tx.objectStore("kv").delete(key); else tx.objectStore("kv").put(value, key);
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error);
  });
}

const EMPTY = () => ({ analyses: [], corrections: [], models: [], jobs: [], audit: [], files: {} });
let S = null;                 // the state, loaded once
let queue = Promise.resolve();  // requests run one at a time, like the server's write lock

async function load() {
  if (!S) S = (await kvGet("state")) || EMPTY();
  return S;
}
const save = () => kvSet("state", S);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class HttpError extends Error { constructor(status, msg) { super(msg); this.status = status; } }
const conflict = m => new HttpError(409, m), bad = m => new HttpError(400, m), missing = m => new HttpError(404, m);
const nowIso = () => new Date().toISOString();
const uuid = () => (crypto.randomUUID ? crypto.randomUUID()
  : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => ((Math.random() * 16) | 0).toString(16)));

function canonical(v) {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (v && typeof v === "object") return `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${canonical(v[k])}`).join(",")}}`;
  return JSON.stringify(v ?? null);
}
async function sha256Hex(data) {
  const buf = typeof data === "string" ? new TextEncoder().encode(data) : data;
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, "0")).join("");
}
const entryBody = e => ({ seq: e.seq, ts: e.ts, actor: e.actor, role: e.role, client: e.client, action: e.action,
                          target_type: e.target_type, target_id: e.target_id, details: e.details });

async function audit(actor, action, targetType, targetId, details = {}) {
  const last = S.audit[S.audit.length - 1];
  const prev = last ? last.hash : GENESIS_HASH;
  const e = { seq: (last?.seq || 0) + 1, ts: nowIso(), actor: actor.name, role: actor.role, client: "this phone",
              action, target_type: targetType ?? null, target_id: targetId ?? null, details: canonical(details) };
  e.prev_hash = prev;
  e.hash = await sha256Hex(prev + canonical(entryBody(e)));
  S.audit.push(e);
  return e;
}
const auditOut = e => ({ ...e, details: JSON.parse(e.details) });

function me() {
  let name = "operator";
  try { name = localStorage.getItem(PERSON_KEY) || "operator"; } catch { /* default */ }
  if (!PEOPLE[name]) name = "operator";
  return { name, role: PEOPLE[name] };
}
const atLeast = (role, need) => ROLES.indexOf(role) >= ROLES.indexOf(need);

// ---------------------------------------------------------------------------
// Analyses and retention (localdb.py: retention, prune_rolling_iq)
// ---------------------------------------------------------------------------

const correctedIds = () => new Set(S.corrections.map(c => c.analysis_id));
function retention(a, corrected) {
  if (a.iq_pruned_at) return ["deleted", "routine signal, removed by the rolling window"];
  if (!a.file_path) return ["none", "no raw signal stored"];
  if (a.used_for_training) return ["kept", `used to train ${a.used_for_training}`];
  if (corrected) return ["kept", "a human corrected it"];
  if (THREAT_VERDICTS.includes(a.verdict) || a.alert_level) return ["kept", "threat detected"];
  if (a.needs_review) return ["kept", "close call, flagged for review"];
  return ["rolling", "routine signal, kept briefly"];
}
function analysisOut(a, corr = correctedIds()) {
  const [kind, why] = retention(a, corr.has(a.id));
  return { ...a, retention: { kind, why }, corrected: corr.has(a.id) };
}
async function pruneRolling(cap = ROLLING_IQ_BYTES) {
  const corr = correctedIds();
  const routine = S.analyses.filter(a => a.file_path && !a.iq_pruned_at && retention(a, corr.has(a.id))[0] === "rolling")
    .sort((x, y) => (x.created_at < y.created_at ? 1 : -1));                  // newest first
  let total = 0; const gone = [];
  for (const a of routine) { total += S.files[a.file_path]?.bytes || 0; if (total > cap) gone.push(a); }
  if (!gone.length) return;
  const stamp = nowIso();
  for (const a of gone) { a.iq_pruned_at = stamp; await kvSet(`iq:${a.file_path}`, undefined); }
  await audit({ name: "system", role: "system" }, "iq.prune", "analysis", null, {
    reason: "rolling window for routine signals", cap_bytes: cap,
    bytes_freed: gone.reduce((n, a) => n + (S.files[a.file_path]?.bytes || 0), 0),
    deleted: gone.map(a => ({ id: a.id, sha256: a.file_sha256 || null, verdict: a.verdict || null })),
  });
}

// ---------------------------------------------------------------------------
// Retraining trigger (localdb.py: retrain_status)
// ---------------------------------------------------------------------------

function retrainStatus() {
  const since = new Date(Date.now() - RULES.window_days * 864e5).toISOString();
  const analysed = S.analyses.filter(a => a.created_at >= since).length;
  const corrected = new Set(S.corrections.filter(c => c.created_at >= since && c.status !== "rejected").map(c => c.analysis_id)).size;
  const last = S.models.reduce((m, x) => (!m || x.created_at > m ? x.created_at : m), null);
  const perClass = Object.fromEntries(CLASSES.map(c => [c, 0]));
  for (const c of S.corrections) if (c.status === "approved" && (!last || (c.reviewed_at || "") > last)) c.corrected_labels.forEach(l => perClass[l]++);
  const rate = analysed ? corrected / analysed : 0;
  const daysSince = last ? (Date.now() - new Date(last).getTime()) / 864e5 : null;
  const best = Object.keys(perClass).reduce((a, b) => (perClass[b] > perClass[a] ? b : a));
  const conditions = [
    { id: "volume", met: analysed >= RULES.min_reviewed, text: `At least ${RULES.min_reviewed} captures analysed in the last ${RULES.window_days} days (now ${analysed})` },
    { id: "rate", met: analysed >= RULES.min_reviewed && rate > RULES.max_rate, text: `Experts corrected more than ${RULES.max_rate * 100}% of them (now ${(rate * 100).toFixed(1)}%)` },
    { id: "data", met: perClass[best] >= RULES.min_per_class, text: `At least ${RULES.min_per_class} new approved corrections in one class `
      + (perClass[best] ? `(most: ${best} ${perClass[best]})` : "(none yet)") },
    { id: "cooldown", met: daysSince === null || daysSince >= RULES.cooldown_days, text: `At least ${RULES.cooldown_days} days since the last retrain `
      + (daysSince === null ? "(never retrained)" : `(${daysSince.toFixed(1)} days)`) },
  ];
  const triggered = conditions[0].met && conditions[1].met, ready = conditions.every(c => c.met);
  return { rules: RULES, analysed, corrected, rate, approved_new_per_class: perClass, last_retrain_at: last,
           days_since_retrain: daysSince, conditions, triggered, recommended: ready,
           summary: ready ? "Retraining recommended: every rule is met." : triggered ? "Triggered, but not every safety check is met yet."
             : "Not triggered: the model is not being corrected often enough to need it." };
}

// ---------------------------------------------------------------------------
// Simulated retrain: the log the real script writes, played out over time
// ---------------------------------------------------------------------------

function simulatedLog(job, approved) {
  const dur = SIM_SECONDS[job.scope] || 45, epochs = 2, runs = job.scope === "both" ? 6 : 1;
  const lines = [
    [0, `Retrain job ${job.id.slice(0, 8)} started by ${job.started_by}: "${job.reason}"${job.override ? " (OVERRIDE: rules not all met)" : ""}`],
    [0, "SIMULATED on this phone: no training runs here. The laptop version fine-tunes the model for real."],
    [0, `Scope: ${job.scope === "both" ? "single model + the 5-model ensemble" : "single model only"}`],
    [1, "[1/4] Collecting approved corrections…"],
    [1.5, `      ${approved.length} corrections -> ${approved.length * 40} windows`],
    [3, "[2/4] Preparing replay data and the fixed exam…"],
    [4, "      exam: shipped ensemble's held-out test split (numbers illustrative on the phone)"],
  ];
  const t0 = 5, t1 = dur * 0.9, per = (t1 - t0) / (runs * epochs);
  let k = 0;
  lines.push([t0, `[3/${job.scope === "both" ? 5 : 4}] Single model: fine-tuning…`]);
  for (let r = 0; r < runs; r++) {
    const who = r === 0 ? "single" : `member ${r}/5`;
    if (r === 1) lines.push([t0 + per * k, "[4/5] Ensemble: fine-tuning all 5 members…"]);
    for (let e = 1; e <= epochs; e++) {
      k++;
      lines.push([t0 + per * k, `      ${who} epoch ${e}/${epochs}: loss ${(0.09 / (k + 1) + 0.02).toFixed(4)} (simulated)`]);
    }
  }
  lines.push([t1 + 1, "      exam: PASS on every rule (illustrative)"]);
  lines.push([dur - 1, `[${job.scope === "both" ? 5 : 4}/${job.scope === "both" ? 5 : 4}] Registering candidates…`]);
  lines.push([dur, `Done in ${dur} s (simulated).`]);
  return lines;
}

function gateFor(kind, i) {
  const nudge = { LFM_RADAR: 0.004 + i * 0.001, FHSS: 0.006, JAMMING: -0.002 };
  const before = { recall: { ...SHIPPED_RECALL }, noise_false_alarm: null };
  const after = { recall: Object.fromEntries(JUDGED.map(c => [c, +(SHIPPED_RECALL[c] + nudge[c]).toFixed(4)])), noise_false_alarm: null };
  const rules = [];
  for (const c of JUDGED) {
    const b = before.recall[c], a = after.recall[c];
    rules.push({ text: `${c}: recall may not drop more than 1 point(s) (${b.toFixed(3)} -> ${a.toFixed(3)})`, met: a >= b - 0.01 });
    rules.push({ text: `${c}: recall stays at or above 0.80 (${a.toFixed(3)})`, met: a >= 0.8 });
  }
  return { set: "SIMULATED on this phone: shipped ensemble's test-split recall, with illustrative changes",
           kind: "synthetic", passed: rules.every(r => r.met), rules, before, after };
}

/** Bring a running simulated job up to date (called before every request). */
async function tickJobs() {
  for (const job of S.jobs.filter(j => j.status === "queued" || j.status === "running")) {
    const elapsed = (Date.now() - new Date(job.created_at).getTime()) / 1000;
    const lines = job._plan.filter(([t]) => t <= elapsed).map(([, l]) => l);
    job.log = lines.join("\n") + "\n";
    job.status = "running";
    if (elapsed < SIM_SECONDS[job.scope]) continue;
    const stamp = nowIso().replace(/[-:T]/g, "").slice(0, 15);
    const kinds = job.scope === "both" ? ["single", "ensemble"] : ["single"];
    const cands = kinds.map((kind, i) => ({
      version: `${kind === "single" ? "ft" : "ens"}-${stamp}`, kind, created_at: nowIso(),
      path: "simulated (the shipped weights stay in use on the phone)", status: "candidate", notes: "",
      approved_by: null, approved_at: null,
      source: { job_id: job.id, corrections: job._approved, replay_windows: 2000, base: SHIPPED[kind] },
      metrics: { gate: gateFor(kind, i), base: SHIPPED[kind],
                 train: { epochs: 2, corrections_used: job._approved, replay_windows: 2000,
                          correction_windows: job._approved.length * 40, seconds: SIM_SECONDS[job.scope] } },
    }));
    S.models.push(...cands);
    job.status = "succeeded"; job.finished_at = nowIso(); job.model_version = cands.map(c => c.version).join(",");
    job.result = { seconds: SIM_SECONDS[job.scope], scope: job.scope, simulated: true };
    const trained = {};
    for (const c of cands) {
      trained[c.version] = S.corrections.filter(x => job._approved.includes(x.id))
        .map(x => ({ correction: x.id, capture: x.analysis_id, iq_sha256: x.iq_sha256, labels: x.corrected_labels }));
      for (const t of trained[c.version]) {
        const a = S.analyses.find(z => z.id === t.capture);
        if (a) a.used_for_training = [...new Set([...(a.used_for_training || "").split(",").filter(Boolean), c.version])].join(",");
      }
    }
    await audit({ name: "system", role: "system" }, "retrain.finish", "job", job.id, {
      candidates: Object.fromEntries(cands.map(c => [c.version, { kind: c.kind, gate_passed: c.metrics.gate.passed }])),
      trained_on: trained, simulated: true, error: null,
    });
    await save();
  }
}
const jobOut = j => { const { _plan, _approved, ...rest } = j; return rest; };

// ---------------------------------------------------------------------------
// Models (localdb.py: review_model, activate_model, rollback_model, retrain_history)
// ---------------------------------------------------------------------------

async function activate(version, note, actor, override = false) {
  const m = S.models.find(x => x.version === version);
  if (!m) throw bad("no such model");
  if (m.status === "active") throw conflict(`${version} is already in use`);
  note = String(note || "").trim();
  const passed = !!m.metrics?.gate?.passed;
  let action;
  if (!passed || m.status === "rejected") {
    if (!override) throw conflict(`${version} ${passed ? "was rejected" : "failed its gate"}. Activating it needs an override and a reason.`);
    if (note.length < 10) throw bad("an override needs a real reason (at least 10 characters)");
    action = "model.override";
  } else if (m.status === "retired") {
    if (note.length < 3) throw bad("say why you are restoring this version");
    action = "model.restore";
  } else action = "model.approve";
  const prev = S.models.find(x => x.status === "active" && x.kind === m.kind);
  if (prev) prev.status = "retired";
  Object.assign(m, { status: "active", approved_by: actor.name, approved_at: nowIso(), notes: note || m.notes });
  await audit(actor, action, "model", version, { note, kind: m.kind, gate_passed: passed, replaced: prev ? prev.version : SHIPPED[m.kind] });
  return m;
}

function history() {
  const events = {};
  for (const e of S.audit) {
    if (e.action.startsWith("model.") && e.target_id) {
      const d = JSON.parse(e.details);
      (events[e.target_id] ||= []).push({ ts: e.ts, actor: e.actor, action: e.action, note: d.note || "", replaced: d.replaced ?? null, seq: e.seq });
    }
  }
  return [...S.jobs].reverse().map(job => ({
    ...jobOut(job),
    candidates: (job.model_version || "").split(",").filter(Boolean).map(v => S.models.find(m => m.version === v)).filter(Boolean).map(m => ({
      version: m.version, kind: m.kind, status: m.status, base: m.metrics.base, gate_passed: m.metrics.gate.passed,
      exam: m.metrics.gate.kind, before: m.metrics.gate.before, after: m.metrics.gate.after,
      corrections: m.metrics.train.corrections_used.length, seconds: m.metrics.train.seconds, events: events[m.version] || [],
    })),
  }));
}

// ---------------------------------------------------------------------------
// Routes (serve_local.py: required_role and _route)
// ---------------------------------------------------------------------------

function requiredRole(method, parts) {
  const head = parts[0] || "";
  if (head === "users" || (method === "POST" && parts.join("/") === "retrain/start")
      || (method === "POST" && parts.join("/") === "models/rollback") || (method === "DELETE" && head === "analyses")) return "analyst";
  if (method === "POST" && (head === "corrections" || head === "models") && parts.length === 3 && ["review", "activate"].includes(parts[2])) return "analyst";
  return "operator";
}

const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });

async function readBody(init) {
  const b = init?.body;
  if (b == null) return {};
  if (typeof b === "string") return b ? JSON.parse(b) : {};
  return b;
}

async function route(method, parts, query, init) {
  const head = parts[0] || "", actor = me();
  if (head === "health") {
    const single = S.models.find(m => m.status === "active" && m.kind === "single");
    const ens = S.models.find(m => m.status === "active" && m.kind === "ensemble");
    return json({ app: "nexa-local", ok: true, demo_in_browser: true, db: "this browser (phone demo)",
                  counts: { analyses: S.analyses.length, audit_log: S.audit.length },
                  active_model: single?.version || null, active_ensemble: ens?.version || null });
  }
  if (head === "auth") {
    if (method === "GET" && parts[1] === "me") {
      return json({ auth_enabled: true, demo: true, demo_in_browser: true, can_setup: false, user: { username: actor.name, role: actor.role },
                    people: Object.entries(PEOPLE).map(([username, role]) => ({ username, role })) });
    }
    if (method === "POST" && parts[1] === "demo") {
      const name = String((await readBody(init)).username || "").toLowerCase();
      if (!PEOPLE[name]) throw bad(`demo person must be one of ${Object.keys(PEOPLE).join(", ")}`);
      try { localStorage.setItem(PERSON_KEY, name); } catch { /* private mode: stays operator */ }
      await audit({ name, role: PEOPLE[name] }, "user.demo_switch", "user", name, { from: actor.name });
      await save();
      return json({ user: { username: name, role: PEOPLE[name] } });
    }
    if (method === "POST" && parts[1] === "logout") return json({ signed_out: true });
    throw missing("sign-in is off in the demo; switch person instead");
  }
  if (!atLeast(actor.role, requiredRole(method, parts))) {
    throw new HttpError(403, `This needs the ${requiredRole(method, parts)} role; you are acting as ${actor.name} (${actor.role}).`);
  }
  if (head === "users") return json([]);

  if (head === "analyses") {
    const corr = correctedIds();
    if (method === "GET" && parts.length === 1) {
      const limit = Number(query.get("limit") || 500);
      return json([...S.analyses].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).slice(0, limit).map(a => analysisOut(a, corr)));
    }
    if (method === "POST" && parts.length === 1) {
      const r = await readBody(init);
      if (!r.id || !r.created_at || !r.source) throw bad("record is missing id, created_at or source");
      if (S.analyses.some(a => a.id === r.id)) throw conflict("that capture is already stored");
      const a = { ...r, operator: actor.name, alert_level: r.alert_level || null, needs_review: !!r.needs_review,
                  reviewed_by: null, reviewed_at: null, iq_pruned_at: null, used_for_training: null };
      const f = Object.values(S.files).find(x => x.analysis_id === a.id);
      if (f && !a.file_path) { a.file_path = f.path; a.file_sha256 = f.sha256; }
      S.analyses.push(a);
      await audit(actor, "analysis.create", "analysis", a.id, { source: a.source, verdict: a.verdict || null,
        classes_detected: a.classes_detected || [], model: a.model || null, file_name: a.file_name || null, file_sha256: a.file_sha256 || null });
      await pruneRolling();
      await save();
      return json(analysisOut(a), 201);
    }
    const a = S.analyses.find(x => x.id === parts[1]);
    if (method === "DELETE" && parts.length === 1) {
      const ids = S.analyses.filter(x => !corr.has(x.id)).map(x => x.id);
      S.analyses = S.analyses.filter(x => corr.has(x.id));
      await audit(actor, "analysis.clear", "analysis", null, { deleted_ids: ids, kept_with_corrections: S.analyses.length });
      await save();
      return json({ deleted: ids.length });
    }
    if (!a) throw missing("no such capture");
    if (method === "DELETE" && parts.length === 2) {
      if (corr.has(a.id)) throw conflict("This capture has human corrections attached and is kept as their evidence; it cannot be deleted.");
      S.analyses = S.analyses.filter(x => x !== a);
      await audit(actor, "analysis.delete", "analysis", a.id, { deleted: a });
      await save();
      return json({ deleted: a.id });
    }
    if (method === "GET" && parts[2] === "iq") {
      const out = analysisOut(a, corr);
      if (out.retention.kind === "deleted") throw new HttpError(410, "this routine signal was deleted by the rolling window; only its result is kept");
      return iqResponse(a.file_path);
    }
    if (method === "POST" && parts[2] === "reviewed") {
      if (a.reviewed_by) throw conflict(`already reviewed by ${a.reviewed_by}`);
      a.reviewed_by = actor.name; a.reviewed_at = nowIso();
      await audit(actor, "analysis.reviewed", "analysis", a.id, { verdict: a.verdict || null, agrees_with_model: true,
        note: String((await readBody(init)).note || "").slice(0, 300) });
      await save();
      return json(analysisOut(a));
    }
  }

  if (head === "captures" && method === "PUT" && parts.length === 2) {
    const id = parts[1], name = (query.get("name") || "capture.bin").replace(/[^\w.\-]/g, "_").slice(0, 120);
    const buf = await new Response(init.body).arrayBuffer();
    const path = `iq/${id}/${name}`, sha = await sha256Hex(buf);
    await kvSet(`iq:${path}`, buf);
    S.files[path] = { path, analysis_id: id, bytes: buf.byteLength, sha256: sha };
    const a = S.analyses.find(x => x.id === id);
    if (a && !a.file_path) { a.file_path = path; a.file_sha256 = sha; }
    await audit(actor, "iq.store", "analysis", id, { name, path, bytes: buf.byteLength, sha256: sha });
    await save();
    return json({ file_path: path, file_sha256: sha, bytes: buf.byteLength }, 201);
  }

  if (head === "corrections") {
    if (method === "POST" && parts.length === 1) {
      const b = await readBody(init);
      const labels = (v, what) => {
        if (!Array.isArray(v)) throw bad(`${what} must be a list of class names`);
        const unknown = v.filter(x => !CLASSES.includes(x));
        if (unknown.length) throw bad(`${what}: unknown class ${unknown.join(", ")}`);
        return [...new Set(v)].sort((x, y) => CLASSES.indexOf(x) - CLASSES.indexOf(y));
      };
      const predicted = labels(b.predicted_labels || [], "predicted_labels"), corrected = labels(b.corrected_labels || [], "corrected_labels");
      if (!corrected.length) throw bad("say what is really there: tick at least one class (NOISE_FLOOR if nothing)");
      if (corrected.join() === predicted.join()) throw bad("that is what the model already said -- nothing to correct");
      const reason = String(b.reason || "").trim();
      if (reason.length < 5) throw bad("a reason is required (at least a few words)");
      const start = Number(b.start_s), end = Number(b.end_s);
      if (!(start >= 0 && start < end)) throw bad("the corrected span must have start < end");
      const a = S.analyses.find(x => x.id === b.analysis_id);
      if (!a) throw bad("no such capture");
      if (end > (a.duration_s ?? Infinity) + 1e-3) throw bad("the corrected span runs past the end of the capture");
      if (!a.file_path) throw bad("this capture has no stored raw IQ; upload it before correcting");
      const c = { id: uuid(), analysis_id: a.id, created_at: nowIso(), operator: actor.name, start_s: start, end_s: end,
                  predicted_labels: predicted, corrected_labels: corrected, reason, model: b.model || a.model,
                  iq_path: a.file_path, iq_sha256: a.file_sha256 || null, status: "pending",
                  reviewed_by: null, reviewed_at: null, review_note: null };
      S.corrections.push(c);
      if (!a.reviewed_by) { a.reviewed_by = actor.name; a.reviewed_at = c.created_at; }
      await audit(actor, "correction.create", "correction", c.id, { analysis_id: a.id, span_s: [start, end], predicted, corrected, reason, iq_sha256: c.iq_sha256 });
      await save();
      return json(c, 201);
    }
    if (method === "GET" && parts.length === 1) {
      const st = query.get("status"), aid = query.get("analysis_id");
      return json(S.corrections.filter(c => (!st || c.status === st) && (!aid || c.analysis_id === aid))
        .sort((x, y) => (x.created_at < y.created_at ? 1 : -1)));
    }
    if (method === "GET" && parts[1] === "stats") {
      const per = Object.fromEntries(CLASSES.map(c => [c, 0]));
      S.corrections.filter(c => c.status === "approved").forEach(c => c.corrected_labels.forEach(l => per[l]++));
      const n = s => S.corrections.filter(c => c.status === s).length;
      return json({ pending: n("pending"), approved: n("approved"), rejected: n("rejected"), approved_per_class: per });
    }
    const c = S.corrections.find(x => x.id === parts[1]);
    if (!c) throw missing("no such correction");
    if (method === "GET" && parts[2] === "iq") return iqResponse(c.iq_path);
    if (method === "POST" && parts[2] === "review") {
      const b = await readBody(init), status = { approve: "approved", reject: "rejected" }[b.decision];
      if (!status) throw bad("decision must be 'approve' or 'reject'");
      const note = String(b.note || "").trim();
      if (status === "rejected" && note.length < 3) throw bad("say why it is rejected");
      if (c.status !== "pending") throw conflict(`already ${c.status} by ${c.reviewed_by}`);
      if (c.operator.toLowerCase() === actor.name.toLowerCase()) {
        throw conflict("Four-eyes rule: a correction must be reviewed by someone other than the person who submitted it.");
      }
      Object.assign(c, { status, reviewed_by: actor.name, reviewed_at: nowIso(), review_note: note });
      await audit(actor, `correction.${b.decision}`, "correction", c.id, { analysis_id: c.analysis_id, submitted_by: c.operator, corrected: c.corrected_labels, note });
      await save();
      return json(c);
    }
  }

  if (head === "retrain") {
    if (method === "GET" && parts[1] === "status") return json(retrainStatus());
    if (method === "GET" && parts[1] === "history") return json(history());
    if (method === "GET" && parts[1] === "jobs" && parts.length === 2) return json([...S.jobs].reverse().slice(0, 20).map(jobOut));
    if (method === "GET" && parts[1] === "jobs") {
      const j = S.jobs.find(x => x.id === parts[2]);
      if (!j) throw missing("no such job");
      return json(jobOut(j));
    }
    if (method === "POST" && parts[1] === "start") {
      const b = await readBody(init), reason = String(b.reason || "").trim(), scope = b.scope || "single";
      if (reason.length < 5) throw bad("say why you are retraining (at least a few words)");
      if (!["single", "both"].includes(scope)) throw bad("scope must be 'single' or 'both'");
      const status = retrainStatus();
      if (!status.recommended && !b.override) {
        throw conflict("The retraining rules are not all met. Tick 'override' to start anyway; your reason is recorded in the audit trail.");
      }
      const busy = S.jobs.find(j => j.status === "queued" || j.status === "running");
      if (busy) throw conflict(`A retrain is already running (job ${busy.id.slice(0, 8)}).`);
      const approved = S.corrections.filter(c => c.status === "approved").map(c => c.id);
      const job = { id: uuid(), created_at: nowIso(), started_by: actor.name, reason, override: !!b.override, scope,
                    status: "queued", finished_at: null, model_version: null, result: {}, log: "" };
      job._plan = simulatedLog(job, approved); job._approved = approved;
      S.jobs.push(job);
      await audit(actor, "retrain.start", "job", job.id, { reason, override: !!b.override, scope, simulated: true,
        rules_met: Object.fromEntries(status.conditions.map(c => [c.id, c.met])), correction_rate: +status.rate.toFixed(4) });
      await save();
      return json(jobOut(job));
    }
  }

  if (head === "models") {
    if (method === "GET" && parts.length === 1) return json([...S.models].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)));
    if (method === "POST" && parts[1] === "rollback") {
      const b = await readBody(init), kind = b.kind || "single";
      const m = S.models.find(x => x.status === "active" && x.kind === kind);
      if (!m) throw bad(`already on the ${SHIPPED[kind]}; nothing to roll back`);
      m.status = "retired";
      await audit(actor, "model.rollback", "model", m.version, { note: String(b.note || "").trim(), kind, now_using: SHIPPED[kind] });
      await save();
      return json({ rolled_back: m.version });
    }
    const b = await readBody(init);
    if (method === "POST" && parts[2] === "activate") {
      const m = await activate(parts[1], b.note, actor, !!b.override);
      await save();
      return json(m);
    }
    if (method === "POST" && parts[2] === "review") {
      const m = S.models.find(x => x.version === parts[1]);
      if (!m) throw bad("no such model");
      if (b.decision === "approve") {
        if (m.status !== "candidate") throw conflict(`${m.version} is ${m.status}, not a candidate`);
        if (!m.metrics.gate.passed) throw conflict("This candidate failed its gate and cannot be activated without an override.");
        const out = await activate(m.version, b.note, actor);
        await save();
        return json(out);
      }
      if (b.decision !== "reject") throw bad("decision must be 'approve' or 'reject'");
      const note = String(b.note || "").trim();
      if (note.length < 3) throw bad("say why it is rejected");
      if (m.status !== "candidate") throw conflict(`${m.version} is ${m.status}, not a candidate`);
      Object.assign(m, { status: "rejected", approved_by: actor.name, approved_at: nowIso(), notes: note });
      await audit(actor, "model.reject", "model", m.version, { note, kind: m.kind });
      await save();
      return json(m);
    }
  }

  if (head === "audit") {
    if (method === "GET" && parts[1] === "verify") {
      let prev = GENESIS_HASH;
      for (const e of S.audit) {
        if (e.prev_hash !== prev) return json({ ok: false, entries: S.audit.length, broken_at: e.seq, reason: "points at a previous entry that is missing or changed" });
        if (await sha256Hex(prev + canonical(entryBody(e))) !== e.hash) return json({ ok: false, entries: S.audit.length, broken_at: e.seq, reason: "contents changed after it was written" });
        prev = e.hash;
      }
      return json({ ok: true, entries: S.audit.length, broken_at: null, reason: null });
    }
    if (method === "GET") {
      const limit = Number(query.get("limit") || 200), pre = query.get("action");
      return json(S.audit.filter(e => !pre || e.action.startsWith(pre)).slice(-limit).reverse().map(auditOut));
    }
    if (method === "POST") {
      const b = await readBody(init);
      if (!String(b.action || "").startsWith("client.")) throw bad("client events must start with 'client.'");
      const e = await audit(actor, b.action, b.target_type, b.target_id, b.details || {});
      await save();
      return json(auditOut(e), 201);
    }
  }

  if (head === "export") throw new HttpError(501, "Excel, CSV, JSON and SigMF reports are built by the laptop version. The PDF report works here.");
  throw missing("no such endpoint");
}

async function iqResponse(path) {
  const buf = path ? await kvGet(`iq:${path}`) : null;
  if (!buf) throw missing("no raw signal was stored for this capture");
  return new Response(buf, { status: 200, headers: { "Content-Type": "application/octet-stream",
    "X-NEXA-Datatype": "cf32_le", "X-NEXA-Sample-Rate": "3200000" } });
}

// ---------------------------------------------------------------------------
// Install: answer /api/* in the browser, pass everything else through
// ---------------------------------------------------------------------------

let installed = false;

/** Start answering /api/*; returns the health reply detectServer() expects. */
export async function startPhoneDemo() {
  if (!installed) {
    installed = true;
    await load();
    const realFetch = globalThis.fetch.bind(globalThis);
    globalThis.fetch = (input, init = {}) => {
      const url = new URL(typeof input === "string" ? input : input.url, location.href);
      if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return realFetch(input, init);
      const method = (init.method || "GET").toUpperCase();
      const parts = url.pathname.split("/").filter(Boolean).slice(1).map(decodeURIComponent);
      const run = queue.then(async () => {
        try {
          await tickJobs();
          return await route(method, parts, url.searchParams, init);
        } catch (e) {
          return json({ error: e.message }, e instanceof HttpError ? e.status : 500);
        }
      });
      queue = run.catch(() => {});
      return run;
    };
  }
  return (await (await fetch("/api/health")).json());
}
