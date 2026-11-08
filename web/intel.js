// Comms Intel page: receiver AUDIO recordings decoded on this machine by
// scripts/comms_intel.py (Morse decoder, Whisper for voice, a local Ollama
// model for the reading), served read-only by scripts/serve_local.py at
// /api/intel. The LLM reading is shown NEXT TO the decoded text it came from,
// and every quote it gives is one the server already checked occurs in that
// text (src/intel/reason.py ground()).

const CONF_COLOR = { high: "#2E7D32", medium: "#B7791F", low: "#9B2C2C" };
const KIND_LABEL = { morse: "Morse (CW)", voice: "Voice", silent: "No signal" };

const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function hms(sec) {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m ${String(s).padStart(2, "0")}s`;
}

function clock(sec) {
  sec = Math.floor(sec || 0);
  return [Math.floor(sec / 3600), Math.floor(sec % 3600 / 60), sec % 60].map(v => String(v).padStart(2, "0")).join(":");
}

function chip(text, color) {
  return `<span class="chip" style="background:${color}1f;color:${color};">${esc(text)}</span>`;
}

// Plain words for the recording panel (shown to non-technical visitors).
const SURE_TEXT = { high: "The AI is sure", medium: "The AI is fairly sure", low: "The AI is not sure" };
const Q_MEANING = {
  QTC: "I have messages for you", QRU: "I have nothing for you", QRV: "I am ready", QSL: "I confirm I received it",
  QRZ: "Who is calling me?", QTH: "My location is…", QRM: "There is interference", QRN: "There is static noise",
  QSY: "Change frequency", QRT: "Stop sending", QRX: "Wait", QSA: "Your signal strength is…", QRK: "How readable am I?",
};
const BIG = "font-size:1.12rem;line-height:1.55;";
const H = "font-size:1.25rem;font-weight:700;margin:18px 0 6px;";

function plainDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleString(undefined, { day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

function plainHours(sec) {
  sec = Math.round(sec || 0);
  if (sec < 60) return `${sec} seconds`;
  const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60);
  return h ? `${h} hour${h > 1 ? "s" : ""} ${m} minutes` : `${m} minutes`;
}

/** The "In short" lines at the top of a recording (exported for tests). */
export function inShort(rec) {
  const r = rec.reading || {}, f = r.facts || {};
  const lines = [];
  const top = (f.call_sign_like || [])[0];
  const kind = rec.kind === "morse" ? "Morse code (beeps)" : rec.kind === "voice" ? "people speaking" : "no clear signal";
  lines.push(["What it is", `${kind}${top ? `, mostly from a station calling itself <b>${esc(top[0])}</b>` : ""}`]);
  const n = (rec.segments || []).length;
  lines.push(["How much", `${n} message${n === 1 ? "" : "s"} found in ${plainHours(rec.analysed_s)} of listening`]);
  const coded = /cod|group|cipher/i.test(`${r.category || ""} ${r.summary || ""}`) || (f.five_char_groups_share || 0) > 0.2;
  if (rec.reading && !r.error) {
    lines.push(["What it says", coded
      ? "Mostly short codes, call signs and numbers, not normal sentences. Professional stations send coded traffic, so this is expected."
      : esc(r.translation_en || r.summary || "")]);
    if (r.confidence) lines.push(["How sure", SURE_TEXT[r.confidence] || esc(r.confidence)]);
  }
  return lines;
}

let state = { rows: [], filter: "", kind: "", category: "", open: null };

/** Pure: rows matching the filters (exported for tests). */
export function filterRows(rows, { filter = "", kind = "", category = "" } = {}) {
  const q = filter.trim().toUpperCase();
  return rows.filter(r => (!kind || r.kind === kind) && (!category || r.category === category)
    && (!q || [r.file, r.summary, r.category, ...(r.call_signs || [])].join(" ").toUpperCase().includes(q)));
}

export async function renderIntel(root, { server }) {
  if (!server) {
    // The page may ask before its own server check has finished: ask the server directly.
    try { server = (await fetch("/api/health", { credentials: "same-origin" })).ok; } catch { server = false; }
  }
  if (!server) {
    root.innerHTML = `<div class="block"><div class="note">Comms Intel needs the local server, because the
      recordings and their decodes stay on this machine. Start it with
      <code>python scripts/serve_local.py</code>, then decode a folder with
      <code>python scripts/comms_intel.py --in "FOLDER"</code>.</div></div>`;
    return;
  }
  root.innerHTML = `<div class="note">Loading…</div>`;
  let data;
  try {
    const res = await fetch("/api/intel", { credentials: "same-origin" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    root.innerHTML = `<div class="note">Could not read the decoded recordings: ${esc(e.message)}</div>`;
    return;
  }
  state.rows = data.recordings || [];
  draw(root, data);
}

function draw(root, data) {
  const rows = state.rows;
  const cats = [...new Set(rows.map(r => r.category).filter(Boolean))].sort();
  const kinds = [...new Set(rows.map(r => r.kind).filter(Boolean))];
  const totalH = rows.reduce((a, r) => a + (r.analysed_s || 0), 0) / 3600;
  const sigMin = rows.reduce((a, r) => a + (r.active_minutes || 0), 0);
  const tx = rows.reduce((a, r) => a + (r.transmissions || 0), 0);
  root.innerHTML = `
    <div class="block">
      <div class="block-label">Decoded recordings</div>
      <div class="note">Receiver audio decoded on this machine: Morse by the CW decoder, speech by Whisper, then
        read by a local language model (Ollama). <b>The model's reading is a suggestion.</b> Every quote it gives
        was checked against the decoded text, and anything it could not back up was removed. The decode itself can
        contain errors (<code>*</code> = unreadable letter). Everything Whisper or the Morse decoder analyses in Signal
        Pipeline, live or from a file, is added here by itself.</div>
      ${rows.length ? `<div class="note" style="margin-top:6px;">${rows.length} recordings · ${totalH.toFixed(1)} h analysed ·
        ${Math.round(sigMin)} min with signal · ${tx} transmissions</div>` : `<div class="note" style="margin-top:6px;">
        Nothing decoded yet in <code>${esc(data.folder)}</code>. ${esc(data.hint || "")}</div>`}
      <div class="filters" style="margin-top:10px;">
        <input id="intelSearch" placeholder="Search call sign, word, file" value="${esc(state.filter)}" style="width:240px;">
        <select id="intelKind"><option value="">All types</option>${kinds.map(k =>
          `<option value="${esc(k)}" ${state.kind === k ? "selected" : ""}>${esc(KIND_LABEL[k] || k)}</option>`).join("")}</select>
        <select id="intelCat"><option value="">All categories</option>${cats.map(c =>
          `<option ${state.category === c ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>
      </div>
      <div class="table-wrap" id="intelTable"></div>
    </div>
    <div class="block" id="intelDetail" hidden></div>`;
  const table = root.querySelector("#intelTable");
  const paint = () => {
    const shown = filterRows(rows, state);
    table.innerHTML = `<table><thead><tr><th>Recorded</th><th>Length · signal</th><th>Type</th>
      <th>Category</th><th>Call signs</th><th style="min-width:200px;">Summary</th></tr></thead><tbody>`
      + shown.map(r => `<tr data-id="${esc(r.id)}" style="cursor:pointer;" title="Open">
        <td style="white-space:nowrap;">${esc(r.recorded_at ? new Date(r.recorded_at).toLocaleString() : "—")}
          <div class="note" style="font-family:var(--mono);font-size:11.5px;">${esc(r.file)}</div></td>
        <td style="white-space:nowrap;">${hms(r.analysed_s)}${r.analysed_s < r.duration_s - 1 ? ` <span class="note">of ${hms(r.duration_s)}</span>` : ""}
          <div class="note">${esc(Math.round(r.active_minutes || 0))} min with signal</div></td>
        <td style="white-space:nowrap;">${esc(KIND_LABEL[r.kind] || r.kind)}${r.wpm_median ? `<div class="note">${Math.round(r.wpm_median)} WPM</div>` : ""}</td>
        <td>${r.error ? chip("LLM failed", "#9B2C2C") : esc(r.category || "—")}
          ${r.confidence ? `<div>${chip(r.confidence, CONF_COLOR[r.confidence] || "#5F6B72")}</div>` : ""}</td>
        <td>${esc((r.call_signs || []).slice(0, 4).join(", "))}</td>
        <td>${esc(r.error || r.summary || "")}</td></tr>`).join("")
      + `</tbody></table>${shown.length ? "" : `<div class="note" style="padding:8px;">No recording matches.</div>`}`;
    for (const tr of table.querySelectorAll("tr[data-id]")) {
      tr.addEventListener("click", () => openRecording(root, tr.dataset.id));
    }
  };
  root.querySelector("#intelSearch").addEventListener("input", e => { state.filter = e.target.value; paint(); });
  root.querySelector("#intelKind").addEventListener("change", e => { state.kind = e.target.value; paint(); });
  root.querySelector("#intelCat").addEventListener("change", e => { state.category = e.target.value; paint(); });
  paint();
  if (state.open) openRecording(root, state.open);
}

async function openRecording(root, id) {
  const box = root.querySelector("#intelDetail");
  state.open = id;
  box.hidden = false;
  box.innerHTML = `<div class="note">Loading ${esc(id)}…</div>`;
  let rec;
  try {
    const res = await fetch(`/api/intel/${encodeURIComponent(id)}`, { credentials: "same-origin" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    rec = await res.json();
  } catch (e) {
    box.innerHTML = `<div class="note">Could not open ${esc(id)}: ${esc(e.message)}</div>`;
    return;
  }
  const r = rec.reading || {};
  const f = r.facts || {};
  const segs = rec.segments || [];
  const LIMIT = 400;
  const pct = v => v != null ? Math.round(v * 100) + "%" : "—";
  const decoded = rec.kind === "voice" ? "transcribed speech" : "Morse text";
  const qCodes = (f.q_codes || []).map(([c, n]) =>
    `<div><b>${esc(c)}</b> = “${esc(Q_MEANING[c] || "a standard radio short code")}” <span class="note">(${n}×)</span></div>`).join("");
  box.innerHTML = `
    <div style="${BIG}">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
      <div>
        <div style="font-size:1.45rem;font-weight:700;">${esc(KIND_LABEL[rec.kind] || rec.kind)} recording · ${esc(plainDate(rec.recorded_at))}</div>
        <div class="note" style="font-size:0.95rem;">File ${esc(rec.file)}</div>
      </div>
      <button class="mini" id="intelClose">Close</button>
    </div>

    <div style="margin-top:12px;padding:14px 16px;border-radius:10px;background:var(--panel);border:1px solid var(--grid);">
      <div style="font-size:1.25rem;font-weight:700;margin-bottom:6px;">In short</div>
      ${inShort(rec).map(([k, v]) => `<div style="margin:4px 0;"><b>${esc(k)}:</b> ${v}</div>`).join("")}
    </div>

    <div style="${H}">When the station was sending</div>
    <canvas id="intelStrip" height="70" style="width:100%;display:block;background:var(--panel);border:1px solid var(--grid);"></canvas>
    <div class="note" style="font-size:1rem;">Each bar is a moment in the recording. <b>Taller, darker bar = more sending and a clearer signal.</b>
      Click a bar to see what was sent at that time.</div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:22px;margin-top:6px;">
      <div>
        <div style="${H}">What the AI thinks it means ${r.confidence ? chip(SURE_TEXT[r.confidence] || r.confidence, CONF_COLOR[r.confidence]) : ""}</div>
        ${r.error ? `<div style="color:#9B2C2C;">The AI could not read this one: ${esc(r.error)}</div>` : !rec.reading
          ? `<div class="note">No AI reading for this recording.</div>` : `
        ${r.translation_en ? `<div><b>In plain English:</b> ${esc(r.translation_en)}</div>` : `<div>${esc(r.summary)}</div>`}
        ${(r.call_signs || []).length ? `<div style="margin-top:8px;"><b>Station names (call signs):</b> ${r.call_signs.map(c => chip(c, "#4A552F")).join(" ")}</div>` : ""}
        ${(r.places || []).length ? `<div style="margin-top:6px;"><b>Places mentioned:</b> ${r.places.map(c => chip(c, "#2B6CB0")).join(" ")}</div>` : ""}
        ${(r.key_points || []).length ? `<ul style="margin:8px 0 0 20px;padding:0;">${r.key_points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
        ${(r.evidence || []).length ? `<div style="margin-top:12px;font-weight:700;">Proof: the exact ${decoded} the AI based this on</div>
          ${r.evidence.map(e => `<div style="margin-top:6px;"><code style="font-size:1.05rem;">${esc(e.quote)}</code><div class="note" style="font-size:1rem;">${esc(e.why)}</div></div>`).join("")}` : ""}
        ${r.caveats ? `<div style="margin-top:10px;"><b>Be careful:</b> ${esc(r.caveats)}</div>` : ""}
        <div class="note" style="margin-top:10px;font-size:0.95rem;">Type: ${esc(r.category || "—")}${r.language ? ` · language: ${esc(r.language)}` : ""} · AI model ${esc(r.model || "—")}
          ${r.removed_ungrounded ? ` · ${r.removed_ungrounded} detail${r.removed_ungrounded > 1 ? "s" : ""} the AI made up (not in the ${decoded}) ${r.removed_ungrounded > 1 ? "were" : "was"} removed` : ""}</div>`}
      </div>
      <div>
        <div style="${H}">Facts measured from the signal <span class="note" style="font-weight:400;">(no AI)</span></div>
        <table style="font-size:1.05rem;"><tbody>
          <tr><td>Sending speed</td><td>${f.wpm_range ? `${Math.round(f.wpm_range[0])}–${Math.round(f.wpm_range[1])} words per minute` : rec.wpm_median ? `${Math.round(rec.wpm_median)} words per minute` : "—"}</td></tr>
          <tr><td>Station names heard</td><td>${esc((f.call_sign_like || []).slice(0, 4).map(([c, n]) => `${c} (${n}×)`).join(", ") || "—")}</td></tr>
          <tr><td>Radio short codes</td><td>${qCodes || "none"}</td></tr>
          <tr><td>“Calling anyone” (CQ)</td><td>${f.cq_calls ? `${f.cq_calls}×` : "none"}</td></tr>
          <tr><td>Secret-code groups <span class="note">(blocks of 5 letters/digits)</span></td><td>${pct(f.five_char_groups_share)} of words</td></tr>
          <tr><td>Letters that could not be read</td><td>${pct(f.unreadable_share)}</td></tr>
        </tbody></table>
        <div class="note" style="margin-top:8px;font-size:0.95rem;">Listened to ${plainHours(rec.analysed_s)} · ${segs.length} messages
          ${rec.dropped_noise_segments ? ` · ${rec.dropped_noise_segments} bursts of noise ignored` : ""} · recording ${rec.sample_rate} Hz, ${rec.bits}-bit</div>
      </div>
    </div>
    ${rec.transcript?.text ? `<div style="${H}">What was said (speech to text, ${esc(rec.transcript.language || "?")})</div>
      <div style="white-space:pre-wrap;">${esc(rec.transcript.text)}</div>` : rec.transcript?.error ? `<div class="note">Speech to text failed: ${esc(rec.transcript.error)}</div>` : rec.note ? `<div style="${H}">What was said</div><div class="note">${esc(rec.note)}</div>` : ""}
    <div style="${H}">Every message, word for word</div>
    <div class="filters" style="margin-top:4px;"><input id="intelSegSearch" placeholder="Search the messages" style="width:260px;font-size:1rem;">
      <span class="note" id="intelSegCount"></span></div>
    <div class="table-wrap" id="intelSegs" style="max-height:460px;"></div>
    ${rec.recorder_log ? `<details style="margin-top:10px;"><summary>Technical: recorder log (${esc(rec.file.replace(/\.wav$/i, ".log"))})</summary>
      <pre style="white-space:pre-wrap;max-height:240px;overflow:auto;">${esc(rec.recorder_log)}</pre></details>` : ""}
    </div>`;
  box.querySelector("#intelClose").addEventListener("click", () => { box.hidden = true; state.open = null; });

  const segBox = box.querySelector("#intelSegs");
  const count = box.querySelector("#intelSegCount");
  const paintSegs = (q = "", focusS = null) => {
    const Q = q.trim().toUpperCase();
    let list = Q ? segs.filter(s => s.text.toUpperCase().includes(Q)) : segs;
    if (focusS != null) {
      const i = Math.max(0, list.findIndex(s => s.end_s >= focusS));
      list = list.slice(i);
    }
    count.textContent = `${Q ? list.length + " found · " : ""}showing ${Math.min(LIMIT, list.length)} of ${list.length}`;
    const hl = t => Q ? esc(t).split(esc(Q)).join(`<mark>${esc(Q)}</mark>`) : esc(t);
    const strength = db => db == null ? "—" : db >= 25 ? "strong" : db >= 15 ? "clear" : db >= 8 ? "weak" : "very weak";
    segBox.innerHTML = `<table style="font-size:1.02rem;"><thead><tr><th style="white-space:nowrap;">Time in recording</th>
      <th>Signal</th><th>Speed</th><th style="width:60%;">What was sent</th></tr></thead><tbody>`
      + list.slice(0, LIMIT).map(s => `<tr style="vertical-align:top;"><td style="white-space:nowrap;"><code>${clock(s.start_s)}</code></td>
        <td style="white-space:nowrap;" title="${s.snr_db != null ? Math.round(s.snr_db) + " dB above the noise" : ""}">${strength(s.snr_db)}</td>
        <td style="white-space:nowrap;">${s.wpm ? `${Math.round(s.wpm)} wpm` : "—"}</td>
        <td style="font-family:var(--mono);word-break:break-word;">${hl(s.text)}</td></tr>`).join("") + `</tbody></table>`;
  };
  box.querySelector("#intelSegSearch").addEventListener("input", e => paintSegs(e.target.value));
  paintSegs();
  drawStrip(box.querySelector("#intelStrip"), rec, t => paintSegs(box.querySelector("#intelSegSearch").value, t));
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawStrip(canvas, rec, onPick) {
  const ch = rec.chunks || {};
  const down = ch.key_down || [], snr = ch.snr_db || [];
  const W = canvas.clientWidth || 800, H = canvas.height;
  canvas.width = W;
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, W, H);
  const n = down.length;
  if (!n) return;
  const bw = W / n;
  for (let i = 0; i < n; i++) {
    const s = snr[i] ?? 0;
    const k = Math.min(1, Math.max(0, (s - 8) / 22));
    g.fillStyle = s < 8 ? "#C9CEC4" : `rgb(${Math.round(220 - 146 * k)},${Math.round(226 - 141 * k)},${Math.round(205 - 158 * k)})`;
    const h = s < 8 ? 2 : Math.max(3, Math.min(1, down[i] * 2) * (H - 4));
    g.fillRect(i * bw, H - h, Math.max(1, bw - (bw > 3 ? 1 : 0)), h);
  }
  canvas.onclick = e => {
    const i = Math.floor(e.offsetX / bw);
    onPick(i * (rec.chunk_s || 15));
  };
  canvas.title = "Click to jump to this time";
}
