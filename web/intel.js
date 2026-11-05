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

let state = { rows: [], filter: "", kind: "", category: "", open: null };

/** Pure: rows matching the filters (exported for tests). */
export function filterRows(rows, { filter = "", kind = "", category = "" } = {}) {
  const q = filter.trim().toUpperCase();
  return rows.filter(r => (!kind || r.kind === kind) && (!category || r.category === category)
    && (!q || [r.file, r.summary, r.category, ...(r.call_signs || [])].join(" ").toUpperCase().includes(q)));
}

export async function renderIntel(root, { server }) {
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
        contain errors (<code>*</code> = unreadable letter).</div>
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
  box.innerHTML = `
    <div class="block-label">${esc(rec.file)} <button class="mini" id="intelClose" style="float:right;">Close</button></div>
    <div class="note">${esc(rec.recorded_at ? new Date(rec.recorded_at).toLocaleString() : "")} · ${hms(rec.analysed_s)} analysed
      · ${rec.sample_rate} Hz ${rec.bits}-bit · ${esc(KIND_LABEL[rec.kind] || rec.kind)}
      · ${segs.length} transmissions${rec.dropped_noise_segments ? ` (${rec.dropped_noise_segments} noise-like left out)` : ""}</div>
    <div style="margin:10px 0 4px;font-weight:600;">Activity over the recording</div>
    <canvas id="intelStrip" height="54" style="width:100%;display:block;background:var(--panel);border:1px solid var(--grid);"></canvas>
    <div class="note">Bar height = share of time the key was down; colour = how far the beep stood above the noise
      (grey &lt; 8 dB, then pale → dark olive up to 30 dB). Click a bar to jump to that time in the list.</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:12px;">
      <div>
        <div style="font-weight:700;margin-bottom:4px;">Model's reading ${r.confidence ? chip(r.confidence + " confidence", CONF_COLOR[r.confidence]) : ""}</div>
        ${r.error ? `<div class="note" style="color:#9B2C2C;">${esc(r.error)}</div>` : !rec.reading
          ? `<div class="note">No LLM reading (run without --no-llm).</div>` : `
        <div>${esc(r.summary)}</div>
        <div class="note" style="margin-top:4px;">Category: <b>${esc(r.category)}</b>${r.language ? ` · Language: ${esc(r.language)}` : ""}
          · Model: ${esc(r.model || "—")}${r.removed_ungrounded ? ` · <span title="Call signs, places or quotes the model gave that are not in the decoded text">${r.removed_ungrounded} unsupported item(s) removed</span>` : ""}</div>
        ${r.translation_en ? `<div style="margin-top:6px;"><b>English:</b> ${esc(r.translation_en)}</div>` : ""}
        ${(r.call_signs || []).length ? `<div style="margin-top:6px;">Call signs: ${r.call_signs.map(c => chip(c, "#4A552F")).join(" ")}</div>` : ""}
        ${(r.places || []).length ? `<div style="margin-top:4px;">Places: ${r.places.map(c => chip(c, "#2B6CB0")).join(" ")}</div>` : ""}
        ${(r.key_points || []).length ? `<ul style="margin:6px 0 0 18px;padding:0;">${r.key_points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
        ${(r.evidence || []).length ? `<div style="margin-top:8px;font-weight:600;">Evidence (checked against the decode)</div>
          ${r.evidence.map(e => `<div style="margin-top:4px;"><code>${esc(e.quote)}</code><div class="note">${esc(e.why)}</div></div>`).join("")}` : ""}
        ${r.caveats ? `<div class="note" style="margin-top:6px;"><b>Caveats:</b> ${esc(r.caveats)}</div>` : ""}`}
      </div>
      <div>
        <div style="font-weight:700;margin-bottom:4px;">Measured (no model involved)</div>
        <table><tbody>
          <tr><td>Speed</td><td>${f.wpm_range ? `${Math.round(f.wpm_range[0])}–${Math.round(f.wpm_range[1])} WPM` : rec.wpm_median ? `${Math.round(rec.wpm_median)} WPM` : "—"}</td></tr>
          <tr><td>Call-sign-like tokens</td><td>${esc((f.call_sign_like || []).map(([c, n]) => `${c} ×${n}`).join(", ") || "—")}</td></tr>
          <tr><td>Q-codes</td><td>${esc((f.q_codes || []).map(([c, n]) => `${c} ×${n}`).join(", ") || "—")}</td></tr>
          <tr><td>CQ calls</td><td>${esc(f.cq_calls ?? "—")}</td></tr>
          <tr><td>Five-character groups</td><td>${f.five_char_groups_share != null ? Math.round(f.five_char_groups_share * 100) + "% of words" : "—"}</td></tr>
          <tr><td>Unreadable letters</td><td>${f.unreadable_share != null ? Math.round(f.unreadable_share * 100) + "%" : "—"}</td></tr>
        </tbody></table>
      </div>
    </div>
    ${rec.transcript?.text ? `<div style="margin-top:12px;font-weight:700;">Whisper transcript (${esc(rec.transcript.language || "?")}, loudest ${(rec.transcript.chunks_s || []).length} × 30 s)</div>
      <div style="white-space:pre-wrap;">${esc(rec.transcript.text)}</div>` : rec.transcript?.error ? `<div class="note">Whisper failed: ${esc(rec.transcript.error)}</div>` : ""}
    <div style="margin-top:12px;font-weight:700;">Decoded transmissions</div>
    <div class="filters" style="margin-top:4px;"><input id="intelSegSearch" placeholder="Find in decoded text" style="width:240px;">
      <span class="note" id="intelSegCount"></span></div>
    <div class="table-wrap" id="intelSegs" style="max-height:420px;"></div>
    ${rec.recorder_log ? `<details style="margin-top:10px;"><summary>Recorder log (${esc(rec.file.replace(/\.wav$/i, ".log"))})</summary>
      <pre style="white-space:pre-wrap;max-height:240px;overflow:auto;">${esc(rec.recorder_log)}</pre></details>` : ""}`;
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
    count.textContent = `${Q ? list.length + " match · " : ""}showing ${Math.min(LIMIT, list.length)} of ${list.length}`;
    const hl = t => Q ? esc(t).split(esc(Q)).join(`<mark>${esc(Q)}</mark>`) : esc(t);
    segBox.innerHTML = `<table><thead><tr><th>Time</th><th>WPM</th><th>Tone</th><th>SNR</th><th>Text</th></tr></thead><tbody>`
      + list.slice(0, LIMIT).map(s => `<tr><td><code>${clock(s.start_s)}</code></td><td>${Math.round(s.wpm)}</td>
        <td>${s.tone_hz ? Math.round(s.tone_hz) + " Hz" : "—"}</td><td>${s.snr_db != null ? Math.round(s.snr_db) + " dB" : "—"}</td>
        <td style="font-family:var(--mono);">${hl(s.text)}</td></tr>`).join("") + `</tbody></table>`;
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
