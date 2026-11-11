// Decoded signals: the Decoded review list in History, and the expanded page one
// row opens. Receiver AUDIO recordings decoded on this machine by
// scripts/comms_intel.py (Morse decoder, Whisper for voice, a local Ollama
// model for the reading), served read-only by scripts/serve_local.py at
// /api/intel. The LLM reading is shown NEXT TO the decoded text it came from,
// and every quote it gives is one the server already checked occurs in that
// text (src/intel/reason.py ground()).

const CONF_COLOR = { high: "#2E7D32", medium: "#B7791F", low: "#9B2C2C" };
const KIND_LABEL = { morse: "Morse (CW)", voice: "Voice", silent: "No signal" };

const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

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
const BIG = "font-size:16px;line-height:1.55;";
const H = "font-size:18px;font-weight:700;margin:26px 0 8px;";

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

let state = { open: null, onBack: null };

/** Pure: rows matching the filters (exported for tests). */
export function filterRows(rows, { filter = "", kind = "", category = "" } = {}) {
  const q = filter.trim().toUpperCase();
  return rows.filter(r => (!kind || r.kind === kind) && (!category || r.category === category)
    && (!q || [r.file, r.summary, r.category, ...(r.call_signs || [])].join(" ").toUpperCase().includes(q)));
}

/** Every decoded recording for the Decoded review list, newest first. */
export async function listDecoded() {
  const res = await fetch("/api/intel", { credentials: "same-origin" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const rows = (await res.json()).recordings || [];
  return rows.filter(r => r.kind !== "silent").reverse();
}

const DECODED_SHOWN = 50;

/** Pure: the Decoded review table (exported for tests). */
export function decodedTableHtml(rows) {
  if (!rows.length) return `<div class="note">Nothing decoded yet.</div>`;
  return (rows.length > DECODED_SHOWN ? `<div class="note">Newest ${DECODED_SHOWN} of ${rows.length} shown.</div>` : "")
    + `<table><thead><tr><th>Recorded</th><th>Type</th><th>Call signs</th><th style="min-width:220px;">Summary</th><th></th></tr></thead><tbody>`
    + rows.slice(0, DECODED_SHOWN).map(r => `<tr data-decoded="${esc(r.id)}" style="cursor:pointer;" title="Open this signal">
      <td style="white-space:nowrap;">${esc(r.recorded_at ? new Date(r.recorded_at).toLocaleString() : "—")}
        <div class="note" style="margin:4px 0 0;overflow-wrap:anywhere;white-space:normal;">${esc(r.file)}</div></td>
      <td style="white-space:nowrap;">${esc(KIND_LABEL[r.kind] || r.kind)}</td>
      <td>${esc((r.call_signs || []).slice(0, 4).join(", "))}</td>
      <td>${esc(r.summary || r.error || "Decoded. Open to read the text.")}</td>
      <td class="row-actions"><button class="mini">Open</button></td></tr>`).join("")
    + `</tbody></table>`;
}

/** The expanded page for one decoded signal. */
export async function renderIntelDetail(root, { id, onBack }) {
  state.onBack = onBack;
  root.innerHTML = `<div class="filters"><button class="secondary" id="intelBack">← Back to History</button></div>
    <div class="block" id="intelDetail"></div>`;
  root.querySelector("#intelBack").addEventListener("click", onBack);
  await openRecording(root, id);
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
        <div style="font-size:20px;font-weight:700;">${esc(KIND_LABEL[rec.kind] || rec.kind)} recording · ${esc(plainDate(rec.recorded_at))}</div>
        <div class="note">File ${esc(rec.file)}</div>
      </div>
      <button class="mini" id="intelClose">Close</button>
    </div>

    <div style="margin-top:12px;padding:14px 16px;border-radius:10px;background:var(--panel);border:1px solid var(--grid);">
      <div style="font-size:18px;font-weight:700;margin-bottom:8px;">In short</div>
      ${inShort(rec).map(([k, v]) => `<div style="margin:4px 0;"><b>${esc(k)}:</b> ${v}</div>`).join("")}
    </div>

    <div style="${H}">When the station was sending</div>
    <div style="margin-bottom:10px;">${activitySummary(rec)}</div>
    <canvas id="intelStrip" height="104" style="width:100%;display:block;background:var(--panel);border:1px solid var(--grid);border-radius:6px;"></canvas>
    <div class="note" style="margin-top:8px;">Time runs left to right. Dark bars: ${rec.kind === "voice" ? "someone speaking" : "the Morse key pressed"}; a darker
      bar is a clearer signal. Light bands mark each message. Click anywhere to jump to that moment in the messages below.</div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:22px;margin-top:6px;">
      <div>
        <div style="${H}">What the AI thinks it means ${r.confidence ? chip(SURE_TEXT[r.confidence] || r.confidence, CONF_COLOR[r.confidence]) : ""}</div>
        ${r.error ? `<div style="color:#9B2C2C;">The AI could not read this one: ${esc(r.error)}</div>` : !rec.reading
          ? `<div class="note">No AI reading for this recording.</div>` : `
        ${r.translation_en ? `<div><b>In plain English:</b> ${esc(r.translation_en)}</div>` : `<div>${esc(r.summary)}</div>`}
        ${(r.call_signs || []).length ? `<div style="margin-top:8px;"><b>Station names (call signs):</b> ${r.call_signs.map(c => chip(c, "#172152")).join(" ")}</div>` : ""}
        ${(r.places || []).length ? `<div style="margin-top:6px;"><b>Places mentioned:</b> ${r.places.map(c => chip(c, "#2B6CB0")).join(" ")}</div>` : ""}
        ${(r.key_points || []).length ? `<ul style="margin:8px 0 0 20px;padding:0;">${r.key_points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
        ${(r.evidence || []).length ? `<div style="margin-top:12px;font-weight:700;">Proof: the exact ${decoded} the AI based this on</div>
          ${r.evidence.map(e => `<div style="margin-top:6px;"><code>${esc(e.quote)}</code><div class="note">${esc(e.why)}</div></div>`).join("")}` : ""}
        ${r.caveats ? `<div style="margin-top:10px;"><b>Be careful:</b> ${esc(r.caveats)}</div>` : ""}
        <div class="note" style="margin-top:10px;">Type: ${esc(r.category || "—")}${r.language ? ` · language: ${esc(r.language)}` : ""} · AI model ${esc(r.model || "—")}
          ${r.removed_ungrounded ? ` · ${r.removed_ungrounded} detail${r.removed_ungrounded > 1 ? "s" : ""} the AI made up (not in the ${decoded}) ${r.removed_ungrounded > 1 ? "were" : "was"} removed` : ""}</div>`}
      </div>
      <div>
        <div style="${H}">Facts measured from the signal <span class="note" style="font-weight:400;">(no AI)</span></div>
        <table><tbody>
          <tr><td>Sending speed</td><td>${f.wpm_range ? `${Math.round(f.wpm_range[0])}–${Math.round(f.wpm_range[1])} words per minute` : rec.wpm_median ? `${Math.round(rec.wpm_median)} words per minute` : "—"}</td></tr>
          <tr><td>Station names heard</td><td>${esc((f.call_sign_like || []).slice(0, 4).map(([c, n]) => `${c} (${n}×)`).join(", ") || "—")}</td></tr>
          <tr><td>Radio short codes</td><td>${qCodes || "none"}</td></tr>
          <tr><td>“Calling anyone” (CQ)</td><td>${f.cq_calls ? `${f.cq_calls}×` : "none"}</td></tr>
          <tr><td>Secret-code groups <span class="note">(blocks of 5 letters/digits)</span></td><td>${pct(f.five_char_groups_share)} of words</td></tr>
          <tr><td>Letters that could not be read</td><td>${pct(f.unreadable_share)}</td></tr>
        </tbody></table>
        <div class="note" style="margin-top:8px;">Listened to ${plainHours(rec.analysed_s)} · ${segs.length} messages
          ${rec.dropped_noise_segments ? ` · ${rec.dropped_noise_segments} bursts of noise ignored` : ""} · recording ${rec.sample_rate} Hz, ${rec.bits}-bit</div>
      </div>
    </div>
    ${rec.transcript?.text ? `<div style="${H}">What was said (speech to text, ${esc(rec.transcript.language || "?")})</div>
      <div style="white-space:pre-wrap;">${esc(rec.transcript.text)}</div>` : rec.transcript?.error ? `<div class="note">Speech to text failed: ${esc(rec.transcript.error)}</div>` : rec.note ? `<div style="${H}">What was said</div><div class="note">${esc(rec.note)}</div>` : ""}
    <div style="${H}">Every message, word for word</div>
    <div class="filters" style="margin-top:4px;"><input id="intelSegSearch" placeholder="Search the messages" style="width:260px;">
      <span class="note" id="intelSegCount"></span></div>
    <div class="table-wrap" id="intelSegs" style="max-height:460px;"></div>
    ${rec.recorder_log ? `<details style="margin-top:10px;"><summary>Technical: recorder log (${esc(rec.file.replace(/\.wav$/i, ".log"))})</summary>
      <pre style="white-space:pre-wrap;max-height:240px;overflow:auto;">${esc(rec.recorder_log)}</pre></details>` : ""}
    </div>`;
  box.querySelector("#intelClose").addEventListener("click", () => { state.open = null; state.onBack?.(); });

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
    segBox.innerHTML = `<table><thead><tr><th style="white-space:nowrap;">Time in recording</th>
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

/** One plain sentence above the timeline (exported for tests). */
export function activitySummary(rec) {
  const ch = rec.chunks || {}, down = ch.key_down || [], step = rec.chunk_s || 15;
  const total = rec.analysed_s || down.length * step;
  const on = Math.min(total, down.reduce((a, d) => a + d * step, 0));
  const n = (rec.segments || []).length;
  const parts = [`<b>${rec.kind === "voice" ? "Speech" : "Key pressed"}</b> for ${plainHours(on)} of ${plainHours(total)}`
    + (total ? ` (${Math.round(100 * on / total)}%)` : "")];
  parts.push(`${n} message${n === 1 ? "" : "s"}`);
  const heard = (ch.snr_db || []).filter((s, i) => s >= 8 && down[i] > 0).sort((p, q) => p - q);
  if (heard.length) {
    const db = heard[Math.floor(heard.length / 2)];
    parts.push(`${db >= 25 ? "strong" : db >= 15 ? "clear" : "weak"} signal, ${Math.round(db)} dB above the noise`);
  } else {
    parts.push("no clear signal");
  }
  return parts.join(" · ");
}

function tickEvery(total, most = 8) {
  for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]) if (total / s <= most) return s;
  return 14400;
}

function timeLabel(sec, total) {
  sec = Math.round(sec);
  const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = String(sec % 60).padStart(2, "0");
  return total >= 3600 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

function drawStrip(canvas, rec, onPick) {
  const ch = rec.chunks || {}, down = ch.key_down || [], snr = ch.snr_db || [], step = rec.chunk_s || 15;
  const total = rec.analysed_s || down.length * step;
  const W = canvas.clientWidth || 800, H = canvas.height, base = H - 26, top = 8;
  canvas.width = W;
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, W, H);
  if (!total) return;
  const x = t => Math.min(W, t / total * W);
  g.fillStyle = "#E8ECF6";                                  // each message, behind the bars
  for (const s of rec.segments || []) g.fillRect(x(s.start_s), top, Math.max(2, x(s.end_s) - x(s.start_s)), base - top);
  down.forEach((d, i) => {
    const s = snr[i] ?? 0;
    if (s < 8 || d <= 0) return;
    const k = Math.min(1, (s - 8) / 22);
    g.fillStyle = `rgb(${Math.round(160 - 86 * k)},${Math.round(172 - 87 * k)},${Math.round(130 - 83 * k)})`;
    const h = Math.max(3, Math.min(1, d * 2) * (base - top));
    const x0 = x(i * step), w = x((i + 1) * step) - x0;
    g.fillRect(x0, base - h, Math.max(1, w > 4 ? w - 1 : w), h);
  });
  g.strokeStyle = "#E3E6EC";
  g.beginPath(); g.moveTo(0, base + 0.5); g.lineTo(W, base + 0.5); g.stroke();
  g.fillStyle = "#5E6673";
  g.font = '13px "Montserrat", "Segoe UI", Arial, sans-serif';
  g.textBaseline = "top";
  const every = tickEvery(total);
  for (let t = 0; t <= total + 1e-6; t += every) {
    const px = x(t);
    g.fillRect(Math.min(W - 1, Math.round(px)), base, 1, 5);
    g.textAlign = px < 20 ? "left" : px > W - 24 ? "right" : "center";
    g.fillText(timeLabel(t, total), Math.max(0, Math.min(W, px)), base + 8);
  }
  const at = e => Math.max(0, Math.min(total, e.offsetX / (canvas.clientWidth || W) * total));
  canvas.style.cursor = "pointer";
  canvas.onclick = e => onPick(at(e));
  canvas.onmousemove = e => {
    const t = at(e), d = down[Math.min(down.length - 1, Math.floor(t / step))] ?? 0;
    canvas.title = `${timeLabel(t, total)} · ${d > 0 ? (rec.kind === "voice" ? "speech" : "key pressed") : "nothing sent"}`;
  };
}
