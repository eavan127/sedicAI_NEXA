// SDR Receiver page, Replay a file (and History → Captures, Decoded recordings): the whole wideband chain on a recording, run by the
// local server (scripts/serve_local.py /api/pipeline -> scripts/run_pipeline.py):
//   IQ file -> YOLO boxes -> 60-class classifier -> tracker (vote) -> resolve
//   -> for voice / Morse tracks: zoom, analog kind, audio, Morse / Whisper, AI reading.
// Everything runs on this machine. The model's reading is shown next to the
// text it came from (src/intel/reason.py keeps only quotes that occur in it).

const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const HDR = { "X-NEXA-Client": "1" };
const KIND_TEXT = {
  NBFM: "Narrow FM (walkie-talkie)", WBFM: "Broadcast FM", AM: "AM (with carrier)", AM_SC: "AM (no carrier)",
  USB: "Upper sideband (SSB)", LSB: "Lower sideband (SSB)", CW: "Morse (CW)", CARRIER: "Plain carrier (no modulation)",
  UNKNOWN: "Too weak to tell", ERROR: "Error",
};
const BAND_COL = { high: "#2E7D32", medium: "#B7791F", uncertain: "#9B2C2C" };
const CONF = c => c >= 0.6 ? "#2E7D32" : c >= 0.3 ? "#B7791F" : "#9B2C2C";


let state = { open: null, timer: null };

function chip(text, color) {
  return `<span class="chip" style="background:${color}1f;color:${color};">${esc(text)}</span>`;
}

/** Seconds to scan: the recording's length, up to SCAN_DEFAULT_MAX_S. The scan costs about
 * 20 s of computing per second on an 8 GB laptop (28 s took 9 min and crashed next to
 * Whisper); the decode, name and AI reading cover the whole file anyway. A longer scan
 * can still be typed in (up to 120 s). */
const SCAN_DEFAULT_MAX_S = 10;

/** What a WAV file holds, from its header: {channels, rate, seconds}, or null when it is
 * not a WAV. An IQ recording has 2 channels (I and Q); an audio recording has 1. */
export async function wavInfo(file) {
  if (!file || !/\.wav$/i.test(file.name)) return null;
  const v = new DataView(await file.slice(0, 4096).arrayBuffer());
  const tag = i => String.fromCharCode(v.getUint8(i), v.getUint8(i + 1), v.getUint8(i + 2), v.getUint8(i + 3));
  if (v.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") return null;
  let pos = 12, rate = 0, align = 0, channels = 0;
  while (pos + 8 <= v.byteLength) {
    const id = tag(pos), size = v.getUint32(pos + 4, true);
    if (id === "fmt ") { channels = v.getUint16(pos + 10, true); rate = v.getUint32(pos + 12, true); align = v.getUint16(pos + 20, true); }
    if (id === "data") {
      return { channels, rate, seconds: rate && align ? Math.min(size, file.size - pos - 8) / (rate * align) : null };
    }
    pos += 8 + size + (size & 1);
  }
  return channels ? { channels, rate, seconds: null } : null;
}

/** Replay a file (SDR Receiver): one IQ WAV through the whole chain on the local server
 * (find each transmitter, name it, demodulate and decode voice / Morse, AI reading), drawn
 * into `root`: progress while it runs, then the waterfall with every transmitter boxed and
 * its frequency, the decoded signals (Open analysis) and the transmitter table.
 * Refuses an audio WAV (1 channel): only IQ recordings carry the radio band. */
export async function replayIqFile(file, root) {
  root.hidden = false;
  root.innerHTML = `<div class="block" id="pipeRunBox"><div class="note">Checking ${esc(file.name)}…</div></div>`;
  const box = root.querySelector("#pipeRunBox");
  const info = await wavInfo(file).catch(() => null);
  if (!info) {
    box.innerHTML = `<div class="run-error"><b>Not a WAV recording</b><span>Replay a file takes IQ recordings saved as WAV
      (in SDRconnect: Recording type = IQ). ${esc(file.name)} is not one.</span></div>`;
    return false;
  }
  if (info.channels !== 2) {
    box.innerHTML = `<div class="run-error"><b>This is an audio file, not IQ</b><span>${esc(file.name)} has ${info.channels} channel${info.channels === 1 ? "" : "s"}.
      An IQ recording has 2 (I and Q) and holds the radio band, so NEXA can draw its waterfall and find each transmitter.
      In SDRconnect, record with Recording type = IQ.</span></div>`;
    return false;
  }
  const scan = info.seconds ? Math.min(SCAN_DEFAULT_MAX_S, Math.max(0.25, Math.ceil(info.seconds * 4) / 4)) : 2;
  const p = new URLSearchParams({ source: "file", name: file.name, max_s: String(scan), detector: "hybrid",
                                  listen: "1", whisper: "1", llm: "1", language: "auto" });
  try {
    const busy = ((await (await fetch("/api/pipeline", { credentials: "same-origin" })).json()).runs || [])
      .find(r => r.status === "running" && !r.live);
    if (busy) {
      box.innerHTML = `<div class="note">Another recording (${esc(busy.file || busy.source)}) is still being analysed. Wait for it to finish.</div>`;
      openRun(root, busy.id);
      return false;
    }
    box.innerHTML = `<div class="note">Uploading ${esc(file.name)} (${(file.size / 1e6).toFixed(1)} MB, IQ at
      ${(info.rate / 1e3).toFixed(0)} kS/s${info.seconds ? `, ${info.seconds.toFixed(1)} s` : ""})…</div>`;
    const res = await fetch(`/api/pipeline?${p}`, { method: "POST", credentials: "same-origin", headers: HDR, body: file });
    const j = await res.json();
    if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
    openRun(root, j.id);
    return true;
  } catch (e) {
    box.innerHTML = `<div class="run-error"><b>Could not start the analysis</b><span>${esc(e.message)}${/fetch/i.test(e.message)
      ? " (is python scripts/serve_local.py running?)" : ""}</span></div>`;
    return false;
  }
}

/** History → Captures, "Decoded recordings": every replayed file and live SDRconnect /
 * KiwiSDR run, newest first; a row opens its waterfall, decoded signals and transmitters. */
export async function renderRunHistory(root) {
  root.innerHTML = `<div class="table-wrap" id="pipeRuns"><div class="note">Loading…</div></div>
    <div class="block" id="pipeRunBox" hidden style="margin-top:12px;"></div>`;
  try {
    await listRuns(root);
    if (state.open && root.querySelector(`tr[data-id="${CSS.escape(state.open)}"]`)) openRun(root, state.open);
  } catch (e) {
    root.querySelector("#pipeRuns").innerHTML = `<div class="note">Recordings are kept by the local server
      (python scripts/serve_local.py): ${esc(e.message)}</div>`;
  }
}

// Run ids start with the local time: 20261108-222233-ff1d.
function runTime(id) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(id);
  if (!m) return "—";
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6])
    .toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function runTook(r) {
  if (r.status === "running") return "running…";
  if (r.took_s == null) return "—";
  if (r.took_s < 60) return `${r.took_s.toFixed(1)} s`;
  const total = Math.round(r.took_s);
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}

async function listRuns(root, runs) {
  const box = root.querySelector("#pipeRuns");
  if (!box) return;
  if (!runs) runs = (await (await fetch("/api/pipeline", { credentials: "same-origin" })).json()).runs;
  box.innerHTML = runs.length ? `<table><thead><tr><th>Run</th><th>Time</th><th>Source</th><th>Seconds</th><th>Analysis took</th><th>Detector</th><th>Status</th></tr></thead><tbody>`
    + runs.map(r => `<tr data-id="${esc(r.id)}" style="cursor:pointer;" title="Open its waterfall and decoded signals"><td><code>${esc(r.id)}</code></td>
      <td>${esc(runTime(r.id))}</td><td>${esc(r.file || r.source)}</td><td>${esc(r.max_s)}</td><td>${esc(runTook(r))}</td><td>${esc(r.detector)}</td>
      <td>${chip(r.status, r.status === "done" ? "#2E7D32" : r.status === "running" ? "#2B6CB0" : r.status === "stopped" ? "#5E6673" : "#9B2C2C")}</td></tr>`).join("")
    + `</tbody></table>` : `<div class="note">No runs yet.</div>`;
  for (const tr of box.querySelectorAll("tr[data-id]")) tr.addEventListener("click", () => {
    openRun(root, tr.dataset.id);
    root.querySelector("#pipeRunBox")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

async function openRun(root, id) {
  state.open = id;
  clearTimeout(state.timer);
  const box = root.querySelector("#pipeRunBox");
  box.hidden = false;
  let r;
  try {
    r = await (await fetch(`/api/pipeline/${encodeURIComponent(id)}`, { credentials: "same-origin" })).json();
  } catch (e) {
    box.innerHTML = `<div class="note">Could not read run ${esc(id)}: ${esc(e.message)}</div>`;
    return;
  }
  if (r.status === "running") {
    const last = (r.log || "").trim().split("\n").slice(-4).join("\n");
    box.innerHTML = `<div class="block-label">Analysing ${esc(r.file || r.source)} ${chip("running", "#2B6CB0")}</div>
      <div class="note">Finding each transmitter, naming it, then listening to voice and Morse (about 30-90 s). The waterfall appears here when it is done.</div>
      <pre style="white-space:pre-wrap;max-height:180px;overflow:auto;">${esc(last || "starting…")}</pre>
      ${r.live ? "" : `<button class="mini" id="pipeStop">Stop this run</button> <span class="note" id="pipeStopMsg"></span>`}`;
    box.querySelector("#pipeStop")?.addEventListener("click", async ev => {
      ev.currentTarget.disabled = true;
      const m = box.querySelector("#pipeStopMsg");
      m.textContent = "Stopping…";
      try {
        const res = await fetch(`/api/pipeline/${encodeURIComponent(id)}/stop`, { method: "POST", credentials: "same-origin", headers: HDR });
        const j = await res.json();
        if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
        clearTimeout(state.timer);
        openRun(root, id).then(() => listRuns(root));
      } catch (e) { m.textContent = `Could not stop: ${e.message}`; }
    });
    state.timer = setTimeout(() => { if (state.open === id) openRun(root, id).then(() => listRuns(root)); }, 3000);
    return;
  }
  if (r.status !== "done") {
    // The log is a Python traceback with terminal colour codes: strip them, lead with the
    // error itself in a sentence, and keep the full log one click away.
    const log = (r.log || "").replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").trim();
    const lines = log.split("\n").map(s => s.trim()).filter(Boolean);
    const err = [...lines].reverse().find(s => /^[A-Za-z_.]*(Error|Exception|Interrupt)\b/.test(s)) || lines[lines.length - 1] || "No log was written.";
    box.innerHTML = `<div class="block-label">Run ${esc(id)} ${chip("failed", "#9B2C2C")}</div>
      <div class="run-error"><b>The run stopped with an error</b><span>${esc(err)}</span></div>
      <details style="margin-top:10px;"><summary class="note">Full log</summary>
        <pre style="white-space:pre-wrap;max-height:260px;overflow:auto;">${esc(log.slice(-4000))}</pre></details>`;
    return;
  }
  drawResult(box, id, r);
}

/** Tracks the zoom-in heard as ONE transmission (Morse keyed on and off, speech with
 * pauses: the tracker cuts them into many) shown as one row: first-last track, whole
 * time span, the strongest confidence. Other tracks are shown as they are. */
function mergeTracks(tracks) {
  const out = [], byGroup = new Map();
  for (const t of tracks) {
    const key = t.listen ? (t.listen.tracks || [t.track]).join(",") : null;
    if (!key || (t.listen.tracks || []).length < 2) { out.push(t); continue; }
    const g = byGroup.get(key);
    if (!g) {
      const row = { ...t, merged: 1, track: t.track };
      byGroup.set(key, row);
      out.push(row);
      continue;
    }
    g.merged += 1;
    g.t_start_ms = Math.min(g.t_start_ms, t.t_start_ms);
    g.t_end_ms = Math.max(g.t_end_ms, t.t_end_ms);
    g.f_lo_khz = Math.min(g.f_lo_khz, t.f_lo_khz);
    g.f_hi_khz = Math.max(g.f_hi_khz, t.f_hi_khz);
    if (t.confidence > g.confidence) Object.assign(g, { confidence: t.confidence, band: t.band });
    g.track = `${(t.listen.tracks || [])[0]}–${(t.listen.tracks || []).slice(-1)[0]}`;
  }
  for (const g of byGroup.values()) if (g.merged < 2) delete g.merged;
  return out;
}

function freqText(a, centreHz) {
  const lo = a.f_lo_khz, hi = a.f_hi_khz;
  if (centreHz) {
    const f = v => ((centreHz + v * 1e3) / 1e6).toFixed(4);
    return `${f(lo)}–${f(hi)} MHz`;
  }
  return `${lo.toFixed(0)}…${hi.toFixed(0)} kHz`;
}

function drawResult(box, id, r) {
  const res = r.result || {};
  const src = res.source || {};
  const tracks = res.tracks || [];
  const centre = src.centre_hz;
  const listens = new Map();                      // one listen result per group of tracks (same frequency)
  for (const t of tracks) if (t.listen) listens.set((t.listen.tracks || [t.track]).join(","), t.listen);
  const heard = [...listens.values()].filter(L => L.kind && !["UNKNOWN", "ERROR"].includes(L.kind));
  box.innerHTML = `
    <div class="block-label">Spectrum · waterfall · every transmitter <span class="note" style="font-weight:400;">(recording ${esc(src.file || r.source)})</span></div>
    <div class="note">${esc(src.file || r.source)}${src.format ? ` · ${esc(src.format)}` : ""}${src.fs_in ? ` · ${(src.fs_in / 1e6).toFixed(3)} MS/s` : ""}
      ${src.resampled ? " (resampled to 3.2)" : ""}${centre ? ` · centre ${(centre / 1e6).toFixed(4)} MHz` : " · centre frequency unknown: offsets shown"}
      · ${(res.duration_ms || 0).toFixed(0)} ms analysed · ${tracks.length} transmitters · ${esc(r.detector)} detector</div>
    <img src="/api/pipeline/${encodeURIComponent(id)}/tracks.png" alt="Waterfall with every transmitter boxed and named, frequency against time"
      style="width:100%;margin-top:10px;border:1px solid var(--grid);background:#000;">
    ${heard.length ? `<div style="margin-top:14px;font-weight:700;">Decoded signals</div>` + heard.map(L => heardCard(id, L, tracks, centre)).join("") : ""}
    <div style="margin-top:14px;font-weight:700;">Every transmitter</div>
    <div class="note">Answer = the 60-class model voted over the whole transmission; when two classes cannot be told apart from
      the signal (e.g. RADAR_CW / JAM_TONE) both are shown with the reason. Confidence band from the track probability: high (≥ 0.40), medium (≥ 0.10), uncertain (below: shown as 'uncertain' with the likeliest class). Voice / Morse tracks on a clear signal are named by the zoom-in, which listens to the whole track.</div>
    <div class="table-wrap"><table class="pipe-tracks"><thead><tr><th>#</th><th>Time (ms)</th><th>Frequency</th><th>Answer</th>
      <th>Confidence</th><th>Why</th><th>Other candidates</th><th>Zoom</th></tr></thead><tbody>
    ${mergeTracks(tracks).map(t => `<tr><td>${t.track}${t.merged ? `<div class="note">${t.merged} bursts</div>` : ""}</td><td>${t.t_start_ms.toFixed(1)}–${t.t_end_ms.toFixed(1)}</td>
      <td>${esc(freqText(t, centre)).replace("–", "–<wbr>")}</td><td>${t.band === 'uncertain' ? `<span class="note">uncertain</span><div class="note">likeliest ${esc(t.answer)}</div>` : `<b>${esc(t.answer)}</b>`}${t.decided_by === 'zoom' ? `<div class="note">by zoom-in (classifier: ${esc(t.classifier_answer)})</div>` : ''}</td>
      <td>${chip(t.band || t.confidence.toFixed(2), BAND_COL[t.band] || CONF(t.confidence))}<div class="note">${t.confidence.toFixed(2)}</div></td><td class="note">${esc(t.reason)}</td>
      <td class="note">${esc((t.top5 || []).slice(1, 4).map(([c, p]) => `${c} ${p.toFixed(2)}`).join(", "))}</td>
      <td>${t.listen ? esc(KIND_TEXT[t.listen.kind] || t.listen.kind) : ""}</td></tr>`).join("")}
    </tbody></table></div>
    <details style="margin-top:8px;"><summary class="note">Run log</summary>
      <pre style="white-space:pre-wrap;max-height:240px;overflow:auto;">${esc(r.log || "")}</pre></details>`;
  for (const form of box.querySelectorAll("form.tdoa")) {
    form.addEventListener("submit", async ev => {
      ev.preventDefault();
      const out = form.querySelector(".tdoa-msg");
      const lat = Number(form.lat.value), lon = Number(form.lon.value);
      if (!(Math.abs(lat) <= 90 && Math.abs(lon) <= 180) || form.lat.value === "" || form.lon.value === "") {
        out.textContent = "Latitude -90..90 and longitude -180..180, e.g. 3.14 and 101.69.";
        return;
      }
      try {
        const res = await fetch("/api/audit", { method: "POST", credentials: "same-origin",
          headers: { ...HDR, "Content-Type": "application/json" },
          body: JSON.stringify({ action: "client.tdoa_location", target_type: "pipeline_run", target_id: id,
            details: { freq_khz: Number(form.dataset.khz), lat, lon, receivers: form.rx.value.trim(),
                       note: form.note.value.trim(), source: "KiwiSDR TDoA" } }) });
        if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
        out.textContent = `Saved in the audit trail: ${lat.toFixed(3)}, ${lon.toFixed(3)}.`;
      } catch (e) { out.textContent = `Could not save: ${e.message}`; }
    });
  }
}

// KiwiSDR direction finding (TDoA) is a KiwiSDR service: NEXA links to it for
// HF signals with a known frequency and records the location the analyst reads
// off its map. Nothing is sent from NEXA; the link opens the receiver's page.
const KIWI_RX = "http://dx6ran.ddns.net:8073";   // DU6/PE1NSQ, Negros, Philippines (team default receiver, GPS-timed)
const HF_MAX_KHZ = 30000;

/** Absolute frequency (kHz) of a heard signal, when the capture's centre is known and it is HF. */
function hfKhz(tracks, L, centre) {
  if (!centre) return null;
  const t = tracks.find(t => t.listen === L) || tracks.find(t => (L.tracks || []).includes(t.track));
  if (!t) return null;
  const khz = (centre + ((t.f_lo_khz + t.f_hi_khz) / 2) * 1e3) / 1e3;
  return khz > 0 && khz <= HF_MAX_KHZ ? khz : null;
}

function locateBlock(khz) {
  const url = `${KIWI_RX}/?f=${khz.toFixed(2)}iqz10`;
  return `<details style="margin-top:8px;"><summary><b>Locate on KiwiSDR</b> <span class="note">(HF ${khz.toFixed(2)} kHz: where is the transmitter?)</span></summary>
    <div class="note" style="margin:6px 0;">
      1. <a href="${esc(url)}" target="_blank" rel="noopener noreferrer">Open the Kuala Lumpur KiwiSDR at ${khz.toFixed(2)} kHz in IQ mode</a>
         (an outside website; only the frequency is in the link).<br>
      2. In its bottom-right panel, choose <b>Extension → TDoA</b>, tick 3 or more receivers on the map that hear the signal, press <b>Submit</b>.<br>
      3. When the map shows the most likely area, enter its centre below. It is saved in NEXA's audit trail next to this detection.<br>
      Short-wave paths bounce off the ionosphere, so expect an area tens to hundreds of km across, not a point.</div>
    <form class="tdoa filters" data-khz="${khz.toFixed(3)}">
      <input name="lat" placeholder="latitude (e.g. 3.14)" style="width:150px;">
      <input name="lon" placeholder="longitude (e.g. 101.69)" style="width:160px;">
      <input name="rx" placeholder="receivers used" style="width:170px;">
      <input name="note" placeholder="note" style="width:180px;">
      <button type="submit">Record location</button> <span class="note tdoa-msg"></span>
    </form></details>`;
}

/** One decoded signal under the waterfall: what it is, where, a line of what was said, and
 * Open analysis (the full page: playback, original text, English, AI reading, facts). */
function heardCard(id, L, tracks, centre) {
  const t = tracks.find(t => t.listen === L) || tracks.find(t => (L.tracks || []).includes(t.track));
  const mid = t ? (t.f_lo_khz + t.f_hi_khz) / 2 : null;
  const where = t ? (centre ? `${((centre + mid * 1e3) / 1e6).toFixed(4)} MHz` : `${mid >= 0 ? "+" : ""}${mid.toFixed(1)} kHz from centre`) : "";
  const text = L.morse || L.transcript || L.digital_text || "";
  const short = text.length > 160 ? text.slice(0, 160).replace(/\s+\S*$/, "") + " …" : text;
  const intelId = `${id}_track_${Math.min(...(L.tracks || [0]))}`;
  const khz = hfKhz(tracks, L, centre);
  return `<div class="block" style="margin:8px 0;padding:12px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;">
      <div><b style="font-size:16px;">${esc(KIND_TEXT[L.kind] || L.kind)}</b>
        <span class="note">${where ? ` · ${esc(where)}` : ""} · ${(L.duration_s || 0).toFixed(1)} s</span>
        ${short ? `<div style="margin-top:4px;font-family:var(--mono);overflow-wrap:anywhere;">${esc(short)}</div>`
          : `<div class="note" style="margin-top:4px;">${esc(L.text_error || "No text read from this signal.")}</div>`}</div>
      ${text ? `<button class="primary" data-open-analysis="${esc(intelId)}">Open analysis</button>` : ""}
    </div>
    ${khz ? locateBlock(khz) : ""}
  </div>`;
}
