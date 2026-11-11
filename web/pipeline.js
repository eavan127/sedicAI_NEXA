// Signal Pipeline page: the whole wideband chain on a recording, run by the
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
const LANG_TEXT = { auto: "Language: detect", ms: "Malay", en: "English", id: "Indonesian", zh: "Mandarin" };

// The transcript line by line, with the lines Whisper itself doubted marked. A line
// with no mark is not known to be right: Whisper can be wrong without saying so.
function speechText(L) {
  const segs = L.segments || [];
  if (!segs.length) return L.transcript ? esc(L.transcript) : "<span class='note'>no words recognised</span>";
  return segs.map(s => s.doubtful
    ? `<span style="color:#9B2C2C;" title="Whisper was unsure of this part">[? ${esc(s.text)}]</span>`
    : esc(s.text)).join(" ");
}

let state = { open: null, timer: null, follow: false, liveTimer: null };

function chip(text, color) {
  return `<span class="chip" style="background:${color}1f;color:${color};">${esc(text)}</span>`;
}

export async function renderPipeline(root, { server }) {
  if (!server) {
    // The page may ask before its own server check has finished: ask the server directly.
    try { server = (await fetch("/api/health", { credentials: "same-origin" })).ok; } catch { server = false; }
  }
  if (!server) {
    root.innerHTML = `<div class="block"><div class="note">The signal pipeline runs on this machine's local server
      (it needs the trained models and Python). Start it with <code>python scripts/serve_local.py</code>.</div></div>`;
    return;
  }
  let info;
  try {
    info = await (await fetch("/api/pipeline", { credentials: "same-origin" })).json();
  } catch (e) {
    root.innerHTML = `<div class="note">Could not reach the pipeline: ${esc(e.message)}</div>`;
    return;
  }
  root.innerHTML = `
    <div class="block">
      <div class="block-label">Analyse a recording</div>
      <div class="note">The whole chain on one recording: <b>find</b> each transmitter (YOLO boxes), <b>name</b> it
        (60-class model, voted over the whole transmission), and for voice or Morse <b>listen</b>: zoom in, demodulate,
        Morse decoder or Whisper, then a local AI reading. Record in SDRconnect with <b>Recording type = IQ</b>.
        On this PC the analysis takes about 50 s per second of recording, so start with 1 s.</div>
      <div class="note" style="margin-top:6px;">Models: classifier ${info.classifiers.length
        ? info.classifiers.map(c => `<code>${esc(c)}</code>`).join(" ") + (info.classifiers.length > 1 ? " (averaged)" : "")
        : `<b style="color:#9B2C2C;">none: put a best.pt in results/classifier/</b>`} ·
        detector ${info.detector ? "<code>emitter_detector.pt</code> (YOLO)" : "<b>energy fallback</b> (no YOLO file)"}</div>
      <div class="filters" style="margin-top:10px;">
        <input type="file" id="pipeFile" accept=".wav,.npy,.bin,.raw,.cf32,.cs16,.cs8,.cu8,.iq">
        <label class="note" style="margin:0;">Seconds <input id="pipeSec" type="number" min="0.05" max="10" step="0.25" value="1" style="width:70px;"></label>
        <label class="note" style="margin:0;" title="Only for raw files without a header">Sample rate <input id="pipeFs" placeholder="from file" style="width:90px;"></label>
        <select id="pipeDtype" title="Only for raw files"><option value="">raw type</option><option>int16</option><option>float32</option><option>int8</option><option>uint8</option></select>
      </div>
      <div class="filters">
        <label class="note" style="margin:0;"><input type="checkbox" id="pipeListen" checked> Listen to voice / Morse</label>
        <label class="note" style="margin:0;"><input type="checkbox" id="pipeWhisper" checked> Whisper</label>
        <select id="pipeLang" title="Language spoken. Setting it is safer on short clips, but a wrong choice gives nonsense">
          ${(info.languages || ["auto"]).map(l => `<option value="${esc(l)}">${esc(LANG_TEXT[l] || l)}</option>`).join("")}</select>
        <label class="note" style="margin:0;"><input type="checkbox" id="pipeLlm" checked> AI reading (Ollama)</label>
        <button id="pipeRun">Analyse file</button>
        <button class="secondary" id="pipeDemo" title="Made-up walkie-talkie speech, Morse and an AM station in a 1.2 s capture">Run voice demo</button>
      </div>
      <div class="note" id="pipeMsg"></div>
    </div>
    <div class="block" id="pipeRunBox" hidden></div>
    <div class="block">
      <div class="block-label">Live decodes</div>
      <div class="note">Each transmission the live receiver records and decodes is added to Recent runs below. With this ticked,
        the page opens each finished one by itself.</div>
      <label class="note" style="margin:6px 0 0;"><input type="checkbox" id="pipeFollow"> Follow live snapshots</label>
      <span class="note" id="pipeLiveMsg" style="margin-left:8px;"></span>
    </div>
    <div class="block">
      <div class="block-label">Recent runs</div>
      <div class="table-wrap" id="pipeRuns"></div>
    </div>`;
  const follow = root.querySelector("#pipeFollow");
  follow.checked = state.follow || false;
  follow.addEventListener("change", () => { state.follow = follow.checked; if (follow.checked) followLive(root); });
  if (state.follow) followLive(root);
  const msg = root.querySelector("#pipeMsg");
  const start = async (demo) => {
    const p = new URLSearchParams({
      source: demo ? "voice-demo" : "file", max_s: root.querySelector("#pipeSec").value || "1",
      detector: "hybrid",                  // always YOLO + the narrow-signal finder
      listen: root.querySelector("#pipeListen").checked ? "1" : "0",
      whisper: root.querySelector("#pipeWhisper").checked ? "1" : "0",
      llm: root.querySelector("#pipeLlm").checked ? "1" : "0",
      language: root.querySelector("#pipeLang").value,
    });
    let body = null;
    if (!demo) {
      const f = root.querySelector("#pipeFile").files[0];
      if (!f) { msg.textContent = "Choose an IQ file first (or run the voice demo)."; return; }
      p.set("name", f.name);
      const fs = root.querySelector("#pipeFs").value.trim();
      if (fs) p.set("fs", String(Number(fs)));
      const dt = root.querySelector("#pipeDtype").value;
      if (dt) p.set("dtype", dt);
      body = f;
      msg.textContent = `Uploading ${f.name} (${(f.size / 1e6).toFixed(1)} MB)…`;
    }
    try {
      const res = await fetch(`/api/pipeline?${p}`, { method: "POST", credentials: "same-origin", headers: HDR, body });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
      msg.textContent = `Started run ${j.id}.`;
      openRun(root, j.id);
      listRuns(root);
    } catch (e) {
      msg.textContent = `Could not start: ${e.message}`;
    }
  };
  root.querySelector("#pipeRun").addEventListener("click", () => start(false));
  root.querySelector("#pipeDemo").addEventListener("click", () => start(true));
  await listRuns(root, info.runs);
  if (state.open) openRun(root, state.open);
  else if (info.runs[0]) openRun(root, info.runs[0].id);
}

// Poll for live-bridge snapshots (scripts/live_sdrconnect.py) and show the newest finished one.
async function followLive(root) {
  clearTimeout(state.liveTimer);
  if (!state.follow || !root.isConnected) return;
  const msg = root.querySelector("#pipeLiveMsg");
  try {
    const runs = (await (await fetch("/api/pipeline", { credentials: "same-origin" })).json()).runs;
    const live = runs.filter(r => r.live && r.status === "done");
    if (!live.length) {
      if (msg) msg.textContent = "waiting for the first snapshot…";
    } else {
      const newest = live[0];
      if (msg) msg.textContent = `newest snapshot ${newest.id}`;
      if (state.open !== newest.id) { await openRun(root, newest.id); await listRuns(root, runs); }
    }
  } catch (e) {
    if (msg) msg.textContent = `cannot reach the server: ${e.message}`;
  }
  state.liveTimer = setTimeout(() => followLive(root), 2000);
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
  if (!runs) runs = (await (await fetch("/api/pipeline", { credentials: "same-origin" })).json()).runs;
  box.innerHTML = runs.length ? `<table><thead><tr><th>Run</th><th>Time</th><th>Source</th><th>Seconds</th><th>Analysis took</th><th>Detector</th><th>Status</th></tr></thead><tbody>`
    + runs.map(r => `<tr data-id="${esc(r.id)}" style="cursor:pointer;"><td><code>${esc(r.id)}</code></td>
      <td>${esc(runTime(r.id))}</td><td>${esc(r.file || r.source)}</td><td>${esc(r.max_s)}</td><td>${esc(runTook(r))}</td><td>${esc(r.detector)}</td>
      <td>${chip(r.status, r.status === "done" ? "#2E7D32" : r.status === "running" ? "#2B6CB0" : "#9B2C2C")}</td></tr>`).join("")
    + `</tbody></table>` : `<div class="note">No runs yet.</div>`;
  for (const tr of box.querySelectorAll("tr[data-id]")) tr.addEventListener("click", () => openRun(root, tr.dataset.id));
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
    box.innerHTML = `<div class="block-label">Run ${esc(id)} ${chip("running", "#2B6CB0")}</div>
      <div class="note">${esc(r.file || r.source)} · ${esc(r.max_s)} s · ${esc(r.detector)} detector. This page refreshes by itself.</div>
      <pre style="white-space:pre-wrap;max-height:180px;overflow:auto;">${esc(last || "starting…")}</pre>`;
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
    <div class="block-label">Run ${esc(id)} ${chip("done", "#2E7D32")}</div>
    <div class="note">${esc(src.file || r.source)}${src.format ? ` · ${esc(src.format)}` : ""}${src.fs_in ? ` · ${(src.fs_in / 1e6).toFixed(3)} MS/s` : ""}
      ${src.resampled ? " (resampled to 3.2)" : ""}${centre ? ` · centre ${(centre / 1e6).toFixed(4)} MHz` : " · centre frequency unknown: offsets shown"}
      · ${(res.duration_ms || 0).toFixed(0)} ms analysed · ${tracks.length} transmitters · ${esc(r.detector)} detector</div>
    <img src="/api/pipeline/${encodeURIComponent(id)}/tracks.png" alt="Waterfall with every transmitter boxed and named"
      style="width:100%;margin-top:10px;border:1px solid var(--grid);background:#000;">
    ${heard.length ? `<div style="margin-top:14px;font-weight:700;">What it says</div>` + heard.map(L => listenCard(id, L, hfKhz(tracks, L, centre))).join("") : ""}
    <div style="margin-top:14px;font-weight:700;">Every transmitter</div>
    <div class="note">Answer = the 60-class model voted over the whole transmission; when two classes cannot be told apart from
      the signal (e.g. RADAR_CW / JAM_TONE) both are shown with the reason. Confidence band from the track probability: high (≥ 0.40), medium (≥ 0.10), uncertain (below: shown as 'uncertain' with the likeliest class). Voice / Morse tracks on a clear signal are named by the zoom-in, which listens to the whole track.</div>
    <div class="table-wrap"><table><thead><tr><th>#</th><th>Time (ms)</th><th>Frequency</th><th>Answer</th>
      <th>Confidence</th><th>Why</th><th>Other candidates</th><th>Zoom</th></tr></thead><tbody>
    ${tracks.map(t => `<tr><td>${t.track}</td><td>${t.t_start_ms.toFixed(1)}–${t.t_end_ms.toFixed(1)}</td>
      <td style="white-space:nowrap;">${esc(freqText(t, centre))}</td><td>${t.band === 'uncertain' ? `<span class="note">uncertain</span><div class="note">likeliest ${esc(t.answer)}</div>` : `<b>${esc(t.answer)}</b>`}${t.decided_by === 'zoom' ? `<div class="note">by zoom-in (classifier: ${esc(t.classifier_answer)})</div>` : ''}</td>
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
const KIWI_RX = "http://21439.proxy2.kiwisdr.com:8073";   // 9M2AMH, Kuala Lumpur (GPS-timed)
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

function listenCard(id, L, khz = null) {
  const rd = L.reading || {};
  const said = L.morse ? `<div><b>Morse:</b> <code>${esc(L.morse)}</code></div>`
    : L.transcript !== undefined ? `<div><b>Speech (Whisper${L.whisper_model ? ` ${esc(L.whisper_model)}` : ""}${L.language ? `, ${esc(L.language)}` : ""}):</b>
        ${speechText(L)}</div>
        <div class="note">Machine transcript: it can be wrong without any mark, so check names and numbers against the audio.</div>` : "";
  return `<div class="block" style="margin:8px 0;padding:10px;">
    <div><b>${esc(KIND_TEXT[L.kind] || L.kind)}</b> · tracks ${esc((L.tracks || []).join(", "))} ·
      ${(L.duration_s || 0).toFixed(1)} s <span class="note">— ${esc(L.why || "")}</span></div>
    ${L.audio ? `<audio controls preload="none" style="width:100%;margin-top:6px;" src="/api/pipeline/${encodeURIComponent(id)}/${encodeURIComponent(L.audio)}"></audio>` : ""}
    ${said}
    ${L.text_error ? `<div class="note" style="color:#9B2C2C;">${esc(L.text_error)}</div>` : ""}
    ${rd.error ? `<div class="note" style="color:#9B2C2C;">AI reading: ${esc(rd.error)}</div>` : rd.summary ? `
      <div style="margin-top:6px;"><b>AI reading</b> ${chip((rd.confidence || "low") + " confidence", rd.confidence === "high" ? "#2E7D32" : rd.confidence === "medium" ? "#B7791F" : "#9B2C2C")}
        <span class="note">${esc(rd.category || "")}</span></div>
      <div>${esc(rd.summary)}</div>
      ${(rd.evidence || []).map(e => `<div class="note"><code>${esc(e.quote)}</code> — ${esc(e.why)}</div>`).join("")}
      ${rd.removed_ungrounded ? `<div class="note">${rd.removed_ungrounded} unsupported item(s) the model made up were removed.</div>` : ""}` : ""}
    ${khz ? locateBlock(khz) : ""}
  </div>`;
}
