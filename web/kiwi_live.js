// Live KiwiSDR listening: type a shortwave frequency, press Start, and the
// server (scripts/sdr_hub.py KiwiHub, /api/kiwi/*) streams that frequency from a
// KiwiSDR without stopping. Every clip (30 s, or when a transmission ends) is
// decoded -- Morse, RTTY, PSK31 or speech (Whisper) -- and read by the local AI,
// then saved as a live Signal Pipeline run and a Comms Intel recording.
// No command line needed. The KiwiSDR page (main.js).

const HDR = { "X-NEXA-Client": "1", "Content-Type": "application/json" };
const DEFAULT_RX = "21439.proxy2.kiwisdr.com:8073";      // 9M2AMH, Kuala Lumpur (GPS-timed: TDoA works)
const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const KIND = { AM: "AM voice", USB: "USB voice", LSB: "LSB voice", NBFM: "FM voice", CW: "Morse", RTTY: "RTTY",
  PSK31: "PSK31", CARRIER: "carrier only", UNKNOWN: "unclear" };

// Saved frequencies: pick one to fill in the frequency, demodulation and language.
const PRESETS = [
  { id: "rtm15295", label: "RTM · 15295 kHz", khz: 15295, demod: "AM", lang: "ms" },
];

let timer = null;
let mounted = null;

async function call(path, body) {
  const r = await fetch(`/api/kiwi/${path}`, body === undefined ? { cache: "no-store" }
    : { method: "POST", headers: HDR, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

export function renderKiwiLive(root, { server }) {
  if (!root) return;
  if (!server) { root.innerHTML = ""; return; }
  if (mounted !== root) {
    mounted = root;
    lastListKey = null;
    root.innerHTML = `
    <div class="block" id="kiwiBlock">
      <div class="block-label">Live listening</div>
      <div class="note">Listens to one shortwave frequency without stopping. Every clip is decoded (Morse, RTTY,
        PSK31 or speech) and read by the local AI. Results appear below and in Intercepts.</div>
      <div class="filters" style="margin-top:12px;flex-wrap:wrap;gap:12px 18px;align-items:flex-end;">
        <label class="field" style="min-width:230px;">Receiver
          <input id="kwRx" value="${DEFAULT_RX}" spellcheck="false"></label>
        <label class="field">Saved frequencies
          <select id="kwPreset"><option value="">Custom</option>
            ${PRESETS.map(p => `<option value="${p.id}">${esc(p.label)}</option>`).join("")}</select></label>
        <label class="field" style="width:130px;">Frequency (kHz)
          <input id="kwFreq" inputmode="decimal" placeholder="e.g. 15295"></label>
        <label class="field">Language
          <select id="kwLang"><option value="auto">Auto</option><option value="ms">Malay</option>
            <option value="en">English</option><option value="zh">Mandarin</option></select></label>
        <label class="field">Speech model
          <select id="kwModel"><option value="base">base (lightest, ~0.5 GB)</option>
            <option value="small" selected>small (~1 GB)</option>
            <option value="medium">medium (best, ~5 GB)</option></select></label>
        <label class="field">AI model
          <select id="kwLlmModel"><option value="llama3.2:3b" selected>llama3.2:3b (better, ~2.5 GB)</option>
            <option value="qwen2.5:1.5b">qwen2.5:1.5b (lighter, ~1 GB)</option></select></label>
        <label class="field">Clip
          <select id="kwClip"><option value="30" selected>30 s</option><option value="60">60 s</option></select></label>
        <label class="field" title="Auto measures each signal. If you know what it is (a short-wave broadcast is AM,
          amateur voice is USB or LSB), choose it: the audio is then demodulated that way.">Demodulate as
          <select id="kwDemod"><option value="auto" selected>Auto (measured)</option>
            <option value="AM">AM (broadcast)</option><option value="USB">USB</option><option value="LSB">LSB</option>
            <option value="NBFM">FM (narrow)</option><option value="CW">CW (Morse)</option></select></label>
        <label style="display:inline-flex;gap:6px;align-items:center;"><input type="checkbox" id="kwLlm" checked> AI reading</label>
        <button class="primary" id="kwStart">Start listening</button>
        <button class="secondary" id="kwStop" disabled>Stop</button>
      </div>
      <div id="kwState" class="note" role="status" style="margin-top:10px;"></div>
    </div>
    <div class="block">
      <div class="block-label">Decoded clips</div>
      <div id="kwList"></div>
    </div>
    <div class="block">
      <div class="block-label">Transmitter location</div>
      <div id="kwLoc" style="margin-top:14px;padding:12px 14px;border:1px solid var(--grid);border-radius:6px;"></div>
      <div class="filters" style="margin-top:10px;flex-wrap:wrap;gap:10px 16px;align-items:flex-end;">
        <button class="secondary" id="kwLocate" title="Records this frequency on 3 GPS-timed KiwiSDRs at once and
          works out where the transmitter is from the arrival-time differences (TDoA)">Locate transmitter</button>
        <label class="field">Sample
          <select id="kwTdoaS"><option value="10">10 s</option><option value="15" selected>15 s</option>
            <option value="30">30 s</option></select></label>
        <label class="field" style="min-width:300px;">TDoA receivers (host:port, comma-separated)
          <input id="kwTdoaRx" spellcheck="false" placeholder="default: DU6/PE1NSQ, HS0ZGZ, JKT (9M2AMH spare)"></label>
        <label style="display:inline-flex;gap:6px;align-items:center;"><input type="checkbox" id="kwAuto" checked>
          Locate automatically when listening starts</label>
      </div>
      <div id="kwTdoa" class="note" role="status" style="margin-top:8px;"></div>
      <details id="kwLocForm" style="margin-top:8px;"><summary class="note" style="cursor:pointer;">Or record a
        location by hand (e.g. from KiwiSDR's own TDoA map)</summary>
        <div class="filters" style="margin-top:8px;">
          <input id="kwLat" placeholder="latitude (e.g. 2.50)" style="width:150px;">
          <input id="kwLon" placeholder="longitude (e.g. 102.50)" style="width:160px;">
          <input id="kwRxUsed" placeholder="receivers used" style="width:220px;">
          <input id="kwNote" placeholder="note" style="width:200px;">
          <button class="secondary" id="kwSaveLoc">Record location</button>
          <span class="note" id="kwLocMsg"></span>
        </div></details>
    </div>`;
    const $ = id => root.querySelector(`#${id}`);
    $("kwStart").addEventListener("click", async () => {
      const [host, port] = $("kwRx").value.trim().split(":");
      $("kwStart").disabled = true;
      try {
        paint(await call("start", { host, port: Number(port) || 8073, freq_khz: Number($("kwFreq").value),
          language: $("kwLang").value, whisper_model: $("kwModel").value, clip_s: Number($("kwClip").value),
          llm: $("kwLlm").checked, llm_model: $("kwLlmModel").value, demod: $("kwDemod").value }));
      } catch (e) {
        $("kwState").textContent = `Could not start: ${e.message}`;
        $("kwStart").disabled = false;
      }
    });
    $("kwStop").addEventListener("click", async () => paint(await call("stop", {}).catch(() => null)));
    $("kwFreq").addEventListener("change", () => {
      const p = PRESETS.find(p => p.id === $("kwPreset").value);
      if (p && Number($("kwFreq").value) !== p.khz) $("kwPreset").value = "";   // typed over it: custom now
      showLocation(root, true);
    });
    $("kwPreset").addEventListener("change", () => {
      const p = PRESETS.find(p => p.id === $("kwPreset").value);
      if (!p) return;
      $("kwFreq").value = p.khz;
      $("kwDemod").value = p.demod;
      $("kwLang").value = p.lang;
      showLocation(root, true);
    });
    $("kwLocate").addEventListener("click", () => locate(Number($("kwFreq").value)));
    $("kwSaveLoc").addEventListener("click", async () => {
      const khz = Number($("kwFreq").value), lat = Number($("kwLat").value), lon = Number($("kwLon").value);
      const msg = $("kwLocMsg");
      if (!(khz > 0) || $("kwLat").value === "" || $("kwLon").value === "" || !(Math.abs(lat) <= 90 && Math.abs(lon) <= 180)) {
        msg.textContent = "Enter the frequency above, and a latitude (-90..90) and longitude (-180..180).";
        return;
      }
      try {
        const r = await fetch("/api/audit", { method: "POST", credentials: "same-origin", headers: HDR,
          body: JSON.stringify({ action: "client.tdoa_location", target_type: "kiwi", target_id: `${khz} kHz`,
            details: { freq_khz: khz, lat, lon, receivers: $("kwRxUsed").value.trim(), note: $("kwNote").value.trim(),
                       source: "KiwiSDR TDoA" } }) });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
        msg.textContent = "Saved in the audit trail.";
        showLocation(root, true);
      } catch (e) {
        msg.textContent = `Could not save: ${e.message}`;
      }
    });
  }
  clearInterval(timer);
  const tick = async () => {
    if (!document.body.contains(root)) { clearInterval(timer); return; }
    try { paint(await call("status")); } catch { /* server restarting: next tick */ }
  };
  tick();
  timer = setInterval(tick, 2000);
  showLocation(root, true);
}

// "Locate transmitter": the server records the frequency on 3 GPS-timed KiwiSDRs
// at once and computes the place from arrival-time differences (TDoA,
// src/kiwi_tdoa.py). Started by the button, or once per frequency when listening
// starts and no location is on record. The result lands in the audit trail as
// client.tdoa_location, so the box below shows it like a hand-recorded one.
const autoTried = new Set();
let lastTdoa = null;
async function locate(khz) {
  const $ = id => mounted.querySelector(`#${id}`);
  if (!(khz > 0)) { $("kwTdoa").textContent = "Type a frequency first."; return; }
  $("kwLocate").disabled = true;
  try {
    paintTdoa(await call("locate", { freq_khz: khz, sample_s: Number($("kwTdoaS").value),
      receivers: $("kwTdoaRx").value.trim() || null }));
  } catch (e) {
    $("kwTdoa").innerHTML = `<span style="color:#9B2C2C;">Could not locate: ${esc(e.message)}</span>`;
    $("kwLocate").disabled = false;
  }
}

function paintTdoa(t) {
  if (!t || !mounted) return;
  const $ = id => mounted.querySelector(`#${id}`);
  const busy = ["checking", "recording", "computing"].includes(t.state);
  $("kwLocate").disabled = busy;
  const skipped = (t.receivers || []).filter(r => !r.ok).map(r => `${esc(r.id)} ${esc(r.why)}`);
  $("kwTdoa").innerHTML = busy
    ? `<b>Locating ${esc(t.freq_khz)} kHz:</b> ${esc(t.message || t.state)}
       <progress max="1" value="${Number(t.progress) || 0}" style="width:160px;vertical-align:middle;"></progress>`
      + (skipped.length ? `<div>Skipped: ${skipped.join("; ")}.</div>` : "")
    : t.state === "error" ? `<span style="color:#9B2C2C;">Locating ${esc(t.freq_khz)} kHz failed: ${esc(t.error)}</span>`
    : t.state === "done" && t.result ? `Located ${esc(t.freq_khz)} kHz at ${esc(t.result.at)} with ${esc(t.result.receivers)}
       (arrival-time differences: ${t.result.pairs.map(p => `${esc(p.a)}−${esc(p.b)} ${p.lag_us} ± ${p.sigma_us} µs`).join(", ")}).`
      + (skipped.length ? ` Skipped: ${skipped.join("; ")}.` : "")
    : "";
  const key = t.state === "done" && t.result ? t.result.at : null;
  if (key && key !== lastTdoa) { lastTdoa = key; showLocation(mounted, true); }
}

// The newest location on record for the frequency in the box (computed by NEXA, or typed in by the analyst).
let locKhz = null;
let locMissing = null;           // the frequency the box last found NO location for
let lastListKey = null;          // what the clip list shows now (paint redraws it only when this changes)
async function showLocation(root, force = false) {
  const box = root.querySelector("#kwLoc"), khz = Number(root.querySelector("#kwFreq").value);
  if (!box) return;
  if (!(khz > 0)) { box.innerHTML = `<span class="note">Type a frequency to see where its transmitter was located.</span>`; locKhz = null; return; }
  locMissing = null;
  if (!force && khz === locKhz) return;
  locKhz = khz;
  let loc = null;
  try { loc = (await (await fetch(`/api/kiwi/location?khz=${khz}`, { cache: "no-store" })).json()).location; } catch { /* server restarting */ }
  if (!loc) {
    locMissing = khz;
    box.innerHTML = `<b>Transmitter location (${khz} kHz):</b> <span class="note">not found yet. Press "Locate
      transmitter" (or start listening with automatic locating on).</span>`;
    return;
  }
  const lat = Number(loc.lat), lon = Number(loc.lon);
  const map = `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=7/${lat}/${lon}`;
  const r = Number(loc.radius_km);
  box.innerHTML = `<b>Transmitter location (${esc(loc.freq_khz)} kHz):</b> most likely ${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"},
    ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}${r > 0 ? `, within about ${r} km` : ""} ·
    <a href="${esc(map)}" target="_blank" rel="noopener noreferrer">show on a map</a>
    <div class="note" style="margin-top:4px;">${esc(loc.source || "KiwiSDR TDoA")}${loc.receivers ? ` with ${esc(loc.receivers)}` : ""} · recorded
      ${esc(new Date(loc.ts).toLocaleString())} by ${esc(loc.actor)}${loc.note ? ` · ${esc(loc.note)}` : ""}.
      An area, not a point: short-wave signals bounce off the ionosphere, so this is the centre of an area tens to
      hundreds of km across.</div>`;
}

function paint(st) {
  if (!st || !mounted) return;
  const $ = id => mounted.querySelector(`#${id}`);
  const on = st.state === "connecting" || st.state === "streaming";
  $("kwStart").disabled = on;
  $("kwStop").disabled = !on;
  const khz = st.decode?.freq_khz;
  if (on && khz && document.activeElement !== $("kwFreq")) $("kwFreq").value = khz;
  const kiwi = (st.label || "").replace(/^KiwiSDR /, "");
  const rx = (/^kiwi:\/\/([^/]+)\//.exec(st.url || "") || [])[1] || DEFAULT_RX;   // the receiver being listened to
  const tdoa = khz ? ` · <a href="http://${esc(rx)}/?f=${Number(khz).toFixed(2)}iqz10" target="_blank"
    rel="noopener noreferrer">open on KiwiSDR for TDoA</a>` : "";
  paintTdoa(st.tdoa);
  const idle = !st.tdoa || ["idle", "done", "error"].includes(st.tdoa.state);
  if (st.state === "streaming" && khz && $("kwAuto").checked && idle && !autoTried.has(khz)
      && locMissing === Number(khz)) {
    autoTried.add(khz);                            // once per frequency: the receivers are shared
    locate(Number(khz));
  }
  $("kwState").innerHTML = st.state === "off" ? "Not listening."
    : st.state === "connecting" ? `Connecting to ${esc(kiwi)}…`
    : st.state === "error" ? `<span style="color:#9B2C2C;">Stopped: ${esc(st.error)}</span>`
    : `<b>Listening</b> to ${esc(kiwi)}${st.on_air ? ` · signal on air, clip ${st.recording_s} s` : " · quiet"}`
      + (st.analysing ? ` · decoding ${st.analysing}` : "")
      + (st.skipped ? ` · ${st.skipped} clip${st.skipped > 1 ? "s" : ""} skipped while busy` : "") + tdoa;
  const rows = [...(st.decodes || [])].reverse();
  // Redraw only when a clip arrives: redrawing every tick would cut off an audio player that is playing.
  const listKey = JSON.stringify([st.state === "streaming", rows.map(d => d.id || d.at)]);
  if (listKey === lastListKey) return;
  lastListKey = listKey;
  // Each clip's demodulated audio (results/pipeline/jobs/<id>/track_N.wav): play it here or save the .wav.
  const players = d => [...new Set(d.tracks.map(t => t.audio).filter(Boolean))].map(a => {
    const src = `/api/pipeline/${encodeURIComponent(d.id)}/${encodeURIComponent(a)}`;
    return `<div style="display:flex;gap:10px;align-items:center;margin-top:6px;flex-wrap:wrap;">
      <audio controls preload="none" style="flex:1;min-width:220px;max-width:480px;" src="${src}"></audio>
      <a href="${src}" download="kiwi_${esc(d.id)}_${esc(a)}">save .wav</a></div>`;
  }).join("");
  $("kwList").innerHTML = rows.map(d => d.error
    ? `<div class="note" style="color:#9B2C2C;">${esc(d.at)} · decode failed: ${esc(d.error)}</div>`
    : `<div style="padding:10px 0;border-top:1px solid var(--grid);">
        <div><b>${esc(d.at)}</b> · ${d.seconds} s · ${d.tracks.length ? d.tracks.map(t =>
          `${esc(KIND[t.kind] || t.kind)} ${d.centre_hz ? ((d.centre_hz / 1e3 + (t.f_lo_khz + t.f_hi_khz) / 2)).toFixed(1) + " kHz" : ""}`).join(", ")
          : "nothing above the noise"}${d.warning ? ` <span class="note">(${esc(d.warning)})</span>` : ""}</div>
        ${d.tracks.filter(t => t.text).map(t => `<div style="margin-top:6px;"><b>${t.kind === "CW" ? "Morse" : ["RTTY", "PSK31"].includes(t.kind) ? "Text" : "Said"}${t.language ? ` (${esc(t.language)})` : ""}:</b>
            <span style="font-family:var(--mono);">${esc(t.text)}</span></div>
          ${t.reading ? `<div style="margin-top:4px;"><b>AI reading:</b> ${esc(t.reading)}</div>` : ""}`).join("")}
        ${d.tracks.filter(t => t.text_error && !t.text).map(t => `<div class="note" style="margin-top:6px;color:#9B2C2C;">
            ${esc(KIND[t.kind] || t.kind)}: not decoded (${esc(t.text_error)})${/memory|alloc/i.test(t.text_error)
              ? ". This laptop ran out of memory: choose a smaller speech model (small or base), or close other programs." : ""}</div>`).join("")}
        ${players(d)}
      </div>`).join("") || (st.state === "streaming" ? `<div class="note">First result after the first clip (about ${Math.round(st.decode?.whisper_model === "medium" ? 90 : 60)} s).</div>`
    : `<div class="note">No clips yet. Choose a frequency and press Start listening.</div>`);
}
