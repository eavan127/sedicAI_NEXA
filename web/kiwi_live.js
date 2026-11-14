// Live KiwiSDR listening: type a shortwave frequency, press Start, and the
// server (scripts/sdr_hub.py KiwiHub, /api/kiwi/*) streams that frequency from a
// KiwiSDR without stopping. Every clip (30 s, or when a transmission ends) is
// decoded -- Morse, RTTY, PSK31 or speech (Whisper) -- and read by the local AI,
// then saved as a live Signal Pipeline run and a Comms Intel recording.
// No command line needed. The KiwiSDR page (main.js).

const HDR = { "X-NEXA-Client": "1", "Content-Type": "application/json" };
const DEFAULT_RX = "dx6ran.ddns.net:8073";              // DU6/PE1NSQ, Negros, Philippines (team default)
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
          <select id="kwClip"><option value="3">3 s</option><option value="5" selected>5 s</option><option value="10">10 s</option><option value="30">30 s</option><option value="60">60 s</option></select></label>
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
        <button class="secondary" id="kwLocate" title="Records this frequency on up to 5 GPS-timed KiwiSDRs at once and
          works out where the transmitter is from the arrival-time differences (TDoA)">Locate transmitter</button>
        <label class="field">Sample
          <select id="kwTdoaS"><option value="10">10 s</option><option value="15" selected>15 s</option>
            <option value="30">30 s</option></select></label>
        <label class="field" title="Repeat the recording and combine: a far receiver often switches between one and two
          ionospheric hops, and any run that catches the direct path corrects the others (14 Nov: one run 440 km off,
          three runs 11 km)">Runs
          <select id="kwTdoaRuns"><option value="1">1 (quickest)</option><option value="2">2</option>
            <option value="3" selected>3 (most accurate)</option></select></label>
        <label class="field" style="min-width:300px;">TDoA receivers (host:port, comma-separated)
          <input id="kwTdoaRx" spellcheck="false" placeholder="empty: NEXA picks (Jakarta, Bangkok, Taiwan, … and further if needed)"></label>
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
    paintTdoa(await call("locate", { freq_khz: khz, sample_s: Number($("kwTdoaS").value), runs: Number($("kwTdoaRuns").value),
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
    : t.state === "done" && t.result ? runSummary(t.result, t.receivers || [])
    : "";
  const key = t.state === "done" && t.result ? t.result.at : null;
  if (key && key !== lastTdoa) { lastTdoa = key; showLocation(mounted, true); }
}

// What the last run did, in sentences; the numbers behind it folded away.
function rxName(id, places) {
  const p = (places || {})[id];
  return p ? `${esc(id)} (${esc(p)})` : esc(id);
}

function runSummary(r, receivers) {
  const used = (r.stations || []).map(s => rxName(s.id, r.places));
  const lines = [`<div>Last run ${esc(r.at)}: ${used.length} receivers heard it${r.runs > 1 ? `, ${r.runs} recordings
    combined` : ""}: ${used.join(", ")}.</div>`];
  if (r.further?.length) lines.push(`<div>The usual receivers were not enough, so NEXA also used ${r.further.map(esc).join(", ")}.</div>`);
  for (const x of r.excluded || []) {
    lines.push(`<div>Left out ${rxName(x.id, r.places)}: its timing disagreed with the others (by about ${Math.abs(x.offset_us)} µs).</div>`);
  }
  const off = (receivers || []).filter(c => !c.ok);
  if (off.length) {
    lines.push(`<div>Could not use: ${off.map(c => `${esc(c.id)} (${esc(c.why)})`).join("; ")}.</div>`);
  }
  const tech = `<details style="margin-top:6px;"><summary style="cursor:pointer;">Technical details</summary>
    <div style="margin-top:4px;">Arrival-time differences (how much later the first receiver heard it):</div>
    <ul style="margin:4px 0 0 18px;padding:0;">${(r.pairs || []).map(p => `<li>${esc(p.a)} − ${esc(p.b)}: ${p.lag_us} µs
      ± ${p.sigma_us}${p.agree != null ? `, ${Math.round(p.agree * 100)} % of seconds agree` : ""}${p.paths > 1
      ? ", two sky-wave paths seen (the earlier kept)" : ""}</li>`).join("")}</ul>
    ${r.dropped_pairs?.length ? `<div style="margin-top:4px;">Ignored (contradicted the other pairs):
      ${r.dropped_pairs.map(p => `${esc(p.a)} − ${esc(p.b)}`).join(", ")}.</div>` : ""}
    <div style="margin-top:4px;">Fit ${r.fit} (0 = all agree)${r.warning ? ` · ${esc(r.warning)}` : ""}.</div></details>`;
  return lines.join("") + tech;
}

const SURE = {
  good: ["Good", "#276749", "4 or more receivers agree on one place."],
  rough: ["Rough", "#B7791F", "the receivers do not fully agree; see below."],
  unchecked: ["Not cross-checked", "#9B2C2C", "only 3 receivers: a timing error would go unseen."],
};
const fmtLat = v => `${Math.abs(v).toFixed(2)}° ${v >= 0 ? "N" : "S"}`;
const fmtLon = v => `${Math.abs(v).toFixed(2)}° ${v >= 0 ? "E" : "W"}`;
const mapLink = (lat, lon) => `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=7/${lat}/${lon}`;

// The newest location on record for the frequency in the box (computed by NEXA, or typed in by the analyst).
let locKhz = null;
let locMissing = null;           // the frequency the box last found NO location for
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
  const lat = Number(loc.lat), lon = Number(loc.lon), r = Number(loc.radius_km);
  const sure = SURE[loc.sure];
  const alt = loc.alternative;
  const computed = (loc.source || "").startsWith("NEXA");
  box.innerHTML = `
    <div style="font-size:1.05em;"><b>${esc(loc.freq_khz)} kHz is most likely transmitted near
      ${fmtLat(lat)}, ${fmtLon(lon)}</b>${r > 0 ? ` (within about ${r} km)` : ""} ·
      <a href="${esc(mapLink(lat, lon))}" target="_blank" rel="noopener noreferrer">show on a map</a></div>
    ${sure ? `<div style="margin-top:6px;"><span style="display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;
      background:${sure[1]};font-size:0.9em;">${sure[0]}</span> <span class="note">${esc(sure[2])}</span></div>` : ""}
    ${alt ? `<div style="margin-top:6px;">Other possible place: <b>${fmtLat(Number(alt.lat))}, ${fmtLon(Number(alt.lon))}</b>
      (${esc(alt.apart_km)} km away) · <a href="${esc(mapLink(alt.lat, alt.lon))}" target="_blank" rel="noopener noreferrer">map</a>
      <div class="note">${esc(alt.why)}. Run it again (or use Runs 3) to tell them apart.</div></div>` : ""}
    <div class="note" style="margin-top:6px;">${computed ? "Worked out by NEXA" : esc(loc.source || "Recorded")} from
      ${(loc.receivers || "").split(", ").filter(Boolean).map(id => rxName(id, loc.places)).join(", ") || "—"}
      on ${esc(new Date(loc.ts).toLocaleString())} (${esc(loc.actor)}). Short-wave bounces off the ionosphere, so this
      is the centre of an area, not a point.</div>`;
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
    : `<b>Listening</b> to ${esc(kiwi)}${st.on_air
        ? ` · signal on air, recording clip ${Math.round(st.recording_s)} / ${Math.round(st.clip_s || 5)} s
           <progress max="1" value="${Math.min(1, st.recording_s / (st.clip_s || 5))}" style="width:110px;vertical-align:middle;"></progress>`
        : " · quiet"}`
      + (st.analysing ? ` · ${st.analysing} clip${st.analysing > 1 ? "s" : ""} being decoded` : "")
      + (st.skipped ? ` · ${st.skipped} not transcribed while busy (audio kept)` : "") + tdoa;
  // One row per clip, newest first, updated in place: the audio player is made once
  // (as soon as the clip's audio exists) and never redrawn, so playback is not cut
  // off while the clip's text is still being worked out.
  const list = $("kwList");
  const rows = [...(st.decodes || [])].reverse();
  const keys = rows.map(d => d.id || `err-${d.at}`);
  [...list.querySelectorAll(":scope > .kwRow")].forEach(el => { if (!keys.includes(el.dataset.key)) el.remove(); });
  list.querySelector(":scope > .kwEmpty")?.remove();
  let prev = null;
  rows.forEach((d, i) => {
    let el = list.querySelector(`:scope > .kwRow[data-key="${CSS.escape(keys[i])}"]`);
    if (!el) {
      el = document.createElement("div");
      el.className = "kwRow";
      el.dataset.key = keys[i];
      el.style.cssText = "padding:10px 0;border-top:1px solid var(--grid);";
      el.innerHTML = `<div class="kwInfo"></div><div class="kwAudio"></div>`;
    }
    if ((prev ? prev.nextSibling : list.firstChild) !== el) list.insertBefore(el, prev ? prev.nextSibling : list.firstChild);
    prev = el;
    const info = clipInfo(d);
    if (el.dataset.info !== info) { el.querySelector(".kwInfo").innerHTML = info; el.dataset.info = info; }
    const box = el.querySelector(".kwAudio");
    if (!box.firstChild && d.id && d.tracks?.some(t => t.audio)) box.innerHTML = players(d);
  });
  if (!rows.length) {
    list.insertAdjacentHTML("beforeend", `<div class="note kwEmpty">${st.state === "streaming"
      ? `Each clip appears here as soon as it is recorded (every ${Math.round(st.clip_s || 5)} s, or when a transmission
         ends): audio first, then what was said.`
      : "No clips yet. Choose a frequency and press Start listening."}</div>`);
  }
}

// Each clip's demodulated audio (results/pipeline/jobs/<id>/track_N.wav): play it here or save the .wav.
function players(d) {
  return [...new Set(d.tracks.map(t => t.audio).filter(Boolean))].map(a => {
    const src = `/api/pipeline/${encodeURIComponent(d.id)}/${encodeURIComponent(a)}`;
    return `<div style="display:flex;gap:10px;align-items:center;margin-top:6px;flex-wrap:wrap;">
      <audio controls preload="none" style="flex:1;min-width:220px;max-width:480px;" src="${src}"></audio>
      <a href="${src}" download="kiwi_${esc(d.id)}_${esc(a)}">save .wav</a></div>`;
  }).join("");
}

// What a clip row says: kinds and frequencies, where it is in the decoding (with
// how long it has been there), then the text and the AI reading once done.
const STAGE_HINT = { "saving audio": "demodulating", "waiting for speech-to-text": "the audio is ready below" };
function clipInfo(d) {
  if (d.error) return `<div class="note" style="color:#9B2C2C;">${esc(d.at)} · decode failed: ${esc(d.error)}</div>`;
  const tracks = d.tracks || [];
  const busy = d.stage && !["done", "not transcribed"].includes(d.stage);
  const head = `<div><b>${esc(d.at)}</b> · ${d.seconds} s · ${tracks.length ? tracks.map(t =>
      `${esc(KIND[t.kind] || t.kind)} ${d.centre_hz ? ((d.centre_hz / 1e3 + (t.f_lo_khz + t.f_hi_khz) / 2)).toFixed(1) + " kHz" : ""}`).join(", ")
      : busy ? "" : "nothing above the noise"}${d.warning ? ` <span class="note">(${esc(d.warning)})</span>` : ""}</div>`;
  const stage = busy ? `<div class="note" style="margin-top:4px;display:flex;gap:8px;align-items:center;">
      <progress style="width:90px;"></progress> ${esc(d.stage)}${STAGE_HINT[d.stage] ? ` (${STAGE_HINT[d.stage]})` : ""}
      · ${Number(d.stage_s) || 0} s</div>` : "";
  const said = tracks.filter(t => t.text).map(t => `<div style="margin-top:6px;"><b>${t.kind === "CW" ? "Morse" : ["RTTY", "PSK31"].includes(t.kind) ? "Text" : "Said"}${t.language ? ` (${esc(t.language)})` : ""}:</b>
      <span style="font-family:var(--mono);">${esc(t.text)}</span></div>
    ${t.reading ? `<div style="margin-top:4px;"><b>AI reading:</b> ${esc(t.reading)}</div>` : ""}
    ${!busy && d.id && t.track != null ? `<button class="mini" style="margin-top:6px;" data-open-analysis="${esc(`${d.id}_track_${t.track}`)}">Open analysis</button>` : ""}`).join("");
  const skipped = d.stage === "not transcribed"
    ? `<div class="note" style="margin-top:6px;">Not transcribed: a newer clip came in while the laptop was busy. The audio is kept.</div>` : "";
  const failed = d.stage === "not transcribed" ? "" : tracks.filter(t => t.text_error && !t.text).map(t => `<div class="note" style="margin-top:6px;color:#9B2C2C;">
      ${esc(KIND[t.kind] || t.kind)}: not decoded (${esc(t.text_error)})${/memory|alloc/i.test(t.text_error)
        ? ". This laptop ran out of memory: choose a smaller speech model (small or base), or close other programs." : ""}</div>`).join("");
  return head + stage + said + skipped + failed;
}
