// Live KiwiSDR listening: type a shortwave frequency, press Start, and the
// server (scripts/sdr_hub.py KiwiHub, /api/kiwi/*) streams that frequency from a
// KiwiSDR without stopping. Every clip (30 s, or when a transmission ends) is
// decoded -- Morse, RTTY, PSK31 or speech (Whisper) -- and read by the local AI,
// then saved as a live Signal Pipeline run and a Comms Intel recording.
// No command line needed. Mounted above the Signal Pipeline runs (main.js).

const HDR = { "X-NEXA-Client": "1", "Content-Type": "application/json" };
const DEFAULT_RX = "21439.proxy2.kiwisdr.com:8073";      // 9M2AMH, Kuala Lumpur (GPS-timed: TDoA works)
const esc = t => String(t ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const KIND = { AM: "AM voice", USB: "USB voice", LSB: "LSB voice", NBFM: "FM voice", CW: "Morse", RTTY: "RTTY",
  PSK31: "PSK31", CARRIER: "carrier only", UNKNOWN: "unclear" };

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
      <div class="block-label">Live KiwiSDR listening</div>
      <div class="note">Listens to one shortwave frequency without stopping. Every clip is decoded (Morse, RTTY,
        PSK31 or speech) and read by the local AI. Results appear here, in the runs below and in Comms Intel.</div>
      <div class="filters" style="margin-top:12px;flex-wrap:wrap;gap:12px 18px;align-items:flex-end;">
        <label class="field" style="min-width:230px;">Receiver
          <input id="kwRx" value="${DEFAULT_RX}" spellcheck="false"></label>
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
        <label style="display:inline-flex;gap:6px;align-items:center;"><input type="checkbox" id="kwLlm" checked> AI reading</label>
        <button class="primary" id="kwStart">Start listening</button>
        <button class="secondary" id="kwStop" disabled>Stop</button>
      </div>
      <div id="kwState" class="note" role="status" style="margin-top:10px;"></div>
      <div id="kwList" style="margin-top:8px;"></div>
    </div>`;
    const $ = id => root.querySelector(`#${id}`);
    $("kwStart").addEventListener("click", async () => {
      const [host, port] = $("kwRx").value.trim().split(":");
      $("kwStart").disabled = true;
      try {
        paint(await call("start", { host, port: Number(port) || 8073, freq_khz: Number($("kwFreq").value),
          language: $("kwLang").value, whisper_model: $("kwModel").value, clip_s: Number($("kwClip").value),
          llm: $("kwLlm").checked, llm_model: $("kwLlmModel").value }));
      } catch (e) {
        $("kwState").textContent = `Could not start: ${e.message}`;
        $("kwStart").disabled = false;
      }
    });
    $("kwStop").addEventListener("click", async () => paint(await call("stop", {}).catch(() => null)));
  }
  clearInterval(timer);
  const tick = async () => {
    if (!document.body.contains(root)) { clearInterval(timer); return; }
    try { paint(await call("status")); } catch { /* server restarting: next tick */ }
  };
  tick();
  timer = setInterval(tick, 2000);
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
  const tdoa = khz ? ` · <a href="http://${esc(DEFAULT_RX)}/?f=${Number(khz).toFixed(2)}iqz10" target="_blank"
    rel="noopener noreferrer">open on KiwiSDR for TDoA</a>` : "";
  $("kwState").innerHTML = st.state === "off" ? "Not listening."
    : st.state === "connecting" ? `Connecting to ${esc(kiwi)}…`
    : st.state === "error" ? `<span style="color:#9B2C2C;">Stopped: ${esc(st.error)}</span>`
    : `<b>Listening</b> to ${esc(kiwi)}${st.on_air ? ` · signal on air, clip ${st.recording_s} s` : " · quiet"}`
      + (st.analysing ? ` · decoding ${st.analysing}` : "")
      + (st.skipped ? ` · ${st.skipped} clip${st.skipped > 1 ? "s" : ""} skipped while busy` : "") + tdoa;
  const rows = [...(st.decodes || [])].reverse();
  $("kwList").innerHTML = rows.map(d => d.error
    ? `<div class="note" style="color:#9B2C2C;">${esc(d.at)} · decode failed: ${esc(d.error)}</div>`
    : `<div style="padding:10px 0;border-top:1px solid var(--grid);">
        <div><b>${esc(d.at)}</b> · ${d.seconds} s · ${d.tracks.length ? d.tracks.map(t =>
          `${esc(KIND[t.kind] || t.kind)} ${d.centre_hz ? ((d.centre_hz / 1e3 + (t.f_lo_khz + t.f_hi_khz) / 2)).toFixed(1) + " kHz" : ""}`).join(", ")
          : "nothing above the noise"}${d.warning ? ` <span class="note">(${esc(d.warning)})</span>` : ""}</div>
        ${d.tracks.filter(t => t.text).map(t => `<div style="margin-top:4px;font-family:var(--mono);">${esc(t.text)}</div>
          ${t.reading ? `<div style="margin-top:4px;"><b>AI:</b> ${esc(t.reading)}</div>` : ""}`).join("")}
      </div>`).join("") || (st.state === "streaming" ? `<div class="note">First result after the first clip (about ${Math.round(st.decode?.whisper_model === "medium" ? 90 : 60)} s).</div>` : "");
}
