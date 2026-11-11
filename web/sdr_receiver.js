// Receiver source 3: the real antenna, through SDRconnect.
//
//   RSPduo + antenna -> SDRconnect (ws://<host>:5454)
//     -> the local server (scripts/sdr_hub.py, ONE connection shared by every page)
//     -> this source: GET /api/sdr/dwell hands the newest slice of IQ, already
//        resampled to the model's 3.2 MS/s, so the waterfall, detector and
//        classifier run on it exactly as on the simulated receiver.
//
// Same interface as SimulatedReceiver / FileReplay in receiver.js:
// describe(), connect(onStep), next() -> block | null, disconnect().
//
// The server ALSO decodes in the background: it records each transmission
// until the air goes quiet, names its analog kind, demodulates and runs Morse
// or Whisper. Those results arrive through status() -> decodes, and land in
// Signal Pipeline, Comms Intel and the audit trail on their own.
//
// Owned by the SDR-connection half of feat/live-sdr-console
// (docs/LIVE_SDR_BRANCH.md). No DOM here.

import { FS } from "./model.js";

const HDR = { "X-NEXA-Client": "1", "Content-Type": "application/json" };
const wait = ms => new Promise(r => setTimeout(r, ms));

async function api(path, opts = {}) {
  const r = await fetch(`/api/sdr/${path}`, { cache: "no-store", ...opts });
  if (!r.ok && r.status !== 204) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).error || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r;
}

export class SdrConnectReceiver {
  /**
   * host/port: the laptop running SDRconnect (127.0.0.1 = this one).
   * decode: also record + decode each transmission on the server (Morse / Whisper).
   * onDecode(entry): called once for every new decoded transmission.
   */
  constructor({ dwellS, host = "127.0.0.1", port = 5454, device = "primary", decode = true,
                whisperModel = "small", language = null, onDecode = () => {}, timeoutS = 25 }) {
    this.dwellS = dwellS;
    this.opts = { host, port, device, decode, whisper_model: whisperModel, language };
    this.onDecode = onDecode;
    this.timeoutS = timeoutS;
    this.index = 0;
    this.after = null;            // the server's sample count at the last dwell: only new air is shown
    this.st = null;
    this.seenDecodes = new Set();
    this.lastStatusAt = 0;
    this.closed = false;
  }

  describe() {
    const s = this.st || {};
    return {
      kind: "sdr", simulated: false, name: "SDRconnect (live antenna)",
      hardware: s.active_device ? `SDRplay ${s.active_device}` : "SDRplay RSPduo via SDRconnect",
      serial: null, link: `${this.opts.host}:${this.opts.port} · WebSocket API ${s.api_version || "?"}`,
      centreHz: s.centre_hz ?? null,
      // The page always gets 3.2 MS/s; the stream's own rate is in the log and status.
      sampleRate: FS, streamRate: s.fs ?? null, gainDb: null, totalS: null,
    };
  }

  /** The server's view of the stream (rate, centre, recording, decodes). */
  async status() {
    this.st = await (await api("status")).json();
    this.lastStatusAt = performance.now();
    for (const d of this.st.decodes || []) {
      const key = d.id || `${d.at}-${d.error}`;
      if (this.seenDecodes.has(key)) continue;
      this.seenDecodes.add(key);
      if (this.primed) this.onDecode(d);
    }
    this.primed = true;           // decodes from before this connect are not announced again
    return this.st;
  }

  async connect(onStep = () => {}) {
    onStep(`Connecting to SDRconnect at ${this.opts.host}:${this.opts.port}…`);
    this.st = await (await api("connect", { method: "POST", headers: HDR, body: JSON.stringify(this.opts) })).json();
    await this.status();
    const deadline = performance.now() + this.timeoutS * 1000;
    let said = false;
    while (this.st.state === "connecting") {
      if (this.closed) return;
      if (performance.now() > deadline) throw new Error("no IQ from SDRconnect: press play / start the radio there");
      if (!said && this.st.total_samples === 0 && performance.now() > deadline - (this.timeoutS - 4) * 1000) {
        onStep("Connected; waiting for IQ (is the radio started in SDRconnect?)…");
        said = true;
      }
      await wait(400);
      await this.status();
    }
    if (this.st.state !== "streaming") throw new Error(this.st.error || `SDRconnect: ${this.st.state}`);
    const s = this.st;
    onStep(`Found ${this.describe().hardware} · API ${s.api_version || "?"}`);
    onStep(`IQ stream ${(s.fs / 1e3).toFixed(1)} kS/s`
      + (s.fs < FS ? ` (shown at ${(FS / 1e6).toFixed(1)} MS/s; outside ±${(s.fs / 2e3).toFixed(0)} kHz is filler)` : ""));
    onStep(s.centre_hz ? `Tuned: ${(s.centre_hz / 1e6).toFixed(4)} MHz (${s.centre_src})` : "Centre frequency not reported");
    onStep(s.decode?.decode
      ? `Decoding on: each transmission is recorded until quiet, then Morse / Whisper (${s.decode.whisper_model})`
      : "Decoding off: waterfall and classifier only");
    onStep(`IQ stream open · ${(this.dwellS * 1000).toFixed(0)} ms dwells`);
    this.index = 0;
  }

  async next() {
    for (;;) {
      if (this.closed) return null;
      if (performance.now() - this.lastStatusAt > 1000) {
        await this.status();
        if (this.st.state === "error") throw new Error(this.st.error || "SDRconnect stream lost");
        if (this.st.state === "off") return null;
      }
      const q = new URLSearchParams({ s: String(this.dwellS) });
      if (this.after != null) q.set("after", String(this.after));
      const r = await api(`dwell?${q}`);
      const total = Number(r.headers.get("X-NEXA-SDR-Total") || 0);
      if (r.status === 204) {
        const state = r.headers.get("X-NEXA-SDR-State");
        if (state === "off") return null;
        if (state === "error") { await this.status(); throw new Error(this.st.error || "SDRconnect stream lost"); }
        await wait(50);           // nothing new on the air yet
        continue;
      }
      const f = new Float32Array(await r.arrayBuffer());
      const n = f.length >> 1;
      const re = new Float32Array(n), im = new Float32Array(n);
      for (let i = 0; i < n; i++) { re[i] = f[2 * i]; im[i] = f[2 * i + 1]; }
      this.after = total;
      const i = this.index++;
      const s = this.st || {};
      const rec = s.on_air ? ` · on air, recording ${s.recording_s} s` : s.analysing ? ` · decoding ${s.analysing}` : "";
      return {
        index: i, re, im, truth: null, source: "sdr",
        caseNote: `LIVE · SDRconnect ${s.centre_hz ? (s.centre_hz / 1e6).toFixed(4) + " MHz" : ""} · dwell ${i + 1}${rec}`,
        snrDb: null, snrCapped: false, requestedSnrDb: null, startS: null, endS: null,
      };
    }
  }

  disconnect() {
    this.closed = true;
    // Fire and forget: the page is already closing the stream.
    fetch("/api/sdr/disconnect", { method: "POST", headers: HDR, body: "{}" }).catch(() => {});
  }
}

/** One line for the receiver log from a decoded transmission. */
export function decodeLine(d) {
  if (d.error) return `Decode failed: ${d.error}`;
  if (!d.tracks?.length) return `Recording ${d.seconds} s (${d.why}): nothing above the noise`;
  const heard = d.tracks.map(t => `${t.kind} ${t.f_lo_khz >= 0 ? "+" : ""}${t.f_lo_khz.toFixed(1)} kHz`
    + (t.text ? ` "${t.text.slice(0, 60)}"` : "")).join("; ");
  return `Decoded ${d.seconds} s: ${heard}${d.intel_ids?.length ? " · saved to Comms Intel" : ""}`;
}
