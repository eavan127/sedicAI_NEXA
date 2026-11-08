"""
Live analysis of what SDRconnect is receiving (or playing back), on this or
another laptop on the same local network. Nothing leaves that network:
SDRconnect's WebSocket API (port 5454) -> this script -> Whisper / Ollama on
this machine.

    SDRconnect (laptop 1, RSPduo or an IQ file playing)
        --  ws://<laptop1>:5454, IQ stream (int16 I/Q)  -->
    this script (laptop 2): keeps the last few seconds, and every cycle takes a
    snapshot, finds the transmissions in it, names the analog kind (NBFM / AM /
    SSB / CW), demodulates, transcribes (Whisper) and optionally reads it
    (Ollama). Each snapshot is saved as an ordinary pipeline run in
    results/pipeline/jobs/, so the dashboard's Signal Pipeline page shows it
    (tick "Follow live" there: it opens every new snapshot by itself).

Modes
  voice (default)  fast path, seconds per snapshot: channel finder + zoom rules
                   + demodulator + Whisper. Names analog voice / Morse only.
  full             the whole wideband chain (YOLO + 60-class model + tracker +
                   listen) on the snapshot. ~50 s of CPU per second of IQ on a
                   laptop without a GPU: use short snapshots (--snapshot-s 0.5).

    # SDRconnect on this laptop
    python scripts/live_sdrconnect.py --language en
    # SDRconnect on another laptop (find its address with ipconfig)
    python scripts/live_sdrconnect.py --host 192.168.137.1 --language ms --whisper-model small
    # no SDRconnect at hand: replay an IQ file as if it were live
    python scripts/live_sdrconnect.py --replay data/local/military_radio/military_net.wav

SDRconnect 1.0.6 or newer (WebSocket API 1.0.3). The API does not state
whether the IQ stream is the whole device band or a narrower channel, so the
sample rate is MEASURED from the stream (override with --fs).
"""
import argparse
import collections
import datetime
import json
import secrets
import sys
import threading
import time
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT  # noqa: E402
from src.pipeline import analog, demod  # noqa: E402

JOBS = REPO_ROOT / "results" / "pipeline" / "jobs"
MSG_IQ_PRIMARY, MSG_IQ_SECONDARY = 2, 5
COMMON_RATES = [48e3, 62.5e3, 96e3, 125e3, 192e3, 250e3, 500e3, 1e6, 1.536e6, 2e6, 2.048e6, 2.4e6, 3e6, 3.2e6,
                4e6, 5e6, 6e6, 8e6, 10e6]
ZOOM_FS = analog.FA                       # 50 kS/s, what the zoom rules and the demodulator expect
KIND_CLASS = {"NBFM": "NBFM", "WBFM": "WBFM", "AM": "AM_DSB_WC", "AM_SC": "AM_DSB_SC",
              "USB": "AM_SSB_SC", "LSB": "AM_SSB_SC", "CW": "CW", "CARRIER": "CARRIER"}


# ---------- IQ sources ----------

class IqBuffer:
    """Ring buffer of complex samples, filled by a receiver thread."""

    def __init__(self, keep_s=10.0):
        self.keep_s = keep_s
        self.chunks = collections.deque()
        self.n = 0
        self.total = 0
        self.t_first = None
        self.lock = threading.Lock()
        self.fs = None
        self.info = {}
        self.error = None

    def add(self, iq):
        with self.lock:
            if self.t_first is None:
                self.t_first = time.monotonic()
            self.chunks.append(iq)
            self.n += len(iq)
            self.total += len(iq)
            limit = int(self.keep_s * (self.fs or 3.2e6))
            while self.n - len(self.chunks[0]) > limit:
                self.n -= len(self.chunks.popleft())

    def measured_rate(self):
        with self.lock:
            if self.t_first is None:
                return None
            dt = time.monotonic() - self.t_first
            return self.total / dt if dt > 0 else None

    def last(self, n):
        with self.lock:
            if self.n < n:
                return None
            out = np.concatenate(list(self.chunks))
        return out[-n:]


def parse_binary(msg, device="primary"):
    """SDRconnect binary frame -> complex64 IQ, or None (audio / spectrum /
    other device). 2-byte little-endian type, then int16 I,Q,I,Q..."""
    if len(msg) < 6:
        return None
    kind = int.from_bytes(msg[:2], "little")
    if kind != (MSG_IQ_PRIMARY if device == "primary" else MSG_IQ_SECONDARY):
        return None
    raw = np.frombuffer(msg, dtype="<i2", offset=2)
    raw = raw[:len(raw) // 2 * 2].astype(np.float32) / 32768
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def command(event_type, value="", prop="", device=None):
    m = {"event_type": event_type, "property": prop, "value": str(value)}
    if device:
        m["device"] = device
    return json.dumps(m)


def sdrconnect_receiver(buf, url, device="primary"):
    """Connect, ask for the radio's settings, switch the IQ stream on, and
    keep filling buf. Runs in a thread."""
    from websockets.sync.client import connect
    try:
        with connect(url, max_size=None, open_timeout=10) as ws:
            for prop in ("api_version", "device_sample_rate", "device_center_frequency", "device_vfo_frequency",
                         "active_device", "demodulator"):
                ws.send(command("get_property", prop=prop, device=device))
            ws.send(command("iq_stream_enable", "true"))
            for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    iq = parse_binary(msg, device)
                    if iq is not None:
                        buf.add(iq)
                else:
                    try:
                        j = json.loads(msg)
                    except ValueError:
                        continue
                    if j.get("event_type") in ("get_property_response", "property_changed") and j.get("property"):
                        buf.info[j["property"]] = j.get("value")
    except Exception as e:                                 # noqa: BLE001 -- shown by the main loop
        buf.error = f"{type(e).__name__}: {e}"


def replay_receiver(buf, path, speed=1.0, loop=True):
    """An IQ WAV played into the buffer at its own pace, as SDRconnect would."""
    with wave.open(str(path), "rb") as w:
        fs, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if ch != 2 or width != 2:
        buf.error = f"{path}: need a 2-channel int16 IQ WAV"
        return
    raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    buf.info.update({"device_sample_rate": str(fs), "active_device": f"replay of {Path(path).name}"})
    step = int(fs * 0.05)
    t0 = time.monotonic()
    sent = 0
    while True:
        for s in range(0, len(iq), step):
            buf.add(iq[s:s + step])
            sent += step
            wait = t0 + sent / fs / speed - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        if not loop:
            return


def settle_rate(buf, given=None, wait_s=2.0):
    """The stream's sample rate: --fs, else measured over wait_s and snapped
    to the device rate or a common rate."""
    if given:
        return float(given)
    time.sleep(wait_s)
    r = buf.measured_rate()
    if not r:
        return None
    dev = float(buf.info.get("device_sample_rate") or 0)
    if dev and abs(r - dev) / dev < 0.1:
        return dev
    best = min(COMMON_RATES, key=lambda c: abs(c - r) / c)
    return best if abs(best - r) / best < 0.1 else round(r, -2)


# ---------- voice path ----------

def find_channels(iq, fs, thr_db=8.0, merge_hz=3e3, min_hz=200.0, max_hz=300e3, max_n=8):
    """Transmissions in a snapshot: frequency spans whose average power sits
    thr_db above the median (noise) level. -> [(lo_hz, hi_hz, peak_db)]."""
    nfft = int(min(max(256, 2 ** np.ceil(np.log2(fs / 250))), 1 << 16))
    f, p = welch(iq, fs=fs, nperseg=nfft, return_onesided=False, scaling="density")
    order = np.argsort(f)
    f, p = f[order], p[order]
    db = 10 * np.log10(p + 1e-20)
    floor = np.median(db)
    hot = db > floor + thr_db
    df = f[1] - f[0]
    grow = max(int(merge_hz / 2 / df), 1)
    hot = np.convolve(hot.astype(int), np.ones(2 * grow + 1, int), mode="same") > 0
    out = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            lo, hi = f[i] + grow * df, f[j - 1] - grow * df
            if hi - lo < min_hz:
                c = (lo + hi) / 2
                lo, hi = c - min_hz / 2, c + min_hz / 2
            if hi - lo <= max_hz:
                out.append((float(lo), float(hi), float(db[i:j].max() - floor)))
            i = j
        else:
            i += 1
    return sorted(out, key=lambda c: -c[2])[:max_n]


def zoom_to(iq, fs, centre_hz):
    """Shift centre_hz to 0 and resample to the 50 kS/s zoom rate."""
    y = iq * np.exp(-2j * np.pi * centre_hz * np.arange(len(iq)) / fs)
    r = Fraction(ZOOM_FS / fs).limit_denominator(1000)
    return resample_poly(y, r.numerator, r.denominator)


def voice_snapshot(iq, fs, out_dir, whisper_model, language, use_whisper, use_llm, llm_model):
    """-> tracks (dashboard schema) for one snapshot, audio files written."""
    tracks = []
    dur_ms = len(iq) / fs * 1e3
    for k, (lo, hi, peak) in enumerate(find_channels(iq, fs), 1):
        y = zoom_to(iq, fs, (lo + hi) / 2)
        kind, feats, why = analog.classify(y, ZOOM_FS)
        L = {"kind": kind, "why": why, "duration_s": round(len(y) / ZOOM_FS, 3), "tracks": [k],
             "t_start_ms": 0.0, "t_end_ms": dur_ms}
        if kind not in ("UNKNOWN", "CARRIER"):
            sound = demod.demodulate(y, ZOOM_FS, kind, feats)
            if sound is not None:
                from src.intel import audio as wavio
                wav = out_dir / f"track_{k}.wav"
                wavio.write_wav(wav, sound, demod.AUDIO_FS, bits=16)
                L["audio"] = wav.name
                segments, text_kind = [], None
                try:
                    if kind == "CW":
                        from src.intel import morse
                        m = morse.analyse(wavio.wav_info(wav))
                        segments, text_kind = m["segments"], "morse"
                        L["morse"] = " / ".join(s["text"] for s in segments)
                    elif use_whisper and L["duration_s"] >= 1.0:
                        from src.intel import voice
                        tr = voice.transcribe(wavio.wav_info(wav), model=whisper_model, max_minutes=1.0,
                                              language=language)
                        L["transcript"], L["language"] = tr.get("text", ""), tr.get("language")
                        segments = [{"start_s": s["clip_start_s"], "text": s["text"]} for s in tr.get("segments", [])]
                        text_kind = "voice"
                except Exception as e:                     # noqa: BLE001 -- keep the audio
                    L["text_error"] = f"{type(e).__name__}: {e}"
                if segments and use_llm:
                    from src.intel import reason
                    kw = {"model": llm_model} if llm_model else {}
                    L["reading"] = reason.read_recording(f"live track {k}", segments, kind=text_kind,
                                                         duration=f"{L['duration_s']:.1f} s", **kw)
        conf = 0.9 if kind not in ("UNKNOWN",) else 0.1
        tracks.append({"track": k, "t_start_ms": 0.0, "t_end_ms": dur_ms, "f_lo_khz": lo / 1e3, "f_hi_khz": hi / 1e3,
                       "answer": KIND_CLASS.get(kind, kind), "confidence": conf,
                       "band": "high" if kind != "UNKNOWN" else "uncertain",
                       "reason": f"live zoom-in, {peak:.0f} dB above the noise: {why}",
                       "decided_by": "zoom", "classifier_answer": "not run (voice mode)", "top5": [], "listen": L})
    return tracks


def plot_snapshot(iq, fs, tracks, path, centre_hz=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    nfft = int(min(max(256, 2 ** np.ceil(np.log2(fs / 500))), 8192))
    f, t, s = spectrogram(iq, fs=fs, nperseg=nfft, noverlap=nfft // 2, return_onesided=False)
    order = np.argsort(f)
    img = 10 * np.log10(s[order] + 1e-20)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(img, aspect="auto", origin="lower", cmap="magma", extent=[0, t[-1] * 1e3, f[order][0] / 1e3, f[order][-1] / 1e3],
              vmin=np.percentile(img, 20), vmax=np.percentile(img, 99.8))
    for a in tracks:
        ax.add_patch(plt.Rectangle((a["t_start_ms"], a["f_lo_khz"]), a["t_end_ms"] - a["t_start_ms"],
                                   a["f_hi_khz"] - a["f_lo_khz"], fill=False, ec="#4cd4ff", lw=1.4))
        said = a.get("listen", {}).get("transcript") or a.get("listen", {}).get("morse") or ""
        ax.text(a["t_start_ms"] + 20, a["f_hi_khz"], f"#{a['track']} {a['listen']['kind']} {said[:40]}",
                color="#4cd4ff", fontsize=8, va="bottom")
    ax.set_xlabel("time in snapshot (ms)")
    ax.set_ylabel("offset (kHz)" + (f" from {centre_hz / 1e6:.4f} MHz" if centre_hz else ""))
    ax.set_title("Live snapshot: transmissions found (boxes) and what they are")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- full path ----------

class FullChain:
    def __init__(self, classifier_paths, detector_path, hybrid=True):
        from src.pipeline.classify import Classifier
        from src.pipeline.run import EnergyDetector, HybridDetector, NetDetector
        if detector_path and Path(detector_path).exists():
            det = NetDetector(detector_path)
            self.detector = HybridDetector(det) if hybrid else det
        else:
            self.detector = EnergyDetector()
        self.classifier = Classifier(classifier_paths)

    def __call__(self, iq, fs, out_dir, rng, **listen_kw):
        from src.pipeline import iqio, listen
        from src.pipeline.run import process_capture
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from run_pipeline import plot
        x = iqio.to_fs(iq, fs, rng)
        answers, _ = process_capture(x, self.detector, self.classifier, rng=rng)
        listen.listen_all(x, answers, out_dir, rng=rng, **listen_kw)
        plot(x, answers, None, out_dir / "tracks.png")
        return answers, len(x) / iqio.FS * 1e3


# ---------- main loop ----------

def finite(o):
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def centre_of(buf, fs):
    """Centre frequency of the stream: the LO when the stream is the whole
    device band, else (assumed) the VFO the stream is cut around."""
    dev = float(buf.info.get("device_sample_rate") or 0)
    lo = buf.info.get("device_center_frequency")
    vfo = buf.info.get("device_vfo_frequency")
    if dev and abs(fs - dev) / dev < 0.1 and lo:
        return float(lo), "device LO"
    if vfo:
        return float(vfo), "VFO (assumed: stream narrower than the device band)"
    return None, "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--host", default="127.0.0.1", help="laptop running SDRconnect (default: this one)")
    src.add_argument("--replay", help="IQ WAV to play as if live (no SDRconnect needed)")
    p.add_argument("--port", type=int, default=5454)
    p.add_argument("--device", choices=["primary", "secondary"], default="primary", help="RSPduo tuner")
    p.add_argument("--fs", type=float, default=None, help="IQ stream sample rate (default: measured)")
    p.add_argument("--mode", choices=["voice", "full"], default="voice")
    p.add_argument("--snapshot-s", type=float, default=None, help="seconds per snapshot (voice 4, full 0.5)")
    p.add_argument("--every-s", type=float, default=0.0, help="minimum time between snapshots")
    p.add_argument("--cycles", type=int, default=0, help="stop after this many snapshots (0 = run until Ctrl+C)")
    p.add_argument("--whisper-model", default="small", help="base (fast) / small / medium (~5 GB RAM)")
    p.add_argument("--language", default=None, help="en / ms / zh (default: Whisper guesses)")
    p.add_argument("--no-whisper", action="store_true")
    p.add_argument("--llm", action="store_true", help="add the Ollama reading (slower)")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--classifier", nargs="+", default=None, help="full mode: best.pt file(s)")
    p.add_argument("--detector", default=str(REPO_ROOT / "results" / "detector" / "emitter_detector.pt"))
    p.add_argument("--keep", type=int, default=40, help="live runs kept in results/pipeline/jobs (oldest deleted)")
    args = p.parse_args()
    snap_s = args.snapshot_s or (4.0 if args.mode == "voice" else 0.5)

    buf = IqBuffer(keep_s=max(10.0, 2 * snap_s))
    if args.replay:
        label = f"replay {Path(args.replay).name}"
        th = threading.Thread(target=replay_receiver, args=(buf, args.replay), daemon=True)
    else:
        url = f"ws://{args.host}:{args.port}"
        label = f"SDRconnect {url}"
        th = threading.Thread(target=sdrconnect_receiver, args=(buf, url, args.device), daemon=True)
    th.start()
    print(f"Listening to {label} ...", flush=True)
    fs = settle_rate(buf, args.fs)
    if buf.error or not fs:
        sys.exit(f"No IQ from {label}: {buf.error or 'nothing received in 2 s'}. "
                 "In SDRconnect: start the radio (or playback); SDRconnect 1.0.6+ is needed for the WebSocket API.")
    buf.fs = fs
    centre, centre_src = centre_of(buf, fs)
    print(f"IQ stream {fs / 1e3:.1f} kS/s ({buf.info.get('active_device', '?')}, API {buf.info.get('api_version', '?')})"
          + (f", centre {centre / 1e6:.4f} MHz ({centre_src})" if centre else ", centre frequency unknown"), flush=True)

    chain = None
    if args.mode == "full":
        cls = args.classifier or [str(c) for c in sorted((REPO_ROOT / "results" / "classifier").glob("*best*.pt"))]
        if not cls:
            sys.exit("full mode needs a classifier best.pt in results/classifier/")
        chain = FullChain(cls, args.detector)
    rng = np.random.default_rng(0)
    JOBS.mkdir(parents=True, exist_ok=True)
    cycle = 0
    while not args.cycles or cycle < args.cycles:
        t_cycle = time.monotonic()
        n = int(snap_s * fs)
        iq = buf.last(n)
        while iq is None:
            if buf.error:
                sys.exit(f"stream stopped: {buf.error}")
            time.sleep(0.1)
            iq = buf.last(n)
        rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        d = JOBS / rid
        d.mkdir(parents=True)
        meta = {"id": rid, "source": "live", "file": f"LIVE · {label}", "max_s": snap_s,
                "detector": "live voice path" if args.mode == "voice" else "live full chain", "status": "running",
                "live": True, "who": "live bridge"}
        (d / "job.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        t0 = time.monotonic()
        kw = dict(whisper_model=args.whisper_model, language=args.language, use_whisper=not args.no_whisper,
                  use_llm=args.llm, llm_model=args.llm_model)
        if chain is None:
            tracks = voice_snapshot(iq, fs, d, **kw)
            plot_snapshot(iq, fs, tracks, d / "tracks.png", centre)
            dur_ms = len(iq) / fs * 1e3
        else:
            tracks, dur_ms = chain(iq, fs, d, rng, **kw)
        source = {"file": f"LIVE · {label}", "format": f"SDRconnect IQ stream ({args.mode} mode)", "fs_in": fs,
                  "centre_hz": centre, "resampled": False, "live": True, "snapshot_at": rid}
        doc = finite({"n_samples": len(iq), "duration_ms": dur_ms, "source": source, "tracks": tracks, "truth": None})
        (d / "tracks.tmp").write_text(json.dumps(doc, indent=1, default=float, allow_nan=False), encoding="utf-8")
        (d / "tracks.tmp").replace(d / "tracks.json")       # the dashboard sees a run as done once this exists
        took = time.monotonic() - t0
        (d / "run.log").write_text(f"live snapshot {snap_s} s at {fs / 1e3:.1f} kS/s, analysed in {took:.1f} s\n",
                                   encoding="utf-8")
        heard = "; ".join(f"{t['listen']['kind']} {t['f_lo_khz']:+.1f}..{t['f_hi_khz']:+.1f} kHz"
                          + (f" \"{(t['listen'].get('transcript') or t['listen'].get('morse') or '')[:60]}\"")
                          for t in tracks) or "nothing above the noise"
        print(f"[{rid}] {took:5.1f} s  {heard}", flush=True)
        live = sorted(x for x in JOBS.iterdir() if (x / "job.json").exists()
                      and json.loads((x / "job.json").read_text(encoding="utf-8")).get("live"))
        for old in live[:-args.keep]:
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        cycle += 1
        rest = args.every_s - (time.monotonic() - t_cycle)
        if rest > 0:
            time.sleep(rest)


if __name__ == "__main__":
    main()
