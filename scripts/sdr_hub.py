"""
Live SDRconnect feed for the dashboard: the RF Replay page's Receiver,
source "SDRconnect (live antenna)".

    RSPduo + antenna -> SDRconnect (ws://<host>:5454, IQ stream)
        -> this hub, inside serve_local.py (ONE connection, shared by every page)
             |-- GET /api/sdr/dwell   newest slice of IQ, resampled to 3.2 MS/s:
             |                        the page draws the waterfall and runs the
             |                        detector + classifier on it, as for the
             |                        simulated receiver
             |-- decode worker        records each transmission until the air goes
                                      quiet, names its analog kind (NBFM / AM / SSB /
                                      CW / RTTY ...), demodulates, and runs Morse or
                                      Whisper. Each one is saved as a live pipeline
                                      run (results/pipeline/jobs) and, when it holds
                                      text, as a Comms Intel recording -- so it shows
                                      in Signal Pipeline, Comms Intel and the audit
                                      trail like any other run.

Everything stays on this machine / local network. Reuses the bridge's tested
parts (scripts/live_sdrconnect.py): the WebSocket receiver, the IQ ring
buffer + disk spool, the rate measurement, the channel finder and the voice
path. Owner: the SDR-connection half of feat/live-sdr-console (see
docs/LIVE_SDR_BRANCH.md).
"""
import datetime
import json
import queue
import re
import secrets
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_sdrconnect as bridge  # noqa: E402
from src.kiwiclient import find_kiwirecorder, not_found as kiwi_not_found  # noqa: E402
from src import kiwi_tdoa  # noqa: E402
from src.pipeline import iqio  # noqa: E402

DWELL_FS = iqio.FS                         # what the page's model expects (3.2 MS/s)
HOST_RE = re.compile(r"[A-Za-z0-9.\-:\[\]]{1,253}")
WHISPER_MODELS = ("base", "small", "medium")
CHECK_S = 1.0                              # how often the decoder looks for a signal on air
IDLE_KEEP_S = 3.0                          # quiet air kept on disk before it is dropped


def tail(buf, n):
    """The newest n samples of buf without joining the whole ring (a 4 s ring
    at a few MS/s is tens of MB). -> (samples or None, total samples so far)."""
    with buf.lock:
        got, parts = 0, []
        for c in reversed(buf.chunks):
            parts.append(c)
            got += len(c)
            if got >= n:
                break
        total = buf.total
    if got < n or n <= 0:
        return None, total
    return np.concatenate(parts[::-1])[-n:], total


def level_db(x):
    p = float(np.mean(np.abs(x) ** 2)) if len(x) else 0.0
    return 10 * np.log10(p) if p > 0 else None


class SdrHub:
    """The one connection to SDRconnect. Thread-safe; every method returns quickly."""

    def __init__(self, jobs_dir=None, intel_dir=None, decode=True, quiet_s=2.0, max_s=60.0, keep_runs=40):
        self.jobs_dir = Path(jobs_dir or bridge.JOBS)
        self.intel_dir = Path(intel_dir) if intel_dir else None
        self.decode_default = decode
        self.quiet_s, self.max_s, self.keep_runs = quiet_s, max_s, keep_runs
        self.lock = threading.Lock()
        self.gen = 0                       # bumped on every connect/disconnect: old threads see it and stop
        self.buf = self.stop = None
        self.state, self.error, self.url, self.device = "off", None, None, "primary"
        self.fs = self.centre = None
        self.centre_src = "unknown"
        self.opts = {}
        self.on_air = False
        self.rec_s = 0.0
        self.label, self.skipped, self.fixed_fs = None, 0, None
        self.decodes = deque(maxlen=20)    # newest last
        self.analysing = 0
        self.work = queue.Queue()
        self.on_result = None              # serve_local sets this to write the audit entry
        threading.Thread(target=self._analyse_worker, daemon=True, name="sdr-decode").start()

    # ---------- control ----------

    def connect(self, host="127.0.0.1", port=5454, device="primary", decode=None,
                whisper_model="small", language=None, use_whisper=True, use_llm=False):
        host = str(host or "127.0.0.1").strip()
        if not HOST_RE.fullmatch(host):
            raise ValueError("host must be an IP address or a machine name")
        port = int(port)
        if not 0 < port < 65536:
            raise ValueError("port must be 1..65535")
        if device not in ("primary", "secondary"):
            raise ValueError("device must be primary or secondary (RSPduo tuner)")
        if whisper_model not in WHISPER_MODELS:
            raise ValueError(f"whisper_model must be one of {list(WHISPER_MODELS)}")
        from src.intel import voice
        if language in ("", "auto"):
            language = None
        if language is not None and language not in voice.LANGUAGES:
            raise ValueError(f"language must be one of {list(voice.LANGUAGES)}")
        url = f"ws://{host if ':' not in host or host.startswith('[') else f'[{host}]'}:{port}"
        opts = {"decode": self.decode_default if decode is None else bool(decode),
                "whisper_model": whisper_model, "language": language,
                "use_whisper": bool(use_whisper), "use_llm": bool(use_llm)}
        return self._open(url, device, f"SDRconnect {url}", bridge.sdrconnect_receiver, (url, device), opts)

    def _open(self, url, device, label, receiver, args, opts, fixed_fs=None):
        """Start (or join) one stream: receiver(buf, *args) fills the buffer in a thread."""
        with self.lock:
            if self.state in ("connecting", "streaming") and self.url == url and self.device == device:
                return self._status_locked()           # a second page joins the running stream
            self._stop_locked()
            self.gen += 1
            gen = self.gen
            self.buf, self.stop = bridge.IqBuffer(keep_s=4.0), threading.Event()
            self.state, self.error, self.url, self.device, self.label = "connecting", None, url, device, label
            self.fs = self.centre = None
            self.centre_src, self.on_air, self.rec_s, self.skipped = "unknown", False, 0.0, 0
            self.opts, self.fixed_fs = opts, fixed_fs
            buf, stop = self.buf, self.stop
            threading.Thread(target=receiver, args=(buf, *args), daemon=True, name="sdr-rx").start()
            threading.Thread(target=self._start, args=(gen, buf, stop), daemon=True, name="sdr-start").start()
            return self._status_locked()

    def disconnect(self):
        with self.lock:
            self.gen += 1
            self._stop_locked()
            self.state, self.error = "off", None
            return self._status_locked()

    def _stop_locked(self):
        if self.stop is not None:
            self.stop.set()
        ws = getattr(self.buf, "ws", None) if self.buf is not None else None
        if ws is not None:
            try:
                ws.close()
            except Exception:                          # noqa: BLE001 -- already closed
                pass
        proc = getattr(self.buf, "proc", None) if self.buf is not None else None
        if proc is not None and proc.poll() is None:   # a KiwiSDR client process
            proc.kill()

    def _start(self, gen, buf, stop):
        if self.fixed_fs:                              # the source states its rate (KiwiSDR): wait for samples only
            end = time.monotonic() + 30
            while buf.t_first is None and buf.error is None and time.monotonic() < end and not stop.is_set():
                time.sleep(0.1)
            fs = self.fixed_fs if buf.t_first is not None else None
        else:
            fs = bridge.settle_rate(buf, wait_s=1.5)
        with self.lock:
            if gen != self.gen:
                return
            if buf.error or not fs:
                self.state = "error"
                if self.fixed_fs:                      # KiwiSDR: its own words, never SDRconnect's
                    self.error = buf.error or ("the KiwiSDR sent no signal in 30 s: it may be busy or slow, or this "
                                               "network may be blocking it. Try again, another receiver, or a phone hotspot.")
                else:
                    self.error = buf.error or ("connected, but no IQ in 20 s: press play / start the radio in SDRconnect"
                                               if buf.info.get("connected") else
                                               "no answer in 20 s: is SDRconnect running, with Settings -> "
                                               "WebSocket interface on?")
                self._stop_locked()
                return
            buf.fs = self.fs = fs
            self.centre, self.centre_src = bridge.centre_of(buf, fs)
            self.state = "streaming"
            decode = self.opts["decode"]
        if decode:
            self._record_loop(gen, buf, fs, stop)

    # ---------- what the page reads ----------

    def status(self):
        with self.lock:
            return self._status_locked()

    def _status_locked(self):
        buf = self.buf
        if buf is not None and self.state == "streaming" and buf.error:
            self.state, self.error = "error", buf.error
        info = dict(buf.info) if buf is not None else {}
        if buf is not None and self.fs:
            self.centre, self.centre_src = bridge.centre_of(buf, self.fs)   # follows a retune in SDRconnect
        return {
            "state": self.state, "error": self.error, "url": self.url, "device": self.device,
            "fs": self.fs, "fs_measured": buf.rate_measured if buf is not None else None, "dwell_fs": DWELL_FS,
            "centre_hz": self.centre, "centre_src": self.centre_src,
            "api_version": info.get("api_version"), "active_device": info.get("active_device"),
            "demodulator": info.get("demodulator"), "device_sample_rate": info.get("device_sample_rate"),
            "total_samples": buf.total if buf is not None else 0,
            "decode": dict(self.opts), "on_air": self.on_air, "recording_s": round(self.rec_s, 1),
            "analysing": self.analysing, "skipped": getattr(self, "skipped", 0), "label": getattr(self, "label", None),
            "decodes": list(self.decodes),
        }

    def dwell(self, seconds=0.05, after=None):
        """The newest `seconds` of IQ at DWELL_FS. -> (complex64 array, meta), or
        (None, meta) when nothing new has arrived since `after` (a total count
        from the previous dwell) or the stream is not up."""
        seconds = min(max(float(seconds), 0.005), 0.5)
        with self.lock:
            buf, fs, state = self.buf, self.fs, self.state
        meta = {"state": state, "fs_in": fs, "total": buf.total if buf is not None else 0}
        if state != "streaming" or buf is None or not fs:
            return None, meta
        if after is not None and buf.total <= int(after):
            return None, meta
        n_in = max(64, int(np.ceil(seconds * fs)))
        x, total = tail(buf, n_in)
        meta["total"] = total
        if x is None:
            return None, meta
        y = iqio.to_fs(x, fs)
        n_out = int(round(seconds * DWELL_FS))
        y = y[:n_out] if len(y) >= n_out else np.pad(y, (0, n_out - len(y)))
        meta.update(level_db=level_db(x), centre_hz=self.centre, upsampled=fs < DWELL_FS - 1)
        return y.astype(np.complex64), meta

    # ---------- decode: record each transmission, then analyse it ----------

    def _record_loop(self, gen, buf, fs, stop):
        bridge.SPOOL.mkdir(parents=True, exist_ok=True)
        buf.rotate_spool(bridge.new_spool())
        checked, heard_at, started = buf.total, None, None
        label = self.label
        while not stop.wait(0.25) and not buf.error:
            now = time.monotonic()
            if buf.total - checked < CHECK_S * fs:
                continue
            checked = buf.total
            x, _ = tail(buf, int(CHECK_S * fs))
            if x is not None and bridge.find_channels(x, fs):
                heard_at = now
                started = started or now
            on_air = heard_at is not None and now - heard_at <= self.quiet_s
            low_disk = shutil.disk_usage(bridge.SPOOL).free < bridge.MIN_FREE_BYTES
            with self.lock:
                if gen != self.gen:
                    break
                self.on_air, self.rec_s = on_air, (buf.spool_n / fs if heard_at else 0.0)
            if heard_at is None:
                if buf.spool_n > IDLE_KEEP_S * fs or low_disk:      # nothing on the air: drop the quiet IQ
                    path, _ = buf.rotate_spool(bridge.new_spool())
                    if path:
                        path.unlink(missing_ok=True)
                continue
            why = ("no signal for %.0f s" % self.quiet_s if not on_air
                   else f"{self.max_s:.0f} s limit" if buf.spool_n >= self.max_s * fs
                   else "disk nearly full" if low_disk else None)
            if why:
                path, n = buf.rotate_spool(bridge.new_spool())
                self._queue(path, n, fs, why, buf, label)
                heard_at = started = None
        path, n = buf.rotate_spool(None)                 # disconnected: what was heard so far is still decoded
        if heard_at is not None:
            self._queue(path, n, fs, "receiver disconnected", buf, label)
        elif path:
            path.unlink(missing_ok=True)
        with self.lock:
            if gen == self.gen:
                self.on_air, self.rec_s = False, 0.0

    def _queue(self, path, n, fs, why, buf, label):
        if not path:
            return
        if n < fs:                                       # a fragment, not a transmission
            path.unlink(missing_ok=True)
            return
        centre, _ = bridge.centre_of(buf, fs)
        problem = bridge.delivery_problem(n, fs, **buf.ended)
        while True:                                      # slower than the air: only the newest clip waits
            try:
                old = self.work.get_nowait()
            except queue.Empty:
                break
            old[0].unlink(missing_ok=True)
            with self.lock:
                self.analysing -= 1
                self.skipped += 1
        with self.lock:
            self.analysing += 1
            opts = dict(self.opts)
        self.work.put((path, fs, centre, why, label, problem, opts))

    def _analyse_worker(self):
        while True:
            job = self.work.get()
            try:
                if not self.work.empty():                # analysis is slower than the air: always take the newest
                    with self.lock:
                        self.skipped += 1
                    continue
                self._analyse(*job)
            except Exception as e:                       # noqa: BLE001 -- the next transmission must still be decoded
                with self.lock:
                    self.decodes.append({"id": None, "at": time.strftime("%H:%M:%S"), "error": f"{type(e).__name__}: {e}"})
            finally:
                job[0].unlink(missing_ok=True)
                with self.lock:
                    self.analysing -= 1

    def _analyse(self, path, fs, centre, why, label, problem, opts):
        from src.intel import store
        rec = bridge.DiskIq(path)
        try:
            rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
            d = self.jobs_dir / rid
            d.mkdir(parents=True)
            meta = {"id": rid, "source": "live", "file": f"LIVE · {label}", "max_s": round(len(rec) / fs, 1),
                    "detector": "live voice path", "status": "running", "live": True, "who": "live receiver"}
            (d / "job.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            done = threading.Event()
            threading.Thread(target=bridge.touch_until, args=(d / "alive", done), daemon=True).start()
            t0 = time.monotonic()
            try:
                tracks = bridge.voice_snapshot(rec, fs, d, opts["whisper_model"], opts["language"],
                                               opts["use_whisper"], opts["use_llm"], opts.get("llm_model"))
                bridge.plot_snapshot(rec, fs, tracks, d / "tracks.png", centre)
                source = {"file": f"LIVE · {label}",
                          "format": f"{label} IQ stream (dashboard receiver)" + (f" · WARNING: {problem}" if problem else ""),
                          "fs_in": fs, "centre_hz": centre, "resampled": False, "live": True, "snapshot_at": rid,
                          "delivery_problem": problem, "ended_because": why}
                doc = bridge.finite({"n_samples": len(rec), "duration_ms": len(rec) / fs * 1e3, "source": source,
                                     "tracks": tracks, "truth": None})
                (d / "tracks.tmp").write_text(json.dumps(doc, indent=1, default=float, allow_nan=False), encoding="utf-8")
                (d / "tracks.tmp").replace(d / "tracks.json")
                kw = {"out": self.intel_dir} if self.intel_dir else {}
                saved = store.save_decoded_tracks(rid, doc["source"], doc["tracks"], d, keep_empty=True, **kw)
            finally:
                done.set()
            took = time.monotonic() - t0
            (d / "run.log").write_text(f"{label} recording {len(rec) / fs:.1f} s at {fs / 1e3:.1f} kS/s "
                                       f"({why}), analysed in {took:.1f} s\n", encoding="utf-8")
            def reading(L):
                r = L.get("reading") or {}
                return (r.get("translation_en") or r.get("summary") or "")[:300] if not r.get("error") else ""
            entry = {"id": rid, "at": time.strftime("%H:%M:%S"), "seconds": round(len(rec) / fs, 1), "why": why,
                     "centre_hz": centre, "intel_ids": list(saved or []), "warning": problem,
                     "tracks": [{"kind": t["listen"]["kind"], "answer": t["answer"],
                                 "f_lo_khz": round(t["f_lo_khz"], 2), "f_hi_khz": round(t["f_hi_khz"], 2),
                                 "text": (t["listen"].get("transcript") or t["listen"].get("morse")
                                          or t["listen"].get("digital_text") or "")[:300],
                                 "language": t["listen"].get("language"), "reading": reading(t["listen"])}
                                for t in tracks]}
            with self.lock:
                self.decodes.append(entry)
            if self.on_result:
                self.on_result(entry)
            self._prune()
        finally:
            rec.close()

    def _prune(self):
        live = []
        for x in self.jobs_dir.iterdir():
            try:
                if json.loads((x / "job.json").read_text(encoding="utf-8")).get("live"):
                    live.append(x)
            except (OSError, ValueError):
                continue
        for old in sorted(live)[:-self.keep_runs]:
            shutil.rmtree(old, ignore_errors=True)


# ---------- KiwiSDR: a public online receiver as a continuous live source ----------

LLM_MODELS = ("llama3.2:3b", "qwen2.5:1.5b")   # local Ollama models: better / lighter (~2.5 GB / ~1 GB RAM)
KIWI_FS = 12000.0                          # a KiwiSDR IQ channel (about +/-6 kHz)
KIWIRECORDER = find_kiwirecorder()           # NEXA_KIWIRECORDER, else ~/tools/kiwiclient, D:/tools/kiwiclient, ...
KIWI_DEFAULT = ("21439.proxy2.kiwisdr.com", 8073)   # 9M2AMH, Kuala Lumpur (GPS-timed, used for TDoA)


def kiwi_unreachable(host, port, timeout=8):
    """None when this machine can open a connection to the KiwiSDR, else why not, in words."""
    import socket
    try:
        socket.create_connection((host, int(port)), timeout=timeout).close()
        return None
    except OSError as e:
        return (f"cannot reach the KiwiSDR {host}:{port} from this network ({type(e).__name__}). "
                "Some networks block KiwiSDR: try a phone hotspot, or another receiver.")


def kiwi_receiver(buf, host, port, khz, user="NEXA"):
    """Stream IQ from a KiwiSDR with the official client (kiwiclient, --nc: raw
    little-endian int16 I/Q on stdout) into buf until the process ends. Runs in a thread."""
    import subprocess
    recorder = KIWIRECORDER if KIWIRECORDER.exists() else find_kiwirecorder()   # cloned since the server started
    if not recorder.exists():
        buf.error = kiwi_not_found(recorder)
        return
    blocked = kiwi_unreachable(host, port)       # a blocked network otherwise looks like a silent receiver
    if blocked:
        buf.error = blocked
        return
    cmd = [sys.executable, str(recorder), "-s", host, "-p", str(port), "-f", f"{khz:g}", "-m", "iq",
           "--nc", "-q", "--user", user]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    except OSError as e:
        buf.error = f"could not start kiwirecorder: {e}"
        return
    buf.proc = proc
    buf.info.update({"connected": "yes", "device_center_frequency": str(khz * 1e3),
                     "device_sample_rate": str(KIWI_FS), "active_device": f"KiwiSDR {host}",
                     "api_version": "kiwiclient"})
    err = deque(maxlen=20)
    threading.Thread(target=lambda: [err.append(line.decode("utf-8", "replace").strip()) for line in proc.stderr],
                     daemon=True).start()
    rest = b""
    while True:
        chunk = proc.stdout.read1(16384)
        if not chunk:
            break
        data = rest + chunk
        n = len(data) // 4 * 4
        raw = np.frombuffer(data[:n], "<i2").astype(np.float32) / 32768
        rest = data[n:]
        if len(raw):
            buf.add((raw[0::2] + 1j * raw[1::2]).astype(np.complex64))
    proc.wait()
    said = " ".join(err).lower()
    buf.error = buf.error or ("the KiwiSDR is full (no free channel): try again in a minute" if "too many" in said
                              or "busy" in said else
                              f"KiwiSDR stream ended: {err[-1] if err else f'kiwirecorder exit {proc.returncode}'}")


class KiwiHub(SdrHub):
    """Continuous live analysis of one frequency on a KiwiSDR: every clip_s seconds
    (or when a transmission ends) the IQ is decoded (Morse / RTTY / PSK31 / Whisper)
    and read by the local AI, then saved like any live run."""

    def __init__(self, jobs_dir=None, intel_dir=None, clip_s=30.0):
        super().__init__(jobs_dir=jobs_dir, intel_dir=intel_dir, decode=True, quiet_s=2.0, max_s=clip_s)
        self.tdoa = TdoaRunner()                     # "Locate transmitter": runs beside the listening stream

    def connect(self, host=KIWI_DEFAULT[0], port=KIWI_DEFAULT[1], freq_khz=None, whisper_model="small",
                language=None, use_llm=True, clip_s=30.0, llm_model=None):
        host = str(host or KIWI_DEFAULT[0]).strip()
        if not HOST_RE.fullmatch(host):
            raise ValueError("host must be a KiwiSDR address, e.g. 21439.proxy2.kiwisdr.com")
        port = int(port or KIWI_DEFAULT[1])
        if not 0 < port < 65536:
            raise ValueError("port must be 1..65535")
        khz = float(freq_khz or 0)
        if not 10 <= khz <= 30000:
            raise ValueError("frequency must be 10..30000 kHz (a KiwiSDR covers shortwave only)")
        if whisper_model not in WHISPER_MODELS:
            raise ValueError(f"whisper_model must be one of {list(WHISPER_MODELS)}")
        from src.intel import voice
        language = None if language in (None, "", "auto") else language
        if language is not None and language not in voice.LANGUAGES:
            raise ValueError(f"language must be one of {list(voice.LANGUAGES)}")
        if llm_model is not None and llm_model not in LLM_MODELS:
            raise ValueError(f"llm_model must be one of {list(LLM_MODELS)}")
        self.max_s = min(max(float(clip_s or 30), 10.0), 120.0)
        opts = {"decode": True, "whisper_model": whisper_model, "language": language,
                "use_whisper": True, "use_llm": bool(use_llm), "freq_khz": khz,
                "llm_model": llm_model}
        key = f"kiwi://{host}:{port}/{khz:g}"
        return self._open(key, "kiwi", f"KiwiSDR {host} {khz:g} kHz", kiwi_receiver, (host, port, khz), opts,
                          fixed_fs=KIWI_FS)

    def status(self):
        st = super().status()
        st["tdoa"] = self.tdoa.status()
        return st


# ---------- KiwiSDR TDoA: where the transmitter is, from 3 GPS-timed KiwiSDRs ----------
# KiwiSDR's own TDoA service needs an access string each KiwiSDR gives only to its
# own web page, so NEXA records the receivers itself and computes the place with
# src/kiwi_tdoa.py (the hcab14/TDoA method). Public receivers are shared: one run
# at a time, short samples, 3 receivers, and a pause between runs.

TDOA_RECEIVERS = (                                  # GPS-timed KiwiSDRs round Malaysia, wide geometry, in order of use
    ("dx6ran.ddns.net", 8073, "DU6/PE1NSQ"),        # Negros, Philippines
    ("21023.proxy.kiwisdr.com", 8073, "HS0ZGZ"),    # Thailand
    ("rajawali.proxy.kiwisdr.com", 8073, "JKT"),    # Jakarta, Indonesia
    ("21439.proxy2.kiwisdr.com", 8073, "9M2AMH"),   # Kuala Lumpur (spare)
)
TDOA_MAX_RX = 3                                     # 4 often timed out on KiwiSDR's own TDoA page
TDOA_SAMPLE_S = (10, 15, 30)
TDOA_PAUSE_S = 60.0                                 # between runs: these receivers are shared
TDOA_WHY = {1: "connection failed", 2: "busy (no free channel)", 3: "no GPS time stamps"}   # kiwirecorder --kiwi-tdoa


def kiwi_status(host, port, timeout=8):
    """The text of a KiwiSDR's /status page (position, GPS, free channels)."""
    import urllib.request
    with urllib.request.urlopen(f"http://{host}:{int(port)}/status", timeout=timeout) as r:
        return r.read(20000).decode("utf-8", "replace")


def parse_receivers(text):
    """'host:port, host:port' (empty: the default list) -> [(host, port, label)]."""
    if not text:
        return list(TDOA_RECEIVERS)
    out = []
    for item in re.split(r"[\s,;]+", str(text).strip()):
        if not item:
            continue
        host, _, port = item.partition(":")
        if not HOST_RE.fullmatch(host) or (port and not port.isdigit()):
            raise ValueError(f"not a KiwiSDR address: {item!r} (use host:port, comma-separated)")
        out.append((host, int(port or 8073), None))
    if not 3 <= len(out) <= 6:
        raise ValueError("give 3 to 6 KiwiSDRs (3 are used; the rest are spares)")
    return out


class TdoaRunner:
    """One background TDoA run at a time: check the receivers, record them all at
    once with GPS time stamps, compute the place. on_located(result, who) is called
    with each result (serve_local writes it to the audit trail)."""

    def __init__(self, workdir=None):
        self.workdir = Path(workdir) if workdir else None
        self.lock = threading.Lock()
        self.st = {"state": "idle"}
        self.finished_at = None
        self.on_located = None

    def status(self):
        with self.lock:
            st = dict(self.st)
        t0 = st.pop("rec_started", None)
        if st.get("state") == "recording" and t0:
            st["progress"] = round(min(0.95, (time.monotonic() - t0) / (st["sample_s"] + 8)), 2)
        return st

    def start(self, freq_khz, receivers=None, sample_s=15, who=None):
        khz = float(freq_khz or 0)
        if not 10 <= khz <= 30000:
            raise ValueError("frequency must be 10..30000 kHz")
        sample_s = int(sample_s or 15)
        if sample_s not in TDOA_SAMPLE_S:
            raise ValueError(f"sample must be one of {list(TDOA_SAMPLE_S)} s")
        cands = parse_receivers(receivers)
        with self.lock:
            if self.st.get("state") in ("checking", "recording", "computing"):
                raise ValueError("a location run is already going: wait for it to finish")
            wait = TDOA_PAUSE_S - (time.monotonic() - self.finished_at) if self.finished_at else 0
            if wait > 0:
                raise ValueError(f"the public receivers are shared: wait {wait:.0f} s before the next run")
            self.st = {"state": "checking", "freq_khz": khz, "sample_s": sample_s, "progress": 0.0,
                       "message": f"checking {len(cands)} KiwiSDRs (GPS, free channel)…",
                       "started": time.strftime("%H:%M:%S"), "receivers": []}
        threading.Thread(target=self._run, args=(khz, cands, sample_s, who), daemon=True, name="kiwi-tdoa").start()
        return self.status()

    def _set(self, **kw):
        with self.lock:
            self.st.update(kw)

    def _check(self, host, port, label):
        name = label or host.split(".")[0]
        if kiwi_unreachable(host, port):
            return {"host": host, "port": port, "id": name, "ok": False, "why": "not reachable from this network"}
        try:
            st = kiwi_tdoa.parse_status(kiwi_status(host, port))
        except Exception as e:                       # noqa: BLE001 -- any failure: this receiver is skipped
            return {"host": host, "port": port, "id": name, "ok": False, "why": f"no /status ({type(e).__name__})"}
        why = kiwi_tdoa.status_problem(st)
        return {"host": host, "port": port, "id": st["tdoa_id"] or name, "ok": why is None,
                "why": why, "lat": st["lat"], "lon": st["lon"], "name": st["name"]}

    def _record(self, use, khz, sample_s, tmp):
        """kiwirecorder records every receiver at once, GPS-stamped. -> {index: problem}"""
        import subprocess
        recorder = KIWIRECORDER if KIWIRECORDER.exists() else find_kiwirecorder()
        if not recorder.exists():
            raise RuntimeError(kiwi_not_found(recorder))
        cmd = [sys.executable, str(recorder), "-s", ",".join(c["host"] for c in use),
               "-p", ",".join(str(c["port"]) for c in use), "-f", f"{khz:g}", "-m", "iq",
               "--kiwi-wav", "--kiwi-tdoa", "--tlimit", str(sample_s), "-d", str(tmp),
               "--station", ",".join(c["sid"] for c in use), "--user", "NEXA-TDoA",
               "--connect-timeout", "10", "--connect-retries", "1", "--busy-retries", "1"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags, cwd=str(tmp))
        try:
            out, err = proc.communicate(timeout=sample_s + 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        codes = {int(m.group(1)): int(m.group(2))
                 for m in re.finditer(r"status=(\d+),(\d+)", out.decode("utf-8", "replace"))}
        last = (err.decode("utf-8", "replace").strip().splitlines() or [""])[-1]
        return {i: TDOA_WHY[c] for i, c in codes.items() if c in TDOA_WHY}, last

    def _run(self, khz, cands, sample_s, who):
        import concurrent.futures as cf
        import tempfile
        tmp = None
        try:
            with cf.ThreadPoolExecutor(len(cands)) as ex:
                checked = list(ex.map(lambda c: self._check(*c), cands))
            use = [c for c in checked if c["ok"]][:TDOA_MAX_RX]
            self._set(receivers=checked)
            if len(use) < 3:
                blocked = any(c["why"] == "not reachable from this network" for c in checked)
                raise RuntimeError("fewer than 3 KiwiSDRs can take part: "
                                   + "; ".join(f"{c['id']} {c['why']}" for c in checked if not c["ok"])
                                   + (". Some networks block KiwiSDR: try a phone hotspot." if blocked else ""))
            for i, c in enumerate(use):                  # kiwirecorder puts the station id in each file name
                c["sid"] = (re.sub(r"[^0-9A-Za-z]", "", c["id"]) or "rx") + str(i)
            base = self.workdir or bridge.SPOOL
            base.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix="tdoa-", dir=base))
            self._set(state="recording", rec_started=time.monotonic(),
                      message=f"recording {khz:g} kHz on {', '.join(c['id'] for c in use)} for {sample_s} s…")
            problems, last_err = self._record(use, khz, sample_s, tmp)
            self._set(state="computing", progress=0.97, message="computing arrival-time differences…")
            recs, failed = [], []
            for i, c in enumerate(use):
                wav = next(iter(sorted(tmp.glob(f"*_{c['sid']}_iq.wav"))), None)
                if i in problems or wav is None:
                    failed.append(f"{c['id']} {problems.get(i, 'sent nothing')}")
                    continue
                try:
                    t, z, fs = kiwi_tdoa.read_kiwi_wav(wav)
                except ValueError as e:
                    failed.append(f"{c['id']}: {e}")
                    continue
                recs.append({"id": c["id"], "lat": c["lat"], "lon": c["lon"], "t": t, "z": z, "fs": fs})
            if len(recs) < 3:
                raise RuntimeError("fewer than 3 usable recordings: " + "; ".join(failed + ([last_err] if last_err else [])))
            shutil.rmtree(tmp, ignore_errors=True)       # the samples are in memory now
            tmp = None
            res = kiwi_tdoa.solve(recs)
            res.update(freq_khz=khz, sample_s=sample_s, receivers=", ".join(r["id"] for r in recs),
                       at=time.strftime("%Y-%m-%d %H:%M:%S"))
            if self.on_located:                          # in the audit trail before the page hears "done"
                self.on_located(res, who)
            final = {"state": "done", "progress": 1.0, "result": res, "message": None}
        except Exception as e:                           # noqa: BLE001 -- shown on the page
            final = {"state": "error", "message": None,
                     "error": str(e) if isinstance(e, (RuntimeError, ValueError)) else f"{type(e).__name__}: {e}"}
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        with self.lock:
            self.finished_at = time.monotonic()
            self.st.update(final)
