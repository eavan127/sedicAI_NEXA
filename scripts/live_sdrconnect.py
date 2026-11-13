"""
Live analysis of what SDRconnect is receiving (or playing back), on this or
another laptop on the same local network. Nothing leaves that network:
SDRconnect's WebSocket API (port 5454) -> this script -> Whisper / Ollama on
this machine.

    SDRconnect (laptop 1, RSPduo or an IQ file playing)
        --  ws://<laptop1>:5454, IQ stream (int16 I/Q)  -->
    this script (laptop 2): records everything the stream sends, with no time
    limit, until the stream stops (the file ended or was paused), then finds
    the transmissions in the whole recording, names the analog kind (NBFM / AM
    / SSB / CW), demodulates, transcribes (Whisper) and optionally reads it
    (Ollama). Each recording is saved as an ordinary pipeline run in
    results/pipeline/jobs/, so the dashboard's Signal Pipeline page shows it
    (tick "Follow live" there: it opens every new run by itself).

    A live radio never stops sending: end a recording with Ctrl+C (it is still
    analysed), or use --quiet-stop-s to end it when the air goes quiet. While
    recording, the IQ goes to disk, not RAM; the recording ends early only if
    the disk is nearly full.

Modes
  voice (default)  channel finder + zoom rules + demodulator + Whisper. Names
                   analog voice / Morse only. --snapshot-s N analyses the newest
                   N seconds in a loop instead of whole recordings.
  full             the whole wideband chain (YOLO + 60-class model + tracker +
                   listen) on short snapshots only: ~50 s of CPU per second of
                   IQ on a laptop without a GPU (--snapshot-s 0.5).

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
import atexit
import collections
import datetime
import json
import re
import secrets
import shutil
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
from src.intel import store  # noqa: E402
from src.pipeline import analog, demod, narrowband  # noqa: E402

JOBS = REPO_ROOT / "results" / "pipeline" / "jobs"
SPOOL = REPO_ROOT / "results" / "pipeline" / "live_spool"   # recordings on their way to analysis, deleted after
MIN_FREE_BYTES = 500_000_000              # a recording ends here rather than fill the disk
BLOCK_S = 6.0                             # a long recording is searched this many seconds at a time
MAX_CLASSIFY_BLOCKS = 20                  # blocks of a channel measured to name its kind
MAX_PLOT_COLS = 1600                      # about the picture's width in pixels
MSG_IQ_PRIMARY, MSG_IQ_SECONDARY = 2, 5
REPLAY_ONCE = [False]                     # --replay-once: play the file once, like SDRconnect without loop
COMMON_RATES = [48e3, 62.5e3, 96e3, 125e3, 192e3, 250e3, 500e3, 1e6, 1.536e6, 2e6, 2.048e6, 2.4e6, 3e6, 3.2e6,
                4e6, 5e6, 6e6, 8e6, 10e6]
ZOOM_FS = analog.FA                       # 50 kS/s, what the zoom rules and the demodulator expect
VOICE_KINDS = {"NBFM", "WBFM", "AM", "AM_SC", "USB", "LSB"}   # what Whisper is run on (as src/pipeline/listen.py)


def kind_answer(kind, feats):
    """The zoom's kind as a 60-class answer, the same table as Signal Pipeline
    (src/pipeline/listen.py zoom_answer): CW -> "OOK (Morse CW)", a bare carrier ->
    "RADAR_CW / JAM_TONE", USB with a carrier -> AM_SSB_WC ..."""
    from src.pipeline import listen
    z = listen.zoom_answer(kind, feats or {})
    return z[0] if z else kind


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
        self.spool = None                  # open file that also receives every sample (whole recordings)
        self.spool_path = None
        self.spool_n = 0
        self.spool_first = self.spool_last = None   # when this recording's first and newest frames arrived
        self.spool_gap = 0.0                        # longest pause between two of its frames
        self.ended = {"span_s": 0.0, "longest_gap_s": 0.0}   # the same for the recording that just ended
        self.rate_measured = None

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
            if self.spool is not None:
                raw = np.empty(2 * len(iq), "<i2")
                raw[0::2] = np.clip(np.round(iq.real * 32768), -32768, 32767)
                raw[1::2] = np.clip(np.round(iq.imag * 32768), -32768, 32767)
                self.spool.write(raw.tobytes())
                self.spool_n += len(iq)
                now = time.monotonic()
                if self.spool_first is None:
                    self.spool_first = now
                else:
                    self.spool_gap = max(self.spool_gap, now - self.spool_last)
                self.spool_last = now

    def rotate_spool(self, next_path=None):
        """End the current recording and start the next, with no sample lost
        between them. -> (path, samples) of the recording that ended."""
        with self.lock:
            ended = self.spool_path, self.spool_n
            if self.spool is not None:
                self.spool.close()
            self.spool = open(next_path, "wb") if next_path else None
            self.spool_path, self.spool_n = next_path, 0
            self.ended = {"span_s": self.spool_last - self.spool_first if self.spool_first is not None else 0.0,
                          "longest_gap_s": self.spool_gap}
            self.spool_first, self.spool_last, self.spool_gap = None, None, 0.0
        return ended

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


class DiskIq:
    """A finished recording, read one slice at a time so that its length is
    not limited by RAM. iq[a:b] and len(iq) work as on an array."""

    def __init__(self, path):
        self.file = open(path, "rb")
        self.n = Path(path).stat().st_size // 4

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        a, b, _ = s.indices(self.n)
        self.file.seek(4 * a)
        raw = np.fromfile(self.file, dtype="<i2", count=2 * max(b - a, 0)).astype(np.float32) / 32768
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    def close(self):
        self.file.close()


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
        with connect(url, max_size=None, open_timeout=15) as ws:
            buf.ws = ws                                   # the auto-tuner sends its commands on this connection
            buf.info["connected"] = "yes"
            for prop in ("api_version", "device_sample_rate", "device_center_frequency", "device_vfo_frequency",
                         "active_device", "demodulator"):
                ws.send(command("get_property", prop=prop, device=device))
            # Ask for the IQ stream ONCE. Asking again while nothing plays crashed SDRconnect
            # (13 Nov: two APPCRASH in its swig_bindings.DLL within a minute), so do not repeat it.
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
        buf.error = buf.error or "SDRconnect closed the connection"
    except Exception as e:                                 # noqa: BLE001 -- shown by the main loop
        buf.error = f"{type(e).__name__}: {e}"


# ---------- auto-tune: put SDRconnect's VFO (the red line) on the signal ----------

# analog kind -> (SDRconnect demodulator, filter bandwidth Hz). API 1.0.3:
# demodulator AM/USB/LSB/CW/SAM/NFM/WFM, filter_bandwidth Uint32 Hz.
CW_MAX_HZ = 1200.0                        # wider than this is not Morse
SDR_MODE = {"NBFM": ("NFM", 12500), "WBFM": ("WFM", 150000), "AM": ("AM", 8000), "AM_SC": ("AM", 8000),
            "USB": ("USB", 2700), "LSB": ("LSB", 2700), "CW": ("CW", 500), "CARRIER": ("AM", 8000),
            "RTTY": ("USB", 2700), "PSK31": ("USB", 2700), "AFSK": ("USB", 2700)}


def tune_plan(iq, fs):
    """Strongest transmission in iq -> {kind, mode, bandwidth, offset_hz,
    peak_db}, or None. offset_hz is where the VFO belongs relative to the
    stream's centre: the carrier for AM/FM/CW, the suppressed carrier just
    below (USB) or above (LSB) the voice band for SSB."""
    ch = find_channels(iq, fs)
    if not ch:
        return None
    lo, hi, peak = ch[0]
    yz = zoom_to(iq, fs, (lo + hi) / 2)
    kind, _, why = analog.classify(yz, ZOOM_FS)
    kind, why, _, _ = narrowband.refine(kind, why, yz, ZOOM_FS)
    if kind == "UNKNOWN":
        return None
    if kind in ("CW", "CARRIER") and hi - lo > CW_MAX_HZ:
        # Morse is a few hundred Hz wide; a "keyed line" this wide is SSB
        # voice whose tones (a roger beep, a held vowel) look like one.
        kind, why = "USB", f"{(hi - lo) / 1e3:.1f} kHz wide with tone-like parts: single-sideband voice"
    mode, bw = SDR_MODE.get(kind, ("AM", 8000))
    off = lo - 300.0 if kind == "USB" else hi + 300.0 if kind == "LSB" else (lo + hi) / 2
    return {"kind": kind, "mode": mode, "bandwidth": bw, "offset_hz": float(off), "peak_db": peak, "why": why}


def tune_commands(plan, centre_hz, device="primary"):
    """The set_property messages that tune SDRconnect to the plan."""
    vfo = int(round(centre_hz + plan["offset_hz"]))
    return vfo, [command("set_property", plan["mode"], "demodulator", device),
                 command("set_property", plan["bandwidth"], "filter_bandwidth", device),
                 command("set_property", vfo, "device_vfo_frequency", device)]


def autotune_loop(buf, fs, device="primary", look_s=4.0, restart_after_s=2.0, once=False):
    """Each time the stream starts (or restarts after a pause, e.g. a new file
    or play pressed again), look at its first look_s seconds and tune
    SDRconnect onto the strongest signal. Runs in a thread."""
    # From the start of the stream: a short transmission may be over by the
    # time this thread starts (the rate is measured first).
    start_total, last_total, last_change = 0, buf.total, time.monotonic()
    tuned = False
    while not buf.error or (not tuned and buf.total - start_total >= int(0.5 * fs)):
        if buf.error and not tuned:
            look_s = min(look_s, (buf.total - start_total) / fs)   # stream over: tune on what came
        time.sleep(0.25)
        now = time.monotonic()
        if buf.total != last_total:
            if now - last_change > restart_after_s:          # data again after a pause: a new start
                start_total, tuned = last_total, False
            last_total, last_change = buf.total, now
        got = buf.total - start_total
        short_and_over = now - last_change > 1.0 and got >= int(0.5 * fs)   # a transmission shorter than look_s ended
        if tuned or (got < look_s * fs and not short_and_over):
            continue
        tuned = True
        plan = tune_plan(buf.last(int(min(look_s * fs, got))), fs)
        if plan is None:
            print("Auto-tune: no clear signal in the first seconds, VFO left where it is.", flush=True)
        else:
            centre, src = centre_of(buf, fs)
            if not centre or centre <= 0:
                # A played-back IQ file without a stored centre frequency is
                # shown by SDRconnect as 0 .. fs, centred on fs / 2.
                centre = fs / 2
            if True:
                vfo, cmds = tune_commands(plan, centre, device)
                ws = getattr(buf, "ws", None)
                if ws is None:                                 # --replay: nothing to tune, say what it would do
                    print(f"Auto-tune (replay, not sent): {plan['mode']} {plan['bandwidth'] / 1e3:g} kHz at "
                          f"{vfo / 1e3:.3f} kHz ({plan['kind']}, {plan['peak_db']:.0f} dB)", flush=True)
                else:
                    try:
                        for c in cmds:
                            ws.send(c)
                        print(f"Auto-tune: SDRconnect -> {plan['mode']}, filter {plan['bandwidth'] / 1e3:g} kHz, "
                              f"VFO {vfo / 1e3:.3f} kHz ({plan['kind']}, {plan['peak_db']:.0f} dB above the noise)",
                              flush=True)
                    except Exception as e:                     # noqa: BLE001 -- tuning is a convenience
                        print(f"Auto-tune: could not send to SDRconnect: {type(e).__name__}: {e}", flush=True)
        if once:
            buf.tuned_once = True
            return


def replay_receiver(buf, path, speed=1.0, loop=True):
    loop = loop and not REPLAY_ONCE[0]
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
    # KiwiSDR recordings (scripts/kiwi_collect.py: <guess>_<kHz>_<UTC>.wav, or kiwirecorder's
    # ..._<Hz>_iq.wav) carry their centre frequency in the name: the dashboard can then show
    # real frequencies and offer "Locate on KiwiSDR".
    m = re.search(r"_(\d{3,5}(?:\.\d+)?)_\d{8}T\d{6}Z", Path(path).name) or re.search(r"_(\d{6,9})_iq", Path(path).name)
    if m:
        hz = float(m.group(1)) * (1e3 if "." in m.group(1) or len(m.group(1)) <= 5 else 1.0)
        buf.info["device_center_frequency"] = str(int(round(hz)))
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


def snap_rate(r, dev=0.0):
    """Measured samples per second -> the stream's real rate. A stream cannot
    deliver faster than its rate for long but often delivers slower (a slow
    start, lost samples), so a common rate up to 20% above the measurement is
    accepted, and only 10% below it (a burst)."""
    if dev and abs(r - dev) / dev < 0.1:
        return dev
    fits = [c for c in COMMON_RATES if -0.1 < (c - r) / c < 0.2]
    return min(fits, key=lambda c: abs(c - r)) if fits else round(r, -2)


def delivery_problem(n, fs, span_s, longest_gap_s, min_span_s=2.0, max_lost=0.05, max_gap_s=0.5):
    """What went wrong while this recording arrived, in words, or None. n
    samples arrived over span_s seconds, and fs of them should arrive per second."""
    problems = []
    lost = 1 - n / (fs * span_s) if span_s >= min_span_s else 0.0
    if lost > max_lost:
        problems.append(f"about {100 * lost:.0f}% of the samples did not arrive "
                        f"({n / fs:.1f} s of IQ in {span_s:.1f} s of streaming)")
    if longest_gap_s > max_gap_s:
        problems.append(f"the stream paused for {longest_gap_s:.1f} s")
    return "; ".join(problems) or None


def settle_rate(buf, given=None, wait_s=2.0):
    """The stream's sample rate: --fs, else measured over wait_s and snapped
    to the device rate or a common rate."""
    if given:
        deadline = time.monotonic() + 3              # let SDRconnect's settings arrive, for the summary line
        while "api_version" not in buf.info and buf.error is None and time.monotonic() < deadline:
            time.sleep(0.1)
        return float(given)
    deadline = time.monotonic() + 20                 # connecting over Wi-Fi can take seconds
    while buf.t_first is None and buf.error is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if buf.t_first is None:
        return None
    time.sleep(wait_s)
    r = buf.measured_rate()
    if not r:
        return None
    buf.rate_measured = r
    dev = float(buf.info.get("device_sample_rate") or 0)
    # SDRconnect sends a played-back file in a fast burst at first (measured 390.7 kS/s for a
    # 250 kS/s file), so for a file its own rate is believed over the 2-second measurement.
    if dev and "file" in str(buf.info.get("active_device", "")).lower() and 0.5 < r / dev < 2.5:
        return dev
    return snap_rate(r, dev)


# ---------- voice path ----------

def spectrum_blocks(iq, fs, block_s=BLOCK_S):
    """Power spectrum of each block_s of iq (the last block takes the
    remainder). -> (freqs, dB[block, bin], block edges in samples)."""
    nfft = int(min(max(256, 2 ** np.ceil(np.log2(fs / 250))), 1 << 16))
    step = int(block_s * fs)
    edges = [i * step for i in range(max(1, len(iq) // step))] + [len(iq)]
    rows = []
    for a, b in zip(edges, edges[1:]):
        # detrend=False: a carrier exactly at the centre (a VFO-centred stream, an auto-tuned
        # signal) is the mean of each segment, and the default detrend would delete it.
        f, p = welch(iq[a:b], fs=fs, nperseg=nfft, return_onesided=False, scaling="density", detrend=False)
        rows.append(p)
    order = np.argsort(f)
    return f[order], 10 * np.log10(np.array(rows)[:, order] + 1e-20), edges


def when_on(f, blocks, lo, hi, thr_db=8.0):
    """The blocks in which the channel lo..hi is on air (at least its strongest one)."""
    i0 = min(int(np.searchsorted(f, lo)), len(f) - 1)
    i1 = max(int(np.searchsorted(f, hi)), i0 + 1)
    level = blocks[:, i0:i1].max(axis=1) - np.median(blocks, axis=1)
    on = np.flatnonzero(level > thr_db)
    return on if len(on) else np.array([int(np.argmax(level))])


def find_channels(iq, fs, thr_db=8.0, merge_hz=3e3, min_hz=200.0, max_hz=300e3, max_n=8, spectra=None):
    """Transmissions in a recording: frequency spans whose power sits thr_db
    above the median (noise) level in at least one block, so that a short
    transmission is not averaged away by a long recording.
    -> [(lo_hz, hi_hz, peak_db)]."""
    f, blocks, _ = spectra or spectrum_blocks(iq, fs)
    db = blocks.max(axis=0)
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
    """Shift centre_hz to 0 and resample to the 50 kS/s zoom rate, in
    overlapping blocks so that a long recording fits in memory."""
    r = Fraction(ZOOM_FS / fs).limit_denominator(1000)
    up, down = r.numerator, r.denominator
    step = (2 ** 21 // down + 1) * down                     # block starts on a multiple of down: seamless output
    margin = (20 * max(up, down) // up // down + 1) * down  # past the resampling filter's reach
    out = []
    for a in range(0, len(iq), step):
        s0, s1 = max(0, a - margin), min(len(iq), a + step + margin)
        x = iq[s0:s1] * np.exp(-2j * np.pi * centre_hz * np.arange(s0, s1) / fs)
        y = resample_poly(x, up, down)
        head = (a - s0) * up // down
        out.append(y[head:head + step * up // down].astype(np.complex64))
    return np.concatenate(out)


FORCE_KINDS = ("AM", "USB", "LSB", "NBFM", "CW")   # what an analyst can tell the voice path a signal is


def voice_snapshot(iq, fs, out_dir, whisper_model, language, use_whisper, use_llm, llm_model, force_kind=None):
    """-> tracks (dashboard schema) for one recording, audio files written.
    force_kind (one of FORCE_KINDS): the analyst knows what the signal is (e.g. AM
    for a short-wave broadcast), so it is demodulated that way whatever the
    measurement says; the measured kind is kept in the reason.
    The two halves can also run apart: voice_audio (seconds) then voice_text (slow)."""
    tracks, extra = voice_audio(iq, fs, out_dir, force_kind=force_kind)
    voice_text(tracks, extra, out_dir, whisper_model, language, use_whisper, use_llm, llm_model)
    return tracks


def voice_audio(iq, fs, out_dir, force_kind=None):
    """The fast half of voice_snapshot: find each channel, name its kind, read
    RTTY / PSK31, demodulate and write track_<k>.wav. No Morse, Whisper or AI
    reading yet. -> (tracks, extra): extra[k] keeps what voice_text needs."""
    if force_kind is not None and force_kind not in FORCE_KINDS:
        raise ValueError(f"force_kind must be one of {list(FORCE_KINDS)}")
    tracks, extra = [], {}
    spectra = spectrum_blocks(iq, fs)
    f, blocks, edges = spectra
    for k, (lo, hi, peak) in enumerate(find_channels(iq, fs, spectra=spectra), 1):
        y = zoom_to(iq, fs, (lo + hi) / 2)
        on = when_on(f, blocks, lo, hi)
        t_on_ms, t_off_ms = edges[on[0]] / fs * 1e3, edges[on[-1] + 1] / fs * 1e3
        # The kind is measured while the channel is on air: a long stretch of
        # silence would make any signal look too weak. The audio stays whole.
        picks = on[np.unique(np.linspace(0, len(on) - 1, MAX_CLASSIFY_BLOCKS).astype(int))]
        scale = ZOOM_FS / fs
        y_on = np.concatenate([y[int(edges[b] * scale):int(edges[b + 1] * scale)] for b in picks])
        kind, feats, why = analog.classify(y_on, ZOOM_FS)
        kind, why, demod_kind, slow = narrowband.refine(kind, why, y_on, ZOOM_FS)
        if force_kind:                                      # the analyst's choice wins; no RTTY / PSK31 guess
            why = (f"set by the analyst: {force_kind}" if kind == force_kind
                   else f"set by the analyst: {force_kind} (measured {kind}: {why})")
            kind = demod_kind = force_kind
            dig = None
        else:
            from src.intel import digital
            was = kind
            kind, dig = digital.read(kind, y, ZOOM_FS, slow)  # RTTY / PSK31 text (and a clean decode names the kind)
            if kind != was:
                why, demod_kind = dig["why"], "USB"
        L = {"kind": kind, "why": why, "duration_s": round(len(y) / ZOOM_FS, 3), "tracks": [k],
             "t_start_ms": t_on_ms, "t_end_ms": t_off_ms}
        if slow:
            L["slow_path"] = slow
        if kind not in ("UNKNOWN", "CARRIER"):
            sound = demod.demodulate(y, ZOOM_FS, demod_kind, feats)
            if sound is not None:
                from src.intel import audio as wavio
                wav = out_dir / f"track_{k}.wav"
                wavio.write_wav(wav, sound, demod.AUDIO_FS, bits=16)
                L["audio"] = wav.name
                extra[k] = {"dig": dig, "peak": peak}
        conf = 0.9 if kind not in ("UNKNOWN",) else 0.1
        tracks.append({"track": k, "t_start_ms": t_on_ms, "t_end_ms": t_off_ms, "f_lo_khz": lo / 1e3, "f_hi_khz": hi / 1e3,
                       "answer": kind_answer(kind, feats), "confidence": conf,
                       "band": "high" if kind != "UNKNOWN" else "uncertain",
                       "reason": f"live zoom-in, {peak:.0f} dB above the noise: {why}",
                       "decided_by": "zoom", "classifier_answer": "not run (voice mode)", "top5": [], "listen": L})
    return tracks, extra


def voice_text(tracks, extra, out_dir, whisper_model, language, use_whisper, use_llm, llm_model, progress=None):
    """The slow half of voice_snapshot, on the audio voice_audio wrote: Morse,
    RTTY / PSK31 text, Whisper for speech, then the AI reading. Fills each
    track's listen in place. progress(stage, track_number) is told each step."""
    from src.intel import audio as wavio
    for t in tracks:
        k, L = t["track"], t["listen"]
        if k not in extra or not L.get("audio"):
            continue
        dig, peak, kind = extra[k]["dig"], extra[k]["peak"], L["kind"]
        wav = out_dir / L["audio"]
        segments, text_kind = [], None
        try:
            if dig is not None:
                from src.intel import digital
                L.update(digital.listen_fields(dig))
                segments, text_kind = dig["segments"], "digital"
            elif kind == "CW":
                from src.intel import morse
                progress and progress("reading Morse", k)
                m = morse.analyse(wavio.wav_info(wav))
                segments, text_kind = m["segments"], "morse"
                L["morse"] = " / ".join(s["text"] for s in segments)
                L["morse_detail"] = m
            # Speech only: RTTY / PSK31 / AFSK are typed data, and Whisper on them gives
            # made-up words (and loads a multi-GB model for nothing).
            elif kind in VOICE_KINDS and use_whisper and L["duration_s"] >= 1.0:
                from src.intel import voice
                progress and progress(f"transcribing (Whisper {whisper_model})", k)
                tr = voice.transcribe(wavio.wav_info(wav), model=whisper_model,
                                      max_minutes=L["duration_s"] / 60 + 1, language=language)
                L["transcript"], L["language"] = tr.get("text", ""), tr.get("language")
                segments = [{"start_s": s["clip_start_s"], "end_s": s["clip_end_s"], "text": s["text"]}
                            for s in tr.get("segments", [])]
                L["speech_segments"], L["snr_db"] = segments, peak
                text_kind = "voice"
        except Exception as e:                     # noqa: BLE001 -- keep the audio
            L["text_error"] = f"{type(e).__name__}: {e}"
        if segments and use_llm:
            from src.intel import reason
            progress and progress(f"AI reading ({llm_model or 'local'})", k)
            kw = {"model": llm_model} if llm_model else {}
            L["reading"] = reason.read_recording(f"live track {k}", segments, kind=text_kind,
                                                 duration=f"{L['duration_s']:.1f} s",
                                                 active_min=L.get("morse_detail", {}).get("active_minutes", 0), **kw)
    return tracks


def plot_snapshot(iq, fs, tracks, path, centre_hz=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    nfft = int(min(max(256, 2 ** np.ceil(np.log2(fs / 500))), 8192))
    if len(iq) <= MAX_PLOT_COLS * nfft // 2:
        f, t, s = spectrogram(iq[0:len(iq)], fs=fs, nperseg=nfft, noverlap=nfft // 2, return_onesided=False)
        t_end = t[-1]
    else:                                                  # long recording: one averaged column per pixel
        # Every column must hold at least one full FFT, or welch shortens it and the
        # columns no longer line up (a 30 s KiwiSDR clip at 12 kS/s has < 256 samples a column).
        cols_n = max(1, min(MAX_PLOT_COLS, len(iq) // nfft))
        cuts = np.linspace(0, len(iq), cols_n + 1).astype(int)
        cols = []
        for a, b in zip(cuts, cuts[1:]):
            f, p = welch(iq[int(a):int(b)], fs=fs, nperseg=nfft, noverlap=0, return_onesided=False, detrend=False)
            cols.append(p)
        s, t_end = np.array(cols).T, len(iq) / fs
    order = np.argsort(f)
    img = 10 * np.log10(s[order] + 1e-20)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(img, aspect="auto", origin="lower", cmap="magma", extent=[0, t_end * 1e3, f[order][0] / 1e3, f[order][-1] / 1e3],
              vmin=np.percentile(img, 20), vmax=np.percentile(img, 99.8))
    for a in tracks:
        ax.add_patch(plt.Rectangle((a["t_start_ms"], a["f_lo_khz"]), a["t_end_ms"] - a["t_start_ms"],
                                   a["f_hi_khz"] - a["f_lo_khz"], fill=False, ec="#4cd4ff", lw=1.4))
        said = a.get("listen", {}).get("transcript") or a.get("listen", {}).get("morse") or ""
        ax.text(a["t_start_ms"] + 20, a["f_hi_khz"], f"#{a['track']} {a['listen']['kind']} {said[:40]}",
                color="#4cd4ff", fontsize=8, va="bottom")
    ax.set_xlabel("time in recording (ms)")
    ax.set_ylabel("offset (kHz)" + (f" from {centre_hz / 1e6:.4f} MHz" if centre_hz else ""))
    ax.set_title("Live recording: transmissions found (boxes) and what they are")
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
        answers, _ = process_capture(x, self.detector, self.classifier, rng=rng, band_hz=fs)
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


def new_spool():
    SPOOL.mkdir(parents=True, exist_ok=True)
    return SPOOL / f"{secrets.token_hex(6)}.iq"


def record_until_stop(buf, fs, stop_after_s=3.0, quiet_stop_s=0.0, exit_when_idle=0.0):
    """Wait, with no time limit, for the current recording to end: the stream
    sends nothing for stop_after_s (file ended or paused), or a signal was
    heard and then nothing for quiet_stop_s, or Ctrl+C, or the disk is nearly
    full. -> (path, samples, why, last): last = no more recordings after this."""
    why, last = None, False
    before, checked, heard_at = buf.total, buf.total, None
    t_new = t_print = time.monotonic()
    said_wait = False
    try:
        while why is None:
            time.sleep(0.2)
            now = time.monotonic()
            if buf.total != before:
                before, t_new, said_wait = buf.total, now, False
            idle = now - t_new
            have = buf.spool_n
            free = shutil.disk_usage(SPOOL).free
            if buf.error:
                why, last = f"stream lost: {buf.error}", True
            elif have == 0:
                if idle > 3 and not said_wait:
                    print("Waiting for IQ (press play in SDRconnect). Ctrl+C to stop.", flush=True)
                    said_wait = True
                if exit_when_idle and idle > exit_when_idle:
                    why, last = f"no new IQ for {exit_when_idle:.0f} s", True
            elif idle > stop_after_s:
                why = "the stream stopped"
            elif free < MIN_FREE_BYTES:
                why = f"the disk is nearly full ({free / 1e6:.0f} MB free)"
            elif quiet_stop_s and buf.total - checked >= fs:
                checked = buf.total
                if find_channels(buf.last(int(fs)), fs):
                    heard_at = now
                elif heard_at and now - heard_at > quiet_stop_s:
                    why = f"no signal for {quiet_stop_s:.0f} s"
            if have and now - t_print >= 10:
                t_print = now
                print(f"Recording: {have / fs:.0f} s so far, {4 * have / 1e6:.0f} MB, {free / 1e9:.1f} GB free", flush=True)
    except KeyboardInterrupt:
        why, last = "Ctrl+C", True
    path, n = buf.rotate_spool(None if last else new_spool())
    return path, n, why, last


def touch_until(path, stop):
    """Tells the dashboard this run is still being worked on."""
    path.touch()
    while not stop.wait(5):
        path.touch()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--host", default="127.0.0.1", help="laptop running SDRconnect (default: this one)")
    src.add_argument("--replay", help="IQ WAV to play as if live (no SDRconnect needed)")
    p.add_argument("--replay-once", action="store_true", help="with --replay: play the file once, then stop sending")
    p.add_argument("--port", type=int, default=5454)
    p.add_argument("--device", choices=["primary", "secondary"], default="primary", help="RSPduo tuner")
    p.add_argument("--fs", type=float, default=None, help="IQ stream sample rate (default: measured)")
    p.add_argument("--mode", choices=["voice", "full"], default="voice")
    p.add_argument("--snapshot-s", type=float, default=None,
                   help="analyse the newest N seconds in a loop (full mode default 0.5); "
                        "voice mode without it records until the stream stops and analyses all of it")
    p.add_argument("--stop-after-s", type=float, default=3.0,
                   help="a recording ends when the stream has sent nothing for this long (default 3)")
    p.add_argument("--quiet-stop-s", type=float, default=0.0, metavar="S",
                   help="live radio: also end a recording when a signal was heard and then nothing for S seconds "
                        "(default off: pauses inside a recording do not split it)")
    p.add_argument("--every-s", type=float, default=0.0, help="minimum time between snapshots")
    p.add_argument("--cycles", type=int, default=0, help="stop after this many snapshots (0 = run until Ctrl+C)")
    p.add_argument("--exit-when-idle", type=float, default=0, metavar="S",
                   help="stop when SDRconnect has sent no IQ for S seconds (default: wait for play again)")
    p.add_argument("--whisper-model", default="small", help="base (fast) / small / medium (~5 GB RAM)")
    p.add_argument("--language", default=None, help="en / ms / zh (default: Whisper guesses)")
    p.add_argument("--no-whisper", action="store_true")
    p.add_argument("--llm", action="store_true", help="add the Ollama reading (slower)")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--classifier", nargs="+", default=None, help="full mode: best.pt file(s)")
    p.add_argument("--detector", default=str(REPO_ROOT / "results" / "detector" / "emitter_detector.pt"))
    p.add_argument("--keep", type=int, default=40, help="live runs kept in results/pipeline/jobs (oldest deleted)")
    p.add_argument("--autotune", action="store_true",
                   help="each time playback starts, tune SDRconnect (VFO, mode, filter) onto the strongest signal")
    p.add_argument("--tune-only", action="store_true",
                   help="only tune SDRconnect onto the strongest signal once, then exit (no analysis)")
    args = p.parse_args()
    whole = args.mode == "voice" and not args.snapshot_s      # record until the stream stops, analyse all of it
    snap_s = args.snapshot_s or 0.5
    REPLAY_ONCE[0] = args.replay_once or whole                # a looping replay would never stop

    buf = IqBuffer(keep_s=max(10.0, 2 * snap_s))
    if whole:
        first = new_spool()                                   # from the first sample, before the rate is known
        for old in SPOOL.glob("*.iq"):                        # left by a run that was killed
            try:
                old.unlink()
            except OSError:
                pass                                          # in use by another bridge
        buf.rotate_spool(first)

        def drop_spool():
            path, _ = buf.rotate_spool(None)
            if path:
                path.unlink(missing_ok=True)
        atexit.register(drop_spool)
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
        why = buf.error or ("connected, but no IQ in 20 s: press play / start the radio in SDRconnect"
                            if buf.info.get("connected") else "no answer in 20 s")
        sys.exit(f"No IQ from {label}: {why}. Check: Test-NetConnection <host> -Port 5454 is True, and "
                 "SDRconnect Settings -> WebSocket interface is on.")
    buf.fs = fs
    centre, centre_src = centre_of(buf, fs)
    print(f"IQ stream {fs / 1e3:.1f} kS/s ({buf.info.get('active_device', '?')}, API {buf.info.get('api_version', '?')})"
          + (f", centre {centre / 1e6:.4f} MHz ({centre_src})" if centre else ", centre frequency unknown"), flush=True)
    if buf.rate_measured and abs(buf.rate_measured - fs) / fs > 0.02:
        print(f"Measured {buf.rate_measured / 1e3:.1f} kS/s at the start and assumed {fs / 1e3:.1f} kS/s. "
              "If the recording's real rate is different, set it with --fs.", flush=True)
    if args.tune_only:
        buf.tuned_once = False
        threading.Thread(target=autotune_loop, args=(buf, fs, args.device), kwargs={"once": True}, daemon=True).start()
        deadline = time.monotonic() + 30
        while not getattr(buf, "tuned_once", False) and not buf.error and time.monotonic() < deadline:
            time.sleep(0.2)
        if not getattr(buf, "tuned_once", False):
            print("Auto-tune: no signal to tune to (is SDRconnect playing?).", flush=True)
        time.sleep(0.5)                                       # let the commands leave before the connection closes
        return
    if args.autotune:
        threading.Thread(target=autotune_loop, args=(buf, fs, args.device), daemon=True).start()

    chain = None
    if args.mode == "full":
        cls = args.classifier or [str(c) for c in sorted((REPO_ROOT / "results" / "classifier").glob("*best*.pt"))]
        if not cls:
            sys.exit("full mode needs a classifier best.pt in results/classifier/")
        chain = FullChain(cls, args.detector)
    rng = np.random.default_rng(0)
    JOBS.mkdir(parents=True, exist_ok=True)
    cycle = 0
    seen = 0                                     # buf.total at the last snapshot
    stopping = False
    if whole:
        print("Recording until the stream stops, then analysing all of it (Ctrl+C: stop and analyse).", flush=True)
    while (not args.cycles or cycle < args.cycles) and not stopping:
        t_cycle = time.monotonic()
        n = int(snap_s * fs)
        rec = rec_path = problem = None
        if whole:
            rec_path, rec_n, why, stopping = record_until_stop(buf, fs, args.stop_after_s, args.quiet_stop_s,
                                                               args.exit_when_idle)
            stopping = stopping or bool(args.replay)  # a file plays once: analyse it, then exit (no SDRconnect to wait for)
            if rec_n < fs:                       # nothing, or a fragment between stop and play
                if rec_n:
                    print(f"Recording of {rec_n / fs:.2f} s ({why}): too short, skipped.", flush=True)
                rec_path.unlink(missing_ok=True)
                continue
            print(f"Recording over ({why}): {rec_n / fs:.1f} s. Analysing all of it ...", flush=True)
            problem = delivery_problem(rec_n, fs, **buf.ended)
            if problem:
                print(f"WARNING: {problem}.", flush=True)
            rec = DiskIq(rec_path)
        # Only NEW samples: once SDRconnect stops sending (playback paused or
        # ended), the buffer still holds the last seconds -- analysing those
        # again and again would look like endless loading.
        idle_since, said_idle = None, False
        while not whole and buf.total - seen < n:
            if buf.error:
                sys.exit(f"stream stopped: {buf.error}")
            before = buf.total
            time.sleep(0.2)
            if buf.total == before:
                idle_since = idle_since or time.monotonic()
                idle = time.monotonic() - idle_since
                fresh = buf.total - seen
                if idle > 1.5 and fresh >= int(1.0 * fs):
                    n = fresh                        # the transmission ended: analyse what came, do not wait for more
                    break
                if idle > 3 and not said_idle:
                    print("No new IQ from SDRconnect (playback paused or ended?): waiting. "
                          "Press play again to continue, Ctrl+C to stop.", flush=True)
                    said_idle = True
                if args.exit_when_idle and idle > args.exit_when_idle:
                    print(f"No new IQ for {args.exit_when_idle:.0f} s: stopping.", flush=True)
                    return
            else:
                idle_since = None
                if said_idle:
                    print("IQ is flowing again.", flush=True)
                    said_idle = False
        iq = rec if whole else buf.last(n)
        seen = buf.total
        rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        d = JOBS / rid
        d.mkdir(parents=True)
        meta = {"id": rid, "source": "live", "file": f"LIVE · {label}", "max_s": round(len(iq) / fs, 1),
                "detector": "live voice path" if args.mode == "voice" else "live full chain", "status": "running",
                "live": True, "who": "live bridge"}
        (d / "job.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        done = threading.Event()
        threading.Thread(target=touch_until, args=(d / "alive", done), daemon=True).start()
        t0 = time.monotonic()
        kw = dict(whisper_model=args.whisper_model, language=args.language, use_whisper=not args.no_whisper,
                  use_llm=args.llm, llm_model=args.llm_model)
        if chain is None:
            tracks = voice_snapshot(iq, fs, d, **kw)
            plot_snapshot(iq, fs, tracks, d / "tracks.png", centre)
            dur_ms = len(iq) / fs * 1e3
        else:
            tracks, dur_ms = chain(iq, fs, d, rng, **kw)
        source = {"file": f"LIVE · {label}",
                  "format": f"SDRconnect IQ stream ({args.mode} mode)" + (f" · WARNING: {problem}" if problem else ""),
                  "fs_in": fs, "centre_hz": centre, "resampled": False, "live": True, "snapshot_at": rid,
                  "delivery_problem": problem}
        doc = finite({"n_samples": len(iq), "duration_ms": dur_ms, "source": source, "tracks": tracks, "truth": None})
        (d / "tracks.tmp").write_text(json.dumps(doc, indent=1, default=float, allow_nan=False), encoding="utf-8")
        (d / "tracks.tmp").replace(d / "tracks.json")       # the dashboard sees a run as done once this exists
        try:
            saved = store.save_decoded_tracks(rid, doc["source"], doc["tracks"], d, keep_empty=whole)
            if saved:
                print(f"Comms Intel: saved {len(saved)} decoded recording(s).", flush=True)
        except Exception as e:                              # noqa: BLE001 -- the run itself is already safe
            print(f"Comms Intel: could not save this run: {type(e).__name__}: {e}", flush=True)
        done.set()
        if rec is not None:
            rec.close()
            rec_path.unlink(missing_ok=True)
        took = time.monotonic() - t0
        (d / "run.log").write_text(f"live {'recording' if whole else 'snapshot'} {len(iq) / fs:.1f} s at {fs / 1e3:.1f} kS/s, "
                                   f"analysed in {took:.1f} s\n", encoding="utf-8")
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
