"""scripts/live_sdrconnect.py against a fake SDRconnect (WebSocket API 1.0.3):
frame parsing, channel finder, and the whole receive -> snapshot -> zoom path."""
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
live = pytest.importorskip("live_sdrconnect")
pytest.importorskip("websockets")

from src.pipeline import analog_synth  # noqa: E402

FS = 250e3


def nbfm_at(offset_hz, seconds=3.0, snr_db=25.0, seed=1):
    rng = np.random.default_rng(seed)
    s, _ = analog_synth.make("NBFM", seconds, snr_db=200, rng=rng, offset_hz=0.0)
    w = resample_poly(s, int(FS / analog_synth.FA), 1)
    w = w / np.sqrt(np.mean(np.abs(w) ** 2)) * np.sqrt(10 ** (snr_db / 10) * 3000 / FS)
    w = w * np.exp(2j * np.pi * offset_hz * np.arange(len(w)) / FS)
    n = (rng.standard_normal(len(w)) + 1j * rng.standard_normal(len(w))) / np.sqrt(2)
    return (w + n).astype(np.complex64)


def frame(iq, kind=2):
    x = iq / (np.abs(np.concatenate([iq.real, iq.imag])).max() + 1e-9) * 0.9
    raw = np.empty(2 * len(x), "<i2")
    raw[0::2], raw[1::2] = np.round(x.real * 32767), np.round(x.imag * 32767)
    return int(kind).to_bytes(2, "little") + raw.tobytes()


def test_parse_binary_types():
    iq = np.array([0.5 + 0.25j, -0.5 - 0.25j], np.complex64)
    out = live.parse_binary(frame(iq))
    assert out is not None and len(out) == 2 and out[0].real > 0 and out[1].imag < 0
    assert live.parse_binary(frame(iq, kind=1)) is None              # audio
    assert live.parse_binary(frame(iq, kind=5)) is None              # secondary tuner IQ
    assert live.parse_binary(frame(iq, kind=5), device="secondary") is not None


def test_find_channels_locates_nbfm():
    ch = live.find_channels(nbfm_at(40e3), FS)
    assert ch, "no channel found"
    lo, hi, _ = ch[0]
    assert lo < 40e3 < hi and hi - lo < 20e3


def test_zoom_in_blocks_matches_zoom_of_the_whole():
    rate = 440e3
    rng = np.random.default_rng(3)
    iq = (rng.standard_normal(5_000_000) + 1j * rng.standard_normal(5_000_000)).astype(np.complex64)
    r = live.Fraction(live.ZOOM_FS / rate).limit_denominator(1000)
    whole = resample_poly(iq * np.exp(-2j * np.pi * 12e3 * np.arange(len(iq)) / rate), r.numerator, r.denominator)
    got = live.zoom_to(iq, rate, 12e3)
    assert len(got) == len(whole)
    assert np.max(np.abs(got - whole)) < 1e-3 * np.max(np.abs(whole))


def test_short_transmission_survives_a_long_recording(tmp_path):
    burst = nbfm_at(40e3, seconds=3.0)
    rng = np.random.default_rng(4)
    n = int(120 * FS)
    iq = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
    at = int(60 * FS)
    iq[at:at + len(burst)] = burst
    tracks = live.voice_snapshot(iq, FS, tmp_path, "base", None, use_whisper=False, use_llm=False, llm_model=None)
    assert tracks, "a 3 s transmission was lost in 120 s of noise"
    t = tracks[0]
    assert t["f_lo_khz"] < 40 < t["f_hi_khz"]
    assert t["listen"]["kind"] == "NBFM"
    assert t["listen"]["duration_s"] == pytest.approx(120, abs=0.01)       # the audio is the whole recording
    assert 54e3 <= t["t_start_ms"] <= 60e3 and 63e3 <= t["t_end_ms"] <= 72e3


def test_snap_rate_accepts_a_slow_start_but_not_a_stranger():
    assert live.snap_rate(500_000) == 500e3
    for slow in (424_600, 434_200, 439_800, 442_500):             # what real 500 kS/s streams measured
        assert live.snap_rate(slow) == 500e3
    assert live.snap_rate(520_000) == 500e3                       # a burst
    assert live.snap_rate(350_000) == 350_000                     # not close to anything: as measured
    assert live.snap_rate(434_200, dev=440_000) == 440_000        # the device's own rate comes first


def test_delivery_problem_names_lost_samples_and_pauses():
    assert live.delivery_problem(6_000_000, 500e3, 12.0, 0.05) is None
    lost = live.delivery_problem(3_000_000, 500e3, 12.0, 0.05)
    assert "50%" in lost and "6.0 s of IQ in 12.0 s" in lost
    assert "paused for 1.5 s" in live.delivery_problem(6_000_000, 500e3, 12.0, 1.5)
    assert live.delivery_problem(100_000, 500e3, 0.5, 0.0) is None   # too short to judge


def test_buffer_measures_how_a_recording_arrived(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "SPOOL", tmp_path)
    buf = live.IqBuffer(keep_s=1)
    buf.fs = 1000.0
    buf.rotate_spool(live.new_spool())
    frame = np.zeros(100, np.complex64)
    buf.add(frame)
    time.sleep(0.15)
    buf.add(frame)
    path, n = buf.rotate_spool(live.new_spool())
    assert n == 200 and buf.ended["span_s"] >= 0.1 and buf.ended["longest_gap_s"] >= 0.1
    buf.add(frame)                                                # the next recording starts clean
    buf.rotate_spool(None)
    assert buf.ended["longest_gap_s"] < 0.1


def test_recording_keeps_every_sample_until_the_stream_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "SPOOL", tmp_path)
    buf = live.IqBuffer(keep_s=0.5)
    buf.fs = 1000.0
    buf.rotate_spool(live.new_spool())
    sent = np.arange(50_000, dtype=np.float32) / 100_000 * (1 + 0.5j)

    def stream():
        for s in range(0, len(sent), 500):
            buf.add(sent[s:s + 500].astype(np.complex64))

    stream()                                                     # far more than the ring buffer keeps
    path, n, why, last = live.record_until_stop(buf, 1000.0, stop_after_s=0.4)
    assert (n, last) == (len(sent), False) and "stopped" in why
    rec = live.DiskIq(path)
    assert len(rec) == len(sent)
    assert np.max(np.abs(rec[0:len(rec)] - sent)) < 1e-4
    assert np.max(np.abs(rec[100:103] - sent[100:103])) < 1e-4   # slices read like an array
    rec.close()
    assert buf.spool_path != path and buf.spool_n == 0           # the next recording is already open
    buf.rotate_spool(None)


def test_websocket_stream_to_snapshot(tmp_path):
    from websockets.sync.server import serve
    sig = nbfm_at(-30e3, seconds=4.0)
    got = []

    def handler(ws):
        started = False
        while not started:
            m = json.loads(ws.recv())
            got.append(m)
            if m["event_type"] == "get_property":
                val = {"device_sample_rate": str(FS), "device_center_frequency": "145000000",
                       "device_vfo_frequency": "145000000", "api_version": "1.0.3"}.get(m["property"], "")
                ws.send(json.dumps({"event_type": "get_property_response", "property": m["property"],
                                    "device": "primary", "value": val}))
            elif m["event_type"] == "iq_stream_enable" and m["value"] == "true":
                started = True
        step = int(FS * 0.02)
        t0 = time.monotonic()
        for k, s in enumerate(range(0, len(sig), step)):
            ws.send(frame(sig[s:s + step]))
            ws.send(int(1).to_bytes(2, "little") + b"\0" * 64)         # an audio frame: must be ignored
            wait = t0 + (k + 1) * 0.02 - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with serve(handler, "127.0.0.1", port) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        buf = live.IqBuffer(keep_s=10)
        threading.Thread(target=live.sdrconnect_receiver, args=(buf, f"ws://127.0.0.1:{port}"), daemon=True).start()
        fs = live.settle_rate(buf, wait_s=1.5)
        assert buf.error is None
        assert fs == FS
        buf.fs = fs
        assert live.centre_of(buf, fs)[0] == 145e6
        deadline = time.monotonic() + 5
        iq = None
        while iq is None and time.monotonic() < deadline:
            iq = buf.last(int(2.0 * fs))
            time.sleep(0.05)
        server.shutdown()
    assert iq is not None
    assert any(m["event_type"] == "iq_stream_enable" for m in got)
    tracks = live.voice_snapshot(iq, fs, tmp_path, "base", None, use_whisper=False, use_llm=False, llm_model=None)
    assert tracks, "no transmission found in the live snapshot"
    t = tracks[0]
    assert t["f_lo_khz"] < -30 < t["f_hi_khz"]
    assert t["listen"]["kind"] == "NBFM"
    assert (tmp_path / t["listen"]["audio"]).exists()
