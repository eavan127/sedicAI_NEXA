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
