"""Auto-tune in scripts/live_sdrconnect.py: from a signal in the IQ stream to
the set_property messages that put SDRconnect's VFO, mode and filter on it,
checked against a fake SDRconnect (WebSocket API 1.0.3)."""
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


def signal_at(kind, offset_hz, seconds=2.5, snr_db=25.0, seed=3):
    rng = np.random.default_rng(seed)
    s, _ = analog_synth.make(kind, seconds, snr_db=200, rng=rng, offset_hz=0.0)
    w = resample_poly(s, int(FS / analog_synth.FA), 1)
    on = np.abs(w) > 1e-3
    w = w / np.sqrt(np.mean(np.abs(w[on]) ** 2)) * np.sqrt(10 ** (snr_db / 10) * 3000 / FS)
    w = w * np.exp(2j * np.pi * offset_hz * np.arange(len(w)) / FS)
    return (w + (rng.standard_normal(len(w)) + 1j * rng.standard_normal(len(w))) / np.sqrt(2)).astype(np.complex64)


@pytest.mark.parametrize("kind,mode,offset", [("NBFM", "NFM", -40e3), ("AM", "AM", 30e3), ("CW", "CW", 55e3)])
def test_tune_plan_finds_signal_and_mode(kind, mode, offset):
    plan = live.tune_plan(signal_at(kind, offset), FS)
    assert plan is not None
    assert plan["mode"] == mode
    assert abs(plan["offset_hz"] - offset) < 2000


def test_tune_commands_are_absolute_and_ordered():
    plan = {"mode": "NFM", "bandwidth": 12500, "offset_hz": -40e3}
    vfo, cmds = live.tune_commands(plan, 145e6)
    assert vfo == 144_960_000
    msgs = [json.loads(c) for c in cmds]
    assert [m["property"] for m in msgs] == ["demodulator", "filter_bandwidth", "device_vfo_frequency"]
    assert all(m["event_type"] == "set_property" for m in msgs)
    assert msgs[2]["value"] == "144960000"


def test_tune_only_sends_set_property_to_sdrconnect():
    from websockets.sync.server import serve
    sig = signal_at("NBFM", -40e3, seconds=7.0)
    got = []

    def handler(ws):
        started = False
        while not started:
            m = json.loads(ws.recv())
            if m["event_type"] == "get_property":
                val = {"device_sample_rate": str(FS), "device_center_frequency": "145000000",
                       "device_vfo_frequency": "145000000", "api_version": "1.0.3"}.get(m["property"], "")
                ws.send(json.dumps({"event_type": "get_property_response", "property": m["property"], "value": val}))
            elif m["event_type"] == "iq_stream_enable":
                started = True
        step = int(FS * 0.02)
        t0 = time.monotonic()
        stop = threading.Event()

        def listen():
            while not stop.is_set():
                try:
                    got.append(json.loads(ws.recv(timeout=0.2)))
                except TimeoutError:
                    continue                      # quiet moment: keep listening
                except Exception:
                    return                        # connection closed
        threading.Thread(target=listen, daemon=True).start()
        peak = np.abs(np.concatenate([sig.real, sig.imag])).max() + 1e-9
        try:
            for k, s in enumerate(range(0, len(sig), step)):
                x = sig[s:s + step] / peak * 0.9
                raw = np.empty(2 * len(x), "<i2")
                raw[0::2], raw[1::2] = np.round(x.real * 32767), np.round(x.imag * 32767)
                ws.send((2).to_bytes(2, "little") + raw.tobytes())
                wait = t0 + (k + 1) * 0.02 - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            time.sleep(0.5)
        except Exception:
            pass                                  # the client hung up once it had tuned: fine
        stop.set()

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with serve(handler, "127.0.0.1", port) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        buf = live.IqBuffer(keep_s=10)
        threading.Thread(target=live.sdrconnect_receiver, args=(buf, f"ws://127.0.0.1:{port}"), daemon=True).start()
        fs = live.settle_rate(buf, wait_s=1.0)
        assert fs == FS
        buf.fs = fs
        buf.tuned_once = False
        t = threading.Thread(target=live.autotune_loop, args=(buf, fs), kwargs={"once": True}, daemon=True)
        t.start()
        t.join(timeout=20)
        assert not t.is_alive(), "auto-tune never finished"
        time.sleep(0.6)
        server.shutdown()
    sets = {m["property"]: m["value"] for m in got if m.get("event_type") == "set_property"}
    assert sets.get("demodulator") == "NFM"
    assert sets.get("filter_bandwidth") == "12500"
    assert abs(int(sets["device_vfo_frequency"]) - 144_960_000) < 2000
