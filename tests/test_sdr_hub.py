"""The dashboard's live SDRconnect receiver (scripts/sdr_hub.py and
/api/sdr/* in serve_local.py), against a fake SDRconnect (WebSocket API
1.0.3) that plays noise, one NBFM transmission, then noise again in real time."""
import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("websockets")
sdr_hub = pytest.importorskip("sdr_hub")

from src.pipeline import analog_synth  # noqa: E402

FS = 250e3


def air(seed=4):
    """1 s noise, 2.5 s NBFM at -40 kHz, then noise for as long as asked."""
    rng = np.random.default_rng(seed)
    s, _ = analog_synth.make("NBFM", 2.5, snr_db=200, rng=rng, offset_hz=0.0)
    w = resample_poly(s, int(FS / analog_synth.FA), 1)
    w = w / np.sqrt(np.mean(np.abs(w) ** 2)) * np.sqrt(10 ** (25 / 10) * 3000 / FS)
    w = w * np.exp(-2j * np.pi * 40e3 * np.arange(len(w)) / FS)
    sig = np.concatenate([np.zeros(int(FS)), w])

    def noise(n):
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return sig, noise


@pytest.fixture
def fake_sdrconnect():
    from websockets.sync.server import serve
    sig, noise = air()
    peak = 8.0

    def handler(ws):
        started = False
        while not started:
            m = json.loads(ws.recv())
            if m["event_type"] == "get_property":
                val = {"device_sample_rate": str(FS), "device_center_frequency": "145000000",
                       "device_vfo_frequency": "145000000", "api_version": "1.0.3",
                       "active_device": "RSPduo"}.get(m["property"], "")
                ws.send(json.dumps({"event_type": "get_property_response", "property": m["property"], "value": val}))
            elif m["event_type"] == "iq_stream_enable":
                started = True
        step = int(FS * 0.02)
        t0 = time.monotonic()
        k = 0
        try:
            while True:                                   # a live radio never stops sending
                a = k * step
                x = sig[a:a + step] if a < len(sig) else np.zeros(step)
                x = (x + noise(len(x))) / peak
                raw = np.empty(2 * len(x), "<i2")
                raw[0::2], raw[1::2] = np.round(x.real * 32767), np.round(x.imag * 32767)
                ws.send((2).to_bytes(2, "little") + raw.tobytes())
                k += 1
                rest = t0 + k * 0.02 - time.monotonic()
                if rest > 0:
                    time.sleep(rest)
        except Exception:
            return                                        # the hub hung up

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with serve(handler, "127.0.0.1", port) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield port
        server.shutdown()


def wait_for(cond, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.2)
    return None


def test_hub_streams_dwells_and_decodes_a_transmission(fake_sdrconnect, tmp_path, monkeypatch):
    monkeypatch.setattr(sdr_hub.bridge, "SPOOL", tmp_path / "spool")
    hub = sdr_hub.SdrHub(jobs_dir=tmp_path / "jobs", intel_dir=tmp_path / "intel", quiet_s=1.0)
    (tmp_path / "jobs").mkdir()
    seen = []
    hub.on_result = seen.append
    st = hub.connect("127.0.0.1", fake_sdrconnect, use_whisper=False)
    assert st["state"] == "connecting"
    assert wait_for(lambda: hub.status()["state"] == "streaming"), hub.status()
    st = hub.status()
    assert st["fs"] == FS and st["centre_hz"] == 145e6 and st["active_device"] == "RSPduo"

    iq, meta = hub.dwell(0.05)
    assert iq is not None and iq.dtype == np.complex64
    assert len(iq) == int(0.05 * sdr_hub.DWELL_FS)            # always the model's 3.2 MS/s
    assert meta["upsampled"] and meta["fs_in"] == FS
    again, _ = hub.dwell(0.05, after=meta["total"] + 10**9)   # nothing new since then: no dwell
    assert again is None

    # The NBFM burst is recorded until 1 s of quiet, then analysed: its audio first
    # (the entry appears, playable), then its text (stage "done").
    entry = wait_for(lambda: [e for e in hub.status()["decodes"] if e.get("stage") == "done"], timeout=40)
    assert entry, hub.status()
    d = entry[-1]
    assert "error" not in d, d
    assert any(t["kind"] == "NBFM" and -48 < t["f_lo_khz"] < -32 for t in d["tracks"]), d
    run = tmp_path / "jobs" / d["id"]
    doc = json.loads((run / "tracks.json").read_text(encoding="utf-8"))
    assert doc["source"]["live"] and doc["source"]["centre_hz"] == 145e6
    assert json.loads((run / "job.json").read_text(encoding="utf-8"))["live"]
    assert (run / "tracks.png").exists() and seen and seen[-1]["id"] == d["id"]

    hub.disconnect()
    assert hub.status()["state"] == "off"
    assert hub.dwell(0.05)[0] is None


def test_api_sdr_endpoints(fake_sdrconnect, tmp_path, monkeypatch):
    import serve_local
    monkeypatch.setattr(sdr_hub.bridge, "SPOOL", tmp_path / "spool")
    monkeypatch.setattr(serve_local, "PIPELINE_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(serve_local.Handler, "_sdr_hub", None)
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0, demo=True, intel_dir=tmp_path / "intel")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/sdr"
    hdr = {"X-NEXA-Client": "1", "Content-Type": "application/json"}

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers=hdr, method="GET" if body is None else "POST")
        return urllib.request.urlopen(req, timeout=10)

    try:
        with pytest.raises(urllib.error.HTTPError) as bad:
            call("/connect", {"host": "x; rm -rf", "port": fake_sdrconnect})
        assert bad.value.code == 400
        st = json.load(call("/connect", {"host": "127.0.0.1", "port": fake_sdrconnect, "decode": False}))
        assert st["state"] == "connecting" and st["decode"]["decode"] is False
        assert wait_for(lambda: json.load(call("/status"))["state"] == "streaming")
        r = call("/dwell?s=0.02")
        assert r.status == 200 and r.headers["X-NEXA-SDR-State"] == "streaming"
        assert len(r.read()) == 2 * 4 * int(0.02 * sdr_hub.DWELL_FS)   # float32 I/Q interleaved
        r = call(f"/dwell?s=0.02&after={10**12}")
        assert r.status == 204
        assert json.load(call("/disconnect", {}))["state"] == "off"
        actions = [e["action"] for e in srv.nexa_db.audit(50)]
        assert "sdr.connect" in actions and "sdr.disconnect" in actions
    finally:
        if serve_local.Handler._sdr_hub:
            serve_local.Handler._sdr_hub.disconnect()
        srv.shutdown()
