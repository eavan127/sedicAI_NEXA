"""Live KiwiSDR listening (scripts/sdr_hub.py KiwiHub, /api/kiwi/*) with a
stand-in for kiwirecorder that streams noise as raw IQ on stdout, like the
real one with --nc. Checks the stream comes up at 12 kS/s on the right
frequency, bad input is refused, and Stop ends the client process."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sdr_hub = pytest.importorskip("sdr_hub")

FAKE = '''
import sys, time, numpy as np
rng = np.random.default_rng(0)
out = sys.stdout.buffer
while True:
    x = (rng.standard_normal(2 * 1200) * 300).astype("<i2")   # 0.1 s of quiet I/Q at 12 kS/s
    out.write(x.tobytes()); out.flush()
    time.sleep(0.1)
'''


@pytest.fixture
def fake_kiwi(tmp_path, monkeypatch):
    p = tmp_path / "fake_kiwirecorder.py"
    p.write_text(FAKE, encoding="utf-8")
    monkeypatch.setattr(sdr_hub, "KIWIRECORDER", p)
    monkeypatch.setattr(sdr_hub, "kiwi_unreachable", lambda host, port: None)   # the fake needs no network
    monkeypatch.setattr(sdr_hub.bridge, "SPOOL", tmp_path / "spool")
    (tmp_path / "jobs").mkdir()
    return tmp_path


def wait_for(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_kiwi_hub_streams_and_stops(fake_kiwi):
    hub = sdr_hub.KiwiHub(jobs_dir=fake_kiwi / "jobs", intel_dir=fake_kiwi / "intel")
    with pytest.raises(ValueError):
        hub.connect(freq_khz=145000)                           # VHF: not on a KiwiSDR
    st = hub.connect(freq_khz=15295, language="ms", use_llm=False)
    assert st["state"] == "connecting" and st["label"].endswith("15295 kHz")
    assert wait_for(lambda: hub.status()["state"] == "streaming"), hub.status()
    st = hub.status()
    assert st["fs"] == sdr_hub.KIWI_FS and st["centre_hz"] == 15295e3
    assert wait_for(lambda: hub.status()["total_samples"] > 12000)
    proc = hub.buf.proc
    hub.disconnect()
    assert wait_for(lambda: proc.poll() is not None, 5), "Stop must end the kiwirecorder process"


def test_blocked_network_is_named_not_waited_on(tmp_path, monkeypatch):
    monkeypatch.setattr(sdr_hub, "KIWIRECORDER", tmp_path / "exists.py")
    (tmp_path / "exists.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sdr_hub, "kiwi_unreachable", lambda host, port: "cannot reach the KiwiSDR x:8073 from this network")
    buf = sdr_hub.bridge.IqBuffer()
    sdr_hub.kiwi_receiver(buf, "x", 8073, 15295)
    assert "cannot reach" in buf.error and "SDRconnect" not in buf.error


def test_kiwi_location_is_the_newest_recorded_for_that_frequency(fake_kiwi):
    import serve_local
    from src.localdb import Actor, LocalDB
    db = LocalDB(fake_kiwi / "nexa.db")
    me = Actor("analyst", "analyst", "127.0.0.1")
    db.log(me, "client.tdoa_location", "kiwi", "15295 kHz", {"freq_khz": 15295, "lat": 1.0, "lon": 100.0})
    db.log(me, "client.tdoa_location", "kiwi", "7040 kHz", {"freq_khz": 7040, "lat": 9.0, "lon": 99.0})
    db.log(me, "client.tdoa_location", "kiwi", "15295 kHz", {"freq_khz": 15294.9, "lat": 2.5, "lon": 102.5,
                                                             "receivers": "DU6/PE1NSQ, HS0ZGZ, JKT"})
    loc = serve_local.kiwi_location(db, 15295)
    assert (loc["lat"], loc["lon"], loc["receivers"]) == (2.5, 102.5, "DU6/PE1NSQ, HS0ZGZ, JKT") and loc["ts"]
    assert serve_local.kiwi_location(db, 9999) is None and serve_local.kiwi_location(db, None) is None


def test_api_kiwi_start_stop(fake_kiwi, monkeypatch):
    import serve_local
    monkeypatch.setattr(serve_local, "PIPELINE_DIR", fake_kiwi / "jobs")
    monkeypatch.setattr(serve_local.Handler, "_kiwi_hub", None)
    srv = serve_local.make_server(fake_kiwi / "nexa.db", "127.0.0.1", 0, demo=True, intel_dir=fake_kiwi / "intel")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/kiwi"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-NEXA-Client": "1", "Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        return json.load(urllib.request.urlopen(req, timeout=10))

    try:
        with pytest.raises(urllib.error.HTTPError) as bad:
            call("/start", {"freq_khz": 0})
        assert bad.value.code == 400
        assert call("/start", {"freq_khz": 7040, "language": "en", "llm": False})["state"] == "connecting"
        assert wait_for(lambda: call("/status")["state"] == "streaming")
        assert call("/stop", {})["state"] == "off"
        actions = [e["action"] for e in srv.nexa_db.audit(50)]
        assert "kiwi.start" in actions and "kiwi.stop" in actions
    finally:
        if serve_local.Handler._kiwi_hub:
            serve_local.Handler._kiwi_hub.disconnect()
        srv.shutdown()
