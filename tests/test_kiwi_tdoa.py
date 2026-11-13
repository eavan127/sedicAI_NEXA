"""Automatic transmitter location (src/kiwi_tdoa.py, scripts/sdr_hub.py
TdoaRunner, POST /api/kiwi/locate) without the network: three receivers round
Malaysia hear one synthetic AM transmitter at Kajang, each with its own delay
(great-circle distance / c), start time, carrier phase and noise, written as
GPS-stamped kiwirecorder --kiwi-wav files. A stand-in kiwirecorder writes those
files; /status pages and reachability are patched. Checks the GPS time stamps
are read back exactly, the arrival-time differences and the place come out
right, and the result reaches the audit trail as client.tdoa_location."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sdr_hub = pytest.importorskip("sdr_hub")
from src import kiwi_tdoa as K  # noqa: E402

TX = (2.99, 101.77)                                    # Kajang, Selangor
STATIONS = {"dx6ran.ddns.net": ("DU6/PE1NSQ", 10.7235, 122.9733),
            "21023.proxy.kiwisdr.com": ("HS0ZGZ", 13.90, 100.60),
            "rajawali.proxy.kiwisdr.com": ("JKT", -6.20, 106.80),
            "21439.proxy2.kiwisdr.com": ("9M2AMH", 3.10, 101.60)}


def synth(lat, lon, seconds=8.0, fs=12000.4, seed=0, start=None):
    """What a KiwiSDR at lat, lon records of the transmitter TX: (GPS start time, IQ)."""
    rng = np.random.default_rng(seed)
    F, n = 8, int((seconds + 1) * fs)
    src = np.random.default_rng(42).standard_normal(n * F)          # the SAME programme at every receiver
    A = np.fft.rfft(src)
    A[np.fft.rfftfreq(n * F, 1 / (fs * F)) > 4000] = 0
    audio = np.fft.irfft(A, n * F)
    audio /= 3 * audio.std()
    t_hi = np.arange(n * F) / (fs * F)
    delay = K.great_circle_km(lat, lon, *TX) / K.C_KM_S
    t0 = 1_700_000_000.0 + (rng.uniform(0, 0.3) if start is None else start)
    tk = t0 + np.arange(int(seconds * fs)) / fs - 1_700_000_000.0 - delay + 0.5
    env = 1 + np.interp(tk, t_hi, audio)
    z = 0.2 * env * np.exp(1j * rng.uniform(0, 2 * np.pi))
    z = z + 0.02 * (rng.standard_normal(len(z)) + 1j * rng.standard_normal(len(z)))
    return t0, z.astype(np.complex64), fs


def test_status_page_is_read_and_judged():
    text = ("status=active\noffline=no\nname=DX6RAN\nusers=1\nusers_max=7\ngps=(10.723528, 122.973320)\n"
            "gps_good=3\nfixes_min=29\ntdoa_id=DU6/PE1NSQ\ntdoa_ch=4\n")
    st = K.parse_status(text)
    assert (st["lat"], st["lon"], st["tdoa_id"], st["users_max"]) == (10.723528, 122.97332, "DU6/PE1NSQ", 7)
    assert K.status_problem(st) is None
    assert "full" in K.status_problem({**st, "users": 7})
    assert "GPS" in K.status_problem({**st, "fixes_min": 0, "gps_good": 0})
    assert "position" in K.status_problem(K.parse_status("status=active\ngps=(0.000000, 0.000000)\n"))
    assert "TDoA" in K.status_problem({**st, "tdoa_ch": 0})


def test_gps_time_stamps_are_read_back(tmp_path):
    t0, z, fs = synth(*STATIONS["dx6ran.ddns.net"][1:], seconds=3)
    K.write_kiwi_wav(tmp_path / "a.wav", t0, z, fs)
    t, zz, fs2 = K.read_kiwi_wav(tmp_path / "a.wav")
    assert abs(fs2 - fs) < 0.01
    assert abs(t[0] - (t0 + 512 / fs)) < 1e-6                       # the first block is dropped, as in hcab14
    assert np.allclose(zz[:100], z[512:612], atol=1e-4)
    assert abs((t[-1] - t[0]) - (len(t) - 1) / fs) < 1e-6
    (tmp_path / "plain.wav").write_bytes(b"RIFF\0\0\0\0WAVEfmt " + bytes(20))
    with pytest.raises(ValueError):
        K.read_kiwi_wav(tmp_path / "plain.wav")


def test_solver_finds_the_transmitter(tmp_path):
    recs = []
    for i, host in enumerate(list(STATIONS)[:3]):
        sid, lat, lon = STATIONS[host]
        t0, z, fs = synth(lat, lon, seed=i)
        K.write_kiwi_wav(tmp_path / f"{i}.wav", t0, z, fs)
        t, zz, fs2 = K.read_kiwi_wav(tmp_path / f"{i}.wav")
        recs.append({"id": sid, "lat": lat, "lon": lon, "t": t, "z": zz, "fs": fs2})
    res = K.solve(recs)
    p = res["pairs"][0]                                             # DU6 hears it later than HS0
    want = (K.great_circle_km(*STATIONS["dx6ran.ddns.net"][1:], *TX)
            - K.great_circle_km(*STATIONS["21023.proxy.kiwisdr.com"][1:], *TX)) / K.C_KM_S * 1e6
    assert abs(p["lag_us"] - want) < 10, (p, want)
    assert K.great_circle_km(res["lat"], res["lon"], *TX) < 20, res
    assert 10 < res["radius_km"] < 300                              # an area, not a point
    with pytest.raises(ValueError):
        K.locate(recs[:2], res["pairs"][:1])                        # 2 receivers: a curve, not a place


FAKE = '''
import sys, json
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
from test_kiwi_tdoa import synth, STATIONS
from src.kiwi_tdoa import write_kiwi_wav
a = sys.argv[1:]
opt = lambda k: a[a.index(k) + 1]
hosts, ids, d = opt("-s").split(","), opt("--station").split(","), opt("-d")
assert "--kiwi-wav" in a and opt("-m") == "iq" and opt("-f") == "15295"
for i, (h, sid) in enumerate(zip(hosts, ids)):
    t0, z, fs = synth(*STATIONS[h][1:], seconds=float(opt("--tlimit")) / 2, seed=i)
    write_kiwi_wav(f"{{d}}/20261113T000000Z_15295000_{{sid}}_iq.wav", t0, z, fs)
for i in range(len(hosts)):
    print("status=%d,0" % i)
'''


@pytest.fixture
def fake_tdoa(tmp_path, monkeypatch):
    p = tmp_path / "fake_kiwirecorder.py"
    p.write_text(FAKE.format(root=str(ROOT), tests=str(ROOT / "tests")), encoding="utf-8")
    monkeypatch.setattr(sdr_hub, "KIWIRECORDER", p)
    monkeypatch.setattr(sdr_hub, "kiwi_unreachable",
                        lambda host, port: "blocked" if host == "21439.proxy2.kiwisdr.com" else None)

    def status(host, port, timeout=8):
        sid, lat, lon = STATIONS[host]
        return (f"status=active\nusers=1\nusers_max=4\ngps=({lat}, {lon})\ngps_good=5\nfixes_min=20\n"
                f"tdoa_id={sid}\ntdoa_ch=4\n")
    monkeypatch.setattr(sdr_hub, "kiwi_status", status)
    monkeypatch.setattr(sdr_hub.bridge, "SPOOL", tmp_path / "spool")
    (tmp_path / "jobs").mkdir()
    return tmp_path


def wait_for(cond, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_runner_locates_and_reports(fake_tdoa):
    run = sdr_hub.TdoaRunner()
    got = []
    run.on_located = lambda res, who: got.append((res, who))
    with pytest.raises(ValueError):
        run.start(15295, sample_s=7)                                # not an offered sample length
    with pytest.raises(ValueError):
        run.start(15295, receivers="a:8073,b:8073")                 # fewer than 3 receivers
    st = run.start(15295, sample_s=10, who="me")
    assert st["state"] == "checking"
    assert wait_for(lambda: run.status()["state"] in ("done", "error")), run.status()
    st = run.status()
    assert st["state"] == "done", st
    res = st["result"]
    assert K.great_circle_km(res["lat"], res["lon"], *TX) < 30, res
    assert res["receivers"] == "DU6/PE1NSQ, HS0ZGZ, JKT" and got[0][1] == "me"
    assert [r["ok"] for r in st["receivers"]] == [True, True, True, False]   # the blocked spare is named, not used
    assert not list((fake_tdoa / "spool").glob("tdoa-*")), "the recordings are deleted"
    with pytest.raises(ValueError, match="wait"):
        run.start(15295)                                            # a pause between runs: shared receivers


def test_api_locate_writes_the_location(fake_tdoa, monkeypatch):
    import serve_local
    monkeypatch.setattr(serve_local, "PIPELINE_DIR", fake_tdoa / "jobs")
    monkeypatch.setattr(serve_local.Handler, "_kiwi_hub", None)
    srv = serve_local.make_server(fake_tdoa / "nexa.db", "127.0.0.1", 0, demo=True, intel_dir=fake_tdoa / "intel")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/kiwi"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"X-NEXA-Client": "1", "Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        return json.load(urllib.request.urlopen(req, timeout=10))

    try:
        with pytest.raises(urllib.error.HTTPError) as bad:
            call("/locate", {"freq_khz": 0})
        assert bad.value.code == 400
        assert call("/locate", {"freq_khz": 15295, "sample_s": 10})["state"] == "checking"
        assert wait_for(lambda: call("/status")["tdoa"]["state"] in ("done", "error"))
        assert call("/status")["tdoa"]["state"] == "done", call("/status")["tdoa"]
        loc = call("/location?khz=15295")["location"]
        assert loc["source"] == "NEXA TDoA (computed)" and loc["receivers"] == "DU6/PE1NSQ, HS0ZGZ, JKT"
        assert K.great_circle_km(loc["lat"], loc["lon"], *TX) < 30 and loc["radius_km"] > 0
        actions = [e["action"] for e in srv.nexa_db.audit(50)]
        assert "kiwi.locate" in actions and "client.tdoa_location" in actions
    finally:
        srv.shutdown()
