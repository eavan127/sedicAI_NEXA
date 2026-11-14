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
            "la6lutha.proxy.kiwisdr.com": ("LA6LU", 12.79, 99.97),
            "hs0zqs.proxy.kiwisdr.com": ("HS0ZQS", 13.87, 100.50),
            "21439.proxy2.kiwisdr.com": ("9M2AMH", 3.10, 101.60),
            "ptlen.proxy.kiwisdr.com": ("ptlen", -6.97, 107.55),          # further receivers (public list)
            "jplin53c.proxy.kiwisdr.com": ("jplin53c", 25.11, 121.00),
            "kch.ddns.net": ("KCH", 1.55, 110.35),                        # Kuching, Borneo (made up)
            "sgp.ddns.net": ("SGP", 1.35, 103.82)}                        # Singapore (made up)
SILENT = "21023.proxy.kiwisdr.com"     # passes every check, then sends nothing while recording


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
from test_kiwi_tdoa import synth, STATIONS, SILENT
from src.kiwi_tdoa import write_kiwi_wav
a = sys.argv[1:]
opt = lambda k: a[a.index(k) + 1]
hosts, ids, d = opt("-s").split(","), opt("--station").split(","), opt("-d")
assert "--kiwi-wav" in a and opt("-m") == "iq" and opt("-f") == "15295"
for i, (h, sid) in enumerate(zip(hosts, ids)):
    if h == SILENT:
        continue
    t0, z, fs = synth(*STATIONS[h][1:], seconds=float(opt("--tlimit")) / 2, seed=sum(map(ord, h)))   # one process per receiver
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
    monkeypatch.setattr(sdr_hub, "kiwi_stream_host",            # its page answers, its audio stream does not
                        lambda host, port: (None, "its audio stream opens but sends nothing")
                        if host == "la6lutha.proxy.kiwisdr.com" else (host, None))

    def status(host, port, timeout=8):
        sid, lat, lon = STATIONS[host]
        return (f"status=active\nusers=1\nusers_max=4\ngps=({lat}, {lon})\ngps_good=5\nfixes_min=20\n"
                f"tdoa_id={sid}\ntdoa_ch=4\n")
    monkeypatch.setattr(sdr_hub, "kiwi_status", status)
    monkeypatch.setattr(sdr_hub, "public_tdoa_receivers", lambda: [])        # no network in tests
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
    # 5 recorded at once; HS0ZGZ fell silent while recording and the other 4 still give the place
    assert res["receivers"] == "JKT, HS0ZQS, jplin53c, ptlen" and got[0][1] == "me"
    assert [r["ok"] for r in st["receivers"]] == [True, True, True, True, True, False, False]   # dead stream, blocked
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
        assert loc["source"] == "NEXA TDoA (computed)" and loc["receivers"] == "JKT, HS0ZQS, jplin53c, ptlen"
        assert K.great_circle_km(loc["lat"], loc["lon"], *TX) < 30 and loc["radius_km"] > 0
        actions = [e["action"] for e in srv.nexa_db.audit(50)]
        assert "kiwi.locate" in actions and "client.tdoa_location" in actions
    finally:
        srv.shutdown()


def test_recorder_errors_are_said_in_words():
    assert sdr_hub.plain_recorder_error("2026-11-14 16:50:19,421 pid 14544 HTTP 307 redirect") == (
        "a receiver's proxy kept redirecting its audio stream and it never started")
    assert sdr_hub.plain_recorder_error("2026-11-14 16:50:19,421 pid 1 Too busy now") == "Too busy now"


def _pairs(stations, tx, offsets_us=None, bad=None):
    """Exact lags for a transmitter at tx; offsets_us: a receiver's clock error; bad: (i, j) pair made wrong."""
    off = offsets_us or {}
    t = [K.great_circle_km(st["lat"], st["lon"], *tx) / K.C_KM_S + off.get(i, 0) * 1e-6 for i, st in enumerate(stations)]
    out = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            lag = t[i] - t[j] + (0.012 if (i, j) == bad else 0.0)
            out.append({"i": i, "j": j, "a": stations[i]["id"], "b": stations[j]["id"], "lag_s": lag,
                        "sigma_s": 21e-6, "lag_us": round(lag * 1e6, 1), "sigma_us": 21.0, "blocks": 9, "used": 9,
                        "peak": 0.4 if (i, j) == bad else 0.7})
    return out


FIVE = [{"id": sid, "lat": lat, "lon": lon} for h, (sid, lat, lon) in STATIONS.items()
        if h in ("dx6ran.ddns.net", "rajawali.proxy.kiwisdr.com", "hs0zqs.proxy.kiwisdr.com",
                 "jplin53c.proxy.kiwisdr.com", "ptlen.proxy.kiwisdr.com")]


def test_a_contradicting_pair_is_dropped():
    res = K.combine(FIVE[:4], _pairs(FIVE[:4], TX, bad=(2, 3)))          # a wrong correlation peak, 12 ms off
    assert [(p["a"], p["b"]) for p in res["dropped_pairs"]] == [(FIVE[2]["id"], FIVE[3]["id"])]
    assert K.great_circle_km(res["lat"], res["lon"], *TX) < 20 and not res["excluded"]


def test_five_receivers_name_the_one_with_wrong_timing_four_cannot():
    """Why 5 are recorded: DU6 0.8 ms early (as measured on 14 Nov). Five name it and
    find the place; with four the point just moves (here ~265 km) and still fits."""
    for off in (500, 800, 2000):
        five = K.combine(FIVE, _pairs(FIVE, TX, offsets_us={0: -off}))
        assert [x["id"] for x in five["excluded"]] == ["DU6/PE1NSQ"], (off, five)
        assert abs(five["excluded"][0]["offset_us"] + off) < 100
        assert K.great_circle_km(five["lat"], five["lon"], *TX) < 20, five
    four = K.combine(FIVE[:4], _pairs(FIVE[:4], TX, offsets_us={0: -800}))
    assert not four["excluded"] and K.great_circle_km(four["lat"], four["lon"], *TX) > 100
    three = K.combine(FIVE[1:4], _pairs(FIVE[1:4], TX))
    assert three["warning"].startswith("3 receivers") and not three["cross_checked"]


def test_further_receivers_surround_the_centre_without_repeats():
    rows = [{"host": "a.proxy.kiwisdr.com", "port": 8073, "id": "A", "lat": -7.0, "lon": 107.5},      # south
            {"host": "a2.proxy.kiwisdr.com", "port": 8073, "id": "A2", "lat": -7.2, "lon": 107.6},    # same site
            {"host": "b.ddns.net", "port": 8073, "id": "B", "lat": 25.1, "lon": 121.0},               # north-east
            {"host": "c.ddns.net", "port": 8073, "id": "C", "lat": -6.0, "lon": 106.0},               # south, further
            {"host": "21023.proxy2.kiwisdr.com", "port": 8073, "id": "dup", "lat": 17, "lon": 103},   # tried already
            {"host": "far.ddns.net", "port": 8073, "id": "F", "lat": -34.3, "lon": 115.1},            # > 3500 km
            {"host": "dx6ran.ddns.net", "port": 8073, "id": "DU6", "lat": 10.7, "lon": 123.0}]        # known bad timing
    got = sdr_hub.further_receivers(rows, ["21023.proxy.kiwisdr.com"], n=5)
    assert [g[2] for g in got] == ["C", "A", "B"], got        # nearest first, one per sector; A2, dup, F, DU6 out


def test_runner_looks_further_when_the_usual_receivers_fail(fake_tdoa, monkeypatch):
    usual = {c[0] for c in sdr_hub.TDOA_RECEIVERS} - {"rajawali.proxy.kiwisdr.com", "hs0zqs.proxy.kiwisdr.com"}
    monkeypatch.setattr(sdr_hub, "kiwi_unreachable", lambda host, port: "blocked" if host in usual else None)
    monkeypatch.setattr(sdr_hub, "public_tdoa_receivers", lambda: [
        {"host": h, "port": 8073, "id": STATIONS[h][0], "lat": STATIONS[h][1], "lon": STATIONS[h][2]}
        for h in ("kch.ddns.net", "sgp.ddns.net", "jplin53c.proxy.kiwisdr.com")])   # jplin53c: tried already
    run = sdr_hub.TdoaRunner()
    run.start(15295, sample_s=10)
    assert wait_for(lambda: run.status()["state"] in ("done", "error")), run.status()
    st = run.status()
    assert st["state"] == "done", st
    assert set(st["result"]["further"]) == {"KCH", "SGP"}
    assert K.great_circle_km(st["result"]["lat"], st["result"]["lon"], *TX) < 30, st["result"]


def _two_paths(seconds=20, direct_share=0.3, extra_s=0.0009):
    """Bangkok hears Kajang directly; Taiwan (far) hears it with an extra hop
    (extra_s later) for most of the time and directly for the last direct_share."""
    sb = STATIONS["hs0zqs.proxy.kiwisdr.com"]
    st = STATIONS["jplin53c.proxy.kiwisdr.com"]
    t0b, zb, fs = synth(sb[1], sb[2], seconds=seconds, seed=1, start=0.1)
    t0t, zt, _ = synth(st[1], st[2], seconds=seconds, seed=2, start=0.1)
    # the same receiver, its signal later by extra_s: what a second hop does
    t0x, zx, _ = synth(st[1], st[2], seconds=seconds, seed=2, start=0.1 - extra_s)
    cut = int(len(zt) * (1 - direct_share))
    z = np.concatenate([zx[:cut], zt[cut:]])
    t = t0t + np.arange(len(z)) / fs
    tb = t0b + np.arange(len(zb)) / fs
    return (tb, zb, fs), (t, z, fs), sb, st


def test_two_ionospheric_paths_the_earliest_at_the_far_receiver_wins():
    b, t, sb, st = _two_paths()
    want = (K.great_circle_km(sb[1], sb[2], *TX) - K.great_circle_km(st[1], st[2], *TX)) / K.C_KM_S
    r = K.pair_lag(b, t, max_dt_s=0.012, far="b")                       # Taiwan is the far one
    assert r["paths"] == 2 and abs(r["lag_s"] - want) < 30e-6, (r, want)
    late = K.pair_lag(b, t, max_dt_s=0.012)                             # no hint: the majority, the late hop
    assert abs(late["lag_s"] - (want - 0.0009)) < 30e-6


def test_runs_are_merged_by_median_per_pair():
    st = FIVE[1:5]
    good = _pairs(st, TX)
    odd = [dict(p, lag_s=p["lag_s"] + (0.0009 if p["j"] == 2 else 0.0)) for p in good]   # Taiwan's hop in one run
    stations, merged = K.merge_runs([(st, good), (st, odd), (st, good)])
    res = K.combine(stations, merged)
    assert all(p["runs"] == 3 for p in res["pairs"]) and K.great_circle_km(res["lat"], res["lon"], *TX) < 20


def test_leaving_out_a_receiver_needs_four_left_to_compare():
    """A noisy receiver paired with only one other: leaving that other out leaves 3,
    which always fit perfectly -- it must not win the comparison (14 Nov, Hua Hin)."""
    st = FIVE[1:5] + [{"id": "LA6LU", "lat": 12.79, "lon": 99.97}]
    ps = _pairs(FIVE[1:5], TX)
    ps.append({"i": 2, "j": 4, "a": st[2]["id"], "b": "LA6LU", "lag_s": 0.0131, "sigma_s": 30e-6, "lag_us": 13100.0,
               "sigma_us": 30.0, "blocks": 9, "used": 6, "peak": 0.5})
    res = K.combine(st, ps)
    assert [x["id"] for x in res["excluded"]] == ["LA6LU"], res["excluded"]
    assert K.great_circle_km(res["lat"], res["lon"], *TX) < 20


def test_a_far_receiver_on_two_hops_gives_a_second_possible_place():
    """14 Nov, set C: Taiwan (3200 km) heard only the two-hop path, ~0.7 ms late; the
    place slid 440 km. Without a second path to compare, NEXA cannot know, so it
    shows where the transmitter would be if Taiwan had heard it in one hop."""
    st = FIVE[1:5]                                          # JKT, HS0ZQS, ptlen, jplin53c
    tw = next(i for i, x in enumerate(st) if x["id"] == "jplin53c")
    d = float(K.great_circle_km(st[tw]["lat"], st[tw]["lon"], *TX))
    hop = (K._skywave_km(d, 2) - K._skywave_km(d, 1)) / K.C_KM_S
    ps = [dict(p, paths=1, lag_s=p["lag_s"] + (hop if p["i"] == tw else -hop if p["j"] == tw else 0.0))
          for p in _pairs(st, TX)]
    res = K.combine(st, ps)
    alt = res["alternative"]
    assert alt and alt["receiver"] == "jplin53c", res
    assert K.great_circle_km(alt["lat"], alt["lon"], *TX) < 40, alt
    seen_both = K.combine(st, [dict(p, paths=2) if tw in (p["i"], p["j"]) else dict(p, paths=1) for p in _pairs(st, TX)])
    assert seen_both["alternative"] is None              # both paths seen: the direct one was already kept


def test_runs_that_disagree_by_a_hop_keep_the_earliest_at_the_far_receiver():
    """14 Nov: Taiwan was on its direct path in two recordings and two hops in a third;
    merged, the direct one must win (11 km from Kajang), not the median."""
    st = FIVE[1:5]
    tw = next(i for i, x in enumerate(st) if x["id"] == "jplin53c")
    direct = [dict(p, far="b" if p["j"] == tw else "a" if p["i"] == tw else None) for p in _pairs(st, TX)]
    hop = 0.0007
    late = [dict(p, lag_s=p["lag_s"] + (hop if p["i"] == tw else -hop if p["j"] == tw else 0.0)) for p in direct]
    for runs in ([late, direct], [late, late, direct]):          # even when the late path is the majority
        stations, merged = K.merge_runs([(st, ps) for ps in runs])
        res = K.combine(stations, merged)
        assert K.great_circle_km(res["lat"], res["lon"], *TX) < 20, res
        assert res["alternative"] is None


def test_a_receiver_at_half_the_sample_rate_still_counts(tmp_path):
    """Some KiwiSDRs send 6 kS/s IQ, most 12 kS/s ("sample rates differ (11999 vs
    6000 S/s)" stopped a whole run on 14 Nov). It is brought down to the common rate."""
    from scipy.signal import resample_poly
    recs = []
    for i, host in enumerate(["dx6ran.ddns.net", "21023.proxy.kiwisdr.com", "rajawali.proxy.kiwisdr.com"]):
        sid, lat, lon = STATIONS[host]
        t0, z, fs = synth(lat, lon, seconds=8, seed=i)
        if i == 1:                                          # this one at half rate
            z, fs = resample_poly(z, 1, 2).astype(np.complex64), fs / 2
        K.write_kiwi_wav(tmp_path / f"{i}.wav", t0, z, fs)
        t, zz, fs2 = K.read_kiwi_wav(tmp_path / f"{i}.wav")
        recs.append({"id": sid, "lat": lat, "lon": lon, "t": t, "z": zz, "fs": fs2})
    res = K.solve(recs)
    assert len(res["pairs"]) == 3 and K.great_circle_km(res["lat"], res["lon"], *TX) < 40, res


def test_last_resort_receivers_rather_than_no_answer(fake_tdoa, monkeypatch):
    """14 Nov evening on FOREFRONT Wi-Fi only 2 usual receivers worked and the run
    stopped. Now the avoided receiver (DU6, known timing problem) is tried last, and
    the answer says it is rough and why."""
    usual = {c[0] for c in sdr_hub.TDOA_RECEIVERS} - {"rajawali.proxy.kiwisdr.com", "hs0zqs.proxy.kiwisdr.com"}
    monkeypatch.setattr(sdr_hub, "kiwi_unreachable", lambda host, port: "blocked" if host in usual else None)
    run = sdr_hub.TdoaRunner()
    run.start(15295, sample_s=10)
    assert wait_for(lambda: run.status()["state"] in ("done", "error")), run.status()
    st = run.status()
    assert st["state"] == "done", st
    res = st["result"]
    assert [x["id"] for x in res["last_resort"]] == ["DU6/PE1NSQ"] and res["sure"] != "good"
    assert "DU6/PE1NSQ" in res["receivers"]


def test_too_few_receivers_is_said_plainly(fake_tdoa, monkeypatch):
    monkeypatch.setattr(sdr_hub, "kiwi_unreachable",
                        lambda host, port: None if host == "rajawali.proxy.kiwisdr.com" else "blocked")
    run = sdr_hub.TdoaRunner()
    run.start(15295, sample_s=10)
    assert wait_for(lambda: run.status()["state"] in ("done", "error")), run.status()
    err = run.status()["error"]
    assert err.startswith("only 1 receiver can be used right now (JKT); at least 3 are needed"), err
