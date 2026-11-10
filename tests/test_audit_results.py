"""The Python half of NEXA (pipeline runs, live bridge recordings, Comms Intel)
reaches the same append-only audit log as the browser half: the server logs
each finished result once, the first time it sees it."""
import json
import sys
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_local  # noqa: E402


def _get(base, path):
    return json.loads(urllib.request.urlopen(base + path).read())


def test_live_and_intel_results_are_audited_once(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    run = jobs / "20261110-101500-abcd"
    run.mkdir(parents=True)
    (run / "job.json").write_text(json.dumps({"id": run.name, "source": "live", "live": True,
                                              "file": "LIVE · SDRconnect ws://10.0.0.2:5454",
                                              "detector": "live voice path"}))
    (run / "tracks.json").write_text(json.dumps({"duration_ms": 6000.0, "tracks": [
        {"answer": "NBFM", "f_lo_khz": -40.5, "f_hi_khz": -38.0,
         "listen": {"kind": "NBFM", "transcript": "Bravo Two, this is Alpha One. Radio check, over."}}]}))
    intel = tmp_path / "intel"
    intel.mkdir()
    (intel / "index.json").write_text(json.dumps({"recordings": [
        {"id": "live-20261110", "file": "track_1.wav", "kind": "voice", "transmissions": 1,
         "category": "radio check", "summary": "Alpha One asks Bravo Two for a radio check."}]}))
    monkeypatch.setattr(serve_local, "PIPELINE_DIR", jobs)
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0, demo=True, intel_dir=intel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        _get(base, "/api/pipeline")
        _get(base, "/api/pipeline")                       # the page polls: still one entry each
        _get(base, "/api/intel")
    finally:
        srv.shutdown()
    log = srv.nexa_db.audit(100)
    live = [e for e in log if e["action"] == "live.result"]
    rec = [e for e in log if e["action"] == "intel.recording"]
    assert len(live) == 1 and len(rec) == 1
    d = live[0]["details"] if isinstance(live[0]["details"], dict) else json.loads(live[0]["details"])
    assert d["n_tracks"] == 1 and d["tracks"][0]["answer"] == "NBFM" and "Radio check" in d["tracks"][0]["text"]
    assert live[0]["actor"] == "live bridge"
    assert srv.nexa_db.verify_audit()["ok"]                # the hash chain still holds
