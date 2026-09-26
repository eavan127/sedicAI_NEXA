"""Local database (src/localdb.py) and its server (scripts/serve_local.py).

The audit log is the part that must not quietly go wrong: these tests edit
the SQLite file behind the app's back and check verify_audit() notices.
"""
import json
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.localdb import Actor, LocalDB, now_iso  # noqa: E402
import serve_local  # noqa: E402

ME = Actor("eavan", "operator", "127.0.0.1")


def record(i="a1", **over):
    return {
        "id": i, "created_at": f"2026-09-26T10:00:0{len(i) % 10}.000Z", "source": "upload",
        "case_note": "", "file_name": "x.sigmf", "model": "ensemble",
        "n_windows": 10, "hop": 256, "snr_db": 2.0, "snr_capped": False,
        "classes_detected": ["JAMMING"], "peak_probability": {"JAMMING": 0.9},
        "tier_counts": {"Hostile": 3}, "verdict": "Hostile", **over,
    }


@pytest.fixture
def db(tmp_path):
    return LocalDB(tmp_path / "nexa.db")


# --- analyses ------------------------------------------------------------------

def test_round_trip_keeps_json_and_bool_columns(db):
    saved = db.save_analysis(record(snr_capped=True), ME)
    assert saved["classes_detected"] == ["JAMMING"]
    assert saved["peak_probability"] == {"JAMMING": 0.9}
    assert saved["snr_capped"] is True
    assert saved["operator"] == "eavan"


def test_unknown_fields_ignored_and_required_enforced(db):
    db.save_analysis(record(not_a_column="x"), ME)
    with pytest.raises(ValueError):
        db.save_analysis({"id": "b"}, ME)


def test_list_newest_first(db):
    db.save_analysis(record("a", created_at="2026-09-01T00:00:00Z"), ME)
    db.save_analysis(record("b", created_at="2026-09-02T00:00:00Z"), ME)
    assert [r["id"] for r in db.list_analyses()] == ["b", "a"]


def test_capture_with_corrections_cannot_be_deleted(db):
    db.save_analysis(record("keep"), ME)
    db.save_analysis(record("drop"), ME)
    with sqlite3.connect(db.path) as con:
        con.execute("INSERT INTO corrections (id, analysis_id, created_at, operator)"
                    " VALUES ('c1', 'keep', '2026-09-26', 'eavan')")
    with pytest.raises(PermissionError):
        db.delete_analysis("keep", ME)
    assert db.clear_analyses(ME) == 1
    assert [r["id"] for r in db.list_analyses()] == ["keep"]


def test_corrections_are_never_deleted(db):
    db.save_analysis(record("a"), ME)
    with sqlite3.connect(db.path) as con:
        con.execute("INSERT INTO corrections (id, analysis_id, created_at, operator)"
                    " VALUES ('c1', 'a', '2026-09-26', 'eavan')")
        with pytest.raises(sqlite3.IntegrityError, match="never deleted"):
            con.execute("DELETE FROM corrections")


# --- audit log -----------------------------------------------------------------

def test_every_change_is_audited_with_who_and_what(db):
    db.save_analysis(record("a"), ME)
    db.delete_analysis("a", ME)
    actions = [e["action"] for e in reversed(db.audit())]
    assert actions == ["db.create", "analysis.create", "analysis.delete"]
    delete = db.audit(limit=1)[0]
    assert delete["actor"] == "eavan" and delete["client"] == "127.0.0.1"
    assert delete["details"]["deleted"]["verdict"] == "Hostile"   # the row survives in the log


def test_chain_verifies(db):
    for i in range(5):
        db.save_analysis(record(f"r{i}"), ME)
    v = db.verify_audit()
    assert v["ok"] and v["entries"] == 6


def test_app_cannot_rewrite_the_log(db):
    db.save_analysis(record(), ME)
    with sqlite3.connect(db.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE audit_log SET actor = 'someone else'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("DELETE FROM audit_log")


def _force(db, sql):
    """Edit the log the way an attacker holding the file would: drop the
    guard triggers first."""
    with sqlite3.connect(db.path) as con:
        con.execute("DROP TRIGGER audit_log_no_update")
        con.execute("DROP TRIGGER audit_log_no_delete")
        con.execute(sql)


def test_edited_entry_is_detected(db):
    for i in range(3):
        db.save_analysis(record(f"r{i}"), ME)
    _force(db, "UPDATE audit_log SET actor = 'mallory' WHERE seq = 2")
    v = db.verify_audit()
    assert not v["ok"] and v["broken_at"] == 2 and "changed" in v["reason"]


def test_deleted_entry_is_detected(db):
    for i in range(3):
        db.save_analysis(record(f"r{i}"), ME)
    _force(db, "DELETE FROM audit_log WHERE seq = 2")
    v = db.verify_audit()
    assert not v["ok"] and v["broken_at"] == 3 and "missing" in v["reason"]


# --- server ----------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()
    srv.server_close()


def call(url, method="GET", body=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"X-NEXA-Client": "1", "Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_health_identifies_the_server(server):
    base, _ = server
    status, _, body = call(base + "/api/health")
    assert status == 200 and json.loads(body)["app"] == "nexa-local"


def test_store_list_and_audit_over_http(server):
    base, _ = server
    status, _, _ = call(base + "/api/analyses", "POST", record("h1"),
                        headers={"X-NEXA-Operator": "jessy"})
    assert status == 201
    rows = json.loads(call(base + "/api/analyses")[2])
    assert rows[0]["id"] == "h1" and rows[0]["operator"] == "jessy"
    assert json.loads(call(base + "/api/audit/verify")[2])["ok"]


def test_writes_need_the_client_header(server):
    base, _ = server
    status, _, _ = call(base + "/api/analyses", "POST", record(), headers={"X-NEXA-Client": ""})
    assert status == 403


def test_foreign_host_header_refused(server):
    base, _ = server
    status, _, _ = call(base + "/api/health", headers={"Host": "evil.example"})
    assert status == 403


def test_capture_upload_is_fingerprinted(server):
    base, tmp = server
    payload = b"\x00\x01" * 1000
    status, _, body = call(base + "/api/captures/abc-1?name=..%2F..%2Fevil%20name.sigmf-data", "PUT",
                           raw=payload, headers={"Content-Type": "application/octet-stream"})
    out = json.loads(body)
    assert status == 201 and out["bytes"] == 2000
    stored = tmp / out["file_path"]
    assert stored.read_bytes() == payload
    assert stored.parent == tmp / "iq" / "abc-1"          # no escaping the folder
    audit = json.loads(call(base + "/api/audit?action=iq.")[2])
    assert audit[0]["details"]["sha256"] == out["file_sha256"]


def test_client_events_cannot_impersonate_server_actions(server):
    base, _ = server
    assert call(base + "/api/audit", "POST", {"action": "analysis.delete"})[0] == 400
    assert call(base + "/api/audit", "POST", {"action": "client.report_export",
                                              "details": {"format": "pdf"}})[0] == 201


def test_static_modules_served_as_javascript(server):
    base, _ = server
    status, headers, _ = call(base + "/main.js")
    assert status == 200 and headers["Content-Type"].startswith("text/javascript")


# --- corrections (human in the loop) --------------------------------------------

EXPERT = Actor("jessy", "operator", "127.0.0.1")


def _with_iq(db, i="c1"):
    db.save_analysis(record(i, duration_s=0.05, file_path=f"iq/{i}/x.f32", file_sha256="ab" * 32), ME)
    return i


def correction(analysis_id, **over):
    return {"analysis_id": analysis_id, "start_s": 0.01, "end_s": 0.02,
            "predicted_labels": ["LFM_RADAR"], "corrected_labels": ["FHSS"],
            "reason": "hops visible on the waterfall", **over}


def test_classes_match_the_model_config():
    from src.config import CLASSES as cfg_classes
    from src.localdb import CLASSES
    assert list(CLASSES) == list(cfg_classes)


def test_correction_created_with_evidence_and_audited(db):
    c = db.create_correction(correction(_with_iq(db)), ME)
    assert c["status"] == "pending" and c["operator"] == "eavan"
    assert c["corrected_labels"] == ["FHSS"] and c["iq_path"] == "iq/c1/x.f32"
    entry = db.audit(limit=1)[0]
    assert entry["action"] == "correction.create" and entry["details"]["reason"].startswith("hops")


@pytest.mark.parametrize("over, msg", [
    ({"corrected_labels": []}, "at least one class"),
    ({"corrected_labels": ["LFM_RADAR"]}, "already said"),
    ({"corrected_labels": ["TANK"]}, "unknown class"),
    ({"reason": ""}, "reason is required"),
    ({"start_s": 0.03, "end_s": 0.02}, "start < end"),
    ({"end_s": 9.0}, "past the end"),
])
def test_bad_corrections_refused(db, over, msg):
    with pytest.raises(ValueError, match=msg):
        db.create_correction(correction(_with_iq(db), **over), ME)


def test_correction_needs_raw_iq(db):
    db.save_analysis(record("noiq", duration_s=0.05), ME)
    with pytest.raises(ValueError, match="no stored raw IQ"):
        db.create_correction(correction("noiq"), ME)
    # ...unless the IQ was uploaded for it afterwards
    db.record_file("noiq", "capture.f32", "iq/noiq/capture.f32", 10, "cd" * 32, ME)
    assert db.create_correction(correction("noiq"), ME)["iq_sha256"] == "cd" * 32


def test_four_eyes_rule(db):
    c = db.create_correction(correction(_with_iq(db)), ME)
    with pytest.raises(PermissionError, match="Four-eyes"):
        db.review_correction(c["id"], "approve", "", ME)
    with pytest.raises(PermissionError, match="Set your name"):
        db.review_correction(c["id"], "approve", "", Actor("operator"))
    done = db.review_correction(c["id"], "approve", "agreed", EXPERT)
    assert done["status"] == "approved" and done["reviewed_by"] == "jessy"
    assert db.audit(limit=1)[0]["action"] == "correction.approve"


def test_reviewed_once_and_content_frozen(db):
    c = db.create_correction(correction(_with_iq(db)), ME)
    db.review_correction(c["id"], "reject", "not convinced", EXPERT)
    with pytest.raises(PermissionError, match="already rejected"):
        db.review_correction(c["id"], "approve", "", Actor("chua"))
    with sqlite3.connect(db.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="read-only"):
            con.execute("UPDATE corrections SET status = 'approved'")
    c2 = db.create_correction(correction(_with_iq(db, "c2")), ME)
    with sqlite3.connect(db.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="cannot be edited"):
            con.execute("UPDATE corrections SET corrected_labels = '[\"JAMMING\"]' WHERE id = ?", (c2["id"],))


def test_reject_needs_a_reason(db):
    c = db.create_correction(correction(_with_iq(db)), ME)
    with pytest.raises(ValueError, match="why"):
        db.review_correction(c["id"], "reject", "", EXPERT)


def test_stats_count_approved_labels(db):
    a = db.create_correction(correction(_with_iq(db, "s1")), ME)
    db.create_correction(correction(_with_iq(db, "s2")), ME)
    db.review_correction(a["id"], "approve", "", EXPERT)
    s = db.correction_stats()
    assert s["pending"] == 1 and s["approved"] == 1 and s["approved_per_class"]["FHSS"] == 1


def test_v1_database_migrates_in_place(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as con:            # a schema-v1 corrections table
        con.execute("CREATE TABLE corrections (id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL,"
                    " created_at TEXT NOT NULL, operator TEXT NOT NULL, start_s REAL, end_s REAL,"
                    " predicted_labels TEXT NOT NULL DEFAULT '[]', corrected_labels TEXT NOT NULL"
                    " DEFAULT '[]', reason TEXT DEFAULT '', model TEXT, status TEXT NOT NULL"
                    " DEFAULT 'pending', reviewed_by TEXT, reviewed_at TEXT, review_note TEXT)")
    db = LocalDB(path)
    with sqlite3.connect(path) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(corrections)")}
    assert {"iq_path", "iq_sha256"} <= cols and db.verify_audit()["ok"]


def test_corrections_over_http(server):
    base, _ = server
    call(base + "/api/analyses", "POST", record("h9", duration_s=0.05, file_path="iq/h9/a.f32"))
    status, _, body = call(base + "/api/corrections", "POST", correction("h9"),
                           headers={"X-NEXA-Operator": "eavan"})
    assert status == 201
    cid = json.loads(body)["id"]
    status, _, body = call(base + f"/api/corrections/{cid}/review", "POST", {"decision": "approve"},
                           headers={"X-NEXA-Operator": "eavan"})
    assert status == 409 and "Four-eyes" in json.loads(body)["error"]
    status, _, _ = call(base + f"/api/corrections/{cid}/review", "POST", {"decision": "approve"},
                        headers={"X-NEXA-Operator": "jessy"})
    assert status == 200
    assert json.loads(call(base + "/api/corrections/stats")[2])["approved"] == 1


# --- retraining trigger (step 5) -------------------------------------------------

def test_trigger_quiet_on_an_empty_database(db):
    s = db.retrain_status()
    assert not s["triggered"] and not s["recommended"] and s["analysed"] == 0
    assert [c["id"] for c in s["conditions"]] == ["volume", "rate", "data", "cooldown"]
    assert s["conditions"][3]["met"]            # never retrained: no cooldown


def test_trigger_fires_on_correction_rate_and_needs_data(db):
    rules = {"min_reviewed": 10, "min_per_class": 2}
    for i in range(10):
        db.save_analysis(record(f"t{i}", created_at=now_iso(),
                                duration_s=0.05, file_path=f"iq/t{i}/x.f32"), ME)
    for i in range(2):                           # 2 of 10 corrected = 20% > 15%
        db.create_correction(correction(f"t{i}"), ME)
    s = db.retrain_status(rules=rules)
    assert s["analysed"] == 10 and s["corrected"] == 2 and abs(s["rate"] - 0.2) < 1e-9
    assert s["triggered"] and not s["recommended"]          # nothing approved yet
    for c in db.list_corrections(status="pending"):
        db.review_correction(c["id"], "approve", "", EXPERT)
    s = db.retrain_status(rules=rules)
    assert s["approved_new_per_class"]["FHSS"] == 2 and s["recommended"]


def test_rejected_corrections_do_not_count_toward_the_rate(db):
    rules = {"min_reviewed": 1}
    db.save_analysis(record("r1", created_at=now_iso(),
                            duration_s=0.05, file_path="iq/r1/x.f32"), ME)
    c = db.create_correction(correction("r1"), ME)
    db.review_correction(c["id"], "reject", "wrong", EXPERT)
    assert db.retrain_status(rules=rules)["corrected"] == 0


def test_trigger_status_over_http(server):
    base, _ = server
    s = json.loads(call(base + "/api/retrain/status")[2])
    assert "conditions" in s and s["rules"]["max_rate"] == 0.15
