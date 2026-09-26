"""Report builder exports (src/report_export.py) -- every format, from a small
local database with real files on disk."""
import csv
import io
import json
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.localdb import Actor, LocalDB, now_iso  # noqa: E402
from src.report_export import build_export, dtg  # noqa: E402
import serve_local  # noqa: E402

EAVAN, JESSY = Actor("eavan"), Actor("jessy")


@pytest.fixture
def setup(tmp_path):
    db = LocalDB(tmp_path / "nexa.db")
    iq = (np.arange(2 * 4000, dtype=np.float32) / 8000).astype("<f4").tobytes()

    def put(rec_id, rel, data):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return rel

    base = {"created_at": now_iso(), "source": "upload", "model": "single", "n_windows": 7,
            "hop": 512, "duration_s": 4000 / 3.2e6, "verdict": "Hostile",
            "classes_detected": ["JAMMING"], "peak_probability": {"JAMMING": 0.9}, "tier_counts": {}}
    db.save_analysis({**base, "id": "a-f32", "file_name": "dwell.f32",
                      "file_path": put("a", "iq/a-f32/dwell.f32", iq), "file_sha256": "x"}, EAVAN)
    put("b", "iq/b-sig/rec.sigmf-meta", json.dumps({
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 3.2e6, "core:version": "1.2.0",
                   "core:hw": "test rig"}, "captures": [{"core:sample_start": 0, "core:frequency": 2.4e9}],
        "annotations": [{"core:sample_start": 0, "core:sample_count": 10, "core:label": "FHSS"}]}))
    db.save_analysis({**base, "id": "b-sig", "file_name": "rec.sigmf-data",
                      "file_path": put("b", "iq/b-sig/rec.sigmf-data", iq)}, EAVAN)
    db.save_analysis({**base, "id": "c-none", "source": "scenario", "verdict": "Military",
                      "classes_detected": ["FHSS"]}, EAVAN)
    c = db.create_correction({"analysis_id": "a-f32", "start_s": 0.0001, "end_s": 0.0009,
                              "predicted_labels": ["JAMMING"], "corrected_labels": ["FHSS"],
                              "reason": "hops on the waterfall"}, EAVAN)
    db.review_correction(c["id"], "approve", "agreed", JESSY)
    return db, tmp_path, iq


def export(db, tmp, fmt, sections=("summary", "captures", "corrections", "audit", "model", "iq"),
           ids=("a-f32", "b-sig", "c-none")):
    return build_export(db, tmp, ids=list(ids), sections=list(sections), fmt=fmt,
                        banner="UNCLASSIFIED // TEST", generated_by="eavan",
                        model_card={"name": "C2 ensemble", "classes": ["BPSK"]})


def test_dtg_format():
    from datetime import datetime, timezone
    assert dtg(datetime(2026, 9, 26, 14, 30, tzinfo=timezone.utc)) == "261430Z SEP 26"


def test_json_has_header_banner_and_sections(setup):
    db, tmp, _ = setup
    data, ctype, name, meta = export(db, tmp, "json")
    body = json.loads(data)
    assert body["report"]["classification"] == body["report_end"] == "UNCLASSIFIED // TEST"
    assert name.startswith("NEXA-") and name.endswith(".json") and ctype == "application/json"
    assert len(body["captures"]) == 3
    assert body["corrections"][0]["Human says"] == "FHSS" and body["corrections"][0]["Status"] == "approved"
    assert any(e["Action"] == "correction.approve" for e in body["audit"])
    assert body["model"][1] == {"Field": "name", "Value": "C2 ensemble"}


def test_only_selected_captures_and_sections(setup):
    db, tmp, _ = setup
    body = json.loads(export(db, tmp, "json", sections=["captures"], ids=["c-none"])[0])
    assert [r["Record id"] for r in body["captures"]] == ["c-none"] and "audit" not in body


def test_csv_zip_one_table_per_section(setup):
    db, tmp, _ = setup
    z = zipfile.ZipFile(io.BytesIO(export(db, tmp, "csv", sections=["captures", "corrections"])[0]))
    assert set(z.namelist()) == {"README.txt", "report_meta.csv", "captures.csv", "corrections.csv"}
    rows = list(csv.reader(io.StringIO(z.read("captures.csv").decode("utf-8-sig"))))
    assert rows[0][0] == "Record id" and len(rows) == 4
    assert "UNCLASSIFIED // TEST" in z.read("README.txt").decode()


def test_xlsx_is_a_real_workbook(setup):
    db, tmp, _ = setup
    data = export(db, tmp, "xlsx", sections=["summary", "captures"])[0]
    z = zipfile.ZipFile(io.BytesIO(data))
    wb = ElementTree.fromstring(z.read("xl/workbook.xml"))
    names = [s.get("name") for s in wb.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet")]
    assert names == ["Report", "Summary", "Captures"]
    try:
        import openpyxl
    except ImportError:
        return
    book = openpyxl.load_workbook(io.BytesIO(data))
    assert book["Captures"]["A1"].value == "Record id" and book["Captures"].max_row == 4
    assert book["Report"]["B2"].value.startswith("NEXA-")


def test_sigmf_bundle_carries_iq_and_approved_corrections(setup):
    db, tmp, iq = setup
    z = zipfile.ZipFile(io.BytesIO(export(db, tmp, "sigmf")[0]))
    manifest = json.loads(z.read("manifest.json"))
    recs = {r["record_id"]: r for r in manifest["recordings"]}
    assert recs["c-none"]["skipped"] == "no stored raw IQ"
    assert recs["a-f32"]["annotations_from_corrections"] == 1
    meta = json.loads(z.read(recs["a-f32"]["file"]))
    ann = meta["annotations"][0]
    assert ann["core:label"] == "FHSS" and ann["core:sample_start"] == 320 and "approved by jessy" in ann["core:comment"]
    assert z.read(recs["a-f32"]["file"].replace("-meta", "-data")) == iq
    # the uploaded SigMF keeps its own metadata and annotations
    meta_b = json.loads(z.read(recs["b-sig"]["file"]))
    assert meta_b["global"]["core:hw"] == "test rig" and meta_b["annotations"][0]["core:label"] == "FHSS"
    assert meta_b["captures"][0]["core:frequency"] == 2.4e9


def test_raw_bundle_is_byte_exact(setup):
    db, tmp, iq = setup
    z = zipfile.ZipFile(io.BytesIO(export(db, tmp, "raw")[0]))
    files = [n for n in z.namelist() if n.endswith((".f32", ".sigmf-data"))]
    assert len(files) == 2 and all(z.read(n) == iq for n in files)


@pytest.mark.parametrize("fmt, kwargs, msg", [
    ("pdf", {}, "format must be"),
    ("json", {"sections": []}, "at least one section"),
    ("json", {"ids": ["nope"]}, "no captures selected"),
])
def test_bad_requests(setup, fmt, kwargs, msg):
    db, tmp, _ = setup
    with pytest.raises(ValueError, match=msg):
        export(db, tmp, fmt, **kwargs)


def test_export_over_http_is_audited(setup):
    db, tmp, _ = setup
    srv = serve_local.make_server(db.path, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.server_address[1]}/api/export", method="POST",
            data=json.dumps({"ids": ["a-f32"], "sections": ["captures"], "format": "json"}).encode(),
            headers={"X-NEXA-Client": "1", "X-NEXA-Operator": "chua", "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert r.headers["Content-Disposition"].startswith("attachment")
            body = r.read()
            report_id = r.headers["X-Report-Id"]
    finally:
        srv.shutdown()
        srv.server_close()
    entry = LocalDB(db.path).audit(limit=1)[0]
    import hashlib
    assert entry["action"] == "report.export" and entry["actor"] == "chua"
    assert entry["target_id"] == report_id and entry["details"]["sha256"] == hashlib.sha256(body).hexdigest()
