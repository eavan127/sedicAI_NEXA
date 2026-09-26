"""Retraining from human feedback, end to end: job rules, the real training
script (small synthetic settings), four-eyes model approval, activation, the
served model file, and roll back."""
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.localdb import Actor, LocalDB, now_iso  # noqa: E402
import serve_local  # noqa: E402

EAVAN, JESSY, CHUA = Actor("eavan"), Actor("jessy"), Actor("chua")


@pytest.fixture
def db(tmp_path):
    return LocalDB(tmp_path / "nexa.db")


def seed_feedback(db, tmp_path):
    """One stored capture with raw IQ and one approved correction on it."""
    from src.generators.jamming import random_jamming_example
    rng = np.random.default_rng(3)
    out = random_jamming_example(3_200_000, 0.004, rng)
    iq = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.complex64)
    iq = iq + (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))).astype(np.complex64) * 0.3
    raw = np.empty(2 * len(iq), "<f4")
    raw[0::2], raw[1::2] = iq.real, iq.imag
    (tmp_path / "iq" / "cap1").mkdir(parents=True)
    (tmp_path / "iq" / "cap1" / "cap.f32").write_bytes(raw.tobytes())
    db.save_analysis({"id": "cap1", "created_at": now_iso(), "source": "upload", "verdict": "Military",
                      "duration_s": len(iq) / 3.2e6, "classes_detected": ["FHSS"],
                      "file_path": "iq/cap1/cap.f32", "file_sha256": "ab" * 32}, EAVAN)
    c = db.create_correction({"analysis_id": "cap1", "start_s": 0.0, "end_s": 0.003,
                              "predicted_labels": ["FHSS"], "corrected_labels": ["JAMMING"],
                              "reason": "wideband energy, jammer"}, EAVAN)
    db.review_correction(c["id"], "approve", "", JESSY)


# --- job rules ---------------------------------------------------------------------

def test_start_needs_name_reason_and_override(db):
    with pytest.raises(PermissionError, match="Set your name"):
        db.start_retrain("because", True, Actor("operator"))
    with pytest.raises(ValueError, match="why"):
        db.start_retrain("", True, EAVAN)
    with pytest.raises(PermissionError, match="override"):
        db.start_retrain("urgent new threat", False, EAVAN)       # rules not met
    job = db.start_retrain("urgent new threat", True, EAVAN)
    assert job["status"] == "queued" and job["override"]
    entry = db.audit(limit=1)[0]
    assert entry["action"] == "retrain.start" and entry["details"]["override"] is True
    assert entry["details"]["rules_met"]["volume"] is False
    with pytest.raises(PermissionError, match="already running"):
        db.start_retrain("again", True, CHUA)


def test_rollback_with_nothing_active(db):
    with pytest.raises(ValueError, match="already on the shipped model"):
        db.rollback_model("", EAVAN)


# --- the real script, end to end -----------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retrain")
    db = LocalDB(tmp / "nexa.db")
    seed_feedback(db, tmp)
    job = db.start_retrain("test run", True, EAVAN)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrain_from_feedback.py"), "--db", str(db.path),
         "--job", job["id"], "--epochs", "1", "--replay", "200", "--gate", "synthetic",
         "--synthetic-per-class", "15"],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    return db, tmp, job, proc


def test_script_registers_a_candidate(trained):
    db, tmp, job, proc = trained
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    done = db.get_job(job["id"])
    assert done["status"] == "succeeded" and done["model_version"].startswith("ft-")
    m = db.get_model(done["model_version"])
    assert m["status"] == "candidate"
    assert (tmp / m["path"]).exists() and (tmp / m["path"]).with_suffix(".pt").exists()
    # one self-contained file: the browser's onnxruntime-web cannot load a .onnx.data sidecar
    assert not (tmp / (m["path"] + ".data")).exists()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp / m["path"]), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["iq", "stft_mag"]
    g = m["metrics"]["gate"]
    assert g["kind"] == "synthetic" and "SYNTHETIC" in g["set"] and g["rules"]
    assert m["metrics"]["corrections_fit"]["windows"] > 0
    assert "[4/5] The gate" in proc.stdout and "Done in" in proc.stdout
    assert db.audit(limit=1)[0]["action"] == "retrain.finish"


def test_four_eyes_and_gate_decide_activation(trained):
    db, tmp, job, _ = trained
    version = db.get_job(job["id"])["model_version"]
    with pytest.raises(PermissionError, match="Four-eyes"):
        db.review_model(version, "approve", "", EAVAN)            # eavan started the retrain
    if db.get_model(version)["metrics"]["gate"]["passed"]:
        m = db.review_model(version, "approve", "looks good", JESSY)
        assert m["status"] == "active" and db.active_model()["version"] == version
    else:
        with pytest.raises(PermissionError, match="failed its gate"):
            db.review_model(version, "approve", "", JESSY)
        # force-activate for the serving test below: the rule is enforced above
        import sqlite3
        with sqlite3.connect(db.path) as con:
            con.execute("UPDATE models SET status = 'active' WHERE version = ?", (version,))


def test_server_swaps_in_the_active_model_and_rollback_restores(trained):
    db, tmp, job, _ = trained
    version = db.get_job(job["id"])["model_version"]
    srv = serve_local.make_server(db.path, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/models/best_model.onnx") as r:
            assert r.headers["X-NEXA-Model"] == version
            assert r.read() == (tmp / db.get_model(version)["path"]).read_bytes()
        assert json.loads(urllib.request.urlopen(base + "/api/health").read())["active_model"] == version
        req = urllib.request.Request(base + "/api/models/rollback", method="POST", data=b'{"note":"test"}',
                                     headers={"X-NEXA-Client": "1", "X-NEXA-Operator": "chua"})
        assert json.loads(urllib.request.urlopen(req).read())["retired"] == version
        with urllib.request.urlopen(base + "/models/best_model.onnx") as r:
            assert r.headers.get("X-NEXA-Model") is None
            assert r.read() == (ROOT / "web" / "models" / "best_model.onnx").read_bytes()
    finally:
        srv.shutdown()
        srv.server_close()
    assert db.audit(limit=1)[0]["action"] == "model.rollback"
