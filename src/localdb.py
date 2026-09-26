"""NEXA's local database: one SQLite file on the operator's machine.

The console used to keep its History in the browser (IndexedDB), which Python
cannot read, a cleared cache wipes, and which never holds the raw IQ -- so a
human correction could never become training data. This module is the
replacement source of truth, served to the page by scripts/serve_local.py.

Standard library only (sqlite3, hashlib, json): the server must start on a
finale laptop with nothing but Python installed.

Tables
    analyses     one row per analysed capture (same columns as the Supabase
                 table in web/supabase/schema.sql, plus operator and
                 file_sha256)
    corrections  a human's correction to a detection; never deleted
    models       every model version: candidate -> approved -> active
    audit_log    every change to the above, append-only and hash-chained

The audit log is tamper-EVIDENT, not tamper-proof: whoever holds the file can
rewrite it, but cannot do so without breaking the hash chain, and
verify_audit() finds the first broken row. Triggers refuse UPDATE and DELETE on
it, so the app itself can never rewrite history even by mistake.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2
GENESIS_HASH = "0" * 64

# The model's classes, in its output order. Duplicated from configs/default.yaml
# (via src.config) so this module stays standard-library only;
# tests/test_localdb.py fails if the two ever disagree.
CLASSES = ("BPSK", "QPSK", "16QAM", "64QAM", "LFM_RADAR", "FHSS", "JAMMING", "NOISE_FLOOR")
REVIEW_DECISIONS = {"approve": "approved", "reject": "rejected"}

# Columns the page may write, and how each is stored. JSON columns hold lists
# and objects as text; the rest are plain SQLite values.
ANALYSIS_COLUMNS = {
    "id": "text", "created_at": "text", "source": "text", "case_note": "text",
    "file_name": "text", "file_bytes": "int", "file_path": "text",
    "file_sha256": "text", "model": "text", "duration_s": "real",
    "n_windows": "int", "hop": "int", "snr_db": "real",
    "requested_snr_db": "real", "snr_capped": "bool",
    "classes_detected": "json", "peak_probability": "json", "n_events": "int",
    "tier_counts": "json", "verdict": "text", "app_version": "text",
    "operator": "text",
}
REQUIRED = ("id", "created_at", "source")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
    id               TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    source           TEXT NOT NULL,
    case_note        TEXT DEFAULT '',
    file_name        TEXT,
    file_bytes       INTEGER,
    file_path        TEXT,
    file_sha256      TEXT,
    model            TEXT,
    duration_s       REAL,
    n_windows        INTEGER,
    hop              INTEGER,
    snr_db           REAL,
    requested_snr_db REAL,
    snr_capped       INTEGER DEFAULT 0,
    classes_detected TEXT DEFAULT '[]',
    peak_probability TEXT DEFAULT '{}',
    n_events         INTEGER,
    tier_counts      TEXT DEFAULT '{}',
    verdict          TEXT,
    app_version      TEXT,
    operator         TEXT
);
CREATE INDEX IF NOT EXISTS analyses_created_at_idx ON analyses (created_at DESC);

-- A correction points at the capture it corrects, so that capture cannot be
-- deleted out from under it (foreign key, no cascade): the evidence behind a
-- training label outlives any tidy-up of the History page.
CREATE TABLE IF NOT EXISTS corrections (
    id               TEXT PRIMARY KEY,
    analysis_id      TEXT NOT NULL REFERENCES analyses(id),
    created_at       TEXT NOT NULL,
    operator         TEXT NOT NULL,
    start_s          REAL,
    end_s            REAL,
    predicted_labels TEXT NOT NULL DEFAULT '[]',
    corrected_labels TEXT NOT NULL DEFAULT '[]',
    reason           TEXT DEFAULT '',
    model            TEXT,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'approved', 'rejected')),
    reviewed_by      TEXT,
    reviewed_at      TEXT,
    review_note      TEXT
);
CREATE INDEX IF NOT EXISTS corrections_status_idx ON corrections (status);

CREATE TABLE IF NOT EXISTS models (
    version      TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    path         TEXT,
    source       TEXT,          -- what it was trained on
    metrics      TEXT DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'candidate'
                 CHECK (status IN ('candidate', 'approved', 'active',
                                   'retired', 'rejected')),
    approved_by  TEXT,
    approved_at  TEXT,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    role        TEXT NOT NULL,
    client      TEXT,
    action      TEXT NOT NULL,
    target_type TEXT,
    target_id   TEXT,
    details     TEXT NOT NULL DEFAULT '{}',
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS corrections_no_delete BEFORE DELETE ON corrections
BEGIN SELECT RAISE(ABORT, 'corrections are never deleted; reject them instead'); END;

-- A correction is written once and reviewed once. After that it is evidence:
-- the database itself refuses a second review or any edit to what the human
-- said, so a label cannot quietly change after it was approved for training.
CREATE TRIGGER IF NOT EXISTS corrections_review_once BEFORE UPDATE ON corrections
WHEN OLD.status != 'pending'
BEGIN SELECT RAISE(ABORT, 'this correction was already reviewed and is now read-only'); END;
CREATE TRIGGER IF NOT EXISTS corrections_content_fixed BEFORE UPDATE ON corrections
WHEN NEW.analysis_id IS NOT OLD.analysis_id OR NEW.created_at IS NOT OLD.created_at
  OR NEW.operator IS NOT OLD.operator OR NEW.start_s IS NOT OLD.start_s
  OR NEW.end_s IS NOT OLD.end_s OR NEW.predicted_labels IS NOT OLD.predicted_labels
  OR NEW.corrected_labels IS NOT OLD.corrected_labels OR NEW.reason IS NOT OLD.reason
  OR NEW.model IS NOT OLD.model
BEGIN SELECT RAISE(ABORT, 'what a human said in a correction cannot be edited'); END;
"""

# Columns added after schema v1 shipped: (table, column, type). Added in
# place on open, so an existing nexa.db keeps its rows and its audit chain.
MIGRATIONS = [
    ("corrections", "iq_path", "TEXT"),
    ("corrections", "iq_sha256", "TEXT"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical(obj) -> str:
    """One fixed JSON spelling, so the same entry always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_hash(prev_hash: str, entry: dict) -> str:
    """SHA-256 over the previous row's hash and this row's content.

    Chaining is what makes an edit visible: change any old row and its hash no
    longer matches, and every row after it points at a hash that is gone.
    """
    body = {k: entry[k] for k in ("seq", "ts", "actor", "role", "client", "action",
                                  "target_type", "target_id", "details")}
    return hashlib.sha256((prev_hash + _canonical(body)).encode("utf-8")).hexdigest()


class Actor:
    """Who did it. Until the login step lands, the page names its operator
    itself (X-NEXA-Operator header) and everyone is role 'operator'; the log
    still records the machine the request came from."""

    def __init__(self, name: str = "operator", role: str = "operator", client: str | None = None):
        self.name = (name or "operator").strip()[:64] or "operator"
        self.role = role
        self.client = client


SYSTEM = Actor("system", "system", "localhost")


class LocalDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One lock for every write: the audit chain must be appended strictly
        # one row at a time, and the HTTP server handles requests on threads.
        self._lock = threading.Lock()
        with closing(self._connect()) as con:
            con.executescript(SCHEMA)
            for table, col, kind in MIGRATIONS:
                have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
                if col not in have:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")
            con.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),))
        if not self.audit(limit=1):
            with self._write() as con:
                self._append_audit(con, SYSTEM, "db.create", "database", str(self.path.name),
                                   {"schema_version": SCHEMA_VERSION})

    # -- plumbing -------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        return con

    class _Tx:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            self.db._lock.acquire()
            self.con = self.db._connect()
            self.con.execute("BEGIN IMMEDIATE")
            return self.con

        def __exit__(self, exc_type, exc, tb):
            try:
                self.con.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self.con.close()
                self.db._lock.release()
            return False

    def _write(self):
        """A write transaction that also holds the chain lock."""
        return LocalDB._Tx(self)

    def _append_audit(self, con, actor: Actor, action: str, target_type: str | None,
                      target_id: str | None, details: dict | None = None) -> dict:
        last = con.execute("SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
        prev = last["hash"] if last else GENESIS_HASH
        entry = {
            "seq": (last["seq"] + 1) if last else 1,
            "ts": now_iso(), "actor": actor.name, "role": actor.role,
            "client": actor.client, "action": action,
            "target_type": target_type, "target_id": target_id,
            "details": _canonical(details or {}),
        }
        entry["prev_hash"] = prev
        entry["hash"] = entry_hash(prev, entry)
        con.execute(
            "INSERT INTO audit_log (seq, ts, actor, role, client, action, target_type,"
            " target_id, details, prev_hash, hash) VALUES (:seq, :ts, :actor, :role,"
            " :client, :action, :target_type, :target_id, :details, :prev_hash, :hash)",
            entry)
        return entry

    # -- analyses -------------------------------------------------------------

    @staticmethod
    def _to_row(record: dict) -> dict:
        missing = [k for k in REQUIRED if not record.get(k)]
        if missing:
            raise ValueError(f"record is missing {', '.join(missing)}")
        row = {}
        for col, kind in ANALYSIS_COLUMNS.items():
            v = record.get(col)
            if v is None:
                row[col] = None
            elif kind == "json":
                row[col] = json.dumps(v)
            elif kind == "bool":
                row[col] = 1 if v else 0
            else:
                row[col] = v
        return row

    @staticmethod
    def _from_row(row: sqlite3.Row) -> dict:
        out = dict(row)
        for col, kind in ANALYSIS_COLUMNS.items():
            if kind == "json" and out.get(col) is not None:
                out[col] = json.loads(out[col])
            elif kind == "bool":
                out[col] = bool(out.get(col))
        return out

    def save_analysis(self, record: dict, actor: Actor) -> dict:
        row = self._to_row({**record, "operator": record.get("operator") or actor.name})
        cols = ", ".join(row)
        with self._write() as con:
            con.execute(f"INSERT INTO analyses ({cols}) VALUES ({', '.join(':' + c for c in row)})", row)
            self._append_audit(con, actor, "analysis.create", "analysis", row["id"], {
                "source": row["source"], "verdict": row["verdict"],
                "classes_detected": record.get("classes_detected") or [],
                "model": row["model"], "file_name": row["file_name"],
                "file_sha256": row["file_sha256"],
            })
        return self.get_analysis(row["id"])

    def get_analysis(self, analysis_id: str) -> dict | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return self._from_row(row) if row else None

    def list_analyses(self, limit: int = 500) -> list[dict]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",
                               (int(limit),)).fetchall()
        return [self._from_row(r) for r in rows]

    def delete_analysis(self, analysis_id: str, actor: Actor) -> bool:
        with self._write() as con:
            row = con.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            if not row:
                return False
            if con.execute("SELECT 1 FROM corrections WHERE analysis_id = ? LIMIT 1",
                           (analysis_id,)).fetchone():
                raise PermissionError("This capture has human corrections attached and is kept "
                                      "as their evidence; it cannot be deleted.")
            con.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            # The deleted row goes into the log: a delete must not erase the
            # fact that the capture was ever analysed, or what was concluded.
            self._append_audit(con, actor, "analysis.delete", "analysis", analysis_id,
                               {"deleted": self._from_row(row)})
        return True

    def clear_analyses(self, actor: Actor) -> int:
        """Delete every capture that no correction depends on."""
        with self._write() as con:
            ids = [r["id"] for r in con.execute(
                "SELECT id FROM analyses WHERE id NOT IN (SELECT analysis_id FROM corrections)")]
            con.executemany("DELETE FROM analyses WHERE id = ?", [(i,) for i in ids])
            kept = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            self._append_audit(con, actor, "analysis.clear", "analysis", None,
                               {"deleted_ids": ids, "kept_with_corrections": kept})
        return len(ids)

    def record_file(self, analysis_id: str, name: str, rel_path: str, n_bytes: int,
                    sha256: str, actor: Actor) -> None:
        """A raw IQ file was written to disk: log its fingerprint, so a later
        swap of the file on disk is detectable against the audit trail."""
        with self._write() as con:
            self._append_audit(con, actor, "iq.store", "analysis", analysis_id,
                               {"name": name, "path": rel_path, "bytes": n_bytes, "sha256": sha256})

    # -- corrections (human in the loop) -------------------------------------

    @staticmethod
    def _labels(value, what) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{what} must be a list of class names")
        unknown = [v for v in value if v not in CLASSES]
        if unknown:
            raise ValueError(f"{what}: unknown class {', '.join(unknown)}")
        return sorted(set(value), key=CLASSES.index)

    @staticmethod
    def _correction_row(row) -> dict:
        d = dict(row)
        for k in ("predicted_labels", "corrected_labels"):
            d[k] = json.loads(d[k])
        return d

    def create_correction(self, body: dict, actor: Actor) -> dict:
        """A human says what was really there in [start_s, end_s) of a capture.

        predicted_labels: what the model reported there ([] for a missed signal).
        corrected_labels: what the human says is there (["NOISE_FLOOR"] = nothing).
        The raw IQ the correction is about is found from the capture's stored
        file, or the file uploaded for it -- a correction nobody could retrain
        on or re-check would be an opinion, not evidence.
        """
        analysis_id = str(body.get("analysis_id") or "")
        predicted = self._labels(body.get("predicted_labels", []), "predicted_labels")
        corrected = self._labels(body.get("corrected_labels", []), "corrected_labels")
        if not corrected:
            raise ValueError("say what is really there: tick at least one class (NOISE_FLOOR if nothing)")
        if corrected == predicted:
            raise ValueError("that is what the model already said -- nothing to correct")
        reason = str(body.get("reason") or "").strip()
        if len(reason) < 5:
            raise ValueError("a reason is required (at least a few words)")
        start_s, end_s = body.get("start_s"), body.get("end_s")
        try:
            start_s, end_s = float(start_s), float(end_s)
        except (TypeError, ValueError):
            raise ValueError("start_s and end_s must be numbers") from None
        if not (0 <= start_s < end_s):
            raise ValueError("the corrected span must have start < end")

        with self._write() as con:
            a = con.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            if not a:
                raise ValueError("no such capture")
            if end_s > (a["duration_s"] or float("inf")) + 1e-3:
                raise ValueError("the corrected span runs past the end of the capture")
            iq_path, iq_sha = a["file_path"], a["file_sha256"]
            if not iq_path:
                last = con.execute(
                    "SELECT details FROM audit_log WHERE action = 'iq.store' AND target_id = ?"
                    " ORDER BY seq DESC LIMIT 1", (analysis_id,)).fetchone()
                if last:
                    d = json.loads(last["details"])
                    iq_path, iq_sha = d.get("path"), d.get("sha256")
            if not iq_path:
                raise ValueError("this capture has no stored raw IQ; upload it before correcting")
            row = {
                "id": str(uuid.uuid4()), "analysis_id": analysis_id, "created_at": now_iso(),
                "operator": actor.name, "start_s": start_s, "end_s": end_s,
                "predicted_labels": json.dumps(predicted), "corrected_labels": json.dumps(corrected),
                "reason": reason, "model": body.get("model") or a["model"],
                "iq_path": iq_path, "iq_sha256": iq_sha,
            }
            con.execute(f"INSERT INTO corrections ({', '.join(row)}) VALUES "
                        f"({', '.join(':' + k for k in row)})", row)
            self._append_audit(con, actor, "correction.create", "correction", row["id"], {
                "analysis_id": analysis_id, "span_s": [start_s, end_s],
                "predicted": predicted, "corrected": corrected, "reason": reason,
                "iq_sha256": iq_sha,
            })
        return self.get_correction(row["id"])

    def get_correction(self, correction_id: str) -> dict | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()
        return self._correction_row(row) if row else None

    def list_corrections(self, status: str | None = None, analysis_id: str | None = None,
                         limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM corrections WHERE 1=1", []
        if status:
            q += " AND status = ?"
            args.append(status)
        if analysis_id:
            q += " AND analysis_id = ?"
            args.append(analysis_id)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with closing(self._connect()) as con:
            return [self._correction_row(r) for r in con.execute(q, args)]

    def review_correction(self, correction_id: str, decision: str, note: str,
                          actor: Actor) -> dict:
        """Approve or reject a pending correction.

        Four-eyes rule: the reviewer must be a named person other than whoever
        submitted it. Nobody marks their own homework, and "operator" (the
        default nobody bothered to change) is not a name.
        """
        status = REVIEW_DECISIONS.get(decision)
        if not status:
            raise ValueError("decision must be 'approve' or 'reject'")
        if actor.name.lower() == "operator":
            raise PermissionError("Set your name (Operator box on the History page) before reviewing.")
        note = str(note or "").strip()
        if status == "rejected" and len(note) < 3:
            raise ValueError("say why it is rejected")
        with self._write() as con:
            row = con.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()
            if not row:
                raise ValueError("no such correction")
            if row["status"] != "pending":
                raise PermissionError(f"already {row['status']} by {row['reviewed_by']}")
            if row["operator"].lower() == actor.name.lower():
                raise PermissionError("Four-eyes rule: a correction must be reviewed by someone "
                                      "other than the person who submitted it.")
            con.execute("UPDATE corrections SET status = ?, reviewed_by = ?, reviewed_at = ?,"
                        " review_note = ? WHERE id = ?",
                        (status, actor.name, now_iso(), note, correction_id))
            self._append_audit(con, actor, f"correction.{decision}", "correction", correction_id, {
                "analysis_id": row["analysis_id"], "submitted_by": row["operator"],
                "corrected": json.loads(row["corrected_labels"]), "note": note,
            })
        return self.get_correction(correction_id)

    def correction_stats(self) -> dict:
        with closing(self._connect()) as con:
            by_status = dict(con.execute("SELECT status, COUNT(*) FROM corrections GROUP BY status").fetchall())
            per_class = {c: 0 for c in CLASSES}
            for (labels,) in con.execute("SELECT corrected_labels FROM corrections WHERE status = 'approved'"):
                for c in json.loads(labels):
                    per_class[c] += 1
        return {"pending": by_status.get("pending", 0), "approved": by_status.get("approved", 0),
                "rejected": by_status.get("rejected", 0), "approved_per_class": per_class}

    def log(self, actor: Actor, action: str, target_type=None, target_id=None, details=None) -> dict:
        """An audited event with no row of its own (e.g. a report export)."""
        with self._write() as con:
            return self._append_audit(con, actor, action, target_type, target_id, details)

    # -- audit ----------------------------------------------------------------

    def audit(self, limit: int = 200, action_prefix: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM audit_log", []
        if action_prefix:
            q += " WHERE action LIKE ?"
            args.append(action_prefix + "%")
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(int(limit))
        with closing(self._connect()) as con:
            rows = con.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d["details"])
            out.append(d)
        return out

    def verify_audit(self) -> dict:
        """Recompute the whole chain. {ok, entries, broken_at, reason}."""
        prev, n = GENESIS_HASH, 0
        with closing(self._connect()) as con:
            for row in con.execute("SELECT * FROM audit_log ORDER BY seq"):
                n += 1
                entry = dict(row)
                if entry["prev_hash"] != prev:
                    return {"ok": False, "entries": n, "broken_at": entry["seq"],
                            "reason": "points at a previous entry that is missing or changed"}
                if entry_hash(prev, entry) != entry["hash"]:
                    return {"ok": False, "entries": n, "broken_at": entry["seq"],
                            "reason": "contents changed after it was written"}
                prev = entry["hash"]
        return {"ok": True, "entries": n, "broken_at": None, "reason": None}

    def counts(self) -> dict:
        with closing(self._connect()) as con:
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("analyses", "corrections", "models", "audit_log")}

