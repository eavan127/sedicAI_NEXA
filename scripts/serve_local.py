"""NEXA local server: the web console plus its local database, on this machine.

Replaces `python -m http.server` (web/start_demo.bat). It still serves web/
exactly as before, and adds a small JSON API over src/localdb.py so every
analysis, raw IQ file and audit entry lands in one SQLite file the training
scripts can read -- nothing is sent anywhere else.

    python scripts/serve_local.py                 # http://localhost:8099
    python scripts/serve_local.py --port 8200 --no-browser
    python scripts/serve_local.py --lan           # other machines on this network

Standard library only, so it runs on any machine with Python and no install.

API (JSON)
    GET    /api/health                 is this the NEXA server? + row counts
    GET    /api/analyses?limit=500     newest first
    POST   /api/analyses               store one record (web/storage.js buildRecord)
    DELETE /api/analyses/<id>          refused if corrections depend on it
    DELETE /api/analyses               delete all that no correction depends on
    PUT    /api/captures/<id>?name=f   raw IQ bytes for analysis <id>
    POST   /api/corrections            a human's correction to a capture
    GET    /api/corrections?status=pending&analysis_id=<id>
    GET    /api/corrections/stats      counts, approved labels per class
    POST   /api/corrections/<id>/review   {decision: approve|reject, note}
    GET    /api/retrain/status         the retraining trigger: numbers + each rule
    POST   /api/retrain/start          {reason, override} -> runs scripts/retrain_from_feedback.py
    GET    /api/retrain/jobs[/<id>]    training runs; one job includes the tail of its log
    GET    /api/models                 retrained versions (candidate/active/retired/rejected)
    POST   /api/models/<v>/review      {decision: approve|reject, note} -- four-eyes, gate must pass
    POST   /api/models/rollback        back to the shipped single model
    POST   /api/export                 report builder: {ids, sections, format, banner, filters}
                                       -> json / csv.zip / xlsx / sigmf.zip / iq.zip
    GET    /api/audit?limit=200        audit trail, newest first
    GET    /api/audit/verify           recompute the hash chain
    POST   /api/audit                  log a client-side event (e.g. report export)

Safety
  * Binds to 127.0.0.1 unless --lan: other machines cannot reach it.
  * Requests must name this machine in their Host header (blocks DNS
    rebinding: a web page on some other site tricking the browser into
    talking to this server).
  * Every write needs the X-NEXA-Client header. A page on another site cannot
    send a custom header without a CORS preflight, which this server never
    approves -- so other sites cannot write here through your browser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.localdb import Actor, LocalDB  # noqa: E402
from src.report_export import build_export  # noqa: E402

WEB = ROOT / "web"
DEFAULT_DB = ROOT / "data" / "local" / "nexa.db"
APP_ID = "nexa-local"
MAX_JSON = 1 << 20            # 1 MB: a record is ~2 KB
MAX_CAPTURE = 1 << 30         # 1 GB: ~40 s of 3.2 MS/s float32
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def safe_name(name: str) -> str:
    name = Path(unquote(name or "")).name        # no directories, ever
    name = re.sub(r"[^\w.\-]", "_", name)[:120]
    return name if name.strip("._") else "capture.bin"


class Handler(SimpleHTTPRequestHandler):
    # The Windows registry often maps .mjs to text/plain, and browsers refuse
    # to run a module served that way -- which is exactly why the ONNX model
    # failed to load under `python -m http.server`.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript", ".mjs": "text/javascript",
        ".wasm": "application/wasm", ".json": "application/json",
        ".onnx": "application/octet-stream",
    }

    db: LocalDB = None
    iq_dir: Path = None
    lan: bool = False
    retrain_args: list = []

    # -- request gate ----------------------------------------------------------

    def _host_ok(self) -> bool:
        if self.lan:
            return True
        host = (self.headers.get("Host") or "").strip()
        host = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        return host.lower() in LOOPBACK_HOSTS

    def _actor(self) -> Actor:
        return Actor(self.headers.get("X-NEXA-Operator", "operator"), "operator",
                     self.client_address[0])

    def _send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, body: bytes, content_type: str, filename: str, extra: dict):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send_json({"error": message}, status)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_JSON:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _dispatch(self, method):
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            if method in ("GET", "HEAD") and url.path == "/models/best_model.onnx" and self._serve_active_model():
                return None
            if method == "GET":
                return super().do_GET()
            if method == "HEAD":
                return super().do_HEAD()
            return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "static files are read-only")
        if method not in ("GET", "HEAD") and self.headers.get("X-NEXA-Client") != "1":
            return self._error(HTTPStatus.FORBIDDEN, "missing X-NEXA-Client header")
        parts = [p for p in url.path.split("/") if p][1:]      # drop "api"
        query = parse_qs(url.query)
        try:
            return self._route(method, parts, query)
        except PermissionError as e:
            return self._error(HTTPStatus.CONFLICT, str(e))
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:                                  # noqa: BLE001
            self.log_error("API error: %r", e)
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    # -- routes ---------------------------------------------------------------

    def _route(self, method, parts, query):
        limit = int((query.get("limit") or ["500"])[0])
        head = parts[0] if parts else ""

        if head == "health" and method == "GET":
            active = self.db.active_model()
            return self._send_json({"app": APP_ID, "ok": True, "db": self.db.path.name,
                                    "counts": self.db.counts(),
                                    "active_model": active["version"] if active else None})

        if head == "analyses":
            if method == "GET" and len(parts) == 1:
                return self._send_json(self.db.list_analyses(limit))
            if method == "POST" and len(parts) == 1:
                return self._send_json(self.db.save_analysis(self._read_json(), self._actor()),
                                       HTTPStatus.CREATED)
            if method == "DELETE" and len(parts) == 2:
                if not self.db.delete_analysis(parts[1], self._actor()):
                    return self._error(HTTPStatus.NOT_FOUND, "no such analysis")
                return self._send_json({"deleted": parts[1]})
            if method == "DELETE" and len(parts) == 1:
                return self._send_json({"deleted": self.db.clear_analyses(self._actor())})

        if head == "corrections":
            if method == "POST" and len(parts) == 1:
                return self._send_json(self.db.create_correction(self._read_json(), self._actor()),
                                       HTTPStatus.CREATED)
            if method == "GET" and len(parts) == 1:
                return self._send_json(self.db.list_corrections(
                    (query.get("status") or [None])[0], (query.get("analysis_id") or [None])[0], limit))
            if method == "GET" and parts[1:] == ["stats"]:
                return self._send_json(self.db.correction_stats())
            if method == "POST" and len(parts) == 3 and parts[2] == "review":
                body = self._read_json()
                return self._send_json(self.db.review_correction(
                    parts[1], str(body.get("decision") or ""), body.get("note"), self._actor()))

        if head == "export" and method == "POST" and len(parts) == 1:
            return self._export(self._read_json())

        if head == "retrain" and method == "GET" and parts[1:] == ["status"]:
            return self._send_json(self.db.retrain_status())
        if head == "retrain" and method == "POST" and parts[1:] == ["start"]:
            body = self._read_json()
            self._check_training_deps(self.training_python())
            job = self.db.start_retrain(body.get("reason"), bool(body.get("override")), self._actor())
            return self._send_json(self._launch(job), HTTPStatus.ACCEPTED)
        if head == "retrain" and method == "GET" and parts[1:2] == ["jobs"]:
            if len(parts) == 2:
                return self._send_json(self.db.list_jobs())
            job = self.db.get_job(parts[2])
            if not job:
                return self._error(HTTPStatus.NOT_FOUND, "no such job")
            log = self.db.path.parent / (job.get("log_path") or "")
            job["log"] = log.read_text(encoding="utf-8", errors="replace")[-12000:] if job.get("log_path") and log.exists() else ""
            return self._send_json(job)

        if head == "models":
            if method == "GET" and len(parts) == 1:
                return self._send_json(self.db.list_models())
            if method == "POST" and parts[1:] == ["rollback"]:
                body = self._read_json()
                return self._send_json({"retired": self.db.rollback_model(body.get("note"), self._actor())})
            if method == "POST" and len(parts) == 3 and parts[2] == "review":
                body = self._read_json()
                return self._send_json(self.db.review_model(
                    parts[1], str(body.get("decision") or ""), body.get("note"), self._actor()))

        if head == "captures" and method == "PUT" and len(parts) == 2:
            return self._put_capture(parts[1], (query.get("name") or [""])[0])

        if head == "audit":
            if method == "GET" and len(parts) == 1:
                prefix = (query.get("action") or [None])[0]
                return self._send_json(self.db.audit(min(limit, 1000), prefix))
            if method == "GET" and parts[1:] == ["verify"]:
                return self._send_json(self.db.verify_audit())
            if method == "POST" and len(parts) == 1:
                body = self._read_json()
                action = str(body.get("action") or "")
                # The page may log what only it knows about (exports, views),
                # but not impersonate the server's own actions.
                if not re.fullmatch(r"client\.[a-z_.]{1,48}", action):
                    raise ValueError("client events must be named client.<something>")
                entry = self.db.log(self._actor(), action, body.get("target_type"),
                                    body.get("target_id"), body.get("details") or {})
                return self._send_json(entry, HTTPStatus.CREATED)

        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    @staticmethod
    def training_python() -> str:
        """The project's virtualenv when there is one: the server itself is
        standard-library only and may run on a bare Python, but training needs
        torch and onnx."""
        for cand in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python",
                     ROOT / "venv" / "Scripts" / "python.exe", ROOT / "venv" / "bin" / "python"):
            if cand.exists():
                return str(cand)
        return sys.executable

    _deps_ok: dict = {}

    def _check_training_deps(self, py: str) -> None:
        """Refuse to start a retrain that would die on an import, and say what to install."""
        if Handler._deps_ok.get(py):
            return
        probe = subprocess.run([py, "-c", "import torch, onnx, onnxruntime, onnxscript"],
                               capture_output=True, text=True, timeout=120)
        if probe.returncode:
            missing = (probe.stderr.strip().splitlines() or ["?"])[-1]
            raise PermissionError(f"Retraining needs the machine-learning packages ({missing}). "
                                  f"Install them into {py}:  pip install -r requirements.txt")
        Handler._deps_ok[py] = True

    def _launch(self, job):
        """Run the training script in the background; its output is the job log."""
        jobs = self.db.path.parent / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        log_rel = f"jobs/{job['id']}.log"
        log = open(self.db.path.parent / log_rel, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                [self.training_python(), "-u", str(ROOT / "scripts" / "retrain_from_feedback.py"),
                 "--db", str(self.db.path), "--job", job["id"], *self.retrain_args],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        finally:
            log.close()
        self.db.update_job(job["id"], log_path=log_rel, pid=proc.pid)
        return self.db.get_job(job["id"])

    def _serve_active_model(self):
        """GET /models/best_model.onnx: the ACTIVE retrained model when one was
        approved, else the shipped file. The ensemble files are never swapped."""
        active = self.db.active_model()
        path = (self.db.path.parent / active["path"]) if active else None
        if not path or not path.exists():
            return False
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-NEXA-Model", active["version"])
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return True

    def _export(self, body):
        card_path = WEB / "data" / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        actor = self._actor()
        data, ctype, fname, meta = build_export(
            self.db, self.db.path.parent, ids=list(body.get("ids") or []),
            sections=list(body.get("sections") or []), fmt=str(body.get("format") or ""),
            banner=str(body.get("banner") or ""), generated_by=actor.name,
            model_card=card, active_model=getattr(self.db, "active_model", lambda: None)(),
            filters=str(body.get("filters") or "")[:300])
        # Logged by the server, with a fingerprint of the exact file handed
        # out: "who took which data off this machine" is the question an
        # audit of a leak asks first.
        self.db.log(actor, "report.export", "report", meta["report_id"], {
            "format": meta["format"], "captures": meta["captures"], "sections": meta["sections"],
            "classification": meta["classification"], "filters": meta["filters"],
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        })
        return self._send_file(data, ctype, fname, {"X-Report-Id": meta["report_id"]})

    def _put_capture(self, analysis_id, name):
        if not re.fullmatch(r"[\w\-]{1,64}", analysis_id):
            raise ValueError("bad analysis id")
        n = int(self.headers.get("Content-Length") or -1)
        if n < 0:
            raise ValueError("Content-Length required")
        if n > MAX_CAPTURE:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               f"capture is {n} bytes; the limit is {MAX_CAPTURE}")
        name = safe_name(name)
        folder = self.iq_dir / analysis_id
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / name
        sha, left = hashlib.sha256(), n
        with open(dest, "wb") as f:
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                sha.update(chunk)
                f.write(chunk)
                left -= len(chunk)
        if left:
            dest.unlink(missing_ok=True)
            raise ValueError("upload ended early")
        rel = dest.relative_to(self.iq_dir.parent).as_posix()
        digest = sha.hexdigest()
        self.db.record_file(analysis_id, name, rel, n, digest, self._actor())
        return self._send_json({"file_path": rel, "file_sha256": digest, "bytes": n},
                               HTTPStatus.CREATED)

    def end_headers(self):
        # Static files: always revalidate (a cheap 304 when unchanged), so a
        # page edited on disk is what the browser runs on the next reload --
        # not a heuristically cached mix of old and new modules.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        # Cross-origin isolation: without these two headers the browser will
        # not give the page SharedArrayBuffer, and onnxruntime-web then runs
        # every model on ONE core. "credentialless" (rather than
        # "require-corp") still lets the lazily loaded PDF library come from
        # its CDN.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Quiet for static files; API calls are the interesting lines.
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)


def pick_port(host: str, preferred: int, tries: int = 20) -> int:
    """The preferred port, or the next free one. (8080 is taken on some
    Windows machines by background services, which is how the old launcher
    failed.)"""
    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {preferred} and {preferred + tries - 1}.")


def make_server(db_path: Path, host: str, port: int, lan: bool = False,
                retrain_args: list | None = None) -> ThreadingHTTPServer:
    db = LocalDB(db_path)
    iq_dir = Path(db_path).parent / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    handler = type("NexaHandler", (Handler,), {"db": db, "iq_dir": iq_dir, "lan": lan,
                                                "retrain_args": list(retrain_args or [])})
    server = ThreadingHTTPServer((host, port), partial(handler, directory=str(WEB)))
    server.nexa_db = db
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--lan", action="store_true",
                   help="listen on all interfaces so other machines on this network can connect")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--quick-retrain", action="store_true",
                   help="demo/test: retrain on a small sample (1 epoch, 2,000 replay windows, "
                        "3,000-window exam) so a retrain takes about a minute")
    p.add_argument("--synthetic-exam", action="store_true",
                   help="demo: judge retrained models on the fixed synthetic exam even when "
                        "data/processed exists (the Model page labels it as synthetic)")
    a = p.parse_args()

    host = "0.0.0.0" if a.lan else "127.0.0.1"
    port = pick_port(host, a.port)
    retrain_args = ["--epochs", "1", "--replay", "2000", "--gate-limit", "3000"] if a.quick_retrain else []
    if a.synthetic_exam:
        retrain_args += ["--gate", "synthetic"]
    server = make_server(a.db, host, port, a.lan, retrain_args)
    url = f"http://localhost:{port}/index.html"
    counts = server.nexa_db.counts()
    print(f"NEXA local server\n  open      {url}\n  database  {a.db}\n"
          f"  stored    {counts.get('analyses', 0)} analyses, {counts.get('audit_log', 0)} audit entries\n"
          f"  network   {'LAN (other machines can connect)' if a.lan else 'this machine only'}\n"
          "Press Ctrl+C to stop.")
    if not a.no_browser:
        threading.Timer(0.8, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
