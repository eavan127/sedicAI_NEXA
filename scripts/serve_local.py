"""NEXA local server: the web console plus its local database, on this machine.

Replaces `python -m http.server` (web/start_demo.bat). It still serves web/
exactly as before, and adds a small JSON API over src/localdb.py so every
analysis, raw IQ file and audit entry lands in one SQLite file the training
scripts can read -- nothing is sent anywhere else.

    python scripts/serve_local.py                 # http://localhost:8099, no sign-in:
                                                  #   buttons switch operator / analyst
    python scripts/serve_local.py --port 8200 --no-browser
    python scripts/serve_local.py --sign-in       # real accounts: login page, passwords, Users page
    python scripts/serve_local.py --lan           # other machines on this network (sign-in forced on)

Standard library only, so it runs on any machine with Python and no install.

API (JSON)
    GET    /api/health                 is this the NEXA server? + row counts
    GET    /api/analyses?limit=500     newest first, each with its retention {kind, why}
    GET    /api/analyses/<id>/iq       a stored capture's raw IQ (410 once the rolling window deleted it)
    POST   /api/analyses/<id>/reviewed {note}: an operator checked a flagged capture; the model was right
    POST   /api/analyses               store one record (web/storage.js buildRecord)
    DELETE /api/analyses/<id>          refused if corrections depend on it
    DELETE /api/analyses               delete all that no correction depends on
    PUT    /api/captures/<id>?name=f   raw IQ bytes for analysis <id>
    POST   /api/corrections            a human's correction to a capture
    GET    /api/corrections?status=pending&analysis_id=<id>
    GET    /api/corrections/stats      counts, approved labels per class
    GET    /api/corrections/<id>/iq    the raw IQ behind a correction (X-NEXA-Datatype, -Sample-Rate)
    POST   /api/corrections/<id>/review   {decision: approve|reject, note}
    GET    /api/retrain/status         the retraining trigger: numbers + each rule
    POST   /api/retrain/start          {reason, override, scope: single|both}
                                       -> runs scripts/retrain_from_feedback.py
    GET    /api/retrain/jobs[/<id>]    training runs; one job includes the tail of its log
    GET    /api/models                 retrained versions (candidate/active/retired/rejected)
    POST   /api/models/<v>/review      {decision: approve|reject, note} -- four-eyes, gate must pass
    POST   /api/models/<v>/activate    {note, override} -- use ANY saved version again; one that
                                       failed its gate or was rejected needs override + reason
    POST   /api/models/rollback        {kind: single|ensemble} back to the shipped files
    GET    /api/retrain/history        every retrain: who, when, why, what, verdicts, decisions
    POST   /api/export                 report builder: {ids, sections, format, banner, filters}
                                       -> json / csv.zip / xlsx / sigmf.zip / iq.zip
    GET    /api/audit?limit=200        audit trail, newest first
    GET    /api/audit/verify           recompute the hash chain
    POST   /api/audit                  log a client-side event (e.g. report export)
    GET    /api/auth/me                sign-in on? who is signed in? may this machine set it up?
    POST   /api/auth/setup             {username, password}: first analyst; turns sign-in on (this machine only)
    POST   /api/auth/login             {username, password} -> session cookie
    POST   /api/auth/logout
    GET    /api/users                  accounts (analyst)
    POST   /api/users                  {username, password, role} (analyst)
    POST   /api/users/<name>           {role?, disabled?, password?} (analyst)

Sign-in (--sign-in, or --lan)
  Off by default: the page opens straight into the console, and buttons at
  the top switch between operator / analyst, so roles
  can be shown without logging out and in. With --sign-in (and always with
  --lan), nothing of the system is served without a session: the page, its scripts,
  the models and every API call except health/auth. A browser that is not
  signed in is sent to /login.html, which on a fresh database offers to
  create the first analyst (on this machine only). The server -- not the page
  -- decides who did what. Roles: operator < analyst (approves,
  retrains, manages users).

Safety
  * Binds to 127.0.0.1 unless --lan: other machines cannot reach it.
  * Requests must name this machine in their Host header (blocks DNS
    rebinding: a web page on some other site tricking the browser into
    talking to this server).
  * The session cookie is HttpOnly + SameSite=Strict: page scripts cannot
    read it and other sites cannot make the browser send it.
  * Every write needs the X-NEXA-Client header. A page on another site cannot
    send a custom header without a CORS preflight, which this server never
    approves -- so other sites cannot write here through your browser.
"""
from __future__ import annotations

import argparse
import hashlib
import http.cookies
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

from src.localdb import ROLLING_IQ_BYTES, SESSION_HOURS, TOP_ROLE, Actor, LocalDB, role_at_least  # noqa: E402
from src.report_export import _iq_payload, build_export  # noqa: E402

WEB = ROOT / "web"
DEFAULT_DB = ROOT / "data" / "local" / "nexa.db"
APP_ID = "nexa-local"
MAX_JSON = 1 << 20            # 1 MB: a record is ~2 KB
MAX_CAPTURE = 1 << 30         # 1 GB: ~40 s of 3.2 MS/s float32
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}
# The model files the page loads; an approved retrained version replaces them.
MODEL_FILE = re.compile(r"/models/(best_model|ensemble_[0-4])\.onnx")
LOOPBACK_CLIENTS = {"127.0.0.1", "::1"}
SESSION_COOKIE = "nexa_session"
LOGIN_PAGE = "/login.html"
LOGO = ROOT / "assets" / "sedic_logo.png"
# All a signed-out browser may fetch: the login page and its logo.
PUBLIC_STATIC = {LOGIN_PAGE, "/brand/logo.png", "/favicon.ico"}

# Default (no --sign-in): no sign-in; the page shows one button per role and whoever was
# clicked last is "who did it" (a cookie). Each role is its own identity, so the
# four-eyes rule still holds on stage: the operator submits, the analyst approves.
DEMO_PEOPLE = {"operator": "operator", "analyst": "analyst"}
DEMO_DEFAULT = "operator"
DEMO_COOKIE = "nexa_demo_as"


def required_role(method: str, parts: list[str]) -> str:
    """The least role an API call needs once sign-in is on."""
    head = parts[0] if parts else ""
    if head == "users" or (method, parts) in (("POST", ["retrain", "start"]), ("POST", ["models", "rollback"]),
                                              ("DELETE", ["analyses"])):
        return TOP_ROLE
    if method == "DELETE" and head == "analyses":
        return "analyst"
    if method == "POST" and head in ("corrections", "models") and len(parts) == 3 and parts[2] == "review":
        return "analyst"
    # Putting a model version into use (approve / restore) is an analyst's
    # decision, like approving one; overriding a FAILED exam is checked
    # separately in the route and needs the analyst role too.
    if method == "POST" and head == "models" and len(parts) == 3 and parts[2] == "activate":
        return "analyst"
    return "operator"


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
    demo: bool = False
    keep_bytes: int = ROLLING_IQ_BYTES
    retrain_args: list = []

    # -- request gate ----------------------------------------------------------

    def _host_ok(self) -> bool:
        if self.lan:
            return True
        host = (self.headers.get("Host") or "").strip()
        host = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        return host.lower() in LOOPBACK_HOSTS

    me: Actor | None = None       # the signed-in person for this request, if any

    def _actor(self) -> Actor:
        """Who is doing this: the signed-in account. Nothing the page sends
        can name someone else."""
        return self.me or Actor("anonymous", "none", self.client_address[0])

    def _cookie_value(self, name: str) -> str | None:
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie") or "")
        except http.cookies.CookieError:
            return None
        return jar[name].value if name in jar else None

    def _session_token(self) -> str | None:
        return self._cookie_value(SESSION_COOKIE)

    def _current_actor(self) -> Actor | None:
        """The signed-in account, or in --demo the person last picked."""
        if self.demo:
            name = self._cookie_value(DEMO_COOKIE)
            name = name if name in DEMO_PEOPLE else DEMO_DEFAULT
            return Actor(name, DEMO_PEOPLE[name], self.client_address[0])
        return self.db.session_actor(self._session_token(), self.client_address[0])

    @staticmethod
    def _cookie(token: str | None) -> dict:
        age = SESSION_HOURS * 3600 if token else 0
        return {"Set-Cookie": f"{SESSION_COOKIE}={token or ''}; HttpOnly; SameSite=Strict; Path=/; Max-Age={age}"}

    def _authorize(self, method: str, parts: list[str]) -> bool:
        """True when the API call may go ahead; otherwise the refusal is sent."""
        self.me = self._current_actor()
        if parts[:1] == ["health"] or (parts[:1] == ["auth"] and parts[1:2] in (["me"], ["login"], ["setup"])):
            return True
        if not self.me:
            self._error(HTTPStatus.UNAUTHORIZED, "Sign in first.")
            return False
        need = required_role(method, parts)
        if not role_at_least(self.me.role, need):
            self._error(HTTPStatus.FORBIDDEN,
                        f"This needs the {need} role; you are signed in as {self.me.name} ({self.me.role}).")
            return False
        return True

    def _send_json(self, obj, status=HTTPStatus.OK, headers: dict | None = None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
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
            # The system itself is only for signed-in people: a page request
            # goes to the login page, anything else (scripts, models, data)
            # is refused.
            if self.demo and url.path == LOGIN_PAGE:       # nothing to sign in to
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/index.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            if url.path not in PUBLIC_STATIC and not self._current_actor():
                if url.path in ("/", "/index.html") or url.path.endswith(".html"):
                    self.send_response(HTTPStatus.FOUND)
                    self.send_header("Location", LOGIN_PAGE)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                return self._error(HTTPStatus.UNAUTHORIZED, "Sign in first.")
            if method in ("GET", "HEAD") and url.path == "/brand/logo.png":
                return self._send_static(LOGO, "image/png")
            # An approved retrained version (single model or ensemble) is
            # served in place of the shipped file; otherwise fall through.
            if method in ("GET", "HEAD") and MODEL_FILE.fullmatch(url.path) and self._serve_active_model(url.path):
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
            if not self._authorize(method, parts):
                return None
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
            single, ens = self.db.active_model("single"), self.db.active_model("ensemble")
            return self._send_json({"app": APP_ID, "ok": True, "db": self.db.path.name,
                                    "counts": self.db.counts(),
                                    "active_model": single["version"] if single else None,
                                    "active_ensemble": ens["version"] if ens else None})

        if head == "auth":
            return self._auth(method, parts[1:])
        if head == "users":
            return self._users(method, parts[1:])

        if head == "analyses":
            if method == "GET" and len(parts) == 1:
                return self._send_json(self.db.list_analyses(limit))
            if method == "POST" and len(parts) == 1:
                saved = self.db.save_analysis(self._read_json(), self._actor())
                # Every capture is stored now, so the rolling window for
                # routine raw IQ moves on with each one (LocalDB.retention).
                self.db.prune_rolling_iq(self.keep_bytes)
                return self._send_json(saved, HTTPStatus.CREATED)
            if method == "GET" and len(parts) == 3 and parts[2] == "iq":
                return self._analysis_iq(parts[1])
            if method == "POST" and len(parts) == 3 and parts[2] == "reviewed":
                return self._send_json(self.db.mark_reviewed(parts[1], self._actor(),
                                                             self._read_json().get("note", "")))
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
            if method == "GET" and len(parts) == 3 and parts[2] == "iq":
                return self._correction_iq(parts[1])
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
            with Handler._jobs_lock:
                self._reap_jobs()
                job = self.db.start_retrain(body.get("reason"), bool(body.get("override")), self._actor(),
                                            scope=str(body.get("scope") or "single"))
                return self._send_json(self._launch(job), HTTPStatus.ACCEPTED)
        if head == "retrain" and method == "GET" and parts[1:] == ["history"]:
            with Handler._jobs_lock:
                self._reap_jobs()
            return self._send_json(self.db.retrain_history())
        if head == "retrain" and method == "GET" and parts[1:2] == ["jobs"]:
            with Handler._jobs_lock:
                self._reap_jobs()
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
                return self._send_json({"retired": self.db.rollback_model(
                    body.get("note"), self._actor(), kind=str(body.get("kind") or "single"))})
            if method == "POST" and len(parts) == 3 and parts[2] == "review":
                body = self._read_json()
                return self._send_json(self.db.review_model(
                    parts[1], str(body.get("decision") or ""), body.get("note"), self._actor()))
            if method == "POST" and len(parts) == 3 and parts[2] == "activate":
                body = self._read_json()
                if body.get("override") and not role_at_least(self._actor().role, TOP_ROLE):
                    raise PermissionError("Activating a version that failed its exam or was rejected "
                                          "is an override and needs the analyst role.")
                return self._send_json(self.db.activate_model(
                    parts[1], body.get("note"), self._actor(), override=bool(body.get("override"))))

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

    def _auth(self, method, rest):
        client = self.client_address[0]
        if self.demo:
            if method == "GET" and rest == ["me"]:
                return self._send_json({"auth_enabled": True, "demo": True, "can_setup": False,
                                        "user": {"username": self.me.name, "role": self.me.role},
                                        "people": [{"username": n, "role": r} for n, r in DEMO_PEOPLE.items()]})
            if method == "POST" and rest == ["demo"]:
                name = str(self._read_json().get("username") or "").lower()
                if name not in DEMO_PEOPLE:
                    raise ValueError(f"demo person must be one of {', '.join(DEMO_PEOPLE)}")
                self.db.log(Actor(name, DEMO_PEOPLE[name], client), "user.demo_switch", "user", name,
                            {"from": self.me.name})
                return self._send_json({"user": {"username": name, "role": DEMO_PEOPLE[name]}}, headers={
                    "Set-Cookie": f"{DEMO_COOKIE}={name}; HttpOnly; SameSite=Strict; Path=/"})
            return self._error(HTTPStatus.NOT_FOUND, "sign-in is off in --demo; switch person instead")
        if method == "GET" and rest == ["me"]:
            enabled = self.db.auth_enabled()
            return self._send_json({"auth_enabled": enabled,
                                    "user": {"username": self.me.name, "role": self.me.role} if self.me else None,
                                    "can_setup": not enabled and client in LOOPBACK_CLIENTS})
        if method == "POST" and rest in (["setup"], ["login"]):
            body = self._read_json()
            if rest == ["setup"]:
                # Only from the machine the server runs on: a laptop on the
                # LAN must not be able to claim the first analyst account.
                if client not in LOOPBACK_CLIENTS:
                    raise PermissionError("Set up sign-in on the machine running the server.")
                self.db.setup_admin(body.get("username"), body.get("password"), client)
            token, user = self.db.login(body.get("username"), body.get("password"), client)
            return self._send_json({"user": {"username": user["username"], "role": user["role"]}},
                                   headers=self._cookie(token))
        if method == "POST" and rest == ["logout"]:
            token = self._session_token()
            if token and self.me:
                self.db.logout(token, self.me)
            return self._send_json({"signed_out": True}, headers=self._cookie(None))
        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _users(self, method, rest):
        actor = self._actor()
        if method == "GET" and not rest:
            if not role_at_least(actor.role, TOP_ROLE):
                raise PermissionError("Only an analyst can list accounts.")
            return self._send_json(self.db.list_users())
        body = self._read_json() if method == "POST" else {}
        if method == "POST" and not rest:
            return self._send_json(self.db.create_user(body.get("username"), body.get("password"),
                                                       str(body.get("role") or "operator"), actor),
                                   HTTPStatus.CREATED)
        if method == "POST" and len(rest) == 1:
            return self._send_json(self.db.update_user(
                rest[0], actor, role=body.get("role"),
                disabled=body["disabled"] if "disabled" in body else None,
                password=body.get("password")))
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

    # Training processes this server started, by job id.
    _procs: dict = {}
    _jobs_lock = threading.Lock()

    def _reap_jobs(self) -> None:
        """A job left 'queued'/'running' whose process is gone (killed, crashed
        before it could report, or started by a server that has since been
        restarted) would block every later retrain. Mark it failed."""
        for job in self.db.list_jobs():
            if job["status"] not in ("queued", "running"):
                continue
            proc = Handler._procs.get(job["id"])
            if proc is None or proc.poll() is not None:
                why = ("the training process exited without reporting a result"
                       f" (exit code {proc.returncode})" if proc is not None else
                       "the training process is no longer running (the server was restarted)")
                self.db.finish_job(job["id"], False, {"error": why})
                Handler._procs.pop(job["id"], None)

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
        Handler._procs[job["id"]] = proc
        self.db.update_job(job["id"], log_path=log_rel, pid=proc.pid)
        return self.db.get_job(job["id"])

    def _serve_active_model(self, path: str) -> bool:
        """/models/best_model.onnx and /models/ensemble_<i>.onnx: the ACTIVE
        approved retrained version when there is one, else the shipped file
        (return False and the static handler serves it). The shipped files in
        web/models/ are never overwritten, so roll back is exact."""
        name = path.rsplit("/", 1)[-1]
        if name == "best_model.onnx":
            active = self.db.active_model("single")
            file = (self.db.path.parent / active["path"]) if active else None
        else:
            active = self.db.active_model("ensemble")
            file = (self.db.path.parent / active["path"] / name) if active else None
        if not file or not file.exists():
            return False
        body = file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-NEXA-Model", active["version"])
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return True

    def _send_static(self, path: Path, content_type: str):
        if not path.exists():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return None

    def _export(self, body):
        card_path = WEB / "data" / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        actor = self._actor()
        data, ctype, fname, meta = build_export(
            self.db, self.db.path.parent, ids=list(body.get("ids") or []),
            sections=list(body.get("sections") or []), fmt=str(body.get("format") or ""),
            banner=str(body.get("banner") or ""), generated_by=actor.name,
            model_card=card, active_model=self.db.active_model("single"),
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

    def _send_iq(self, payload):
        """Raw IQ for the page, with its format -- and a SigMF file's
        annotations, so a capture saved before its truth was stored still
        gets its dashed truth boxes when someone views it again."""
        datatype, rate, data, meta = payload
        headers = {"X-NEXA-Datatype": str(datatype), "X-NEXA-Sample-Rate": str(rate)}
        notes = (meta or {}).get("annotations") or []
        if notes:
            text = json.dumps(notes, separators=(",", ":"))
            if len(text) <= 6000:                   # stays a header, not a second request
                headers["X-NEXA-Annotations"] = text
        return self._send_file(data, "application/octet-stream", "capture.iq", headers)

    def _analysis_iq(self, analysis_id):
        """A stored capture's raw IQ, to view it again and flag a correction."""
        a = self.db.get_analysis(analysis_id)
        if not a:
            return self._error(HTTPStatus.NOT_FOUND, "no such capture")
        if a["retention"]["kind"] == "deleted":
            return self._error(HTTPStatus.GONE, "this routine signal was deleted by the rolling window; "
                                                "only its result is kept")
        payload = _iq_payload(self.db.path.parent, a["file_path"]) if a.get("file_path") else None
        if not payload:
            return self._error(HTTPStatus.NOT_FOUND, "no raw signal was stored for this capture")
        return self._send_iq(payload)

    def _correction_iq(self, correction_id):
        """The raw IQ a correction points at, so a reviewer can look at the
        signal before approving it instead of trusting the text alone."""
        c = self.db.get_correction(correction_id)
        if not c:
            return self._error(HTTPStatus.NOT_FOUND, "no such correction")
        payload = _iq_payload(self.db.path.parent, c["iq_path"]) if c.get("iq_path") else None
        if not payload:
            return self._error(HTTPStatus.NOT_FOUND, "the raw IQ behind this correction is missing")
        return self._send_iq(payload)

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
    failed.)

    A port counts as taken when something already ANSWERS on it, not only
    when bind() fails: on Windows a server bound to 0.0.0.0 with
    SO_REUSEADDR (e.g. `python -m http.server 8099`) does not stop us binding
    127.0.0.1:8099, and the browser then reaches the plain file server --
    no database, no sign-in -- with no error anywhere."""
    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                print(f"  port {port} is already in use by another program; trying {port + 1}")
                continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {preferred} and {preferred + tries - 1}.")


def make_server(db_path: Path, host: str, port: int, lan: bool = False,
                retrain_args: list | None = None, demo: bool = False,
                keep_bytes: int = ROLLING_IQ_BYTES) -> ThreadingHTTPServer:
    db = LocalDB(db_path)
    iq_dir = Path(db_path).parent / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    handler = type("NexaHandler", (Handler,), {"db": db, "iq_dir": iq_dir, "lan": lan, "demo": demo, "keep_bytes": keep_bytes,
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
    p.add_argument("--sign-in", action="store_true",
                   help="require real accounts (login page, passwords, Users page). Without it, "
                        "buttons at the top switch between an operator and an analyst")
    p.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)   # the default now; kept so old commands work
    p.add_argument("--keep-gb", type=float, default=ROLLING_IQ_BYTES / 1024 ** 3,
                   help="rolling window for ROUTINE raw signals (civilian/empty, confident, uncorrected); "
                        "threats, close calls, corrected and training captures are always kept")
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
    if a.demo and a.sign_in:
        raise SystemExit("--demo and --sign-in contradict each other: pick one.")
    if a.demo and a.lan:
        raise SystemExit("--demo turns sign-in off, so anyone who can reach the server could act as "
                         "the analyst. It is refused together with --lan.")
    # Without sign-in anyone who can reach the server could act as the analyst,
    # so on the network (--lan) sign-in is always on.
    demo = not (a.sign_in or a.lan)
    server = make_server(a.db, host, port, a.lan, retrain_args, demo=demo,
                         keep_bytes=int(a.keep_gb * 1024 ** 3))
    url = f"http://localhost:{port}/index.html"          # with sign-in on, redirects to the login page first
    counts = server.nexa_db.counts()
    print(f"NEXA local server\n  open      {url}\n  database  {a.db}\n"
          f"  stored    {counts.get('analyses', 0)} analyses, {counts.get('audit_log', 0)} audit entries\n"
          f"  network   {'LAN (other machines can connect)' if a.lan else 'this machine only'}\n"
          + ("  sign-in   OFF: switch person with the buttons at the top of the page (--sign-in for accounts)\n"
             if demo else "  sign-in   ON (login page; accounts in the database)\n")
          + "Press Ctrl+C to stop.")
    if not demo and not server.nexa_db.auth_enabled():
        print("  FIRST RUN: no accounts yet. Open the page on THIS machine to create the first analyst account.")
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
