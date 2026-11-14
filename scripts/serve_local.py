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
    python scripts/serve_local.py --lan --allow 172.22.80.3   # only that teammate's laptop: no sign-in
    python scripts/serve_local.py --lan --no-sign-in          # whole network, no sign-in (own hotspot only)

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
    GET    /api/intel                  Comms Intel: decoded receiver-audio recordings
                                       (written by scripts/comms_intel.py into --intel-dir)
    GET    /api/intel/<id>             one recording: transmissions, activity, LLM reading
    POST   /api/pipeline?source=voice-demo|file&name=f&max_s=1&detector=hybrid|yolo|energy
                                       &listen=1&whisper=1&llm=1[&fs=2e6&dtype=int16]
                                       body = the IQ file (source=file). Runs the wideband
                                       pipeline (scripts/run_pipeline.py) in the background
    GET    /api/pipeline               pipeline runs, newest first
    GET    /api/pipeline/<id>          one run: status, log tail, tracks (when done)
    GET    /api/pipeline/<id>/<file>   tracks.png or track_<n>.wav of a run

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
import datetime
import threading
import time
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.localdb import CLASSES, ROLLING_IQ_BYTES, SESSION_HOURS, SYSTEM, TOP_ROLE, Actor, LocalDB, role_at_least  # noqa: E402
from src.report_export import _iq_payload, build_export  # noqa: E402

WEB = ROOT / "web"
DEFAULT_DB = ROOT / "data" / "local" / "nexa.db"
DEFAULT_INTEL = ROOT / "results" / "comms_intel"
PIPELINE_DIR = ROOT / "results" / "pipeline" / "jobs"
CLASSIFIER_DIR = ROOT / "results" / "classifier"
DETECTOR_PT = ROOT / "results" / "detector" / "emitter_detector.pt"
PIPE_FILE = re.compile(r"(tracks\.png|track_\d{1,6}\.wav)")
PIPE_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
INTEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,200}")
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
    # Declaring a flagged capture correct is a judgement like approving one:
    # an operator views it and corrects it; only an analyst closes it as right.
    if method == "POST" and head == "analyses" and len(parts) == 3 and parts[2] == "reviewed":
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


def kiwi_location(db, khz, within_khz=2.0):
    """The newest transmitter location recorded (KiwiSDR TDoA, "Record location")
    for a frequency within within_khz of khz, or None."""
    if not khz:
        return None
    for e in db.audit(1000, "client.tdoa_location"):
        d = e["details"] or {}
        try:
            if abs(float(d.get("freq_khz")) - float(khz)) <= within_khz:
                return {"freq_khz": d.get("freq_khz"), "lat": d.get("lat"), "lon": d.get("lon"),
                        "receivers": d.get("receivers"), "note": d.get("note"), "source": d.get("source"),
                        "radius_km": d.get("radius_km"), "ts": e["ts"], "actor": e["actor"]}
        except (TypeError, ValueError):
            continue
    return None


def tdoa_details(res):
    """A computed TDoA result (scripts/sdr_hub.py TdoaRunner) as the audit entry
    client.tdoa_location -- the same fields the manual "Record location" form
    writes, so kiwi_location() finds either."""
    return {"freq_khz": res["freq_khz"], "lat": res["lat"], "lon": res["lon"], "receivers": res["receivers"],
            "note": "; ".join([f"most likely point; 95 % area about {res['radius_km']} km radius",
                               f"{len(res['stations'])} receivers, {res.get('runs', 1)} x {res['sample_s']} s recordings"]
                              + [f"{x['id']} left out: {x['why']} (about {x['offset_us']} µs)"
                                 for x in res.get("excluded") or []]
                              + ([res["warning"]] if res.get("warning") else [])),
            "source": "NEXA TDoA (computed)", "radius_km": res["radius_km"], "area": res["area"],
            "fit": res["fit"], "pairs": res["pairs"], "stations": res["stations"],
            "excluded": res.get("excluded") or [], "dropped_pairs": res.get("dropped_pairs") or [],
            "warning": res.get("warning")}


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
    intel_dir: Path = DEFAULT_INTEL

    # -- request gate ----------------------------------------------------------

    allow_ips: frozenset = frozenset()    # --allow: only these machines (and this one) may connect

    def _client_ok(self) -> bool:
        if not self.allow_ips:
            return True
        return self.client_address[0] in LOOPBACK_CLIENTS or self.client_address[0] in self.allow_ips

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
        if not self._client_ok():
            return self._error(HTTPStatus.FORBIDDEN, "this machine is not on the --allow list")
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

    # -- comms intel (read-only files from scripts/comms_intel.py) --------------

    def _intel(self, rest):
        """The decoded recordings. Files only, never written here; the id is
        checked so a request cannot reach outside the folder."""
        d = self.intel_dir
        self._audit_results()
        if not rest:
            index = d / "index.json"
            if not index.is_file():
                return self._send_json({"recordings": [], "folder": str(d),
                                        "hint": "run: python scripts/comms_intel.py --in <folder of .wav>"})
            data = json.loads(index.read_text(encoding="utf-8"))
            data["folder"] = str(d)
            return self._send_json(data)
        rid = rest[0]
        if len(rest) != 1 or not INTEL_ID.fullmatch(rid) or rid == "index":
            return self._error(HTTPStatus.NOT_FOUND, "no such recording")
        f = d / f"{rid}.json"
        if not f.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "no such recording")
        rec = json.loads(f.read_text(encoding="utf-8"))
        from src.intel import store
        audio = store.audio_of(rec, PIPELINE_DIR)
        if audio:                                # served by /api/pipeline/<run>/<file>
            rec["audio_url"] = f"/api/pipeline/{rec['pipeline_run']}/{audio}"
        return self._send_json(rec)

    def _simulate(self, query):
        """The simulated receiver's "Random mix" case: a fresh scene of the current signal
        groups (radar, spread spectrum, digital comms, jammers, pulses -- src/detect/scenes.py,
        the generator the detector is trained on), several emitters overlapping in time and
        frequency. -> {fs, iq: base64 float32 I/Q interleaved, truth: [{className, startS, endS,
        fLoHz, fHiHz}], classes}. Generated here, never stored."""
        import base64
        import numpy as np
        from src.detect import scenes
        q = lambda k, d: (query.get(k) or [d])[0]
        seed = int(q("seed", "0")) or None
        dur = min(max(float(q("duration_s", "0.05")), 0.005), 0.2)
        snr = q("snr_db", "")
        rng = np.random.default_rng(seed)
        n_scenes = max(1, round(dur * scenes.FS / scenes.SCENE_SAMPLES))
        parts, truth = [], []
        for k in range(n_scenes):
            iq, _, _, meta = scenes.make_scene(
                rng, n_emitters=int(rng.integers(1, 4)),
                snr_db=float(snr) if snr not in ("", "random") else float(rng.uniform(5, 20)))
            off = k * scenes.SCENE_SAMPLES
            for i, cls in enumerate(meta["cls"]):
                if meta["ignore"][i]:
                    continue                                   # too faint to see: not a fair "truth"
                truth.append({"className": cls, "startS": (off + meta["t_start"][i]) / scenes.FS,
                              "endS": (off + meta["t_end"][i]) / scenes.FS,
                              "fLoHz": float(meta["f_lo"][i]), "fHiHz": float(meta["f_hi"][i])})
            parts.append(iq)
        iq = np.concatenate(parts).astype(np.complex64)
        inter = np.empty(iq.size * 2, dtype=np.float32)
        inter[0::2], inter[1::2] = iq.real, iq.imag
        return self._send_json({"fs": scenes.FS, "iq": base64.b64encode(inter.tobytes()).decode("ascii"),
                                "truth": truth, "classes": sorted({t["className"] for t in truth})})

    def _assistant(self):
        """The Assistant page's free questions: the local AI answers from NEXA's own
        records only (src/intel/assistant.py). Nothing leaves this machine."""
        from src.intel import assistant
        body = self._read_json()
        ctx = assistant.build_context(self.db, self.intel_dir, WEB / "data")
        try:
            out = assistant.answer(body.get("question"), ctx)
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except (OSError, TimeoutError) as e:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"the local AI (Ollama) did not answer: {e}")
        return self._send_json(out)

    def _intel_translate(self, rid):
        """Translate a decoded recording into English with the local AI (Ollama, this
        machine) and keep the translation in the recording's file, next to the original."""
        from src.intel import reason, store
        if not INTEL_ID.fullmatch(rid) or rid == "index":
            return self._error(HTTPStatus.NOT_FOUND, "no such recording")
        f = self.intel_dir / f"{rid}.json"
        if not f.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "no such recording")
        body = self._read_json()
        model = str(body.get("model") or reason.DEFAULT_MODEL)
        if not re.fullmatch(r"[A-Za-z0-9._:\-]{1,64}", model):
            raise ValueError("model must be an Ollama model name, e.g. llama3.2:3b")
        rec = json.loads(f.read_text(encoding="utf-8"))
        tr = rec.get("transcript") or {}
        text = tr.get("text") or "\n".join(s.get("text", "") for s in rec.get("segments") or [])
        try:
            out = reason.translate(text, tr.get("language"), model=model)
        except (OSError, ValueError) as e:      # Ollama not running / model missing / nothing to translate
            return self._error(HTTPStatus.BAD_GATEWAY if isinstance(e, OSError) else HTTPStatus.BAD_REQUEST,
                               f"could not translate: {e}" + (" (is Ollama running? ollama serve)"
                                                              if isinstance(e, OSError) else ""))
        out["at"] = datetime.datetime.now().isoformat(timespec="seconds")
        rec = json.loads(f.read_text(encoding="utf-8"))       # re-read: another writer may have touched it
        rec["translation"] = out
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        store._replace(tmp, f)
        self.db.log(self._actor(), "intel.translate", "intel", rid,
                    {"language": out["language"], "model": model, "chars": len(text)})
        return self._send_json(out)

    # -- wideband pipeline runs (scripts/run_pipeline.py, results/pipeline/jobs) --

    _pipe_procs: dict = {}
    _pipe_py: str | None = None

    def pipeline_python(self) -> str:
        """An interpreter that can run the whole chain, Whisper included: the
        project venv may have torch but not openai-whisper (or the other way
        round). First one that imports all of them; else the training one."""
        if Handler._pipe_py is None:
            Handler._pipe_py = self.training_python()
            for py in dict.fromkeys([self.training_python(), sys.executable]):
                ok = subprocess.run([py, "-c", "import torch, scipy, whisper"], capture_output=True, timeout=180)
                if ok.returncode == 0:
                    Handler._pipe_py = py
                    break
        return Handler._pipe_py

    def _pipe_status(self, d: Path) -> dict:
        meta = json.loads((d / "job.json").read_text(encoding="utf-8")) if (d / "job.json").exists() else {}
        proc = Handler._pipe_procs.get(d.name)
        if proc is not None and proc.poll() is None:
            status = "running"
        elif (d / "tracks.json").exists():
            status = "done"
        elif meta.get("live") and (d / "alive").exists() and time.time() - (d / "alive").stat().st_mtime < 60:
            status = "running"          # the live bridge is its own process, and it keeps this file fresh
        elif (d / "stopped").exists():
            status = "stopped"          # Stop pressed on the page
        else:
            status = "failed"           # exited without results, or started by a server that has stopped
        meta["status"] = status
        if status == "done" and (d / "job.json").exists():   # job.json is written at the start, tracks.json at the end
            meta["took_s"] = round(max((d / "tracks.json").stat().st_mtime - (d / "job.json").stat().st_mtime, 0.0), 1)
        return meta

    _audit_lock = threading.Lock()

    def _audit_results(self):
        """Enter Python-side results into the audit log. Pipeline runs, live
        bridge recordings (scripts/live_sdrconnect.py) and Comms Intel
        recordings are written by other processes that do not touch the
        database, so the server logs each one the first time it sees it
        finished: what came in, what was found, what was heard. A marker
        keeps every result to one entry."""
        # The marker names THIS database: two servers with different databases (the
        # demo database and a test one, say) each log every result once.
        db_tag = hashlib.sha1(str(Path(self.db.path).resolve()).lower().encode()).hexdigest()[:10]
        with Handler._audit_lock:
            try:
                runs = sorted(p for p in PIPELINE_DIR.iterdir() if p.is_dir() and PIPE_ID.fullmatch(p.name)) \
                    if PIPELINE_DIR.is_dir() else []
                for d in runs:
                    mark = d / f".audited-{db_tag}"
                    if mark.exists() or not (d / "tracks.json").exists():
                        continue
                    meta = json.loads((d / "job.json").read_text(encoding="utf-8")) if (d / "job.json").exists() else {}
                    doc = json.loads((d / "tracks.json").read_text(encoding="utf-8"), parse_constant=lambda c: None)
                    tracks = doc.get("tracks") or []
                    heard = [{"answer": t.get("answer"), "kind": (t.get("listen") or {}).get("kind"),
                              "f_khz": [round(t.get("f_lo_khz") or 0, 1), round(t.get("f_hi_khz") or 0, 1)],
                              "text": ((t.get("listen") or {}).get("transcript")
                                       or (t.get("listen") or {}).get("morse") or "")[:200]} for t in tracks[:40]]
                    live = bool(meta.get("live"))
                    actor = Actor("live bridge", "system", "localhost") if live else \
                        Actor(meta.get("who") or "operator", meta.get("role") or "operator", meta.get("client"))
                    self.db.log(actor, "live.result" if live else "pipeline.result", "pipeline_run", d.name,
                                {"source": meta.get("file") or meta.get("source"), "detector": meta.get("detector"),
                                 "classifiers": meta.get("classifiers"), "duration_ms": doc.get("duration_ms"),
                                 "n_tracks": len(tracks), "tracks": heard})
                    mark.write_text(datetime.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
                index = self.intel_dir / "index.json"
                if index.is_file():
                    done_f = self.intel_dir / f".audited-{db_tag}.json"
                    done = set(json.loads(done_f.read_text(encoding="utf-8"))) if done_f.is_file() else set()
                    new = [r for r in json.loads(index.read_text(encoding="utf-8")).get("recordings", [])
                           if r.get("id") and r["id"] not in done]
                    for r in new:
                        self.db.log(SYSTEM, "intel.recording", "intel", r["id"],
                                    {k: r.get(k) for k in ("file", "kind", "recorded_at", "duration_s", "transmissions",
                                                           "category", "confidence", "call_signs", "model")}
                                    | {"summary": (r.get("summary") or "")[:300]})
                        done.add(r["id"])
                    if new:
                        done_f.write_text(json.dumps(sorted(done)), encoding="utf-8")
            except Exception as e:                       # noqa: BLE001 -- auditing must not break the page
                print(f"audit of pipeline / intel results failed: {type(e).__name__}: {e}", file=sys.stderr)

    def _pipeline(self, method, rest, query):
        """Start a pipeline run on an uploaded IQ file (or the voice demo) and
        read its results. Runs are folders under results/pipeline/jobs/."""
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        self._audit_results()
        if method == "GET" and not rest:
            from src.intel import voice
            runs = sorted((d for d in PIPELINE_DIR.iterdir() if d.is_dir() and PIPE_ID.fullmatch(d.name)),
                          reverse=True)[:30]
            return self._send_json({"runs": [self._pipe_status(d) for d in runs],
                                    "classifiers": [c.name for c in sorted(CLASSIFIER_DIR.glob("*best*.pt"))],
                                    "detector": DETECTOR_PT.exists(),
                                    "whisper_models": voice.installed_models(),
                                    "whisper_default": voice.DEFAULT_MODEL,
                                    "languages": list(voice.LANGUAGES)})
        if rest and not PIPE_ID.fullmatch(rest[0]):
            return self._error(HTTPStatus.NOT_FOUND, "no such run")
        if method == "GET" and len(rest) == 1:
            d = PIPELINE_DIR / rest[0]
            if not d.is_dir():
                return self._error(HTTPStatus.NOT_FOUND, "no such run")
            out = self._pipe_status(d)
            log = d / "run.log"
            out["log"] = log.read_text(encoding="utf-8", errors="replace")[-3000:] if log.exists() else ""
            if out["status"] == "done":
                out["result"] = json.loads((d / "tracks.json").read_text(encoding="utf-8"),
                                           parse_constant=lambda c: None)     # NaN/Infinity from older runs
            return self._send_json(out)
        if method == "GET" and len(rest) == 2:
            f = PIPELINE_DIR / rest[0] / rest[1]
            if not PIPE_FILE.fullmatch(rest[1]) or not f.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "no such file")
            body = f.read_bytes()
            # Byte ranges: without them the browser's audio player cannot seek (a click
            # on the Intercepts timeline would restart the audio from 0).
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
            status, first, last = HTTPStatus.OK, 0, len(body) - 1
            if m and body and (m.group(1) or m.group(2)):
                if m.group(1):
                    first = int(m.group(1))
                    last = min(int(m.group(2)), len(body) - 1) if m.group(2) else len(body) - 1
                else:                                   # bytes=-N: the last N bytes
                    first = max(0, len(body) - int(m.group(2)))
                if first > last or first >= len(body):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{len(body)}")
                    self.end_headers()
                    return None
                status = HTTPStatus.PARTIAL_CONTENT
            part = body[first:last + 1]
            self.send_response(status)
            self.send_header("Content-Type", "image/png" if f.suffix == ".png" else "audio/wav")
            self.send_header("Content-Length", str(len(part)))
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(part)
            return None
        if method == "POST" and not rest:
            return self._pipeline_start(query)
        if method == "POST" and len(rest) == 2 and rest[1] == "stop":
            proc = Handler._pipe_procs.get(rest[0])
            if proc is None or proc.poll() is not None:
                return self._error(HTTPStatus.CONFLICT, "this run is not running")
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True) \
                if os.name == "nt" else proc.kill()      # the run and its helpers (Whisper, Ollama calls)
            proc.wait(timeout=10)
            (PIPELINE_DIR / rest[0] / "stopped").write_text(self._actor().name, encoding="utf-8")
            self.db.log(self._actor(), "pipeline.stop", "pipeline_run", rest[0], {})
            return self._send_json({"stopped": rest[0]})
        return self._error(HTTPStatus.NOT_FOUND, "unknown pipeline request")

    def _pipeline_start(self, query):
        def q(k, default=None):
            return (query.get(k) or [default])[0]

        if any(proc.poll() is None for proc in Handler._pipe_procs.values()):
            return self._error(HTTPStatus.CONFLICT, "a pipeline run is already going; wait for it to finish")
        classifiers = sorted(CLASSIFIER_DIR.glob("*best*.pt"))
        if not classifiers:
            return self._error(HTTPStatus.CONFLICT, f"no classifier: put a week-1 best.pt into {CLASSIFIER_DIR}")
        from src.intel import voice
        use_whisper = q("listen", "1") == "1" and q("whisper", "1") == "1"
        model, language = q("whisper_model", voice.model_for_this_pc()), q("language", "auto")
        if use_whisper:
            # Checked before anything is created. These reach a command line, and a
            # model that is not on this machine must never trigger a download.
            if model not in voice.installed_models():
                return self._error(HTTPStatus.BAD_REQUEST, f"Whisper model '{model}' is not on this machine; "
                                   f"it has {voice.installed_models()}")
            if language not in voice.LANGUAGES:
                return self._error(HTTPStatus.BAD_REQUEST, f"language must be one of {list(voice.LANGUAGES)}")
        import datetime
        import secrets
        rid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        d = PIPELINE_DIR / rid
        d.mkdir(parents=True)
        source = q("source", "file")
        max_s = min(max(float(q("max_s", "1.0")), 0.05), 120.0)   # read in pieces: the length does not cost memory
        cmd = [self.pipeline_python(), "-u", str(ROOT / "scripts" / "run_pipeline.py"),
               "--classifier", *map(str, classifiers), "--out", str(d), "--max-seconds", str(max_s)]
        actor = self._actor()
        meta = {"id": rid, "source": source, "max_s": max_s, "status": "running",
                "classifiers": [c.name for c in classifiers], "who": actor.name, "role": actor.role,
                "client": actor.client}
        if source == "voice-demo":
            cmd.append("--voice-demo")
        else:
            n = int(self.headers.get("Content-Length") or -1)
            if n <= 0:
                return self._error(HTTPStatus.BAD_REQUEST, "send the IQ file as the request body")
            if n > MAX_CAPTURE:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"file is {n} bytes; limit {MAX_CAPTURE}")
            name = safe_name(q("name", "capture.bin"))
            dest = d / ("input_" + name)
            left = n
            with open(dest, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            cmd += ["--iq", str(dest)]
            if q("fs"):
                cmd += ["--fs", str(float(q("fs")))]
            if q("dtype") in ("float32", "int16", "int8", "uint8"):
                cmd += ["--dtype", q("dtype")]
            meta["file"] = name
        mode = q("detector", "hybrid")
        if mode == "energy" or not DETECTOR_PT.exists():
            cmd.append("--energy")
            meta["detector"] = "energy"
        else:
            if mode == "hybrid":
                cmd.append("--hybrid")
            meta["detector"] = "yolo+narrow" if mode == "hybrid" else "yolo"
        if q("listen", "1") == "1":
            cmd.append("--listen")
            if not use_whisper:
                cmd.append("--no-whisper")
            else:
                cmd += ["--whisper-model", model, "--language", language]
                meta["whisper_model"], meta["language"] = model, language
            if q("llm", "1") != "1":
                cmd.append("--no-llm")
        meta["listen"] = "--listen" in cmd
        (d / "job.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log = open(d / "run.log", "w", encoding="utf-8")
        try:
            Handler._pipe_procs[rid] = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        finally:
            log.close()
        self.db.log(actor, "pipeline.start", "pipeline_run", rid,
                    {k: meta.get(k) for k in ("source", "file", "max_s", "detector", "classifiers", "listen",
                                              "whisper_model", "language")})
        return self._send_json(meta, HTTPStatus.CREATED)

    # -- routes ---------------------------------------------------------------

    def _route(self, method, parts, query):
        limit = int((query.get("limit") or ["500"])[0])
        head = parts[0] if parts else ""

        if head == "health" and method == "GET":
            single, ens = self._active_single(), self.db.active_model("ensemble")
            return self._send_json({"app": APP_ID, "ok": True, "db": self.db.path.name,
                                    "counts": self.db.counts(),
                                    "active_model": single["version"] if single else None,
                                    "active_ensemble": ens["version"] if ens else None})

        if head == "intel" and method == "GET":
            return self._intel(parts[1:])
        if head == "intel" and method == "POST" and len(parts) == 3 and parts[2] == "translate":
            return self._intel_translate(parts[1])
        if head == "assistant" and method == "POST" and len(parts) == 1:
            return self._assistant()
        if head == "simulate" and method == "GET" and len(parts) == 1:
            return self._simulate(query)
        if head == "pipeline":
            return self._pipeline(method, parts[1:], query)
        if head == "sdr":            # live SDRconnect receiver -- scripts/sdr_hub.py (feat/live-sdr-console)
            return self._sdr(method, parts[1:], query)
        if head == "kiwi":           # continuous KiwiSDR listening -- scripts/sdr_hub.py KiwiHub
            return self._kiwi(method, parts[1:], query)
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

    # -- live SDRconnect receiver (scripts/sdr_hub.py) ---------------------------
    # One connection to SDRconnect for the whole server; every page reads it.
    # Owned by the SDR-connection half of feat/live-sdr-console: see docs/LIVE_SDR_BRANCH.md.

    _sdr_hub = None
    _sdr_lock = threading.Lock()

    def _hub(self):
        with Handler._sdr_lock:
            if Handler._sdr_hub is None:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from sdr_hub import SdrHub
                hub = SdrHub(jobs_dir=PIPELINE_DIR, intel_dir=self.intel_dir, max_s=20.0)   # a signal that never goes quiet is named every 20 s
                # Each decoded transmission goes into the audit trail as soon as it is
                # done (live.result + intel.recording), not only when a page next opens
                # Signal Pipeline. _audit_results only reads class-level state.
                hub.on_result = lambda _entry, h=self: h._audit_results()
                Handler._sdr_hub = hub
            return Handler._sdr_hub

    _kiwi_hub = None

    def _kiwi(self, method, rest, query=None):
        """Listen to one KiwiSDR frequency continuously: decode + AI reading every clip."""
        if method == "GET" and rest == ["location"]:      # newest transmitter location recorded for a frequency
            try:
                khz = float((query or {}).get("khz", [""])[0])
            except ValueError:
                khz = None
            return self._send_json({"location": kiwi_location(self.db, khz)})
        with Handler._sdr_lock:
            if Handler._kiwi_hub is None:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from sdr_hub import KiwiHub
                hub = KiwiHub(jobs_dir=PIPELINE_DIR, intel_dir=self.intel_dir)
                hub.on_result = lambda _entry, h=self: h._audit_results()
                hub.tdoa.on_located = lambda res, who, db=self.db: db.log(
                    who, "client.tdoa_location", "kiwi", f"{res['freq_khz']:g} kHz", tdoa_details(res))
                Handler._kiwi_hub = hub
        hub = Handler._kiwi_hub
        if method == "GET" and rest == ["status"]:
            return self._send_json(hub.status())
        if method == "POST" and rest == ["start"]:
            b = self._read_json()
            st = hub.connect(host=b.get("host"), port=b.get("port"), freq_khz=b.get("freq_khz"),
                             whisper_model=b.get("whisper_model") or "small", language=b.get("language"),
                             use_llm=b.get("llm", True), clip_s=b.get("clip_s") or 30,
                             llm_model=b.get("llm_model") or None, demod=b.get("demod") or None)
            self.db.log(self._actor(), "kiwi.start", "kiwi", st["url"], {"label": st["label"], "decode": st["decode"]})
            return self._send_json(st)
        if method == "POST" and rest == ["locate"]:       # TDoA from 3 GPS-timed KiwiSDRs, in the background
            b = self._read_json()
            who = self._actor()
            st = hub.tdoa.start(b.get("freq_khz"), receivers=b.get("receivers") or None,
                                sample_s=b.get("sample_s") or 30, who=who, runs=b.get("runs") or 1)
            self.db.log(who, "kiwi.locate", "kiwi", f"{st['freq_khz']:g} kHz",
                        {"receivers": b.get("receivers") or "default", "sample_s": st["sample_s"], "runs": st["runs"]})
            return self._send_json(st)
        if method == "POST" and rest == ["stop"]:
            was = hub.status()
            st = hub.disconnect()
            if was["state"] != "off":
                self.db.log(self._actor(), "kiwi.stop", "kiwi", was["url"],
                            {"label": was["label"], "decodes": len(was["decodes"]), "skipped": was["skipped"]})
            return self._send_json(st)
        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _sdr(self, method, rest, query):
        hub = self._hub()
        if method == "GET" and rest == ["status"]:
            return self._send_json(hub.status())
        if method == "POST" and rest == ["connect"]:
            body = self._read_json()
            st = hub.connect(host=body.get("host") or "127.0.0.1", port=body.get("port") or 5454,
                             device=body.get("device") or "primary", decode=body.get("decode"),
                             whisper_model=body.get("whisper_model") or "small",
                             language=body.get("language"), use_whisper=body.get("whisper", True),
                             use_llm=bool(body.get("llm")))
            self.db.log(self._actor(), "sdr.connect", "sdr", st["url"],
                        {"device": st["device"], "decode": st["decode"]})
            return self._send_json(st)
        if method == "POST" and rest == ["disconnect"]:
            was = hub.status()
            st = hub.disconnect()
            if was["state"] != "off":
                self.db.log(self._actor(), "sdr.disconnect", "sdr", was["url"],
                            {"samples": was["total_samples"], "decodes": len(was["decodes"])})
            return self._send_json(st)
        if method == "GET" and rest == ["dwell"]:
            after = (query.get("after") or [None])[0]
            iq, meta = hub.dwell(float((query.get("s") or ["0.05"])[0]), int(after) if after else None)
            headers = {"X-NEXA-SDR-State": meta["state"], "X-NEXA-SDR-Total": str(meta["total"])}
            if iq is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return None
            headers |= {"X-NEXA-Sample-Rate": str(int(meta.get("fs_in") or 0)),
                        "X-NEXA-Level-Db": "" if meta.get("level_db") is None else f"{meta['level_db']:.1f}",
                        "X-NEXA-Upsampled": "1" if meta.get("upsampled") else "0"}
            # complex64 is float32 I, Q, I, Q ... (little-endian): the page's interleaved format as it is.
            return self._send_file(iq.view("float32").tobytes(), "application/octet-stream", "dwell.f32", headers)
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

    def _active_single(self):
        """The active retrained single model, if the page can run it: one with
        the 60-class model's outputs. A version retrained before the switch to
        60 classes (8 outputs, no n_classes in its metrics) is never served --
        the page would read its scores against the wrong class names -- so the
        shipped web/models/best_model.onnx is served instead."""
        active = self.db.active_model("single")
        if active and (active.get("metrics") or {}).get("n_classes") == len(CLASSES):
            return active
        return None

    def _serve_active_model(self, path: str) -> bool:
        """/models/best_model.onnx and /models/ensemble_<i>.onnx: the ACTIVE
        approved retrained version when there is one, else the shipped file
        (return False and the static handler serves it). The shipped files in
        web/models/ are never overwritten, so roll back is exact."""
        name = path.rsplit("/", 1)[-1]
        if name == "best_model.onnx":
            active = self._active_single()
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
                keep_bytes: int = ROLLING_IQ_BYTES, intel_dir: Path = DEFAULT_INTEL,
                allow_ips=()) -> ThreadingHTTPServer:
    db = LocalDB(db_path)
    iq_dir = Path(db_path).parent / "iq"
    iq_dir.mkdir(parents=True, exist_ok=True)
    handler = type("NexaHandler", (Handler,), {"db": db, "iq_dir": iq_dir, "lan": lan, "demo": demo, "keep_bytes": keep_bytes,
                                                "allow_ips": frozenset(allow_ips),
                                                "retrain_args": list(retrain_args or []),
                                                "intel_dir": Path(intel_dir)})
    server = ThreadingHTTPServer((host, port), partial(handler, directory=str(WEB)))
    server.nexa_db = db
    return server


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--lan", action="store_true",
                   help="listen on all interfaces so other machines on this network can connect")
    p.add_argument("--allow", nargs="+", metavar="IP", default=[],
                   help="with --lan: only these machines (and this one) may connect, and then no sign-in is "
                        "needed, e.g. --lan --allow 172.22.80.3 (a teammate's laptop)")
    p.add_argument("--no-sign-in", action="store_true",
                   help="never show the login page, also with --lan. Without --allow, ANYONE on the same "
                        "network can then use the system as the analyst: use it only on your own hotspot")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--sign-in", action="store_true",
                   help="require real accounts (login page, passwords, Users page). Without it, "
                        "buttons at the top switch between an operator and an analyst")
    p.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)   # the default now; kept so old commands work
    p.add_argument("--keep-gb", type=float, default=ROLLING_IQ_BYTES / 1024 ** 3,
                   help="rolling window for ROUTINE raw signals (civilian/empty, confident, uncorrected); "
                        "threats, low-confidence, corrected and training captures are always kept")
    p.add_argument("--quick-retrain", action="store_true",
                   help="demo/test: retrain on a small sample (1 epoch, 2,000 replay windows, "
                        "3,000-window exam) so a retrain takes about a minute")
    p.add_argument("--synthetic-exam", action="store_true",
                   help="demo: judge retrained models on the fixed synthetic exam even when "
                        "data/processed exists (the Model page labels it as synthetic)")
    p.add_argument("--intel-dir", type=Path, default=DEFAULT_INTEL,
                   help="results of scripts/comms_intel.py shown on the Comms Intel page")
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
    if a.allow and not a.lan:
        raise SystemExit("--allow only means something with --lan (without it, only this machine can connect).")
    import ipaddress
    for ip in a.allow:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            raise SystemExit(f"--allow {ip}: not an IP address (use the number from ipconfig, e.g. 172.22.80.3)")
    # Without sign-in anyone who can reach the server could act as the analyst,
    # so on the network (--lan) sign-in is on -- unless --allow names the only
    # machines that may connect at all.
    if a.no_sign_in and a.sign_in:
        raise SystemExit("--no-sign-in and --sign-in contradict each other: pick one.")
    demo = a.no_sign_in or (not a.sign_in and (not a.lan or bool(a.allow)))
    if a.no_sign_in and a.lan and not a.allow:
        print("WARNING: --lan --no-sign-in: anyone on this network can use the system as the analyst. "
              "Safer: --allow <teammate IP>.")
    server = make_server(a.db, host, port, a.lan, retrain_args, demo=demo,
                         keep_bytes=int(a.keep_gb * 1024 ** 3), intel_dir=a.intel_dir, allow_ips=a.allow)
    url = f"http://localhost:{port}/index.html"          # with sign-in on, redirects to the login page first
    counts = server.nexa_db.counts()
    print(f"NEXA local server\n  open      {url}\n  database  {a.db}\n"
          f"  stored    {counts.get('analyses', 0)} analyses, {counts.get('audit_log', 0)} audit entries\n"
          f"  network   {('only this machine + ' + ', '.join(a.allow)) if a.allow else 'LAN (other machines can connect)' if a.lan else 'this machine only'}\n"
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
