"""Sign-in and roles (src/localdb.py accounts, scripts/serve_local.py gate).

Sign-in is always on. With no accounts yet, only the login page is served
(it offers to create the first admin, on the server machine only); after
that every page, script, model and API call needs a session, and the server
decides who did what.
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

from src.localdb import MAX_FAILED_LOGINS, Actor, LocalDB  # noqa: E402
import serve_local  # noqa: E402

ADMIN = Actor("ava", "admin", "127.0.0.1")


@pytest.fixture
def db(tmp_path):
    return LocalDB(tmp_path / "nexa.db")


@pytest.fixture
def accounts(db):
    db.setup_admin("ava", "admin-pass-1", "127.0.0.1")
    db.create_user("olly", "operator-pass", "operator", ADMIN)
    db.create_user("nina", "analyst-pass", "analyst", ADMIN)
    return db


# --- database ----------------------------------------------------------------

def test_sign_in_is_off_until_the_first_admin(db):
    assert not db.auth_enabled()
    user = db.setup_admin("Ava", "admin-pass-1")
    assert user == {**user, "username": "ava", "role": "admin", "disabled": False}
    assert db.auth_enabled()
    with pytest.raises(PermissionError, match="already set up"):
        db.setup_admin("eve", "another-pass")


def test_passwords_are_never_stored(accounts):
    with sqlite3.connect(accounts.path) as con:
        dump = "\n".join(con.iterdump())
    assert "admin-pass-1" not in dump and "operator-pass" not in dump


def test_login_gives_a_session_for_the_right_password_only(accounts):
    token, user = accounts.login("OLLY", "operator-pass")
    me = accounts.session_actor(token)
    assert (me.name, me.role, user["role"]) == ("olly", "operator", "operator")
    with pytest.raises(PermissionError, match="Wrong username or password"):
        accounts.login("olly", "nope-nope")
    with pytest.raises(PermissionError, match="Wrong username or password"):
        accounts.login("nobody", "whatever1")
    assert accounts.session_actor("forged-token") is None


def test_failed_logins_are_audited_and_lock_the_account(accounts):
    for _ in range(MAX_FAILED_LOGINS):
        with pytest.raises(PermissionError):
            accounts.login("olly", "wrong-guess")
    with pytest.raises(PermissionError, match="Too many"):
        accounts.login("olly", "operator-pass")         # even the right one, while locked
    fails = accounts.audit(limit=50, action_prefix="user.login_failed")
    assert len(fails) == MAX_FAILED_LOGINS + 1
    assert accounts.get_user("olly")["locked"]
    # an admin password reset unlocks it
    accounts.update_user("olly", ADMIN, password="fresh-pass-1")
    assert accounts.login("olly", "fresh-pass-1")[0]
    assert accounts.verify_audit()["ok"]


def test_only_admins_manage_accounts(accounts):
    with pytest.raises(PermissionError):
        accounts.create_user("zed", "zed-pass-123", "admin", Actor("nina", "analyst"))
    with pytest.raises(ValueError, match="reserved"):
        accounts.create_user("operator", "some-pass-1", "operator", ADMIN)
    with pytest.raises(ValueError, match="at least"):
        accounts.create_user("zed", "short", "operator", ADMIN)


def test_the_last_admin_cannot_be_removed_and_nobody_disables_themselves(accounts):
    with pytest.raises(PermissionError, match="last active admin"):
        accounts.update_user("ava", ADMIN, role="analyst")
    with pytest.raises(PermissionError, match="your own account"):
        accounts.update_user("ava", ADMIN, disabled=True)
    accounts.update_user("nina", ADMIN, role="admin")
    assert accounts.update_user("ava", Actor("nina", "admin"), role="analyst")["role"] == "analyst"


def test_disabling_an_account_ends_its_sessions(accounts):
    token, _ = accounts.login("olly", "operator-pass")
    accounts.update_user("olly", ADMIN, disabled=True)
    assert accounts.session_actor(token) is None
    with pytest.raises(PermissionError, match="disabled"):
        accounts.login("olly", "operator-pass")


# --- server --------------------------------------------------------------------

@pytest.fixture
def server(tmp_path):
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv.nexa_db
    srv.shutdown()
    srv.server_close()


def call(url, method="GET", body=None, cookie=None, operator=None):
    h = {"X-NEXA-Client": "1", "Content-Type": "application/json"}
    if cookie:
        h["Cookie"] = cookie
    if operator:
        h["X-NEXA-Operator"] = operator
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers=h)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.headers, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.headers, json.loads(e.read() or b"null")


def sign_in(base, username, password):
    status, headers, body = call(base + "/api/auth/login", "POST", {"username": username, "password": password})
    assert status == 200, body
    cookie = headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    return cookie.split(";")[0]


def record(i):
    return {"id": i, "created_at": "2026-09-28T10:00:00.000Z", "source": "upload", "verdict": "Hostile",
            "classes_detected": ["JAMMING"], "duration_s": 0.05}


def fetch(base, path, cookie=None):
    """GET without following redirects: (status, headers)."""
    import http.client
    from urllib.parse import urlparse
    u = urlparse(base)
    con = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
    con.request("GET", path, headers={"Host": "localhost", **({"Cookie": cookie} if cookie else {})})
    r = con.getresponse()
    r.read()
    con.close()
    return r.status, r.headers


def test_nothing_of_the_system_is_served_before_sign_in(server):
    """No demo mode: even with no accounts yet, the system is closed. Only
    the login page (which offers to create the first admin) is served."""
    base, _ = server
    assert call(base + "/api/auth/me")[2] == {"auth_enabled": False, "user": None, "can_setup": True}
    assert call(base + "/api/analyses", "POST", record("r1"), operator="jessy")[0] == 401
    assert call(base + "/api/analyses")[0] == 401
    status, headers = fetch(base, "/index.html")
    assert status == 302 and headers["Location"] == "/login.html"
    assert fetch(base, "/")[0] == 302
    for path in ("/main.js", "/models/ensemble_0.onnx", "/data/model_card.json"):
        assert fetch(base, path)[0] == 401, path
    assert fetch(base, "/login.html")[0] == 200
    status, headers = fetch(base, "/brand/logo.png")
    assert status == 200 and headers["Content-Type"] == "image/png"


def test_after_sign_in_the_system_is_served(server):
    base, db = server
    db.setup_admin("ava", "admin-pass-1")
    ava = sign_in(base, "ava", "admin-pass-1")
    assert fetch(base, "/index.html", ava)[0] == 200
    assert fetch(base, "/main.js", ava)[0] == 200
    assert fetch(base, "/index.html", "nexa_session=forged")[0] == 302


def test_after_setup_every_call_needs_a_session(server):
    base, _ = server
    status, headers, body = call(base + "/api/auth/setup", "POST", {"username": "ava", "password": "admin-pass-1"})
    assert status == 200 and body["user"] == {"username": "ava", "role": "admin"}
    assert "nexa_session=" in headers["Set-Cookie"]
    assert call(base + "/api/analyses")[0] == 401
    assert call(base + "/api/health")[0] == 200                 # the page must still find the server
    me = call(base + "/api/auth/me")[2]
    assert me["auth_enabled"] and me["user"] is None and not me["can_setup"]
    assert call(base + "/api/auth/setup", "POST", {"username": "eve", "password": "eve-pass-12"})[0] == 409


def test_the_server_not_the_page_names_who_did_it(server):
    base, db = server
    db.setup_admin("ava", "admin-pass-1")
    db.create_user("olly", "operator-pass", "operator", ADMIN)
    olly = sign_in(base, "olly", "operator-pass")
    assert call(base + "/api/analyses", "POST", record("r2"), cookie=olly, operator="ava")[0] == 201
    assert db.get_analysis("r2")["operator"] == "olly"          # the header's "ava" is ignored
    assert db.audit(limit=1)[0]["actor"] == "olly" and db.audit(limit=1)[0]["role"] == "operator"


def test_roles_gate_what_each_person_can_do(server):
    base, db = server
    db.setup_admin("ava", "admin-pass-1")
    db.create_user("olly", "operator-pass", "operator", ADMIN)
    db.create_user("nina", "analyst-pass", "analyst", ADMIN)
    olly, nina = sign_in(base, "olly", "operator-pass"), sign_in(base, "nina", "analyst-pass")
    ava = sign_in(base, "ava", "admin-pass-1")
    call(base + "/api/analyses", "POST", record("r3"), cookie=olly)
    # an operator may not approve, delete, retrain or manage users
    assert call(base + "/api/corrections/x/review", "POST", {"decision": "approve"}, cookie=olly)[0] == 403
    assert call(base + "/api/analyses/r3", "DELETE", cookie=olly)[0] == 403
    status, _, body = call(base + "/api/retrain/start", "POST", {"reason": "because"}, cookie=nina)
    assert status == 403 and "admin role" in body["error"]
    assert call(base + "/api/users", cookie=nina)[0] == 403
    # an analyst may delete; an admin manages users
    assert call(base + "/api/analyses/r3", "DELETE", cookie=nina)[0] == 200
    assert [u["username"] for u in call(base + "/api/users", cookie=ava)[2]] == ["ava", "nina", "olly"]
    assert call(base + "/api/users", "POST", {"username": "sam", "password": "sam-pass-12", "role": "analyst"},
                cookie=ava)[0] == 201


def test_logout_ends_the_session(server):
    base, db = server
    db.setup_admin("ava", "admin-pass-1")
    ava = sign_in(base, "ava", "admin-pass-1")
    assert call(base + "/api/auth/me", cookie=ava)[2]["user"]["username"] == "ava"
    status, headers, _ = call(base + "/api/auth/logout", "POST", {}, cookie=ava)
    assert status == 200 and "Max-Age=0" in headers["Set-Cookie"]
    assert call(base + "/api/analyses", cookie=ava)[0] == 401


# --- --demo: no sign-in, buttons switch person ----------------------------------

@pytest.fixture
def demo_server(tmp_path):
    srv = serve_local.make_server(tmp_path / "nexa.db", "127.0.0.1", 0, demo=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv.nexa_db
    srv.shutdown()
    srv.server_close()


def test_demo_serves_the_system_without_sign_in(demo_server):
    base, db = demo_server
    assert not db.auth_enabled()                         # no accounts were ever created
    assert fetch(base, "/index.html")[0] == 200
    status, headers = fetch(base, "/login.html")
    assert status == 302 and headers["Location"] == "/index.html"
    me = call(base + "/api/auth/me")[2]
    assert me["demo"] and me["user"] == {"username": "admin", "role": "admin"}
    assert {p["role"] for p in me["people"]} == {"operator", "analyst", "admin"}
    assert call(base + "/api/auth/login", "POST", {"username": "x", "password": "y"})[0] == 404


def test_demo_switch_changes_who_did_it_and_keeps_roles_and_four_eyes(demo_server):
    base, db = demo_server
    status, headers, _ = call(base + "/api/auth/demo", "POST", {"username": "operator"})
    assert status == 200
    op_cookie = headers["Set-Cookie"].split(";")[0]
    assert call(base + "/api/analyses", "POST", record("d1"), cookie=op_cookie, operator="admin")[0] == 201
    assert db.get_analysis("d1")["operator"] == "operator"
    # still an operator: may not approve or retrain
    assert call(base + "/api/corrections/x/review", "POST", {"decision": "approve"}, cookie=op_cookie)[0] == 403
    assert call(base + "/api/retrain/start", "POST", {"reason": "because"}, cookie=op_cookie)[0] == 403
    an_cookie = call(base + "/api/auth/demo", "POST", {"username": "analyst"})[1]["Set-Cookie"].split(";")[0]
    assert call(base + "/api/auth/me", cookie=an_cookie)[2]["user"]["role"] == "analyst"
    assert call(base + "/api/auth/demo", "POST", {"username": "mallory"})[0] == 400
    assert db.audit(limit=5, action_prefix="user.demo_switch")[0]["actor"] == "analyst"
