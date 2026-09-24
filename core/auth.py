"""Authentication module + lightweight data layer (SQLite).

FR-1.3: passwords are never stored in plain text — PBKDF2-HMAC-SHA256 with a per-user random salt.
The prediction log stores NO patient-identifying data (NFR Security): only who ran which tab and
what came out.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone

import streamlit as st

from .config import DATA_DIR, DB_PATH, LOCKOUT_SECONDS, MAX_FAILED_LOGINS

_ITERATIONS = 240_000
_USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")

# Demo accounts created on first run (change / delete them from the Admin panel).
DEMO_USERS = [
    ("admin", "Admin@2026", "admin", "System Admin"),
    ("staff_sara", "Sara@2026", "nurse", "Sara — Ward Nurse"),
    ("staff_omar", "Omar@2026", "radiology", "Omar — Radiology Technician"),
]

ROLE_LABELS = {"admin": "System Admin", "nurse": "Ward / triage nurse", "radiology": "Radiology technician"}


# --------------------------------------------------------------------------- hashing
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


# --------------------------------------------------------------------------- database
def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource(show_spinner=False)
def init_db() -> bool:
    """Create tables once per server process and seed demo users on first run."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users(
                   username TEXT PRIMARY KEY,
                   password_hash TEXT NOT NULL,
                   role TEXT NOT NULL,
                   full_name TEXT NOT NULL,
                   active INTEGER NOT NULL DEFAULT 1,
                   created_at TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS prediction_log(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts TEXT NOT NULL,
                   username TEXT NOT NULL,
                   tab TEXT NOT NULL,
                   result TEXT NOT NULL,
                   score REAL)"""
        )
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            now = _now()
            for username, pw, role, name in DEMO_USERS:
                conn.execute(
                    "INSERT INTO users(username, password_hash, role, full_name, active, created_at) VALUES(?,?,?,?,1,?)",
                    (username, hash_password(pw), role, name, now),
                )
    return True


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def authenticate(username: str, password: str) -> dict | None:
    username = (username or "").strip().lower()
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ? AND active = 1", (username,)).fetchone()
    if row is None:
        # Burn comparable time so response time doesn't reveal whether the username exists.
        verify_password(password or "", hash_password("dummy-password"))
        return None
    if not verify_password(password or "", row["password_hash"]):
        return None
    return {"username": row["username"], "role": row["role"], "name": row["full_name"]}


def create_user(username: str, password: str, role: str, full_name: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()
    if not _USERNAME_RE.match(username):
        return False, "Username must be 3–32 characters: lowercase letters, digits, dot, dash or underscore."
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters."
    if role not in ROLE_LABELS:
        return False, "Unknown role."
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO users(username, password_hash, role, full_name, active, created_at) VALUES(?,?,?,?,1,?)",
                (username, hash_password(password), role, (full_name or username).strip(), _now()),
            )
    except sqlite3.IntegrityError:
        return False, "That username already exists."
    return True, f"User '{username}' created."


def set_user_active(username: str, active: bool) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE users SET active = ? WHERE username = ?", (1 if active else 0, username))


def list_users() -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT username, full_name, role, active, created_at FROM users ORDER BY created_at, username"
        ).fetchall()
    return [dict(r) for r in rows]


def log_prediction(username: str, tab: str, result: str, score: float | None) -> None:
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO prediction_log(ts, username, tab, result, score) VALUES(?,?,?,?,?)",
                (_now(), username, tab, result, None if score is None else float(score)),
            )
    except Exception:
        pass  # logging is best-effort and must never break a prediction


def recent_logs(limit: int = 100) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT ts, username, tab, result, score FROM prediction_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- session (FR-1.2 / 1.4 / 1.5)
def current_user() -> dict | None:
    return st.session_state.get("auth")


def is_authenticated() -> bool:
    return current_user() is not None


def login(username: str, password: str) -> tuple[bool, str]:
    """Returns (ok, message). Throttles repeated failures per browser session."""
    locked_until = st.session_state.get("locked_until", 0.0)
    if time.time() < locked_until:
        wait = int(locked_until - time.time()) + 1
        return False, f"Too many failed attempts. Try again in {wait}s."
    user = authenticate(username, password)
    if user is None:
        fails = st.session_state.get("failed_logins", 0) + 1
        st.session_state["failed_logins"] = fails
        if fails >= MAX_FAILED_LOGINS:
            st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
            st.session_state["failed_logins"] = 0
            return False, f"Too many failed attempts. Locked for {LOCKOUT_SECONDS}s."
        return False, "Wrong username or password."
    st.session_state["failed_logins"] = 0
    st.session_state["auth"] = user
    return True, "ok"


def logout() -> None:
    """Drop the session completely so the next user of this browser tab starts clean."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
