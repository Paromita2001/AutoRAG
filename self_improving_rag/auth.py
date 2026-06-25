"""
User authentication — PBKDF2-SHA256 password hashing, JSON-backed user store.

users.json format:
{
  "alice": "pbkdf2:sha256:260000$<salt_hex>$<hash_hex>",
  ...
}

The admin account from .env (AUTH_USERNAME / AUTH_PASSWORD) is seeded
automatically on first load if users.json does not exist.
"""

import hashlib
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_USERS_FILE = "users.json"
_ITERATIONS = 260_000


# ── Password hashing ─────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS)
    return f"pbkdf2:sha256:{_ITERATIONS}${salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _, _, rest = stored.split(":", 2)
        iterations_str, salt, stored_hash = rest.split("$")
        iterations = int(iterations_str)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
        return secrets.compare_digest(dk.hex(), stored_hash)
    except Exception:
        return False


# ── User store ────────────────────────────────────────────────────────────────

def _load_users() -> Dict[str, str]:
    if not Path(_USERS_FILE).exists():
        return {}
    try:
        with open(_USERS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to read users.json: %s", exc)
        return {}


def _save_users(users: Dict[str, str]) -> None:
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _seed_admin_from_env(users: Dict[str, str]) -> bool:
    """
    If no users exist yet, create the admin account from .env AUTH_* vars.
    Returns True if the file was written.
    """
    if users:
        return False
    username = os.environ.get("AUTH_USERNAME", "admin")
    password = os.environ.get("AUTH_PASSWORD", "autorag2024")
    users[username] = _hash_password(password)
    _save_users(users)
    logger.info("Seeded initial admin account '%s' from .env", username)
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_store_initialized() -> None:
    """Call once at startup to create users.json with the admin seed if needed."""
    users = _load_users()
    _seed_admin_from_env(users)


def verify_user(username: str, password: str) -> bool:
    """Return True if username exists and password matches."""
    users = _load_users()
    stored = users.get(username)
    if not stored:
        return False
    return _verify_password(password, stored)


def register_user(username: str, password: str) -> Optional[str]:
    """
    Create a new user. Returns None on success, or an error string.
    - username must be 3–32 chars, alphanumeric + underscores
    - password must be at least 8 chars
    - username must not already exist
    """
    username = username.strip()
    if not username:
        return "Username cannot be empty."
    if not (3 <= len(username) <= 32):
        return "Username must be 3–32 characters."
    if not all(c.isalnum() or c == "_" for c in username):
        return "Username can only contain letters, numbers, and underscores."
    if len(password) < 8:
        return "Password must be at least 8 characters."

    users = _load_users()
    if username in users:
        return f"Username '{username}' is already taken."

    users[username] = _hash_password(password)
    _save_users(users)
    logger.info("Registered new user: %s", username)
    return None


def change_password(username: str, old_password: str, new_password: str) -> Optional[str]:
    """Change a user's password. Returns None on success, or an error string."""
    if not verify_user(username, old_password):
        return "Current password is incorrect."
    if len(new_password) < 8:
        return "New password must be at least 8 characters."

    users = _load_users()
    users[username] = _hash_password(new_password)
    _save_users(users)
    logger.info("Password changed for user: %s", username)
    return None


def list_users() -> list:
    return list(_load_users().keys())


def delete_user(username: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    del users[username]
    _save_users(users)
    return True
