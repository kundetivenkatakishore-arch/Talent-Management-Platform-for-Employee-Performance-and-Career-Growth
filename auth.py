"""Authentication core: PBKDF2 password hashing, credential checks, seeding.

Framework-free — the Flask layer (``webapp.security``) owns sessions and
request guards. Passwords are stored as
``pbkdf2_sha256$<iterations>$<salt-hex>$<hash-hex>``; no plaintext password is
ever persisted.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from src import db

_ITERATIONS = 200_000

# Default admin seeded on first run — change the password after first login.
DEFAULT_ADMIN_EMAIL = "admin@talentsphere.com"
DEFAULT_ADMIN_PASSWORD = "Admin@123"


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random salt."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verification of a password against a stored hash."""
    try:
        _, iterations, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


def seed_admin() -> None:
    """Create the default admin account if no admin exists yet."""
    if not db.list_users(role="admin"):
        db.create_user(
            emp_id="ADMIN-001",
            name="Administrator",
            email=DEFAULT_ADMIN_EMAIL,
            password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            role="admin",
        )


def login(email: str, password: str) -> tuple[bool, dict | str]:
    """Validate credentials. Returns (True, user_dict) or (False, error_msg)."""
    user = db.get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return False, "Invalid email or password."
    db.touch_last_login(user["id"])
    return True, user
