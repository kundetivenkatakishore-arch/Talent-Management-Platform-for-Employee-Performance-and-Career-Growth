"""Request-level auth: session loading and route guards."""

from __future__ import annotations

from functools import wraps

from flask import g, jsonify, redirect, request, session, url_for

from src import db


def load_current_user() -> None:
    """Populate ``g.user`` from the signed session cookie (before every request)."""
    g.user = None
    user_id = session.get("user_id")
    if user_id:
        user = db.get_user(user_id)
        if user:
            user.pop("password_hash", None)
            g.user = user


def _deny(as_json: bool):
    if as_json or request.path.startswith("/api/"):
        return jsonify({"error": "unauthorized"}), 401
    return redirect(url_for("auth.login_page", next=request.path))


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not g.user:
            return _deny(request.is_json)
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not g.user:
            return _deny(request.is_json)
        if g.user["role"] != "admin":
            if request.path.startswith("/api/"):
                return jsonify({"error": "forbidden"}), 403
            return redirect(url_for("main.home"))
        return fn(*args, **kwargs)

    return wrapper


def sign_in(user: dict) -> None:
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = False


def sign_out() -> None:
    if g.user:
        db.update_user_status(g.user["id"], "inactive")
    session.clear()
