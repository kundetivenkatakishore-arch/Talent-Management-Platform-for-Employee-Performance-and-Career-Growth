"""Talent Sphere Elevate — Flask application factory."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

from flask import Flask, g, redirect, render_template, url_for

from src import db
from src.auth import seed_admin

log = logging.getLogger("talentsphere.web")

_SECRET_FILE = Path(".flask_secret")


def _secret_key() -> str:
    """Stable secret key: env var wins; else persist a random one next to the app."""
    env = os.getenv("SECRET_KEY")
    if env:
        return env
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    try:
        _SECRET_FILE.write_text(key, encoding="utf-8")
    except OSError:  # read-only FS — fall back to per-process key
        pass
    return key


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.update(
        SECRET_KEY=_secret_key(),
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,  # uploads (PDFs, audio clips)
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        JSON_SORT_KEYS=False,
    )

    db.init_db()
    seed_admin()

    from webapp.security import load_current_user
    from webapp.auth_routes import bp as auth_bp
    from webapp.main_routes import bp as main_bp
    from webapp.chat_routes import bp as chat_bp
    from webapp.exam_routes import bp as exam_bp
    from webapp.admin_routes import bp as admin_bp

    app.before_request(load_current_user)
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(exam_bp)
    app.register_blueprint(admin_bp)

    @app.context_processor
    def _globals():
        unread = 0
        if getattr(g, "user", None):
            try:
                unread = db.unread_notification_count(g.user["id"])
            except Exception:  # noqa: BLE001 - badge is cosmetic
                unread = 0
        return {"current_user": getattr(g, "user", None), "unread_count": unread}

    @app.errorhandler(404)
    def _not_found(_e):
        if getattr(g, "user", None):
            return render_template("error.html", code=404,
                                   message="That page doesn't exist."), 404
        return redirect(url_for("auth.login_page"))

    @app.errorhandler(500)
    def _server_error(_e):
        return render_template("error.html", code=500,
                               message="Something went wrong on our side."), 500

    return app
