"""Sign-in / sign-out."""

from __future__ import annotations

from flask import Blueprint, g, redirect, render_template, request, url_for

from src.auth import login as check_login
from webapp.security import sign_in, sign_out

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
def login_page():
    if g.user:
        return redirect(url_for("main.home"))

    error = ""
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        if not email or not password:
            error = "Please enter both email and password."
        else:
            ok, result = check_login(email, password)
            if ok:
                sign_in(result)
                target = request.args.get("next") or url_for("main.home")
                return redirect(target)
            error = str(result)
    return render_template("login.html", error=error)


@bp.route("/logout")
def logout():
    sign_out()
    return redirect(url_for("auth.login_page"))
