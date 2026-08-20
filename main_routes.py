"""Dashboards, documents (+PDF preview), knowledge search, announcements,
and the notification API that powers the bell + settings badge."""

from __future__ import annotations

from pathlib import Path

from flask import (
    Blueprint,
    abort,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from src import db
from src.config import TOP_K
from webapp.security import login_required

bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Home dashboards
# ---------------------------------------------------------------------------


@bp.route("/")
@login_required
def home():
    if g.user["role"] == "admin":
        return redirect(url_for("admin.dashboard"))

    from src.vectorstore import stats

    assignments = db.assignments_for_user(g.user["id"])
    pending = [a for a in assignments if a["status"] == "assigned"]
    completed = [a for a in assignments if a["status"] == "completed" and a["max_score"]]
    avg_pct = (
        sum((a["total_score"] or 0) * 100.0 / a["max_score"] for a in completed)
        / len(completed)
        if completed
        else None
    )
    return render_template(
        "dashboard_user.html",
        pending_exams=len(pending),
        avg_pct=avg_pct,
        sessions=len(db.list_sessions(g.user["id"])),
        index=stats(),
        announcements=db.list_announcements(limit=5),
        assignments=assignments[:5],
    )


# ---------------------------------------------------------------------------
# Documents + preview
# ---------------------------------------------------------------------------


@bp.route("/documents")
@login_required
def documents():
    docs = db.list_documents()
    selected_id = request.args.get("doc", type=int)
    selected = next((d for d in docs if d["id"] == selected_id), docs[0] if docs else None)
    return render_template("documents.html", documents=docs, selected=selected)


@bp.route("/documents/<int:doc_id>/file")
@login_required
def document_file(doc_id: int):
    doc = next((d for d in db.list_documents() if d["id"] == doc_id), None)
    if not doc:
        abort(404)
    path = db.resolve_document_path(doc)
    if not path:
        abort(404)
    return send_file(path, mimetype="application/pdf", as_attachment=False,
                     download_name=doc["filename"], conditional=True)


# ---------------------------------------------------------------------------
# Knowledge search
# ---------------------------------------------------------------------------


@bp.route("/search")
@login_required
def search_page():
    from src.vectorstore import stats

    return render_template("search.html", index=stats(), top_k=TOP_K)


@bp.route("/api/search", methods=["POST"])
@login_required
def api_search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    top_k = min(int(payload.get("top_k", TOP_K) or TOP_K), 15)
    if not query:
        return jsonify({"results": []})
    from src.embeddings import embed_query
    from src.vectorstore import search, stats

    if stats()["total_chunks"] == 0:
        return jsonify({"results": [], "empty_index": True})
    try:
        hits = search(embed_query(query), top_k, query_text=query)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Search failed: {exc}"}), 500
    return jsonify({"results": hits})


# ---------------------------------------------------------------------------
# Announcements feed
# ---------------------------------------------------------------------------


@bp.route("/announcements")
@login_required
def announcements():
    return render_template("announcements.html",
                           announcements=db.list_announcements(limit=100))


# ---------------------------------------------------------------------------
# Notifications API (bell + settings dot)
# ---------------------------------------------------------------------------


@bp.route("/api/notifications")
@login_required
def api_notifications():
    return jsonify(
        {
            "unread": db.unread_notification_count(g.user["id"]),
            "items": db.list_notifications(g.user["id"], limit=25),
        }
    )


@bp.route("/api/notifications/read", methods=["POST"])
@login_required
def api_notifications_read():
    payload = request.get_json(silent=True) or {}
    ids = payload.get("ids")
    db.mark_notifications_read(g.user["id"], ids if isinstance(ids, list) else None)
    return jsonify({"ok": True, "unread": db.unread_notification_count(g.user["id"])})
