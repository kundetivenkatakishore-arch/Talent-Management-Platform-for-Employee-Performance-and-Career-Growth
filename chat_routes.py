"""AI Assistant: sessions, streamed answers, voice, wizards, mock interview.

The answer pipeline runs the role-scoped function-calling agent
(:mod:`src.agent`) — platform questions use tools, content questions use RAG —
then streams the finished answer to the browser as NDJSON so the UI types it
out like ChatGPT. The final event carries sources, follow-ups, wizard signals
and the (auto-generated) session title.
"""

from __future__ import annotations

import json

from flask import Blueprint, Response, g, jsonify, render_template, request, session
from flask import stream_with_context

from src import agent, announcement_wizard, db, exam_wizard, voice
from src.languages import LANGUAGES
from src.llm import (
    llm_available,
    make_session_title,
    mock_interview_reply,
    sanitize_mermaid,
    suggest_followups,
)
from webapp.security import admin_required, login_required

bp = Blueprint("chat", __name__)

_STREAM_CHUNK_WORDS = 3  # words per NDJSON delta — smooth typing effect


@bp.route("/chat")
@login_required
def chat_page():
    return render_template(
        "chat.html",
        llm_ready=llm_available(),
        languages=LANGUAGES,
        is_admin=g.user["role"] == "admin",
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@bp.route("/api/chat/sessions")
@login_required
def sessions_list():
    return jsonify({"sessions": db.list_sessions(g.user["id"])})


@bp.route("/api/chat/sessions", methods=["POST"])
@login_required
def sessions_create():
    session_id = db.create_session(g.user["id"])
    return jsonify({"id": session_id})


@bp.route("/api/chat/sessions/<int:session_id>", methods=["PATCH", "DELETE"])
@login_required
def sessions_modify(session_id: int):
    owned = {s["id"] for s in db.list_sessions(g.user["id"])}
    if session_id not in owned:
        return jsonify({"error": "not found"}), 404
    if request.method == "DELETE":
        db.delete_session(session_id)
        return jsonify({"ok": True})
    name = str((request.get_json(silent=True) or {}).get("name", "")).strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    db.rename_session(session_id, name[:60])
    return jsonify({"ok": True})


@bp.route("/api/chat/sessions/<int:session_id>/messages")
@login_required
def session_messages(session_id: int):
    owned = {s["id"] for s in db.list_sessions(g.user["id"])}
    if session_id not in owned:
        return jsonify({"error": "not found"}), 404
    return jsonify({"messages": db.get_messages(session_id)})


# ---------------------------------------------------------------------------
# Send + stream
# ---------------------------------------------------------------------------


def _chunk_words(text: str, size: int):
    words = text.split(" ")
    for i in range(0, len(words), size):
        piece = " ".join(words[i : i + size])
        if i + size < len(words):
            piece += " "
        yield piece


@bp.route("/api/chat/send", methods=["POST"])
@login_required
def chat_send():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    session_id = payload.get("session_id")
    if not message:
        return jsonify({"error": "empty message"}), 400

    user = dict(g.user)  # capture for the generator (g is request-scoped)

    owned = {s["id"] for s in db.list_sessions(user["id"])}
    if session_id not in owned:
        session_id = db.create_session(user["id"])

    history = db.get_messages(session_id)
    is_first = not history
    db.add_message(session_id, "user", message)

    def generate():
        yield json.dumps({"type": "start", "session_id": session_id}) + "\n"
        try:
            result = agent.ask(
                user,
                message,
                [{"role": m["role"], "content": m["content"]} for m in history],
            )
        except Exception as exc:  # noqa: BLE001 - never leak a traceback mid-stream
            result = {
                "answer": f"⚠️ The assistant hit an unexpected error: {exc}",
                "signals": {},
                "trace": [],
            }

        answer = sanitize_mermaid(str(result.get("answer") or ""))
        signals = result.get("signals") or {}
        sources = signals.get("sources") or []
        # De-duplicate sources while keeping best-score order.
        seen, unique_sources = set(), []
        for s in sorted(sources, key=lambda x: -float(x.get("score", 0))):
            key = (s.get("source"), s.get("page"))
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
        unique_sources = unique_sources[:6]

        for piece in _chunk_words(answer, _STREAM_CHUNK_WORDS):
            yield json.dumps({"type": "delta", "text": piece}) + "\n"

        db.add_message(session_id, "assistant", answer, unique_sources)

        title = None
        if is_first:
            title = make_session_title(message)
            db.rename_session(session_id, title)

        followups = suggest_followups(message, answer)

        yield json.dumps(
            {
                "type": "done",
                "session_id": session_id,
                "sources": unique_sources,
                "followups": followups,
                "title": title,
                "signals": {
                    "start_exam_wizard": bool(signals.get("start_exam_wizard")),
                    "exam_title": signals.get("exam_title") or "",
                    "start_announcement_wizard": bool(
                        signals.get("start_announcement_wizard")
                    ),
                    "announcement_title": signals.get("announcement_title") or "",
                    "announcement_category": signals.get("announcement_category") or "",
                    "start_mock_interview": bool(signals.get("start_mock_interview")),
                    "interview_topic": signals.get("interview_topic") or "",
                },
            }
        ) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Voice: speech-to-text (Groq Whisper)
# ---------------------------------------------------------------------------


@bp.route("/api/voice/transcribe", methods=["POST"])
@login_required
def voice_transcribe():
    clip = request.files.get("audio")
    if clip is None:
        return jsonify({"error": "no audio"}), 400
    text = voice.transcribe(clip.read(), filename=clip.filename or "speech.webm")
    return jsonify({"text": text})


# ---------------------------------------------------------------------------
# Mock interview (spoken, turn-based)
# ---------------------------------------------------------------------------


@bp.route("/api/interview/turn", methods=["POST"])
@login_required
def interview_turn():
    payload = request.get_json(silent=True) or {}
    history = payload.get("history") or []
    topic = str(payload.get("topic") or "")
    count = int(payload.get("count") or 0)
    language = str(payload.get("language") or "English")
    result = mock_interview_reply(
        history, topic=topic, count=count, max_questions=5, language=language
    )
    return jsonify(result)


# ---------------------------------------------------------------------------
# Exam wizard (admin, driven from chat) — deterministic, server-side state
# ---------------------------------------------------------------------------


def _wizard_options(state: dict) -> dict:
    step = state.get("step")
    options: dict = {}
    if step == exam_wizard.PICK_DOCS:
        options["documents"] = [
            {"id": d["id"], "filename": d["filename"], "pages": d["pages"]}
            for d in exam_wizard.available_documents()
        ]
    if step == exam_wizard.CONFIGURE:
        options["question_counts"] = exam_wizard.QUESTION_COUNT_OPTIONS
        options["marks_options"] = exam_wizard.MARKS_OPTIONS
    if step == exam_wizard.ASSIGN:
        options["users"] = [
            {"id": u["id"], "name": u["name"], "email": u["email"]}
            for u in exam_wizard.assignable_users()
        ]
    if step == exam_wizard.CONFIRM:
        options["summary"] = exam_wizard.summary(state)
    return options


@bp.route("/api/wizard/exam", methods=["POST"])
@admin_required
def wizard_exam():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "")
    state = session.get("exam_wizard")
    message = ""

    if action == "start":
        state = exam_wizard.new_state(str(payload.get("title") or ""))
        if state["title"]:
            state["step"] = exam_wizard.PICK_DOCS
    elif not state:
        return jsonify({"error": "no active wizard"}), 400
    elif action == "title":
        exam_wizard.set_title(state, str(payload.get("title") or ""),
                              str(payload.get("description") or ""))
    elif action == "documents":
        exam_wizard.set_documents(state, payload.get("doc_ids") or [])
    elif action == "configure":
        exam_wizard.set_config(
            state,
            int(payload.get("num_questions") or 5),
            int(payload.get("marks_per_question") or 10),
            int(payload.get("duration") or 30),
        )
        ok, message = exam_wizard.generate(state)
        if not ok:
            state["error"] = message
    elif action == "questions":
        ok, message = exam_wizard.set_questions(state, payload.get("questions") or [])
        if not ok:
            state["error"] = message
    elif action == "assign":
        exam_wizard.set_assignment(
            state, payload.get("user_ids") or [], payload.get("due_date") or None
        )
    elif action == "confirm":
        ok, message = exam_wizard.commit(state, g.user["email"])
        if ok and state.get("user_ids"):
            db.notify_users(
                state["user_ids"],
                f"New exam assigned: {state['title']}",
                f"{len(state.get('questions', []))} questions · "
                f"due {state.get('due_date') or 'no due date'}. Open My Exams to start.",
                kind="exam",
                link="/exams",
            )
    elif action == "cancel":
        session.pop("exam_wizard", None)
        return jsonify({"ok": True, "cancelled": True})
    else:
        return jsonify({"error": f"unknown action {action}"}), 400

    session["exam_wizard"] = state
    if state.get("step") == exam_wizard.DONE:
        session.pop("exam_wizard", None)
    return jsonify({"state": state, "options": _wizard_options(state), "message": message})


# ---------------------------------------------------------------------------
# Announcement wizard (admin, driven from chat)
# ---------------------------------------------------------------------------


@bp.route("/api/wizard/announcement", methods=["POST"])
@admin_required
def wizard_announcement():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "")
    state = session.get("announcement_wizard")
    message = ""

    if action == "start":
        state = announcement_wizard.new_state(
            str(payload.get("title") or ""), str(payload.get("category") or "General")
        )
        if state["title"]:
            state["step"] = announcement_wizard.ASK_BODY
    elif not state:
        return jsonify({"error": "no active wizard"}), 400
    elif action == "title":
        ok, message = announcement_wizard.set_title(state, str(payload.get("title") or ""))
        if not ok:
            state["error"] = message
    elif action == "body":
        ok, message = announcement_wizard.set_body(state, str(payload.get("body") or ""))
        if not ok:
            state["error"] = message
    elif action == "category":
        announcement_wizard.set_category(state, str(payload.get("category") or "General"))
    elif action == "confirm":
        ok, message = announcement_wizard.commit(state, g.user["email"])
        if ok:
            db.notify_all_users(
                f"📣 {state['title']}",
                state["body"][:180],
                kind="announcement",
                link="/announcements",
            )
    elif action == "cancel":
        session.pop("announcement_wizard", None)
        return jsonify({"ok": True, "cancelled": True})
    else:
        return jsonify({"error": f"unknown action {action}"}), 400

    session["announcement_wizard"] = state
    options = {"categories": announcement_wizard.CATEGORIES}
    if state.get("step") == announcement_wizard.CONFIRM:
        options["summary"] = announcement_wizard.summary(state)
    if state.get("step") == announcement_wizard.DONE:
        session.pop("announcement_wizard", None)
    return jsonify({"state": state, "options": options, "message": message})
