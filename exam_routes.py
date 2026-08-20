"""Trainee exams: list, fullscreen proctored runner, violations, grading.

Proctoring contract (enforced client-side, recorded server-side):
- The exam opens in fullscreen. Leaving the tab, minimising, or exiting
  fullscreen raises a violation.
- Violations 1 and 2 show an error notification (max two warnings).
- Violation 3 terminates the attempt: the exam auto-submits with whatever
  answers exist, flagged with the violation count.
"""

from __future__ import annotations

from flask import Blueprint, g, jsonify, redirect, render_template, request, session, url_for

from src import db, voice
from src.config import TOP_K
from src.llm import grade_answer, grade_voice_answer, llm_available
from webapp.security import login_required

bp = Blueprint("exams", __name__)

MAX_WARNINGS = 2  # third violation terminates


def _own_assignment(assignment_id: int) -> dict | None:
    assignment = db.get_assignment(assignment_id)
    if not assignment or assignment["user_id"] != g.user["id"]:
        return None
    return assignment


def _question_context(question_text: str) -> str:
    """Grounding passages for grading (best-effort)."""
    if not llm_available():
        return ""
    try:
        from src.embeddings import embed_query
        from src.vectorstore import search, stats

        if stats()["total_chunks"] == 0:
            return ""
        hits = search(embed_query(question_text), TOP_K, query_text=question_text)
        return "\n\n".join(h["text"] for h in hits[:3])
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@bp.route("/exams")
@login_required
def my_exams():
    assignments = db.assignments_for_user(g.user["id"])
    pending = [a for a in assignments if a["status"] == "assigned"]
    done = [a for a in assignments if a["status"] == "completed"]
    details = {}
    for a in done:
        if a.get("submission_id"):
            details[a["assignment_id"]] = db.get_submission_detail(a["submission_id"])
    return render_template("exams.html", pending=pending, done=done, details=details)


@bp.route("/exams/take/<int:assignment_id>")
@login_required
def take_exam(assignment_id: int):
    assignment = _own_assignment(assignment_id)
    if not assignment:
        return redirect(url_for("exams.my_exams"))
    if assignment["status"] != "assigned":
        return redirect(url_for("exams.my_exams"))
    questions = db.get_exam_questions(assignment["exam_id"])
    session[f"viol_{assignment_id}"] = 0
    return render_template(
        "exam_runner.html",
        assignment=assignment,
        questions=questions,
        max_warnings=MAX_WARNINGS,
        is_voice=assignment.get("exam_type") == "voice",
    )


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


@bp.route("/api/exams/<int:assignment_id>/violation", methods=["POST"])
@login_required
def report_violation(assignment_id: int):
    assignment = _own_assignment(assignment_id)
    if not assignment:
        return jsonify({"error": "not found"}), 404
    key = f"viol_{assignment_id}"
    count = int(session.get(key, 0)) + 1
    session[key] = count
    payload = request.get_json(silent=True) or {}
    reason = str(payload.get("reason") or "tab switch")[:80]
    terminate = count > MAX_WARNINGS
    return jsonify(
        {
            "count": count,
            "max_warnings": MAX_WARNINGS,
            "remaining": max(0, MAX_WARNINGS - count),
            "terminate": terminate,
            "message": (
                f"Violation {count}: {reason}. "
                + (
                    "Your exam is being submitted automatically."
                    if terminate
                    else f"{MAX_WARNINGS - count + 1} more and the exam auto-submits."
                    if count == MAX_WARNINGS
                    else "Please stay in fullscreen on this tab."
                )
            ),
        }
    )


# ---------------------------------------------------------------------------
# Written exam submit (+ AI grading)
# ---------------------------------------------------------------------------


@bp.route("/api/exams/<int:assignment_id>/submit", methods=["POST"])
@login_required
def submit_exam(assignment_id: int):
    assignment = _own_assignment(assignment_id)
    if not assignment:
        return jsonify({"error": "not found"}), 404
    if assignment["status"] != "assigned":
        return jsonify({"error": "already submitted"}), 409

    payload = request.get_json(silent=True) or {}
    raw_answers = payload.get("answers") or {}
    terminated = bool(payload.get("terminated"))
    violations = int(session.pop(f"viol_{assignment_id}", 0))

    questions = db.get_exam_questions(assignment["exam_id"])
    answers = [
        {"question_id": q["id"], "answer": str(raw_answers.get(str(q["id"]), "")).strip()}
        for q in questions
    ]
    submission_id = db.submit_exam(
        assignment_id, answers, float(assignment["total_marks"]), violations=violations
    )

    grades = []
    for q in questions:
        answer_text = next(
            (a["answer"] for a in answers if a["question_id"] == q["id"]), ""
        )
        context = _question_context(q["question"]) if answer_text else ""
        result = grade_answer(q["question"], answer_text, float(q["marks"]), context)
        grades.append(
            {"question_id": q["id"], "score": result["score"], "feedback": result["feedback"]}
        )
    db.grade_submission(submission_id, grades)

    submission = db.submission_for_assignment(assignment_id) or {}
    total = submission.get("total_score", 0) or 0
    max_score = submission.get("max_score", 0) or 0
    pct = (total * 100.0 / max_score) if max_score else 0

    db.notify_admins(
        f"Exam submitted: {assignment['title']}",
        f"{g.user['name']} scored {total:.1f}/{max_score:.0f} ({pct:.0f}%)"
        + (f" · {violations} violation(s)" if violations else "")
        + (" · terminated for violations" if terminated else ""),
        kind="exam",
        link="/admin/exams",
    )

    return jsonify(
        {
            "ok": True,
            "submission_id": submission_id,
            "total_score": total,
            "max_score": max_score,
            "pct": round(pct, 1),
            "violations": violations,
            "terminated": terminated,
            "detail": db.get_submission_detail(submission_id),
        }
    )


# ---------------------------------------------------------------------------
# Voice exam: per-question spoken answer → transcribe → grade
# ---------------------------------------------------------------------------


@bp.route("/api/exams/<int:assignment_id>/voice-answer", methods=["POST"])
@login_required
def voice_answer(assignment_id: int):
    assignment = _own_assignment(assignment_id)
    if not assignment:
        return jsonify({"error": "not found"}), 404
    clip = request.files.get("audio")
    question_id = request.form.get("question_id", type=int)
    if clip is None or question_id is None:
        return jsonify({"error": "audio and question_id required"}), 400
    question = next(
        (q for q in db.get_exam_questions(assignment["exam_id"]) if q["id"] == question_id),
        None,
    )
    if not question:
        return jsonify({"error": "unknown question"}), 404

    transcript = voice.transcribe(clip.read(), filename=clip.filename or "answer.webm")
    if not transcript:
        return jsonify({"ok": False, "retry": True,
                        "message": "I couldn't hear that — please record again."})

    grade = grade_voice_answer(
        question["question"], transcript, _question_context(question["question"])
    )
    return jsonify({"ok": True, "transcript": transcript, **grade})


@bp.route("/api/exams/<int:assignment_id>/voice-submit", methods=["POST"])
@login_required
def voice_submit(assignment_id: int):
    """Persist a completed voice exam: one point per question."""
    assignment = _own_assignment(assignment_id)
    if not assignment:
        return jsonify({"error": "not found"}), 404
    if assignment["status"] != "assigned":
        return jsonify({"error": "already submitted"}), 409

    payload = request.get_json(silent=True) or {}
    results = payload.get("results") or []
    violations = int(session.pop(f"viol_{assignment_id}", 0))
    questions = db.get_exam_questions(assignment["exam_id"])
    by_id = {q["id"]: q for q in questions}

    answers, grades = [], []
    for r in results:
        qid = int(r.get("question_id") or 0)
        if qid not in by_id:
            continue
        answers.append({"question_id": qid, "answer": str(r.get("answer") or "")})
        feedback = str(r.get("feedback") or "")
        correct = str(r.get("correct_answer") or "")
        grades.append(
            {
                "question_id": qid,
                "score": float(r.get("score") or 0),
                "feedback": (f"[{r.get('verdict','')}] {feedback}"
                             + (f" Correct answer: {correct}" if correct else "")),
            }
        )

    submission_id = db.submit_exam(
        assignment_id, answers, float(len(questions)), violations=violations
    )
    db.grade_submission(submission_id, grades)

    submission = db.submission_for_assignment(assignment_id) or {}
    total = submission.get("total_score", 0) or 0
    pct = (total * 100.0 / len(questions)) if questions else 0
    db.notify_admins(
        f"Voice exam submitted: {assignment['title']}",
        f"{g.user['name']} scored {pct:.0f}%",
        kind="exam",
        link="/admin/exams",
    )
    return jsonify({"ok": True, "pct": round(pct, 1),
                    "detail": db.get_submission_detail(submission_id)})
