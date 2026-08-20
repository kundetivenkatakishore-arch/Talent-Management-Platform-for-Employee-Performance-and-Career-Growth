"""Deterministic exam-builder wizard.

The AI Assistant decides *that* an admin wants to build an exam; this module
decides *how*. Every step is plain Python driven by real widget input, because
an LLM asked to orchestrate a multi-step form will skip steps, forget
selections and invent questions.

The LLM appears here in exactly one place: :func:`generate`, which drafts
questions from the chosen documents. Everything the trainee is eventually
graded against is reviewed and editable by the admin before it is saved.

This module holds no Streamlit imports so it can be tested headlessly. The
chat view owns rendering; this owns state and the two writes (create + assign).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src import db
from src.ingest import extract_pages
from src.llm import generate_exam_questions

log = logging.getLogger("talentsphere.exam_wizard")

# Wizard steps, in order.
ASK_TITLE = "ask_title"
PICK_DOCS = "pick_docs"
CONFIGURE = "configure"
REVIEW = "review"
ASSIGN = "assign"
CONFIRM = "confirm"
DONE = "done"

# Choices offered in the dropdowns. Marks-per-question drives how deep the AI
# makes each question (see src.llm.generate_exam_questions).
QUESTION_COUNT_OPTIONS = [3, 5, 8, 10, 15]
MARKS_OPTIONS = [5, 10, 15]

# How many pages per document to feed the question generator. Enough for good
# coverage without blowing the model's context on a large manual.
_PAGES_PER_DOC = 8


def new_state(title: str = "") -> dict:
    return {
        "step": ASK_TITLE,
        "title": (title or "").strip(),
        "description": "",
        "doc_ids": [],
        "num_questions": 5,
        "marks_per_question": 10,
        "duration": 30,
        "questions": [],       # list[str], editable before save
        "user_ids": [],
        "due_date": None,      # ISO string or None
        "exam_id": None,       # set once created
        "assigned_count": 0,
        "error": "",
    }


def is_active(state) -> bool:
    return bool(state) and state.get("step") != DONE


# ---------------------------------------------------------------------------
# step 1 — title
# ---------------------------------------------------------------------------
def set_title(state: dict, title: str, description: str = "") -> None:
    state["title"] = (title or "").strip()
    state["description"] = (description or "").strip()
    state["step"] = PICK_DOCS
    state["error"] = ""


# ---------------------------------------------------------------------------
# step 2 — source documents (one or many)
# ---------------------------------------------------------------------------
def available_documents() -> list[dict]:
    """Ingested documents the exam can be generated from."""
    return db.list_documents()


def set_documents(state: dict, doc_ids: list[int]) -> None:
    state["doc_ids"] = [int(d) for d in doc_ids]
    state["step"] = CONFIGURE
    state["error"] = ""


# ---------------------------------------------------------------------------
# step 3 — question count + marks + duration
# ---------------------------------------------------------------------------
def set_config(state: dict, num_questions: int, marks_per_question: int, duration: int) -> None:
    """Store the question/marks/duration choices. Stays on CONFIGURE until
    :func:`generate` succeeds, so a failed AI draft never strands the wizard."""
    state["num_questions"] = int(num_questions)
    state["marks_per_question"] = int(marks_per_question)
    state["duration"] = int(duration)
    state["error"] = ""


def _material_from_docs(doc_ids: list[int]) -> str:
    """Concatenate extractable text from the chosen documents."""
    wanted = set(doc_ids)
    docs = [d for d in db.list_documents() if d["id"] in wanted]
    parts: list[str] = []
    for doc in docs:
        path = db.resolve_document_path(doc)
        if not path:
            log.warning("exam_wizard: file missing on disk for %s", doc.get("filename"))
            continue
        try:
            pages = extract_pages(str(path))
        except Exception:  # noqa: BLE001 - one bad PDF shouldn't kill generation
            log.exception("exam_wizard: failed to read %s", doc.get("filename"))
            continue
        parts.append(" ".join(p["text"] for p in pages[:_PAGES_PER_DOC]))
    return "\n\n".join(parts)


def generate(state: dict) -> tuple[bool, str]:
    """Draft questions from the chosen documents. Returns (ok, message)."""
    if not state.get("doc_ids"):
        return False, "Pick at least one document first."
    material = _material_from_docs(state["doc_ids"])
    if not material.strip():
        return False, "Couldn't read any text from the selected document(s)."

    questions = generate_exam_questions(
        material,
        num_questions=state["num_questions"],
        marks=state["marks_per_question"],
    )
    if not questions:
        return False, "AI question generation failed. Try again, or edit questions by hand."
    state["questions"] = questions
    state["step"] = REVIEW
    state["error"] = ""
    return True, f"Drafted {len(questions)} question(s). Review and edit before assigning."


def set_questions(state: dict, questions: list[str]) -> tuple[bool, str]:
    """Accept the admin's reviewed/edited questions and move to assignment."""
    cleaned = [q.strip() for q in questions if q and q.strip()]
    if not cleaned:
        return False, "An exam needs at least one question."
    state["questions"] = cleaned
    state["step"] = ASSIGN
    state["error"] = ""
    return True, ""


# ---------------------------------------------------------------------------
# step 4 — assign to trainees
# ---------------------------------------------------------------------------
def assignable_users() -> list[dict]:
    return db.list_users(role="user")


def set_assignment(state: dict, user_ids: list[int], due_date: str | None) -> None:
    state["user_ids"] = [int(u) for u in user_ids]
    state["due_date"] = due_date or None
    state["step"] = CONFIRM
    state["error"] = ""


# ---------------------------------------------------------------------------
# step 5 — confirm & commit
# ---------------------------------------------------------------------------
def summary(state: dict) -> dict:
    """Human-readable recap shown before the admin commits."""
    docs = {d["id"]: d["filename"] for d in db.list_documents()}
    users = {u["id"]: u["name"] for u in db.list_users(role="user")}
    marks = state["marks_per_question"]
    return {
        "title": state.get("title") or "-",
        "documents": [docs.get(i, f"#{i}") for i in state.get("doc_ids", [])],
        "num_questions": len(state.get("questions", [])),
        "marks_each": marks,
        "total_marks": len(state.get("questions", [])) * marks,
        "duration": state.get("duration"),
        "recipients": [users.get(i, f"#{i}") for i in state.get("user_ids", [])],
        "due_date": state.get("due_date") or "no due date",
    }


def commit(state: dict, admin_email: str) -> tuple[bool, str]:
    """Create the exam and assign it. The wizard's only writes."""
    if not state.get("title"):
        return False, "The exam needs a title."
    if not state.get("questions"):
        return False, "The exam needs at least one question."

    marks = float(state["marks_per_question"])
    questions = [{"question": q, "marks": marks} for q in state["questions"]]
    exam_id = db.create_exam(
        state["title"],
        state.get("description", ""),
        int(state.get("duration", 30)),
        admin_email,
        questions,
    )
    state["exam_id"] = exam_id

    added = 0
    if state.get("user_ids"):
        added = db.assign_exam(exam_id, state["user_ids"], state.get("due_date"))
    state["assigned_count"] = added
    state["step"] = DONE

    log.info(
        "exam_created_via_chat exam=%s title=%r questions=%s assigned=%s by=%s",
        exam_id, state["title"], len(questions), added, admin_email,
    )
    if added:
        return True, (
            f"Exam **{state['title']}** created with {len(questions)} question(s) "
            f"and assigned to {added} trainee(s)."
        )
    return True, (
        f"Exam **{state['title']}** created with {len(questions)} question(s). "
        "No trainees were selected, so it is saved but unassigned."
    )
