"""Deterministic announcement wizard.

Mirrors the exam wizard: the AI Assistant detects the intent to publish an
announcement, this module walks the admin through the fields with real widgets,
and commits only on an explicit confirmation click. No Streamlit imports here —
the chat view renders, this owns state.
"""

from __future__ import annotations

import logging

from src import db

log = logging.getLogger("talentsphere.announcement_wizard")

# Steps, in order.
ASK_TITLE = "ask_title"
ASK_BODY = "ask_body"
ASK_CATEGORY = "ask_category"
CONFIRM = "confirm"
DONE = "done"

# Must match the categories used on the admin Announcements page.
CATEGORIES = ["General", "Training", "Exam", "Policy", "Event"]


def new_state(title: str = "", category: str = "General") -> dict:
    return {
        "step": ASK_TITLE,
        "title": (title or "").strip(),
        "body": "",
        "category": category if category in CATEGORIES else "General",
        "error": "",
    }


def is_active(state) -> bool:
    return bool(state) and state.get("step") != DONE


def set_title(state: dict, title: str) -> tuple[bool, str]:
    if not (title or "").strip():
        return False, "The announcement needs a title."
    state["title"] = title.strip()
    state["step"] = ASK_BODY
    state["error"] = ""
    return True, ""


def set_body(state: dict, body: str) -> tuple[bool, str]:
    if not (body or "").strip():
        return False, "The announcement needs a message."
    state["body"] = body.strip()
    state["step"] = ASK_CATEGORY
    state["error"] = ""
    return True, ""


def set_category(state: dict, category: str) -> None:
    state["category"] = category if category in CATEGORIES else "General"
    state["step"] = CONFIRM
    state["error"] = ""


def summary(state: dict) -> dict:
    return {
        "title": state.get("title") or "-",
        "body": state.get("body") or "-",
        "category": state.get("category") or "General",
    }


def commit(state: dict, admin_email: str) -> tuple[bool, str]:
    """Publish the announcement. The wizard's only write."""
    if not state.get("title") or not state.get("body"):
        return False, "Both a title and a message are required."
    db.create_announcement(state["title"], state["body"], state["category"], admin_email)
    state["step"] = DONE
    log.info(
        "announcement_published_via_chat title=%r category=%s by=%s",
        state["title"], state["category"], admin_email,
    )
    return True, f"Announcement **{state['title']}** published to all trainees."
