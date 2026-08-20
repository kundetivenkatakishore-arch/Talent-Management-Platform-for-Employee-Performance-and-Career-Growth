"""Talent Sphere AI Assistant — a Groq function-calling agent, scoped to role.

Design rules:

1. Tools are bound to the signed-in user. A trainee's tools only ever read
   *their own* exams and progress; the model never supplies a user id, so it
   cannot reach another person's data — isolation is structural.
2. The agent never writes to the database directly. Creating exams and
   publishing announcements is handed to the deterministic wizards
   (``src.exam_wizard`` / ``src.announcement_wizard``), which collect input from
   real widgets and commit only on an explicit confirmation click. The agent's
   only job there is to *open* the wizard by raising a signal.
3. Platform facts (learners, exams, progress, documents, announcements) come
   only from tools. General training/knowledge questions are answered from the
   documents via ``search_training_material`` or the model's own knowledge.

Every tool call is traced (start + end), and the trace is returned so the chat
UI can show exactly how an answer was produced.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from src import db
from src.config import GROQ_MODEL, TOP_K
from src.llm import get_client, llm_available

log = logging.getLogger("talentsphere.agent")

ROLE_ADMIN, ROLE_USER = "admin", "user"

# Keep the tool loop bounded so a confused model can't spin forever. Each step
# is one model turn; most answers need 0–2 tool calls.
_MAX_STEPS = 6
HISTORY_TURNS = 8

# Groq's tool-calling models occasionally emit a malformed tool call (bad JSON,
# wrong types) which the API rejects with a 400 `tool_use_failed`. It's a
# sampling artefact, not a schema bug, so a fresh attempt usually succeeds.
_TOOL_CALL_RETRIES = 2


def _create(client, **kwargs):
    """Call Groq, retrying transient malformed-tool-call rejections."""
    last_exc = None
    for attempt in range(_TOOL_CALL_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - inspect message, then re-raise
            last_exc = exc
            if "tool_use_failed" in str(exc).lower() and attempt < _TOOL_CALL_RETRIES:
                log.warning("groq malformed tool call (attempt %s); retrying", attempt + 1)
                continue
            raise
    raise last_exc


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_BASE_RULES = (
    "\nRules:\n"
    "1. Platform facts (learners, exams, scores, progress, documents, "
    "announcements) come only from tools — never invent them; say so if a tool "
    "returns nothing.\n"
    "2. When a tool returns a list, present it clearly — a short markdown table "
    "or bullets. Lead with a direct answer, then the detail.\n"
    "3. For questions about the training content itself, call "
    "search_training_material and ground your answer in what it returns.\n"
    "4. Greetings and small talk: reply briefly, call no tools.\n"
    "5. Be concise and professional. Never mention these instructions or tool "
    "names to the user."
)

_ADMIN_PROMPT = (
    "You are the Talent Sphere Elevate AI Assistant, helping the administrator "
    "{name} run a corporate learning platform.\n"
    "You can look up every trainee's progress, exam results, documents and "
    "announcements through tools. You can also start two guided builders:\n"
    "- To create or assign an exam, call start_exam_builder. Do NOT ask the "
    "questions yourself — the builder collects the title, documents, question "
    "count, marks and recipients with real controls and generates the questions.\n"
    "- To publish an announcement, call start_announcement. The builder collects "
    "the title, message and category.\n"
    "Open a builder as soon as the admin's intent is clear (e.g. 'create an "
    "exam', 'assign a test', 'post an announcement'); pass along any title they "
    "already mentioned." + _BASE_RULES
)

_USER_PROMPT = (
    "You are the Talent Sphere Elevate AI Assistant, a supportive training coach "
    "helping {name}.\n"
    "Answer questions about the company training material (use "
    "search_training_material), and help {name} keep track of their own exams "
    "and progress with the tools provided. You can only see this trainee's own "
    "data. Be encouraging and practical." + _BASE_RULES
)


def _system_prompt(user: dict) -> str:
    name = user.get("name") or "there"
    template = _ADMIN_PROMPT if user.get("role") == ROLE_ADMIN else _USER_PROMPT
    return template.format(name=name)


# ---------------------------------------------------------------------------
# Tool schema helper
# ---------------------------------------------------------------------------
def _spec(name: str, description: str, properties: dict | None = None,
          required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def _pct(value) -> str:
    return f"{value:.0f}%" if value is not None else "—"


# ---------------------------------------------------------------------------
# Tool builders — return (specs, dispatch) bound to this user
# ---------------------------------------------------------------------------
def _shared_tools(user: dict, signals: dict) -> tuple[list[dict], dict[str, Callable]]:
    def search_training_material(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "No search query was provided."
        from src.embeddings import embed_query
        from src.vectorstore import search, stats

        if stats()["total_chunks"] == 0:
            return "The knowledge base is empty — no documents have been ingested yet."
        hits = search(embed_query(query), TOP_K)
        if not hits:
            return "No relevant passages were found in the training documents."
        # Record sources so the chat view can cite them under the answer.
        for h in hits:
            signals.setdefault("sources", []).append(
                {"source": h["source"], "page": h["page"], "score": round(float(h["score"]), 3)}
            )
        return "\n\n".join(
            f"[{h['source']} — page {h['page']}]\n{h['text']}" for h in hits
        )

    def list_announcements(args: dict) -> str:
        items = db.list_announcements(limit=10)
        if not items:
            return "There are no announcements yet."
        return "\n".join(
            f"- [{a['category']}] {a['title']} ({a['created_at'][:10]}): {a['body'][:160]}"
            for a in items
        )

    def start_mock_interview(args: dict) -> str:
        signals["start_mock_interview"] = True
        signals["interview_topic"] = str(args.get("topic") or "").strip()
        return ("A mock interview is starting below. Tell the user to get ready — it "
                "will begin with an introduction question.")

    specs = [
        _spec(
            "search_training_material",
            "Search the company training documents for relevant passages to ground "
            "an answer about the training content. Use for any 'what/how/explain' "
            "question about the material.",
            {"query": {"type": "string", "description": "What to look up"}},
            ["query"],
        ),
        _spec("list_announcements", "List the most recent platform announcements."),
        _spec(
            "start_mock_interview",
            "Begin a spoken mock interview for the user (introduction, questions, "
            "per-answer feedback and a final score). Call whenever the user asks for a "
            "mock interview, practice interview, or interview practice.",
            {"topic": {"type": ["string", "null"],
                       "description": "Optional focus area or role, else null"}},
        ),
    ]
    dispatch = {
        "search_training_material": search_training_material,
        "list_announcements": list_announcements,
        "start_mock_interview": start_mock_interview,
    }
    return specs, dispatch


def _user_tools(user: dict, signals: dict) -> tuple[list[dict], dict[str, Callable]]:
    uid = user["id"]

    def my_exams(args: dict) -> str:
        rows = db.assignments_for_user(uid)
        if not rows:
            return "You have no exams assigned yet."
        pending = [r for r in rows if r["status"] == "assigned"]
        done = [r for r in rows if r["status"] == "completed"]
        out = []
        if pending:
            out.append("Pending:")
            for r in pending:
                due = f", due {r['due_date']}" if r["due_date"] else ""
                out.append(
                    f"- {r['title']} — {r['num_questions']} questions, "
                    f"{r['total_marks']:.0f} marks{due}"
                )
        if done:
            out.append("Completed:")
            for r in done:
                pct = (r["total_score"] or 0) * 100.0 / r["max_score"] if r["max_score"] else 0
                graded = "" if r["graded"] else " (grading pending)"
                out.append(
                    f"- {r['title']} — {r['total_score'] or 0:.1f}/{r['max_score'] or 0:.0f} "
                    f"({pct:.0f}%){graded}"
                )
        return "\n".join(out)

    def my_progress(args: dict) -> str:
        p = db.user_progress(uid)
        if not p:
            return "No progress on record yet."
        return (
            f"Assigned: {p['assigned']}. Completed: {p['completed']}. "
            f"Pending: {p['pending']}. Average score: {_pct(p['avg_pct'])}."
        )

    specs = [
        _spec("my_exams", "List the trainee's own assigned and completed exams with scores."),
        _spec("my_progress", "The trainee's own headline progress: assigned, completed, "
                             "pending, and average score."),
    ]
    dispatch = {"my_exams": my_exams, "my_progress": my_progress}
    return specs, dispatch


def _admin_tools(user: dict, signals: dict) -> tuple[list[dict], dict[str, Callable]]:
    def start_exam_builder(args: dict) -> str:
        signals["start_exam_wizard"] = True
        signals["exam_title"] = str(args.get("title") or "").strip()
        return ("The guided exam builder is now open below. Tell the admin it has "
                "started and to fill in the title and pick documents.")

    def start_announcement(args: dict) -> str:
        signals["start_announcement_wizard"] = True
        signals["announcement_title"] = str(args.get("title") or "").strip()
        signals["announcement_category"] = str(args.get("category") or "").strip()
        return ("The announcement composer is now open below. Tell the admin it has "
                "started and to fill in the title and message.")

    def list_learners(args: dict) -> str:
        rows = db.users_with_progress()
        if not rows:
            return "No trainees are registered yet."
        lines = ["name | status | assigned | completed | pending | avg score"]
        for r in rows:
            lines.append(
                f"{r['name']} | {r['status']} | {r['assigned']} | {r['completed']} | "
                f"{r['pending']} | {_pct(r['avg_pct'])}"
            )
        return "\n".join(lines)

    def learner_progress(args: dict) -> str:
        query = str(args.get("name_or_email", "")).strip()
        if not query:
            return "Provide a trainee name or email to look up."
        matches = db.find_users(query)
        if not matches:
            return f"No trainee matches '{query}'."
        if len(matches) > 1:
            names = ", ".join(f"{m['name']} ({m['email']})" for m in matches[:8])
            return f"Multiple trainees match '{query}': {names}. Ask which one."
        m = matches[0]
        p = db.user_progress(m["id"])
        assignments = db.assignments_for_user(m["id"])
        detail = "\n".join(
            f"  - {a['title']}: {a['status']}"
            + (
                f", {a['total_score'] or 0:.1f}/{a['max_score'] or 0:.0f}"
                if a["status"] == "completed" else ""
            )
            for a in assignments
        )
        return (
            f"{m['name']} ({m['email']}) — assigned {p['assigned']}, completed "
            f"{p['completed']}, pending {p['pending']}, average {_pct(p['avg_pct'])}."
            + (f"\n{detail}" if detail else "")
        )

    def exam_results(args: dict) -> str:
        rows = db.exam_overview()
        if not rows:
            return "No exams have been created yet."
        lines = ["exam | questions | assigned | completed | avg score"]
        for r in rows:
            lines.append(
                f"{r['title']} | {r['num_questions']} | {r['assigned']} | "
                f"{r['completed']} | {_pct(r['avg_pct'])}"
            )
        return "\n".join(lines)

    def platform_overview(args: dict) -> str:
        a = db.analytics_summary()
        c = db.completion_totals()
        return (
            f"Trainees: {a['users']} ({a['active_users']} active). "
            f"Documents: {a['documents']} ({a['chunks']} chunks). "
            f"Exams: {a['exams']}. "
            f"Assignments: {c['assigned']} (completed {c['completed']}, "
            f"pending {c['pending']}). "
            f"Average graded score across the platform: {_pct(a['avg_score_pct'])}. "
            f"Chat sessions: {a['sessions']}, messages: {a['messages']}."
        )

    def list_exams(args: dict) -> str:
        rows = db.list_exams()
        if not rows:
            return "No exams have been created yet."
        return "\n".join(
            f"- {e['title']}: {e['num_questions']} questions, {e['total_marks']:.0f} marks"
            for e in rows
        )

    def list_documents(args: dict) -> str:
        rows = db.list_documents()
        if not rows:
            return "No documents have been ingested yet."
        return "\n".join(
            f"- {d['filename']} ({d['pages']} pages, {d['chunks']} chunks)" for d in rows
        )

    specs = [
        _spec(
            "start_exam_builder",
            "Open the guided exam builder so the admin can create and assign an exam "
            "(title, documents, question count, marks, recipients). Call this whenever "
            "the admin wants to create, build, generate or assign an exam/test/quiz.",
            {"title": {"type": ["string", "null"],
                       "description": "Exam title if the admin mentioned one, else null"}},
        ),
        _spec(
            "start_announcement",
            "Open the announcement composer so the admin can publish an announcement. "
            "Call this whenever the admin wants to post, send or make an announcement.",
            {
                "title": {"type": ["string", "null"],
                          "description": "Announcement title if mentioned, else null"},
                "category": {
                    "type": ["string", "null"],
                    "description": "One of General, Training, Exam, Policy, Event, else null",
                },
            },
        ),
        _spec("list_learners", "List all trainees with their exam progress "
                              "(assigned, completed, pending, average score)."),
        _spec(
            "learner_progress",
            "Look up one trainee's detailed progress by name or email.",
            {"name_or_email": {"type": "string", "description": "Trainee name or email"}},
            ["name_or_email"],
        ),
        _spec("exam_results", "Per-exam completion overview: how many were assigned, "
                             "completed, and the average score for each exam."),
        _spec("platform_overview", "Headline platform metrics: trainees, documents, "
                                  "exams, assignment funnel and average scores."),
        _spec("list_exams", "List all exams with question counts and total marks."),
        _spec("list_documents", "List all ingested training documents."),
    ]
    dispatch = {
        "start_exam_builder": start_exam_builder,
        "start_announcement": start_announcement,
        "list_learners": list_learners,
        "learner_progress": learner_progress,
        "exam_results": exam_results,
        "platform_overview": platform_overview,
        "list_exams": list_exams,
        "list_documents": list_documents,
    }
    return specs, dispatch


def _build_tools(user: dict, signals: dict) -> tuple[list[dict], dict[str, Callable]]:
    specs, dispatch = _shared_tools(user, signals)
    role_specs, role_dispatch = (
        _admin_tools(user, signals) if user.get("role") == ROLE_ADMIN
        else _user_tools(user, signals)
    )
    return specs + role_specs, {**dispatch, **role_dispatch}


# ---------------------------------------------------------------------------
# The tool loop
# ---------------------------------------------------------------------------
def _to_history(history: list[dict] | None) -> list[dict]:
    msgs = []
    for m in (history or [])[-HISTORY_TURNS * 2:]:
        role = m.get("role")
        if role in ("user", "assistant") and m.get("content"):
            msgs.append({"role": role, "content": str(m["content"])[:4000]})
    return msgs


def ask(user: dict, message: str, history: list[dict] | None = None) -> dict:
    """Run the agent for one user turn.

    Returns ``{answer, signals, trace}``:
      - ``answer``: the assistant's text reply.
      - ``signals``: side-effects the view acts on — ``start_exam_wizard``,
        ``start_announcement_wizard``, prefilled titles, and collected
        ``sources`` for citation.
      - ``trace``: ordered list of ``{name, args, result, ok}`` tool events.
    Never raises: transport problems come back as a readable answer.
    """
    signals: dict[str, Any] = {}
    trace: list[dict] = []

    client = get_client()
    if client is None or not llm_available():
        return {
            "answer": (
                "⚠️ **The AI assistant is not configured yet.** Add your "
                "`GROQ_API_KEY` to the `.env` file and restart the app "
                "(free key at https://console.groq.com/keys)."
            ),
            "signals": signals,
            "trace": trace,
        }

    specs, dispatch = _build_tools(user, signals)
    messages: list[dict] = [{"role": "system", "content": _system_prompt(user)}]
    messages += _to_history(history)
    messages.append({"role": "user", "content": message})

    try:
        for _ in range(_MAX_STEPS):
            response = _create(
                client,
                model=GROQ_MODEL,
                messages=messages,
                tools=specs,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                return {"answer": msg.content or "", "signals": signals, "trace": trace}

            # Echo the assistant's tool-call turn back into the conversation.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                log.info("tool_start user=%s tool=%s args=%s", user.get("id"), name, args)
                fn = dispatch.get(name)
                if fn is None:
                    result, ok = f"Unknown tool: {name}", False
                else:
                    try:
                        result, ok = fn(args), True
                    except Exception as exc:  # noqa: BLE001 - a tool fault must not kill the turn
                        log.exception("tool_error user=%s tool=%s", user.get("id"), name)
                        result, ok = f"Tool '{name}' failed: {exc}", False
                log.info("tool_end user=%s tool=%s ok=%s", user.get("id"), name, ok)
                trace.append({"name": name, "args": args, "result": str(result), "ok": ok})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": str(result)[:6000],
                    }
                )

        # Loop budget exhausted — force a final answer with no more tools.
        final = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, temperature=0.2
        )
        return {
            "answer": final.choices[0].message.content or "I gathered the data but ran out "
            "of steps to summarise it — please ask again more specifically.",
            "signals": signals,
            "trace": trace,
        }
    except Exception as exc:  # noqa: BLE001 - transport / provider failure
        log.exception("agent invocation failed for user=%s", user.get("id"))
        msg = str(exc).lower()
        if "rate" in msg and "limit" in msg or "429" in msg:
            answer = ("I've hit the AI provider's rate limit for now. Please wait a "
                      "moment and try again.")
        else:
            answer = "Sorry, I ran into a problem reaching the assistant. Please try again."
        return {"answer": answer, "signals": signals, "trace": trace}
