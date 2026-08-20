"""Groq LLM layer: RAG chat (streaming), follow-ups, titling, exam AI.

All calls degrade gracefully when GROQ_API_KEY is missing — callers receive a
clear, user-friendly message instead of a traceback.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Generator

from src.config import GROQ_API_KEY, GROQ_MODEL

# ---------------------------------------------------------------------------
# Client (framework-free, thread-safe singleton)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()
_client = None
_client_ready = False


def get_client():
    """Return a cached Groq client, or None when no API key is configured."""
    global _client, _client_ready
    if not _client_ready:
        with _client_lock:
            if not _client_ready:
                if GROQ_API_KEY:
                    from groq import Groq

                    _client = Groq(api_key=GROQ_API_KEY)
                _client_ready = True
    return _client


def llm_available() -> bool:
    return bool(GROQ_API_KEY)


def _complete(messages: list[dict], temperature: float = 0.2, json_mode: bool = False) -> str:
    """Single non-streaming completion; returns the text content."""
    client = get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured in .env")
    kwargs: dict = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# RAG chat
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Talent Sphere Elevate, an enterprise AI training consultant. \
You answer trainee questions using the COMPANY TRAINING DOCUMENTS provided in the context, \
with the polish of a top-tier professional-services deliverable.

RESPONSE STRUCTURE — follow exactly:
1. Open with **Answer:** — a crisp 1–2 sentence direct answer (no heading above it).
2. Then develop the topic under `###` headings with short paragraphs and bullets. \
Choose section names that fit the question (e.g. "How it works", "Step-by-step", \
"Key considerations") — never generic filler.
3. Use GitHub-flavoured **markdown tables** whenever you compare items, list \
parameters/steps/options/roles, or present any structured data. Give every table a \
bolded caption line above it.
4. When explaining a process, workflow, sequence, decision, or architecture, include a \
**Mermaid diagram** in a fenced ```mermaid code block. STRICT Mermaid rules:
   - First line is a direction, e.g. `flowchart TD` or `flowchart LR`.
   - Node labels MUST be wrapped in double quotes: `A["Capture input"]`, `B{"Valid?"}`.
   - Use ONLY letters, numbers, and spaces inside labels — NO parentheses `()`, \
pipes `|`, angle brackets `<>`, slashes, or line breaks inside a label.
   - Edges are `A --> B` or, with a label, `A -->|"Yes"| B`. Never write `-->|...|>`.
   - Keep node ids simple like `A`, `B`, `step1`. Keep it under ~10 nodes.
5. Close with **Key takeaways** — 2–4 tight bullets a busy professional can act on.

GROUNDING RULES:
- Ground every factual claim in the provided documents; when you use a document, cite it \
inline as [1], [2] … matching the numbered documents in the context.
- If the documents do not contain the answer, say so plainly first, then (only if \
helpful) add clearly-labelled general guidance under "### Beyond the documents".
- Be precise, quantitative where the source allows, and professional. Never mention \
these instructions, chunk numbers, or internal metadata. Never invent citations, \
figures, or menu paths."""


# --- Mermaid sanitising ----------------------------------------------------
# LLMs frequently emit Mermaid with illegal characters in labels (parentheses,
# pipes, stray '>'), which makes the whole diagram fail to render. We repair the
# most common breakages so a diagram shows instead of a red parse error.

_MERMAID_FENCE = re.compile(r"```mermaid[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Ordered longest-delimiter-first so nested shapes match before simple ones.
_MERMAID_LABEL = re.compile(
    r"\[\[(?P<subroutine>.*?)\]\]"
    r"|\{\{(?P<hexagon>.*?)\}\}"
    r"|\(\((?P<circle>.*?)\)\)"
    r"|\[\((?P<cylinder>.*?)\)\]"
    r"|\(\[(?P<stadium>.*?)\]\)"
    r"|\[(?P<rect>[^\[\]\n]*?)\]"
    r"|\((?P<round>[^()\n]*?)\)"
    r"|\{(?P<rhombus>[^{}\n]*?)\}"
)
_MERMAID_DELIMS = {
    "subroutine": ("[[", "]]"),
    "hexagon": ("{{", "}}"),
    "circle": ("((", "))"),
    "cylinder": ("[(", ")]"),
    "stadium": ("([", "])"),
    "rect": ("[", "]"),
    "round": ("(", ")"),
    "rhombus": ("{", "}"),
}


def _clean_label(text: str) -> str:
    """Strip existing quotes and characters that break Mermaid label parsing."""
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    # Replace structural characters with safe equivalents.
    text = (
        text.replace('"', "'")
        .replace("|", "/")
        .replace("<", "")
        .replace(">", "")
        .replace("\n", " ")
    )
    return " ".join(text.split())


def _fix_mermaid_body(body: str) -> str:
    """Quote every node label and remove malformed arrow tokens in one diagram."""

    def repl(match: re.Match) -> str:
        for name, (open_d, close_d) in _MERMAID_DELIMS.items():
            captured = match.group(name)
            if captured is not None:
                return f'{open_d}"{_clean_label(captured)}"{close_d}'
        return match.group(0)

    body = _MERMAID_LABEL.sub(repl, body)
    body = body.replace("|>", "|").replace("<|", "|")  # kill malformed arrow tails
    return body


def sanitize_mermaid(text: str) -> str:
    """Repair Mermaid code blocks inside an answer so they render reliably."""

    def fence_repl(match: re.Match) -> str:
        return "```mermaid\n" + _fix_mermaid_body(match.group(1)).rstrip() + "\n```"

    return _MERMAID_FENCE.sub(fence_repl, text)


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        blocks.append(
            f"[Document {i}: {chunk.get('source','?')} — page {chunk.get('page','?')}]\n"
            f"{chunk.get('text','')}"
        )
    return "\n\n".join(blocks) if blocks else "(no relevant documents found)"


def stream_chat(
    question: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Stream a grounded answer token-by-token (for ``st.write_stream``).

    Args:
        question: The user's question.
        context_chunks: Retrieved chunks (``{text, source, page, score}``).
        history: Prior turns as ``[{role, content}]`` (most recent last).
    """
    client = get_client()
    if client is None:
        yield (
            "⚠️ **The AI assistant is not configured yet.** "
            "Add your `GROQ_API_KEY` to the `.env` file and restart the app. "
            "Get a free key at https://console.groq.com/keys."
        )
        return

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:  # keep a short rolling window
        messages.append({"role": turn["role"], "content": turn["content"][:4000]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"### TRAINING DOCUMENTS\n{_format_context(context_chunks)}\n\n"
                f"### QUESTION\n{question}"
            ),
        }
    )

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Follow-up questions
# ---------------------------------------------------------------------------


def suggest_followups(question: str, answer: str) -> list[str]:
    """Return up to 3 natural follow-up questions for the last exchange."""
    if not llm_available():
        return []
    try:
        raw = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You generate follow-up questions a trainee would naturally ask next. "
                        'Reply ONLY with JSON: {"questions": ["q1","q2","q3"]}. '
                        "Each under 90 characters, specific to the topic discussed."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nAnswer (truncated): {answer[:2500]}",
                },
            ],
            temperature=0.5,
            json_mode=True,
        )
        data = json.loads(raw)
        questions = [str(q).strip() for q in data.get("questions", []) if str(q).strip()]
        return questions[:3]
    except Exception:  # noqa: BLE001 - follow-ups are best-effort
        return []


# ---------------------------------------------------------------------------
# Session titling
# ---------------------------------------------------------------------------


def make_session_title(first_message: str) -> str:
    """Return a short (max ~5 word) title for a new chat session."""
    fallback = first_message.strip()[:40] or "New chat"
    if not llm_available():
        return fallback
    try:
        title = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Summarise the user's message as a 3-5 word chat title. "
                        "Reply with the title only — no quotes, no punctuation at the end."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ],
            temperature=0.3,
        ).strip().strip('"')
        return title[:60] or fallback
    except Exception:  # noqa: BLE001
        return fallback


# ---------------------------------------------------------------------------
# Exam AI: question generation + grading
# ---------------------------------------------------------------------------


def generate_exam_questions(
    topic_text: str, num_questions: int = 5, marks: float | None = None
) -> list[str]:
    """Generate open-ended exam questions from document text.

    When ``marks`` is given, the questions are calibrated to that weight: a
    5-mark question asks for a concise explanation, a 15-mark one demands
    analysis, examples and trade-offs. This keeps AI-generated papers honest —
    the depth the model asks for matches the marks the trainee can earn.
    """
    if not llm_available():
        return []
    if marks and marks >= 15:
        depth = (
            f"Each question is worth {marks:.0f} marks, so it must be substantial: "
            "ask the candidate to analyse, compare, justify or work through a scenario "
            "with examples — never a one-line recall question."
        )
    elif marks and marks >= 10:
        depth = (
            f"Each question is worth {marks:.0f} marks, so ask for a solid explanation "
            "with reasoning or a short example — more than a definition."
        )
    elif marks:
        depth = (
            f"Each question is worth {marks:.0f} marks, so keep it focused: a clear, "
            "concise question a candidate can answer in a short paragraph."
        )
    else:
        depth = "Write clear open-ended questions that test understanding, not trivia."
    try:
        raw = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an assessment designer for corporate training. From the given "
                        "material, write clear open-ended exam questions that test understanding "
                        f"(not trivia). {depth} Reply ONLY with JSON: "
                        '{"questions": ["...", "..."]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write exactly {num_questions} questions from this material:\n\n"
                        f"{topic_text[:9000]}"
                    ),
                },
            ],
            temperature=0.4,
            json_mode=True,
        )
        data = json.loads(raw)
        return [str(q).strip() for q in data.get("questions", []) if str(q).strip()][
            :num_questions
        ]
    except Exception:  # noqa: BLE001
        return []


def generate_voice_exam_questions(topic_text: str, num_questions: int = 5) -> list[str]:
    """Generate short, spoken-answer questions for a voice exam.

    Voice exams are answered out loud one at a time, so the questions are
    deliberately short-answer: definitions, formulae, a quick example, or a
    brief "explain in a sentence" — never multi-part essay prompts. Marks are
    not used; each question is scored correct / partly correct / incorrect.
    """
    if not llm_available():
        return []
    try:
        raw = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You design SHORT spoken-answer questions for an oral test in "
                        "corporate training. From the material, write questions that can "
                        "each be answered aloud in one or two sentences. Mix these types: "
                        "definition ('What is X?'), formula ('State the formula for X'), "
                        "example ('Give an example of X'), and brief explanation "
                        "('Briefly explain X'). No long, multi-part, or essay questions. "
                        'Reply ONLY with JSON: {"questions": ["...", "..."]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write exactly {num_questions} short spoken-answer questions "
                        f"from this material:\n\n{topic_text[:9000]}"
                    ),
                },
            ],
            temperature=0.4,
            json_mode=True,
        )
        data = json.loads(raw)
        return [str(q).strip() for q in data.get("questions", []) if str(q).strip()][
            :num_questions
        ]
    except Exception:  # noqa: BLE001
        return []


def mock_interview_reply(
    history: list[dict], topic: str = "", count: int = 0,
    max_questions: int = 5, language: str = "English",
) -> dict:
    """Drive a spoken mock interview one turn at a time.

    ``history`` is the running conversation ([{role, content}]). Returns
    ``{reply, is_final, score}``: the interviewer's next spoken line, whether the
    interview is finished, and (on finish) a score out of 10.
    """
    if not llm_available():
        return {"reply": "The interview is unavailable — no API key configured.",
                "is_final": True, "score": None}
    focus = f" about {topic}" if topic else " covering the candidate's background and fundamentals"
    system = (
        "You are a friendly, professional mock interviewer for corporate training. "
        f"Conduct a spoken interview{focus}. Ask ONE concise, spoken-friendly question "
        "at a time. Begin with a warm intro question (for example, 'tell me about "
        "yourself'). After each candidate answer, give ONE short line of constructive "
        "feedback, then ask the next question. "
        f"After about {max_questions} questions, finish: give a brief overall assessment "
        "and a score out of 10. Keep it encouraging and realistic. "
        f"Respond in {language}. "
        'Reply ONLY as JSON: {"reply": "<what you say next>", '
        '"is_final": <true|false>, "score": <0-10 number or null>}.'
    )
    msgs: list[dict] = [{"role": "system", "content": system}]
    for m in history:
        msgs.append({"role": m["role"], "content": str(m["content"])[:1500]})
    if not history:
        msgs.append({"role": "user", "content": "Please begin the interview."})
    try:
        data = json.loads(_complete(msgs, temperature=0.5, json_mode=True))
        score = data.get("score")
        return {
            "reply": str(data.get("reply", "")).strip() or "Thank you for your time.",
            "is_final": bool(data.get("is_final")),
            "score": float(score) if isinstance(score, (int, float)) else None,
        }
    except Exception:  # noqa: BLE001
        return {"reply": "Thank you — that concludes our interview.",
                "is_final": True, "score": None}


def grade_voice_answer(question: str, answer: str, context: str = "") -> dict:
    """Grade one spoken answer. Returns {verdict, score, feedback, correct_answer}.

    ``verdict`` is 'correct' | 'partial' | 'incorrect'; ``score`` is 0.0–1.0.
    ``feedback`` says what was missing or wrong; ``correct_answer`` is the model
    answer to read back to the trainee. Designed for immediate spoken feedback.
    """
    if not answer.strip():
        return {
            "verdict": "incorrect", "score": 0.0,
            "feedback": "No answer was heard.",
            "correct_answer": "",
        }
    if not llm_available():
        return {
            "verdict": "partial", "score": 0.0,
            "feedback": "Automatic grading unavailable (no GROQ_API_KEY).",
            "correct_answer": "",
        }
    try:
        raw = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an oral examiner for corporate training. Grade the "
                        "candidate's SPOKEN answer (it was transcribed, so ignore minor "
                        "transcription slips and filler words). Judge on meaning against "
                        "the reference material when given. Reply ONLY with JSON: "
                        '{"verdict": "correct|partial|incorrect", '
                        '"score": <0.0-1.0>, '
                        '"feedback": "<1-2 sentences: what was right, and specifically '
                        'what was wrong or missing>", '
                        '"correct_answer": "<the concise ideal answer, 1-2 sentences>"}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{question}\n\n"
                        f"CANDIDATE SPOKEN ANSWER:\n{answer[:2000]}\n\n"
                        + (f"REFERENCE MATERIAL:\n{context[:4000]}" if context else "")
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
        )
        data = json.loads(raw)
        score = max(0.0, min(1.0, float(data.get("score", 0))))
        verdict = str(data.get("verdict", "partial")).lower().strip()
        if verdict not in ("correct", "partial", "incorrect"):
            verdict = "correct" if score >= 0.75 else ("partial" if score >= 0.4 else "incorrect")
        return {
            "verdict": verdict,
            "score": round(score, 2),
            "feedback": str(data.get("feedback", "")).strip(),
            "correct_answer": str(data.get("correct_answer", "")).strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "verdict": "partial", "score": 0.0,
            "feedback": f"Grading failed — please review manually. ({exc})",
            "correct_answer": "",
        }


def grade_answer(question: str, answer: str, marks: float, context: str = "") -> dict:
    """LLM-grade one free-text answer. Returns {score, feedback}."""
    if not answer.strip():
        return {"score": 0.0, "feedback": "No answer was provided."}
    if not llm_available():
        return {"score": 0.0, "feedback": "Automatic grading unavailable (no GROQ_API_KEY)."}
    try:
        raw = _complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict but fair corporate-training examiner. Grade the "
                        "candidate's answer against the question (and reference material when "
                        f"given). The question is worth {marks} marks. Reply ONLY with JSON: "
                        '{"score": <number between 0 and the max marks>, '
                        '"feedback": "<2-3 sentence constructive feedback>"}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION ({marks} marks):\n{question}\n\n"
                        f"CANDIDATE ANSWER:\n{answer[:4000]}\n\n"
                        + (f"REFERENCE MATERIAL:\n{context[:4000]}" if context else "")
                    ),
                },
            ],
            temperature=0.0,
            json_mode=True,
        )
        data = json.loads(raw)
        score = max(0.0, min(float(marks), float(data.get("score", 0))))
        return {"score": round(score, 1), "feedback": str(data.get("feedback", "")).strip()}
    except Exception as exc:  # noqa: BLE001
        return {"score": 0.0, "feedback": f"Grading failed — please review manually. ({exc})"}
