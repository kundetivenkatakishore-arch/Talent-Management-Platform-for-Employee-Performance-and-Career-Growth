"""SQLite persistence layer for Talent Sphere Elevate.

One connection per call (context-managed) keeps things safe across Streamlit
reruns and threads. WAL mode allows concurrent readers while a write is in
flight. All timestamps are stored as ISO-8601 UTC strings.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import DB_PATH, DOCUMENTS_DIR, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a WAL-mode connection with dict-like rows; commit/close on exit."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id        TEXT UNIQUE,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),
    domain        TEXT DEFAULT 'general',
    status        TEXT NOT NULL DEFAULT 'inactive',
    created_at    TEXT NOT NULL,
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content    TEXT NOT NULL,
    sources    TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    file_hash   TEXT NOT NULL UNIQUE,
    pages       INTEGER DEFAULT 0,
    chunks      INTEGER DEFAULT 0,
    size_kb     REAL DEFAULT 0,
    file_path   TEXT,
    uploaded_by TEXT,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exams (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    description      TEXT DEFAULT '',
    duration_minutes INTEGER DEFAULT 30,
    exam_type        TEXT NOT NULL DEFAULT 'write',  -- 'write' | 'voice'
    created_by       TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exam_questions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id  INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    marks    REAL NOT NULL DEFAULT 10,
    ord      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exam_assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id     INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assigned_at TEXT NOT NULL,
    due_date    TEXT,
    status      TEXT NOT NULL DEFAULT 'assigned' CHECK (status IN ('assigned','completed')),
    UNIQUE (exam_id, user_id)
);

CREATE TABLE IF NOT EXISTS exam_submissions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL UNIQUE REFERENCES exam_assignments(id) ON DELETE CASCADE,
    submitted_at  TEXT NOT NULL,
    total_score   REAL DEFAULT 0,
    max_score     REAL DEFAULT 0,
    graded        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exam_answers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES exam_submissions(id) ON DELETE CASCADE,
    question_id   INTEGER NOT NULL REFERENCES exam_questions(id) ON DELETE CASCADE,
    answer        TEXT DEFAULT '',
    score         REAL,
    feedback      TEXT
);

CREATE TABLE IF NOT EXISTS announcements (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    category   TEXT DEFAULT 'General',
    created_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notifications (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    body       TEXT DEFAULT '',
    kind       TEXT DEFAULT 'info',
    link       TEXT DEFAULT '',
    is_read    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, is_read);
"""


def init_db() -> None:
    """Create all tables if they do not exist, then run lightweight migrations."""
    with get_conn() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive, idempotent schema migrations for databases created earlier."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(exams)").fetchall()}
    if "exam_type" not in cols:
        conn.execute(
            "ALTER TABLE exams ADD COLUMN exam_type TEXT NOT NULL DEFAULT 'write'"
        )
    sub_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(exam_submissions)").fetchall()
    }
    if "violations" not in sub_cols:
        conn.execute(
            "ALTER TABLE exam_submissions ADD COLUMN violations INTEGER NOT NULL DEFAULT 0"
        )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def create_user(
    emp_id: str,
    name: str,
    email: str,
    password_hash: str,
    role: str = "user",
    domain: str = "general",
) -> tuple[bool, str]:
    """Insert a user. Returns (ok, message)."""
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (emp_id, name, email, password_hash, role, domain, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (emp_id.strip(), name.strip(), email.strip().lower(), password_hash, role, domain, _now()),
            )
        return True, "User created successfully."
    except sqlite3.IntegrityError:
        return False, "Email or Employee ID already exists."


def get_user_by_email(email: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)
        ).fetchone()
    return dict(row) if row else None


def get_user(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def list_users(role: str | None = "user") -> list[dict]:
    query = "SELECT id, emp_id, name, email, role, domain, status, created_at, last_login FROM users"
    params: tuple = ()
    if role:
        query += " WHERE role = ?"
        params = (role,)
    query += " ORDER BY created_at DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def update_user_status(user_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))


def touch_last_login(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = ?, status = 'active' WHERE id = ?", (_now(), user_id)
        )


def update_user_password(user_id: int, password_hash: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))


def delete_user(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Chat sessions & messages
# ---------------------------------------------------------------------------


def list_sessions(user_id: int) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        ]


def create_session(user_id: int, name: str | None = None) -> int:
    with get_conn() as conn:
        if not name:
            count = conn.execute(
                "SELECT COUNT(*) FROM chat_sessions WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            name = f"New chat {count + 1}"
        cur = conn.execute(
            "INSERT INTO chat_sessions (user_id, name, created_at, updated_at) VALUES (?,?,?,?)",
            (user_id, name, _now(), _now()),
        )
        return int(cur.lastrowid)


def rename_session(session_id: int, name: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE chat_sessions SET name = ? WHERE id = ?", (name.strip(), session_id))


def delete_session(session_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


def get_messages(session_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
    messages = []
    for row in rows:
        msg = dict(row)
        msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else []
        messages.append(msg)
    return messages


def add_message(session_id: int, role: str, content: str, sources: list | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, created_at)"
            " VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(sources or []), _now()),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def document_by_hash(file_hash: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
    return dict(row) if row else None


def add_document(
    filename: str,
    file_hash: str,
    pages: int,
    chunks: int,
    size_kb: float,
    file_path: str,
    uploaded_by: str,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO documents"
            " (filename, file_hash, pages, chunks, size_kb, file_path, uploaded_by, uploaded_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (filename, file_hash, pages, chunks, size_kb, file_path, uploaded_by, _now()),
        )


def list_documents() -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        ]


def clear_documents() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents")


def resolve_document_path(doc: dict) -> Path | None:
    """Best-effort absolute path to a document's file on disk.

    Tolerant of two historical issues: ``file_path`` values stored as paths
    relative to a *different* working directory (the CLI saved uploads relative
    to the project root, while the Flask app resolves relative paths against its
    own package dir), and files that were moved into ``DOCUMENTS_DIR`` by hand.
    Returns ``None`` only when nothing matches.
    """
    root = Path(PROJECT_ROOT)
    candidates: list[Path] = []

    raw = (doc.get("file_path") or "").strip()
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else root / p)

    filename = (doc.get("filename") or "").strip()
    if filename:
        docs_dir = Path(DOCUMENTS_DIR)
        if not docs_dir.is_absolute():
            docs_dir = root / docs_dir
        candidates.append(docs_dir / filename)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Exams
# ---------------------------------------------------------------------------


def create_exam(
    title: str,
    description: str,
    duration_minutes: int,
    created_by: str,
    questions: list[dict],
    exam_type: str = "write",
) -> int:
    """Create an exam with its questions. Each question: {question, marks}.

    ``exam_type`` is 'write' (typed answers, marks) or 'voice' (spoken answers
    graded correct/partial/incorrect, one at a time).
    """
    exam_type = exam_type if exam_type in ("write", "voice") else "write"
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO exams (title, description, duration_minutes, exam_type,"
            " created_by, created_at) VALUES (?,?,?,?,?,?)",
            (title.strip(), description.strip(), duration_minutes, exam_type,
             created_by, _now()),
        )
        exam_id = int(cur.lastrowid)
        for i, q in enumerate(questions):
            conn.execute(
                "INSERT INTO exam_questions (exam_id, question, marks, ord) VALUES (?,?,?,?)",
                (exam_id, q["question"].strip(), float(q.get("marks", 10)), i),
            )
        return exam_id


def list_exams() -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT e.*, COUNT(q.id) AS num_questions, COALESCE(SUM(q.marks),0) AS total_marks
                   FROM exams e LEFT JOIN exam_questions q ON q.exam_id = e.id
                   GROUP BY e.id ORDER BY e.created_at DESC"""
            ).fetchall()
        ]


def get_exam(exam_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
    return dict(row) if row else None


def get_exam_questions(exam_id: int) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM exam_questions WHERE exam_id = ? ORDER BY ord ASC", (exam_id,)
            ).fetchall()
        ]


def delete_exam(exam_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM exams WHERE id = ?", (exam_id,))


def assign_exam(exam_id: int, user_ids: list[int], due_date: str | None = None) -> int:
    """Assign an exam to users; returns number of new assignments (dupes skipped)."""
    added = 0
    with get_conn() as conn:
        for uid in user_ids:
            try:
                conn.execute(
                    "INSERT INTO exam_assignments (exam_id, user_id, assigned_at, due_date)"
                    " VALUES (?,?,?,?)",
                    (exam_id, uid, _now(), due_date),
                )
                added += 1
            except sqlite3.IntegrityError:
                continue  # already assigned
    return added


def assignments_for_user(user_id: int) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT a.id AS assignment_id, a.status, a.assigned_at, a.due_date,
                          e.id AS exam_id, e.title, e.description, e.duration_minutes,
                          e.exam_type,
                          (SELECT COUNT(*) FROM exam_questions q WHERE q.exam_id = e.id) AS num_questions,
                          (SELECT COALESCE(SUM(marks),0) FROM exam_questions q WHERE q.exam_id = e.id) AS total_marks,
                          s.total_score, s.max_score, s.graded, s.id AS submission_id
                   FROM exam_assignments a
                   JOIN exams e ON e.id = a.exam_id
                   LEFT JOIN exam_submissions s ON s.assignment_id = a.id
                   WHERE a.user_id = ?
                   ORDER BY a.assigned_at DESC""",
                (user_id,),
            ).fetchall()
        ]


def assignments_for_exam(exam_id: int) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT a.id AS assignment_id, a.status, a.assigned_at, a.due_date,
                          u.name, u.email, u.emp_id,
                          s.total_score, s.max_score, s.graded, s.submitted_at,
                          s.violations, s.id AS submission_id
                   FROM exam_assignments a
                   JOIN users u ON u.id = a.user_id
                   LEFT JOIN exam_submissions s ON s.assignment_id = a.id
                   WHERE a.exam_id = ?
                   ORDER BY a.assigned_at DESC""",
                (exam_id,),
            ).fetchall()
        ]


def submit_exam(
    assignment_id: int, answers: list[dict], max_score: float, violations: int = 0
) -> int:
    """Store a submission with ungraded answers. answers: [{question_id, answer}]."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO exam_submissions (assignment_id, submitted_at, max_score, violations)"
            " VALUES (?,?,?,?)",
            (assignment_id, _now(), max_score, int(violations)),
        )
        submission_id = int(cur.lastrowid)
        for ans in answers:
            conn.execute(
                "INSERT INTO exam_answers (submission_id, question_id, answer) VALUES (?,?,?)",
                (submission_id, ans["question_id"], ans["answer"]),
            )
        conn.execute(
            "UPDATE exam_assignments SET status = 'completed' WHERE id = ?", (assignment_id,)
        )
        return submission_id


def grade_submission(submission_id: int, grades: list[dict]) -> None:
    """Apply grades: [{question_id, score, feedback}]. Marks submission graded."""
    total = sum(float(g["score"]) for g in grades)
    with get_conn() as conn:
        for g in grades:
            conn.execute(
                "UPDATE exam_answers SET score = ?, feedback = ?"
                " WHERE submission_id = ? AND question_id = ?",
                (float(g["score"]), g.get("feedback", ""), submission_id, g["question_id"]),
            )
        conn.execute(
            "UPDATE exam_submissions SET total_score = ?, graded = 1 WHERE id = ?",
            (total, submission_id),
        )


def get_submission_detail(submission_id: int) -> list[dict]:
    """Return per-question rows: question, marks, answer, score, feedback."""
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT q.question, q.marks, a.answer, a.score, a.feedback
                   FROM exam_answers a JOIN exam_questions q ON q.id = a.question_id
                   WHERE a.submission_id = ?
                   ORDER BY q.ord ASC""",
                (submission_id,),
            ).fetchall()
        ]


# ---------------------------------------------------------------------------
# Announcements
# ---------------------------------------------------------------------------


def create_announcement(title: str, body: str, category: str, created_by: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO announcements (title, body, category, created_by, created_at)"
            " VALUES (?,?,?,?,?)",
            (title.strip(), body.strip(), category, created_by, _now()),
        )


def list_announcements(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM announcements ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        ]


def delete_announcement(announcement_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM announcements WHERE id = ?", (announcement_id,))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def analytics_summary() -> dict:
    """Aggregate counts for the admin dashboard."""
    with get_conn() as conn:
        def one(q: str, params: tuple = ()) -> Any:
            return conn.execute(q, params).fetchone()[0]

        return {
            "users": one("SELECT COUNT(*) FROM users WHERE role='user'"),
            "active_users": one("SELECT COUNT(*) FROM users WHERE role='user' AND status='active'"),
            "documents": one("SELECT COUNT(*) FROM documents"),
            "chunks": one("SELECT COALESCE(SUM(chunks),0) FROM documents"),
            "sessions": one("SELECT COUNT(*) FROM chat_sessions"),
            "messages": one("SELECT COUNT(*) FROM chat_messages"),
            "exams": one("SELECT COUNT(*) FROM exams"),
            "submissions": one("SELECT COUNT(*) FROM exam_submissions"),
            "avg_score_pct": one(
                """SELECT COALESCE(AVG(total_score * 100.0 / NULLIF(max_score,0)),0)
                   FROM exam_submissions WHERE graded = 1"""
            ),
        }


def messages_per_day(days: int = 30) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT substr(created_at,1,10) AS day, COUNT(*) AS count
                   FROM chat_messages
                   GROUP BY day ORDER BY day DESC LIMIT ?""",
                (days,),
            ).fetchall()
        ]


def score_distribution() -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT e.title, u.name, s.total_score, s.max_score,
                          ROUND(s.total_score * 100.0 / NULLIF(s.max_score,0), 1) AS pct
                   FROM exam_submissions s
                   JOIN exam_assignments a ON a.id = s.assignment_id
                   JOIN exams e ON e.id = a.exam_id
                   JOIN users u ON u.id = a.user_id
                   WHERE s.graded = 1"""
            ).fetchall()
        ]


# ---------------------------------------------------------------------------
# Chatbot analytics — learner progress & exam completion
#
# These power the AI Assistant's admin tools ("list learners and their
# progress", "who is pending", "how did everyone do"). They are read-only
# aggregates: no tool the assistant calls can ever mutate the database.
# ---------------------------------------------------------------------------


def users_with_progress() -> list[dict]:
    """Every trainee with their exam progress, one row per user.

    Columns: id, emp_id, name, email, status, domain, assigned, completed,
    pending, avg_pct (average graded score %, rounded, or None if nothing
    graded yet).
    """
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """
                SELECT u.id, u.emp_id, u.name, u.email, u.status, u.domain,
                       COUNT(a.id) AS assigned,
                       COALESCE(SUM(a.status = 'completed'), 0) AS completed,
                       COALESCE(SUM(a.status = 'assigned'), 0) AS pending,
                       ROUND(AVG(
                           CASE WHEN s.graded = 1 AND s.max_score > 0
                                THEN s.total_score * 100.0 / s.max_score END
                       ), 1) AS avg_pct
                FROM users u
                LEFT JOIN exam_assignments a ON a.user_id = u.id
                LEFT JOIN exam_submissions s ON s.assignment_id = a.id
                WHERE u.role = 'user'
                GROUP BY u.id
                ORDER BY u.name COLLATE NOCASE
                """
            ).fetchall()
        ]


def user_progress(user_id: int) -> dict | None:
    """One trainee's headline progress, or None if the user does not exist."""
    with get_conn() as conn:
        row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        stats = conn.execute(
            """
            SELECT COUNT(a.id) AS assigned,
                   COALESCE(SUM(a.status = 'completed'), 0) AS completed,
                   COALESCE(SUM(a.status = 'assigned'), 0) AS pending,
                   ROUND(AVG(
                       CASE WHEN s.graded = 1 AND s.max_score > 0
                            THEN s.total_score * 100.0 / s.max_score END
                   ), 1) AS avg_pct
            FROM exam_assignments a
            LEFT JOIN exam_submissions s ON s.assignment_id = a.id
            WHERE a.user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return {**dict(row), **dict(stats)}


def find_users(query: str, limit: int = 10) -> list[dict]:
    """Fuzzy-match trainees by name, email or employee id."""
    like = f"%{query.strip()}%"
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """SELECT id, emp_id, name, email, status FROM users
                   WHERE role = 'user'
                     AND (name LIKE ? COLLATE NOCASE
                          OR email LIKE ? COLLATE NOCASE
                          OR IFNULL(emp_id,'') LIKE ? COLLATE NOCASE)
                   ORDER BY name COLLATE NOCASE LIMIT ?""",
                (like, like, like, limit),
            ).fetchall()
        ]


def exam_overview() -> list[dict]:
    """Every exam with how many it was assigned to, completed, and average %."""
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                """
                SELECT e.id, e.title,
                       (SELECT COUNT(*) FROM exam_questions q WHERE q.exam_id = e.id) AS num_questions,
                       (SELECT COALESCE(SUM(marks),0) FROM exam_questions q WHERE q.exam_id = e.id) AS total_marks,
                       COUNT(a.id) AS assigned,
                       COALESCE(SUM(a.status = 'completed'), 0) AS completed,
                       ROUND(AVG(
                           CASE WHEN s.graded = 1 AND s.max_score > 0
                                THEN s.total_score * 100.0 / s.max_score END
                       ), 1) AS avg_pct
                FROM exams e
                LEFT JOIN exam_assignments a ON a.exam_id = e.id
                LEFT JOIN exam_submissions s ON s.assignment_id = a.id
                GROUP BY e.id
                ORDER BY e.created_at DESC
                """
            ).fetchall()
        ]


def completion_totals() -> dict:
    """Platform-wide assignment funnel: assigned / completed / pending / graded."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(a.id) AS assigned,
                   COALESCE(SUM(a.status = 'completed'), 0) AS completed,
                   COALESCE(SUM(a.status = 'assigned'), 0) AS pending,
                   (SELECT COUNT(*) FROM exam_submissions WHERE graded = 1) AS graded
            FROM exam_assignments a
            """
        ).fetchone()
    return dict(row)


# ---------------------------------------------------------------------------
# Notifications (bell + settings badge in the top bar)
# ---------------------------------------------------------------------------


def notify_users(user_ids: list[int], title: str, body: str = "",
                 kind: str = "info", link: str = "") -> int:
    """Create one notification per user. Returns how many were created."""
    if not user_ids:
        return 0
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO notifications (user_id, title, body, kind, link, created_at)"
            " VALUES (?,?,?,?,?,?)",
            [(uid, title, body, kind, link, _now()) for uid in user_ids],
        )
    return len(user_ids)


def notify_all_users(title: str, body: str = "", kind: str = "info", link: str = "") -> int:
    """Notify every trainee (role='user')."""
    ids = [u["id"] for u in list_users(role="user")]
    return notify_users(ids, title, body, kind, link)


def notify_admins(title: str, body: str = "", kind: str = "info", link: str = "") -> int:
    """Notify every admin account."""
    ids = [u["id"] for u in list_users(role="admin")]
    return notify_users(ids, title, body, kind, link)


def list_notifications(user_id: int, limit: int = 30) -> list[dict]:
    with get_conn() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM notifications WHERE user_id = ?"
                " ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        ]


def unread_notification_count(user_id: int) -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM notifications WHERE user_id = ? AND is_read = 0",
            (user_id,),
        ).fetchone()[0]


def mark_notifications_read(user_id: int, ids: list[int] | None = None) -> None:
    """Mark the given notifications read (or all of the user's when ids is None)."""
    with get_conn() as conn:
        if ids:
            conn.executemany(
                "UPDATE notifications SET is_read = 1 WHERE user_id = ? AND id = ?",
                [(user_id, nid) for nid in ids],
            )
        else:
            conn.execute(
                "UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user_id,)
            )


def submission_for_assignment(assignment_id: int) -> dict | None:
    """The submission row for an assignment, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM exam_submissions WHERE assignment_id = ?", (assignment_id,)
        ).fetchone()
    return dict(row) if row else None


def get_assignment(assignment_id: int) -> dict | None:
    """One assignment joined with its exam metadata (ownership checks + runner)."""
    with get_conn() as conn:
        row = conn.execute(
            """SELECT a.id AS assignment_id, a.user_id, a.status, a.due_date,
                      e.id AS exam_id, e.title, e.description, e.duration_minutes,
                      e.exam_type,
                      (SELECT COUNT(*) FROM exam_questions q WHERE q.exam_id = e.id)
                          AS num_questions,
                      (SELECT COALESCE(SUM(marks),0) FROM exam_questions q
                          WHERE q.exam_id = e.id) AS total_marks
               FROM exam_assignments a JOIN exams e ON e.id = a.exam_id
               WHERE a.id = ?""",
            (assignment_id,),
        ).fetchone()
    return dict(row) if row else None
