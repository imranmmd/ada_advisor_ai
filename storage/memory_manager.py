import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

try:
    from storage.repositories import ChatHistoryRepository
except Exception:
    ChatHistoryRepository = None  # type: ignore

SESSION_ID_FILE = Path("data/session_id.txt")
FILE_HISTORY_DIR = Path("data/chat_history")
_DB_WARNING_PRINTED = False


def resolve_session_id(cli_session_id: Optional[str]) -> str:
    """
    Pick a session id in order of priority:
    1) CLI flag --session-id
    2) RETRIEVAL_SESSION_ID env var
    3) Previously persisted id at data/session_id.txt
    4) Fresh UUID (also persisted for reuse)
    """
    if cli_session_id:
        return cli_session_id

    env_session = os.getenv("RETRIEVAL_SESSION_ID")
    if env_session:
        return env_session

    if SESSION_ID_FILE.exists():
        content = SESSION_ID_FILE.read_text(encoding="utf-8").strip()
        if content:
            return content

    new_session = str(uuid4())
    SESSION_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_ID_FILE.write_text(new_session, encoding="utf-8")
    return new_session


def _warn_db_failure_once(msg: str) -> None:
    global _DB_WARNING_PRINTED
    if _DB_WARNING_PRINTED:
        return
    _DB_WARNING_PRINTED = True
    print(msg)


def _save_history_file(session_id: str, messages: List[Dict[str, str]]) -> None:
    FILE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = FILE_HISTORY_DIR / f"{session_id}.json"
    timestamp = datetime.now(timezone.utc).isoformat()
    messages_with_ts: List[Dict[str, str]] = []
    for msg in messages:
        msg_copy = dict(msg)
        msg_copy.setdefault("created_at", timestamp)
        messages_with_ts.append(msg_copy)
    try:
        existing: List[Dict[str, str]] = []
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        combined = messages_with_ts + existing  # newest-first
        path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _parse_created_at(value: str) -> Optional[datetime]:
    try:
        # Support ISO strings with optional trailing Z
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_history_file(session_id: str, limit: int, days: Optional[int]) -> List[Dict[str, str]]:
    path = FILE_HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            filtered: List[Dict[str, str]] = []
            for msg in data:
                created_raw = msg.get("created_at")
                if not created_raw:
                    filtered.append(msg)
                    continue
                created_dt = _parse_created_at(created_raw)
                if not created_dt or created_dt >= cutoff:
                    filtered.append(msg)
            data = filtered
        return data[:limit]
    except Exception:
        return []


def persist_messages(session_id: str, messages: List[Dict[str, str]]) -> None:
    """
    Persist a batch of messages (newest-first). Tries DB first, falls back to file.
    """
    if ChatHistoryRepository:
        try:
            repo = ChatHistoryRepository()
            for msg in messages:
                repo.add_message(msg)
        except Exception:
            _warn_db_failure_once("⚠️ Could not persist chat history to DB; falling back to local file.")
            _save_history_file(session_id, messages)
            return
    else:
        _save_history_file(session_id, messages)


def fetch_recent_messages(session_id: str, limit: int = 50, days: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Retrieve recent chat history for the session.
    - Tries DB first.
    - Falls back to local file cache if DB unavailable or empty.
    """
    if ChatHistoryRepository:
        try:
            repo = ChatHistoryRepository()
            msgs = repo.recent_messages(session_id, limit=limit, days=days)
            if msgs:
                return msgs
        except Exception:
            _warn_db_failure_once("⚠️ Could not load chat history from DB; falling back to local file.")

    return _load_history_file(session_id, limit, days)


def trim_history(
    history: List[Dict[str, str]], max_chars: int = 4000
) -> List[Dict[str, str]]:
    """
    Trim history so the combined content stays within max_chars (rough token proxy).
    Assumes history is ordered newest-first; preserves order in the returned list.
    """
    trimmed: List[Dict[str, str]] = []
    total = 0
    for msg in history:
        content = msg.get("content") or ""
        projected = total + len(content)
        if projected > max_chars and trimmed:
            break
        total = projected
        trimmed.append(msg)
    return trimmed
