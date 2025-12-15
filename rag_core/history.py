from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from storage.memory_manager import (
    fetch_recent_messages,
    persist_messages,
    resolve_session_id,
    trim_history as _trim_history,
)

History = List[Dict[str, str]]


class HistoryStore(Protocol):
    """Abstraction over history persistence to support dependency inversion."""

    def resolve_session(self, session_id: Optional[str]) -> str:  # pragma: no cover - protocol
        ...

    def fetch(self, session_id: str, *, days: Optional[int], limit: int) -> History:  # pragma: no cover - protocol
        ...

    def persist(self, session_id: str, messages: History) -> None:  # pragma: no cover - protocol
        ...


class DefaultHistoryStore:
    """Bridges the protocol to the existing storage layer."""

    def resolve_session(self, session_id: Optional[str]) -> str:
        return resolve_session_id(session_id)

    def fetch(self, session_id: str, *, days: Optional[int], limit: int) -> History:
        return fetch_recent_messages(session_id, limit=limit, days=days)

    def persist(self, session_id: str, messages: History) -> None:
        persist_messages(session_id, messages)


def trim_history(history: History, max_chars: int = 4000) -> History:
    """Thin wrapper for history trimming to keep consumers decoupled from storage implementation."""
    return _trim_history(history, max_chars=max_chars)


__all__ = ["History", "HistoryStore", "DefaultHistoryStore", "trim_history"]
