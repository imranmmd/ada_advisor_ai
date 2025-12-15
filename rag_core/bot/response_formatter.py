from dataclasses import dataclass
from typing import List, Optional

from rag_core.models import RAGResult

MAX_MESSAGE_LENGTH = 3900


@dataclass
class FormattedResponse:
    text: str
    parse_mode: Optional[str] = None


def _split_message(text: str) -> List[str]:
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]
    parts: List[str] = []
    remaining = text
    while remaining:
        parts.append(remaining[:MAX_MESSAGE_LENGTH])
        remaining = remaining[MAX_MESSAGE_LENGTH:]
    return parts


def format_result(result: RAGResult) -> List[FormattedResponse]:
    """
    Convert a RAGResult into Telegram-friendly responses following the exact
    output structure requested by the user.
    """
    answer = result.answer or ""
    rewritten = result.rewritten_query or ""
    body = f"Answer:\n{answer}\n\nRewritten query: {rewritten}"
    return [FormattedResponse(text=part) for part in _split_message(body)]


__all__ = ["FormattedResponse", "format_result"]
