from dataclasses import dataclass
from html import escape
from typing import List

from telegram.constants import ParseMode

from rag_core.models import RAGResult, RetrievedChunk

MAX_MESSAGE_LENGTH = 3900


@dataclass
class FormattedResponse:
    text: str
    parse_mode: str = ParseMode.HTML


def _describe_chunk(chunk: RetrievedChunk, index: int) -> str:
    label_parts = []
    header = chunk.metadata.get("header") or chunk.metadata.get("title")
    if header:
        label_parts.append(str(header))

    doc = chunk.metadata.get("doc_id") or chunk.metadata.get("document_id") or chunk.source
    if doc:
        label_parts.append(str(doc))

    page = chunk.page_number or chunk.metadata.get("page_number")
    if page:
        label_parts.append(f"p.{page}")

    label = " - ".join(label_parts) if label_parts else "Context"

    snippet = chunk.text or str(chunk.metadata.get("text") or "")
    snippet = snippet.strip()
    if len(snippet) > 220:
        snippet = snippet[:220].rstrip() + "..."

    escaped_label = escape(label)
    escaped_snippet = escape(snippet) if snippet else ""
    if escaped_snippet:
        return f"<b>[{index}] {escaped_label}</b>\n{escaped_snippet}"
    return f"<b>[{index}] {escaped_label}</b>"


def _format_sources(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    lines = []
    for idx, chunk in enumerate(chunks[:5], start=1):
        lines.append(_describe_chunk(chunk, idx))
    return "\n".join(lines)


def _split_message(text: str) -> List[str]:
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]
    parts = []
    remaining = text
    while remaining:
        parts.append(remaining[:MAX_MESSAGE_LENGTH])
        remaining = remaining[MAX_MESSAGE_LENGTH:]
    return parts


def format_result(result: RAGResult) -> List[FormattedResponse]:
    """Convert a RAGResult into Telegram-friendly responses."""
    answer_html = escape(result.answer or "")
    sources = _format_sources(result.retrieved_chunks)
    body = answer_html
    if sources:
        body = f"{answer_html}\n\n"
    print(body)
    return [FormattedResponse(text=part) for part in _split_message(body)]


__all__ = ["FormattedResponse", "format_result"]