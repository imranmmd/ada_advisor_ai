from __future__ import annotations

from typing import List, Optional

from rag_core.llm import ChatModel
from rag_core.models import RetrievedChunk
from rag_core.history import History, trim_history


def format_history(history: History) -> str:
    return "\n".join(
        f"- {msg.get('role', 'unknown')}: {msg.get('content', '').strip()}"
        for msg in history
        if msg.get("content")
    )


def condense_text(text: str, max_len: int, keep_start: int, keep_end: int) -> str:
    """Keep head and tail slices of long text to preserve key facts."""
    clean = text.strip().replace("\n", " ")
    if len(clean) <= max_len:
        return clean
    if keep_start + keep_end + 3 > max_len:
        budget = max_len - 3
        keep_start = min(keep_start, budget // 2)
        keep_end = budget - keep_start
    head = clean[:keep_start].rstrip()
    tail = clean[-keep_end:].lstrip()
    return f"{head} ... {tail}"


def format_context(context: List[RetrievedChunk]) -> str:
    parts = []
    for idx, chunk in enumerate(context, start=1):
        snippet = condense_text(chunk.text, max_len=800, keep_start=450, keep_end=250)
        page_label = f"page {chunk.page_number}" if chunk.page_number is not None else "page unknown"
        score_label = f"{chunk.score:.4f}" if isinstance(chunk.score, (int, float)) else "n/a"
        parts.append(
            f"[{idx}] id={chunk.chunk_id} source={chunk.source} {page_label} score={score_label}\n{snippet}"
        )
    return "\n\n".join(parts)


class PromptedTask:
    """Base class for chat-based tasks to enable polymorphic prompt execution."""

    def __init__(self, chat_model: ChatModel, system_prompt: str) -> None:
        self._chat_model = chat_model
        self._system_prompt = system_prompt

    def _complete(self, user_prompt: str) -> str:
        return self._chat_model.complete(self._system_prompt, user_prompt)


class QueryRewriter(PromptedTask):
    """Rewrites the user question using available memory."""

    def rewrite(self, query: str, history: History) -> str:
        if not history:
            return query

        trimmed_history = trim_history(history, max_chars=4000)
        formatted_history = format_history(trimmed_history)
        user_prompt = (
            f"USER QUESTION:\n{query}\n\nCONVERSATION MEMORY (newest first):\n"
            f"{formatted_history or '- none'}"
        )

        try:
            rewritten = self._complete(user_prompt)
            return rewritten or query
        except Exception:
            return query


class GroundedAnswerTask(PromptedTask):
    """Produces a grounded answer using provided context snippets."""

    def answer(
        self,
        user_query: str,
        rewritten_query: str,
        context: List[RetrievedChunk],
    ) -> str:
        context_block = format_context(context)
        user_prompt = (
            f"USER QUESTION:\n{user_query}\n"
            f"REWRITTEN FOR RETRIEVAL:\n{rewritten_query}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block or '- none -'}"
        )

        try:
            return self._complete(user_prompt)
        except Exception:
            return "The system could not generate an answer right now."


class CitationTask(PromptedTask):
    """Asks the model to append citations when context is available."""

    def append_citations(self, answer: str, context: List[RetrievedChunk]) -> str:
        if not context:
            return answer

        meta_lines = []
        for idx, chunk in enumerate(context, start=1):
            snippet = condense_text(chunk.text, max_len=400, keep_start=220, keep_end=140)
            page_label = f"page {chunk.page_number}" if chunk.page_number is not None else "page unknown"
            meta_lines.append(
                f"[{idx}] chunk_id={chunk.chunk_id} source={chunk.source} {page_label} text={snippet}"
            )

        user_prompt = f"ANSWER:\n{answer}\n\nRETRIEVED CONTEXT METADATA:\n" + "\n".join(meta_lines)

        try:
            return self._complete(user_prompt)
        except Exception:
            return answer


__all__ = [
    "PromptedTask",
    "QueryRewriter",
    "GroundedAnswerTask",
    "CitationTask",
    "format_history",
    "format_context",
    "condense_text",
]
