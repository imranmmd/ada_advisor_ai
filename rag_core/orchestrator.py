"""
RAG orchestrator that:
- accepts a user query
- rewrites it with conversation memory
- retrieves top-K chunks via hybrid search
- queries the chat model with a grounded prompt
- (optionally) adds citations and logs history
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from config.settings import OPENAI_API_KEY
from rag_core.retrievers.hybrid_retriever import HybridRetriever
from storage.memory_manager import (
    fetch_recent_messages,
    persist_messages,
    resolve_session_id,
    trim_history,
)

try:
    from storage.repositories import RetrievalEventRepository
except Exception:
    RetrievalEventRepository = None  # type: ignore

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CONTEXT_TEMPLATE_PATH = PROMPTS_DIR / "context_template.txt"
MEMORY_TEMPLATE_PATH = PROMPTS_DIR / "memory_template.txt"
FALLBACK_CONTEXT_PROMPT = "Answer the question concisely using only the provided context."
FALLBACK_MEMORY_PROMPT = "Rewrite the user question so it is standalone. Return only the rewritten question."
FALLBACK_CITATION_PROMPT = (
    "Attach citations in [n] format to the answer using the provided metadata. "
    "Return the answer followed by a 'Sources' section."
)

DEFAULT_LLM_MODEL = "gpt-4o-mini"


def _read_prompt(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return fallback


class RAGOrchestrator:
    """
    Minimal orchestrator for end-to-end RAG.
    Usage:
        orchestrator = RAGOrchestrator()
        response = orchestrator.run("What is the deadline?", session_id="abc123")
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        top_k: int = 10,
        history_days: int = 10,
        client: Optional[OpenAI] = None,
        add_citations: bool = True,
    ) -> None:
        self.retriever = retriever or HybridRetriever(limit=top_k)
        self.llm_model = llm_model
        self.top_k = top_k
        self.history_days = history_days
        self.client = client or OpenAI(api_key=OPENAI_API_KEY)
        self.add_citations = add_citations

        self.context_prompt = _read_prompt(CONTEXT_TEMPLATE_PATH, FALLBACK_CONTEXT_PROMPT)
        self.memory_prompt = _read_prompt(MEMORY_TEMPLATE_PATH, FALLBACK_MEMORY_PROMPT)
        self.citation_prompt = _read_prompt(
            PROMPTS_DIR / "citation_template.txt", FALLBACK_CITATION_PROMPT
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        persist_history: bool = True,
    ) -> Dict[str, object]:
        """
        Execute the full RAG flow.
        Returns a dict with the answer, retrieved context, and metadata.
        """
        requested_k = top_k or self.top_k
        session = resolve_session_id(session_id)
        history = fetch_recent_messages(session, days=self.history_days)

        rewritten_query = self._rewrite_with_memory(query, history)
        retrieved = self._retrieve(rewritten_query, top_k=requested_k)

        answer = self._answer(query, rewritten_query, retrieved)

        # If the model claims there is no context but we did retrieve chunks,
        # rerun with a deeper top_k to reduce false negatives.
        if retrieved and self._is_insufficient_answer(answer):
            deep_k = max(requested_k * 2, requested_k + 5)
            retry_results = self._retrieve(rewritten_query, top_k=deep_k)
            if retry_results:
                retry_answer = self._answer(query, rewritten_query, retry_results)
                if not self._is_insufficient_answer(retry_answer):
                    retrieved = retry_results
                    answer = retry_answer

        cited_answer = answer
        if self.add_citations and retrieved:
            cited_answer = self._attach_citations(answer, retrieved) or answer

        retrieval_event_id = self._log_retrieval(rewritten_query, retrieved, original_query=query)
        if persist_history:
            self._log_chat(session, query, cited_answer, retrieval_event_id)

        return {
            "answer": cited_answer,
            "raw_answer": answer,
            "retrieved_chunks": retrieved,
            "session_id": session,
            "rewritten_query": rewritten_query,
            "retrieval_event_id": retrieval_event_id,
        }

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    def _rewrite_with_memory(self, query: str, history: List[Dict[str, str]]) -> str:
        """Use conversation memory to make the query standalone for retrieval."""
        if not history:
            return query

        trimmed_history = trim_history(history, max_chars=4000)
        formatted_history = self._format_history(trimmed_history)
        user_prompt = (
            f"USER QUESTION:\n{query}\n\nCONVERSATION MEMORY (newest first):\n"
            f"{formatted_history or '- none'}"
        )

        try:
            rewritten = self._chat(self.memory_prompt, user_prompt)
            return rewritten or query
        except Exception:
            return query

    def _retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, object]]:
        """Hybrid retrieval with configurable top_k."""
        limit = top_k or self.top_k
        # Update limit on the shared retriever instance for this call.
        self.retriever.limit = limit
        try:
            return self.retriever.search(query)
        except Exception:
            return []

    def _answer(
        self,
        user_query: str,
        rewritten_query: str,
        context: List[Dict[str, object]],
    ) -> str:
        context_block = self._format_context(context)
        user_prompt = (
            f"USER QUESTION:\n{user_query}\n"
            f"REWRITTEN FOR RETRIEVAL:\n{rewritten_query}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block or '- none -'}"
        )

        try:
            return self._chat(self.context_prompt, user_prompt)
        except Exception:
            return "The system could not generate an answer right now."

    def _attach_citations(self, answer: str, context: List[Dict[str, object]]) -> Optional[str]:
        """Ask the model to append citations using the citation template."""
        if not context:
            return answer

        meta_lines = []
        for idx, chunk in enumerate(context, start=1):
            snippet = self._condense_text(chunk.get("text") or "", max_len=400, keep_start=220, keep_end=140)
            page = chunk.get("page_number")
            page_label = f"page {page}" if page is not None else "page unknown"
            meta_lines.append(
                f"[{idx}] chunk_id={chunk.get('chunk_id')} "
                f"source={chunk.get('source', 'retriever')} "
                f"{page_label} "
                f"text={snippet}"
            )

        user_prompt = (
            f"ANSWER:\n{answer}\n\n"
            "RETRIEVED CONTEXT METADATA:\n" + "\n".join(meta_lines)
        )

        try:
            return self._chat(self.citation_prompt, user_prompt)
        except Exception:
            return None

    def _format_context(self, context: List[Dict[str, object]]) -> str:
        parts = []
        for idx, chunk in enumerate(context, start=1):
            snippet = self._condense_text(chunk.get("text") or "", max_len=800, keep_start=450, keep_end=250)
            page = chunk.get("page_number")
            page_label = f"page {page}" if page is not None else "page unknown"
            score = chunk.get("score")
            score_label = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            parts.append(
                f"[{idx}] id={chunk.get('chunk_id')} "
                f"source={chunk.get('source', 'retriever')} "
                f"{page_label} score={score_label}\n{snippet}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"- {msg.get('role', 'unknown')}: {msg.get('content', '').strip()}"
            for msg in history
            if msg.get("content")
        )

    @staticmethod
    def _is_insufficient_answer(answer: str) -> bool:
        """Detect the fallback 'insufficient context' response."""
        normalized = (answer or "").strip().lower()
        return "context does not contain enough information" in normalized

    @staticmethod
    def _condense_text(text: str, max_len: int, keep_start: int, keep_end: int) -> str:
        """
        Preserve both the beginning and the end of long chunks to reduce
        accidentally trimming the key fact. Keeps head and tail slices with an ellipsis.
        """
        clean = text.strip().replace("\n", " ")
        if len(clean) <= max_len:
            return clean
        # Adjust if keep ranges exceed budget
        if keep_start + keep_end + 3 > max_len:
            budget = max_len - 3
            keep_start = min(keep_start, budget // 2)
            keep_end = budget - keep_start
        head = clean[:keep_start].rstrip()
        tail = clean[-keep_end:].lstrip()
        return f"{head} ... {tail}"

    def _log_retrieval(
        self,
        query: str,
        results: List[Dict[str, object]],
        original_query: Optional[str] = None,
    ) -> Optional[str]:
        """Persist retrieval metadata to the DB if available."""
        if not RetrievalEventRepository:
            return None

        try:
            repo = RetrievalEventRepository()
            event_id = str(uuid4())
            repo.log_event(
                {
                    "event_id": event_id,
                    "query_text": query,
                    "query_embedding": None,
                    "retrieved_chunk_ids": [r.get("chunk_id") for r in results],
                    "top_k": len(results),
                    "scores": [r.get("score") for r in results],
                    "original_query": original_query,
                }
            )
            return event_id
        except Exception:
            return None

    def _log_chat(
        self,
        session_id: str,
        query: str,
        answer: str,
        retrieval_event_id: Optional[str],
    ) -> None:
        """Persist the user/assistant turn for the session."""
        messages = [
            {
                "message_id": str(uuid4()),
                "session_id": session_id,
                "role": "user",
                "content": query,
                "retrieval_event_id": retrieval_event_id,
            },
            {
                "message_id": str(uuid4()),
                "session_id": session_id,
                "role": "assistant",
                "content": answer,
                "retrieval_event_id": retrieval_event_id,
            },
        ]
        persist_messages(session_id, messages)


__all__ = ["RAGOrchestrator"]
