"""
RAG orchestrator that:
- accepts a user query
- rewrites it with conversation memory
- retrieves top-K chunks via hybrid search
- queries the chat model with a grounded prompt
- (optionally) adds citations and logs history
"""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from rag_core.llm import ChatModel, OpenAIChatModel
from rag_core.history import DefaultHistoryStore, HistoryStore
from rag_core.models import OrchestratorConfig, PromptSet, RAGResult, RetrievedChunk
from rag_core.ports import Retriever
from rag_core.prompts import PromptLoader, PromptProvider
from rag_core.retrievers.hybrid_retriever import HybridRetriever
from rag_core.tasks import (
    CitationTask,
    GroundedAnswerTask,
    QueryRewriter,
)

try:
    from storage.repositories import RetrievalEventRepository
except Exception:
    RetrievalEventRepository = None  # type: ignore


def _is_insufficient_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return "context does not contain enough information" in normalized


class RAGOrchestrator:
    """
    Minimal orchestrator for end-to-end RAG.
    Usage:
        orchestrator = RAGOrchestrator()
        response = orchestrator.run("What is the deadline?", session_id="abc123")
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        chat_model: ChatModel | None = None,
        config: OrchestratorConfig | None = None,
        prompts: PromptSet | None = None,
        prompt_loader: PromptLoader | None = None,
        prompt_provider: PromptProvider | None = None,
        history_store: HistoryStore | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        prompt_source = prompt_provider or prompt_loader or PromptLoader()
        self.prompts = prompts or prompt_source.load()
        self.retriever = retriever or HybridRetriever(limit=self.config.top_k)
        self.history_store = history_store or DefaultHistoryStore()

        active_chat_model = chat_model or OpenAIChatModel(model=self.config.llm_model)
        self._query_rewriter = QueryRewriter(active_chat_model, self.prompts.memory)
        self._answer_task = GroundedAnswerTask(active_chat_model, self.prompts.context)
        self._citation_task = CitationTask(active_chat_model, self.prompts.citation)

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        persist_history: bool = True,
    ) -> RAGResult:
        """
        Execute the full RAG flow and return a descriptive object rather than a loose dict.
        """
        active_config = self.config.with_top_k(top_k)
        session = self.history_store.resolve_session(session_id)
        history = self.history_store.fetch(
            session,
            days=active_config.history_days,
            limit=active_config.history_limit,
        )

        rewritten_query = self._query_rewriter.rewrite(query, history)
        retrieved = self._retrieve(rewritten_query, limit=active_config.top_k)

        answer = self._answer_task.answer(query, rewritten_query, retrieved)

        if retrieved and _is_insufficient_answer(answer):
            deep_k = max(active_config.top_k * 2, active_config.top_k + 5)
            retry_results = self._retrieve(rewritten_query, limit=deep_k)
            if retry_results:
                retry_answer = self._answer_task.answer(query, rewritten_query, retry_results)
                if not _is_insufficient_answer(retry_answer):
                    retrieved = retry_results
                    answer = retry_answer

        cited_answer = answer
        if active_config.add_citations and retrieved:
            cited_answer = self._citation_task.append_citations(answer, retrieved)

        retrieval_event_id = self._log_retrieval(
            rewritten_query,
            retrieved,
            original_query=query,
        )
        if persist_history:
            self._log_chat(session, query, cited_answer, retrieval_event_id)

        return RAGResult(
            answer=cited_answer,
            raw_answer=answer,
            retrieved_chunks=retrieved,
            session_id=session,
            rewritten_query=rewritten_query,
            retrieval_event_id=retrieval_event_id,
        )

    def _log_retrieval(
        self,
        query: str,
        results: List[RetrievedChunk],
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
                    "retrieved_chunk_ids": [r.chunk_id for r in results],
                    "top_k": len(results),
                    "scores": [r.score for r in results],
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
        self.history_store.persist(session_id, messages)

    def _retrieve(self, query: str, limit: int) -> List[RetrievedChunk]:
        """Hybrid retrieval with configurable limit."""
        try:
            results = self.retriever.search(query, limit=limit)
        except Exception:
            return []
        return [RetrievedChunk.from_mapping(r) for r in results]


__all__ = [
    "RAGOrchestrator",
    "OrchestratorConfig",
    "RAGResult",
    "RetrievedChunk",
]
