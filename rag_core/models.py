from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_LLM_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class OrchestratorConfig:
    """Immutable configuration for the orchestrator runtime."""

    llm_model: str = DEFAULT_LLM_MODEL
    top_k: int = 10
    history_days: int = 10
    add_citations: bool = True
    history_limit: int = 50

    def with_top_k(self, top_k: Optional[int]) -> "OrchestratorConfig":
        if top_k is None or top_k == self.top_k:
            return self
        return OrchestratorConfig(
            llm_model=self.llm_model,
            top_k=top_k,
            history_days=self.history_days,
            add_citations=self.add_citations,
            history_limit=self.history_limit,
        )


@dataclass(frozen=True)
class PromptSet:
    """Container for the prompts used across the RAG flow."""

    context: str
    memory: str
    citation: str


@dataclass(frozen=True)
class RetrievedChunk:
    """Normalized representation of a retrieved chunk with preserved metadata."""

    chunk_id: Optional[str]
    text: str
    score: Optional[float] = None
    source: str = "retriever"
    page_number: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Dict[str, object]) -> "RetrievedChunk":
        return cls(
            chunk_id=str(raw.get("chunk_id")) if raw.get("chunk_id") is not None else None,
            text=str(raw.get("text") or ""),
            score=raw.get("score") if isinstance(raw.get("score"), (int, float)) else None,
            source=str(raw.get("source") or "retriever"),
            page_number=raw.get("page_number"),
            metadata={
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "chunk_id",
                    "text",
                    "score",
                    "source",
                    "page_number",
                }
            },
        )

    def to_mapping(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "page_number": self.page_number,
        }
        payload.update(self.metadata)
        return payload


@dataclass(frozen=True)
class RAGResult:
    """Descriptive return object for the RAG orchestrator."""

    answer: str
    raw_answer: str
    retrieved_chunks: List[RetrievedChunk]
    session_id: str
    rewritten_query: str
    retrieval_event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "retrieved_chunks": [chunk.to_mapping() for chunk in self.retrieved_chunks],
            "session_id": self.session_id,
            "rewritten_query": self.rewritten_query,
            "retrieval_event_id": self.retrieval_event_id,
        }
