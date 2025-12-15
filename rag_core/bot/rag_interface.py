from functools import lru_cache
from typing import Optional

from rag_core import OrchestratorConfig, RAGOrchestrator, RAGResult

# Mirror the simple orchestrator usage requested by the user.
_DEFAULT_CONFIG = OrchestratorConfig(top_k=20)


@lru_cache(maxsize=1)
def _get_orchestrator() -> RAGOrchestrator:
    return RAGOrchestrator(config=_DEFAULT_CONFIG)


def run_rag(query: str, session_id: Optional[str] = None) -> RAGResult:
    """Execute the RAG pipeline and return the structured result."""
    orchestrator = _get_orchestrator()
    return orchestrator.run(query, session_id=session_id)


__all__ = ["run_rag"]
