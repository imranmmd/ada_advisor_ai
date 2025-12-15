import os
from functools import lru_cache
from typing import Optional

from rag_core import OrchestratorConfig, RAGOrchestrator, RAGResult


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _config_from_env() -> OrchestratorConfig:
    base = OrchestratorConfig()
    return OrchestratorConfig(
        llm_model=os.getenv("TELEGRAM_LLM_MODEL") or base.llm_model,
        top_k=_int_env("TELEGRAM_RAG_TOP_K", base.top_k),
        history_days=_int_env("TELEGRAM_HISTORY_DAYS", base.history_days),
        add_citations=_bool_env("TELEGRAM_ADD_CITATIONS", base.add_citations),
        history_limit=_int_env("TELEGRAM_HISTORY_LIMIT", base.history_limit),
    )


@lru_cache(maxsize=1)
def _get_orchestrator() -> RAGOrchestrator:
    return RAGOrchestrator(config=_config_from_env())


def run_rag(query: str, session_id: Optional[str] = None, top_k: Optional[int] = None) -> RAGResult:
    """
    Execute the RAG pipeline and return the structured result.
    top_k overrides the config for a single request when provided.
    """
    orchestrator = _get_orchestrator()
    return orchestrator.run(query, session_id=session_id, top_k=top_k)


__all__ = ["run_rag"]
