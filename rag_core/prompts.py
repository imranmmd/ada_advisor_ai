from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag_core.models import PromptSet

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FALLBACK_CONTEXT_PROMPT = "Answer the question concisely using only the provided context."
FALLBACK_MEMORY_PROMPT = "Rewrite the user question so it is standalone. Return only the rewritten question."
FALLBACK_CITATION_PROMPT = (
    "Attach citations in [n] format to the answer using the provided metadata. "
    "Return the answer followed by a 'Sources' section."
)


class PromptProvider(Protocol):
    """Interface for supplying prompts, supporting Open/Closed + dependency inversion."""

    def load(self) -> PromptSet:  # pragma: no cover - protocol
        ...


class PromptLoader:
    """Loads prompts from disk with sensible fallbacks."""

    def __init__(
        self,
        prompts_dir: Path = PROMPTS_DIR,
        context_fallback: str = FALLBACK_CONTEXT_PROMPT,
        memory_fallback: str = FALLBACK_MEMORY_PROMPT,
        citation_fallback: str = FALLBACK_CITATION_PROMPT,
    ) -> None:
        self._prompts_dir = prompts_dir
        self._context_fallback = context_fallback
        self._memory_fallback = memory_fallback
        self._citation_fallback = citation_fallback

    def load(self) -> PromptSet:
        return PromptSet(
            context=self._read("context_template.txt", self._context_fallback),
            memory=self._read("memory_template.txt", self._memory_fallback),
            citation=self._read("citation_template.txt", self._citation_fallback),
        )

    def _read(self, filename: str, fallback: str) -> str:
        path = self._prompts_dir / filename
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return fallback


__all__ = [
    "PromptProvider",
    "PromptLoader",
    "PROMPTS_DIR",
    "FALLBACK_CONTEXT_PROMPT",
    "FALLBACK_MEMORY_PROMPT",
    "FALLBACK_CITATION_PROMPT",
]
