from __future__ import annotations

from typing import List, Protocol


class Retriever(Protocol):
    """Port interface for retrieval engines to enable clear UML-friendly dependencies."""

    def search(self, query: str, limit: int | None = None) -> List[dict[str, object]]:  # pragma: no cover - protocol
        ...


__all__ = ["Retriever"]
