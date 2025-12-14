"""
Semantic retriever over FAISS: query -> embedding -> FAISS search -> ranked results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from storage.faiss_loader import (
    FAISS_INDEX_DIR,
    FAISS_INDEX_NAME,
    FAISS_TOP_K,
    hydrate_metadata,
    load_faiss_index,
    search as faiss_search,
)


class SemanticRetriever:
    """
    Lightweight wrapper around the FAISS utilities to provide a clean search API.
    - Loads FAISS index + chunk ids (lazy load).
    - Embeds query and runs FAISS search.
    - Hydrates results with metadata when available.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        index_name: str = FAISS_INDEX_NAME,
        cached_metadata: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
        self.index_name = index_name
        self.cached_metadata = cached_metadata or {}
        self._index = None
        self._chunk_ids: Optional[List[str]] = None

    def _ensure_loaded(self) -> Tuple[object, List[str]]:
        if self._index is not None and self._chunk_ids is not None:
            return self._index, self._chunk_ids
        index, chunk_ids = load_faiss_index(self.index_name, index_dir=self.index_dir)
        self._index = index
        self._chunk_ids = chunk_ids
        return index, chunk_ids

    def search(self, query: str, top_k: int = FAISS_TOP_K) -> List[Dict[str, object]]:
        """Embed the query, run FAISS search, and hydrate metadata."""
        index, chunk_ids = self._ensure_loaded()
        results = faiss_search(index, chunk_ids, query, top_k=top_k, cached_metadata=self.cached_metadata)
        return hydrate_metadata(results, self.cached_metadata)


def semantic_search(
    query: str,
    top_k: int = FAISS_TOP_K,
    index_dir: Optional[Path] = None,
    index_name: str = FAISS_INDEX_NAME,
    cached_metadata: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, object]]:
    """
    Convenience functional API for one-off searches without keeping state.
    Loads the index, embeds the query, runs FAISS search, and hydrates metadata.
    """
    retriever = SemanticRetriever(index_dir=index_dir, index_name=index_name, cached_metadata=cached_metadata)
    try:
        return retriever.search(query, top_k=top_k)
    except Exception:
        # Fail silently in one-off usage; callers can handle empty results.
        return []
 