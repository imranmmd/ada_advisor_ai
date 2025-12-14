"""
BM25 retriever: tokenizes the query, scores BM25, returns top-K ranked results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from storage.bm25_loader import (
    BM25_INDEX_DIR,
    BM25_INDEX_NAME,
    BM25_TOP_K,
    load_bm25_index,
    search_bm25,
)
from storage.faiss_loader import hydrate_metadata  # reuse hydrator for metadata dicts


class BM25Retriever:
    """
    Lightweight wrapper around the BM25 utilities to provide a clean search API.
    - Loads BM25 index + chunk ids (lazy load).
    - Tokenizes/scored via BM25.
    - Hydrates results with metadata when available.
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        index_name: str = BM25_INDEX_NAME,
        cached_metadata: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.index_dir = Path(index_dir) if index_dir else BM25_INDEX_DIR
        self.index_name = index_name
        self.cached_metadata = cached_metadata or {}
        self._bm25 = None
        self._chunk_ids: Optional[List[str]] = None
        self._metadata_loaded = False

    def _ensure_loaded(self) -> Tuple[object, List[str], Dict[str, Dict]]:
        if self._bm25 is not None and self._chunk_ids is not None and self._metadata_loaded:
            return self._bm25, self._chunk_ids, self.cached_metadata
        bm25, chunk_ids, metadata = load_bm25_index(self.index_name, index_dir=self.index_dir)
        self._bm25 = bm25
        self._chunk_ids = chunk_ids
        # Prefer provided cached_metadata if present; otherwise use loaded metadata
        if not self.cached_metadata:
            self.cached_metadata = metadata or {}
        self._metadata_loaded = True
        return bm25, chunk_ids, self.cached_metadata

    def search(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict[str, object]]:
        """Tokenize the query, run BM25, and hydrate metadata."""
        bm25, chunk_ids, meta = self._ensure_loaded()
        results = search_bm25(bm25, chunk_ids, query, top_k=top_k)
        return hydrate_metadata(results, meta)


def bm25_search(
    query: str,
    top_k: int = BM25_TOP_K,
    index_dir: Optional[Path] = None,
    index_name: str = BM25_INDEX_NAME,
    cached_metadata: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, object]]:
    """
    Convenience functional API for one-off BM25 searches.
    Loads the index, runs BM25 scoring, hydrates metadata, and returns ranked results.
    """
    retriever = BM25Retriever(index_dir=index_dir, index_name=index_name, cached_metadata=cached_metadata)
    try:
        return retriever.search(query, top_k=top_k)
    except Exception:
        return []
