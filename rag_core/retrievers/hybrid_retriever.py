"""
Hybrid retrieval with weighted fusion or Reciprocal Rank Fusion (RRF).
Combines semantic (FAISS) and lexical (BM25) results, deduplicates, and returns ranked hits.

To improve recall, we over-fetch from each retriever (prefetch_factor) before
trimming to the requested limit. This helps pull in relevant chunks that sit
just outside the naive top-k cut-off.
"""

from typing import Dict, List, Optional

from .bm25_retriever import BM25Retriever
from .semantic_retriever import SemanticRetriever


def _dedupe_preserve_order(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    deduped = []
    for r in results:
        cid = r.get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(r)
    return deduped


def reciprocal_rank_fusion(
    faiss_results: List[Dict[str, object]],
    bm25_results: List[Dict[str, object]],
    k: float = 60.0,
    limit: int = 10,
) -> List[Dict[str, object]]:
    """
    Apply Reciprocal Rank Fusion to two ranked lists.
    score = sum(1 / (k + rank))
    """
    scores: Dict[str, float] = {}
    meta: Dict[str, Dict[str, object]] = {}

    def add_scores(results: List[Dict[str, object]], source_label: str):
        for rank, r in enumerate(results, start=1):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Keep the first occurrence metadata
            meta.setdefault(cid, dict(r, source=source_label))

    add_scores(faiss_results, "faiss")
    add_scores(bm25_results, "bm25")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = []
    for cid, score in ranked[:limit]:
        r = meta[cid]
        r["score"] = score
        fused.append(r)
    return fused


def weighted_fusion(
    faiss_results: List[Dict[str, object]],
    bm25_results: List[Dict[str, object]],
    w_faiss: float = 0.5,
    w_bm25: float = 0.5,
    limit: int = 10,
) -> List[Dict[str, object]]:
    """
    Linear weighted fusion of FAISS and BM25 scores after min-max normalization.
    Assumes higher scores are better.
    """
    def _norm(results: List[Dict[str, object]]) -> Dict[str, float]:
        scores = [r["score"] for r in results if r.get("score") is not None]
        if not scores:
            return {}
        s_min, s_max = min(scores), max(scores)
        if s_max == s_min:
            return {r["chunk_id"]: 1.0 for r in results}
        return {r["chunk_id"]: (r["score"] - s_min) / (s_max - s_min) for r in results}

    faiss_norm = _norm(faiss_results)
    bm25_norm = _norm(bm25_results)

    combined: Dict[str, float] = {}
    meta: Dict[str, Dict[str, object]] = {}

    for r in faiss_results:
        cid = r["chunk_id"]
        meta.setdefault(cid, dict(r, source="faiss"))
        combined[cid] = combined.get(cid, 0.0) + w_faiss * faiss_norm.get(cid, 0.0)
    for r in bm25_results:
        cid = r["chunk_id"]
        meta.setdefault(cid, dict(r, source="bm25"))
        combined[cid] = combined.get(cid, 0.0) + w_bm25 * bm25_norm.get(cid, 0.0)

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    fused = []
    for cid, score in ranked[:limit]:
        r = meta[cid]
        r["score"] = score
        fused.append(r)
    return fused


class HybridRetriever:
    """
    Combines semantic and BM25 retrievers with fusion + deduplication.
    """

    def __init__(
        self,
        semantic: Optional[SemanticRetriever] = None,
        bm25: Optional[BM25Retriever] = None,
        fusion: str = "rrf",  # "rrf" or "weighted"
        w_faiss: float = 0.6,
        w_bm25: float = 0.4,
        limit: int = 10,
        prefetch_factor: float = 2.0,
    ) -> None:
        self.semantic = semantic or SemanticRetriever()
        self.bm25 = bm25 or BM25Retriever()
        self.fusion = fusion
        self.w_faiss = w_faiss
        self.w_bm25 = w_bm25
        self.limit = limit
        self.prefetch_factor = prefetch_factor

    def search(self, query: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
        # Over-fetch candidates from each retriever to avoid missing near-miss hits.
        search_limit = limit or self.limit
        search_k = max(search_limit, int(search_limit * self.prefetch_factor))

        faiss_results = self.semantic.search(query, top_k=search_k)
        bm25_results = self.bm25.search(query, top_k=search_k)

        if self.fusion == "weighted":
            fused = weighted_fusion(
                faiss_results,
                bm25_results,
                w_faiss=self.w_faiss,
                w_bm25=self.w_bm25,
                limit=search_limit,
            )
        else:
            fused = reciprocal_rank_fusion(faiss_results, bm25_results, limit=search_limit)

        return _dedupe_preserve_order(fused)


def hybrid_search(
    query: str,
    limit: int = 10,
    fusion: str = "rrf",
    w_faiss: float = 0.6,
    w_bm25: float = 0.4,
    prefetch_factor: float = 2.0,
) -> List[Dict[str, object]]:
    """Convenience functional API for hybrid retrieval."""
    retriever = HybridRetriever(
        fusion=fusion,
        w_faiss=w_faiss,
        w_bm25=w_bm25,
        limit=limit,
        prefetch_factor=prefetch_factor,
    )
    try:
        return retriever.search(query, limit=limit)
    except Exception:
        return []
