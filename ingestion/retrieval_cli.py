"""
Hybrid retrieval smoke test (FAISS + BM25).

How it works:
- Loads saved FAISS and BM25 indexes plus their chunk_id mappings.
- Takes user input, embeds it with OpenAI, and queries both indexes.
- Merges top results (5 FAISS + 5 BM25) and asks the chat model to pick the best
  answer using the retrieved chunk text when available.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config.settings import OPENAI_API_KEY

try:
    from storage.repositories import (
        RetrievalEventRepository,
        ChatHistoryRepository,
    )
except Exception:
    RetrievalEventRepository = None  # type: ignore
    ChatHistoryRepository = None  # type: ignore

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
FAISS_INDEX_DIR = Path("data/faiss_index")
INDEX_DIR = FAISS_INDEX_DIR  # kept for compatibility with existing callers/tests
BM25_INDEX_DIR = Path("data/bm25_index")
FAISS_INDEX_NAME = "faiss_hnsw"  # change to "faiss_flat" if you prefer
BM25_INDEX_NAME = "bm25"
FAISS_TOP_K = 5
BM25_TOP_K = 5
HYBRID_LIMIT = 10
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)


def load_faiss_index(index_name: str = FAISS_INDEX_NAME):
    """Load FAISS index file and the parallel chunk_id list."""
    index_dir = INDEX_DIR
    index_path = index_dir / f"{index_name}.bin"
    ids_path = index_dir / f"{index_name}_chunk_ids.json"

    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing index or id file. Expected {index_path} and {ids_path}"
        )

    index = faiss.read_index(str(index_path))
    with ids_path.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    print(f"Loaded index {index_name} with {index.ntotal} vectors (dim={index.d})")
    return index, chunk_ids


def load_bm25_index(
    index_name: str = BM25_INDEX_NAME,
) -> Tuple[BM25Okapi, List[str], Dict[str, Dict]]:
    """Load BM25 pickle plus chunk ids and cached metadata."""
    index_path = BM25_INDEX_DIR / f"{index_name}.pkl"
    ids_path = BM25_INDEX_DIR / f"{index_name}_chunk_ids.json"
    metadata_path = BM25_INDEX_DIR / f"{index_name}_chunks.json"

    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing BM25 index or id file. Expected {index_path} and {ids_path}"
        )

    with index_path.open("rb") as f:
        bm25 = pickle.load(f)
    with ids_path.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    metadata: Dict[str, Dict] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    print(f"Loaded BM25 index {index_name} with {len(chunk_ids)} documents")
    return bm25, chunk_ids, metadata


def embed(text: str) -> np.ndarray:
    """Embed a single string using the configured OpenAI model."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def fetch_chunk_texts(chunk_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Try to fetch chunk text from Postgres; fall back to empty dict if DB is
    unavailable. Only queries the chunk_embeddings table for the requested ids.
    """
    try:
        from storage.db.connection import get_connection
    except Exception:
        return {}

    if not chunk_ids:
        return {}

    try:
        conn = get_connection()
    except Exception:
        return {}

    placeholders = ",".join(["%s"] * len(chunk_ids))
    sql = (
        "SELECT chunk_id, page_number, text "
        "FROM chunk_embeddings WHERE chunk_id IN ({placeholders})"
    ).format(placeholders=placeholders)

    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, chunk_ids)
            return {
                row[0]: {"page_number": row[1], "text": row[2]}
                for row in cur.fetchall()
            }
    except Exception:
        return {}


def search_faiss(index, chunk_ids: List[str], query: str, top_k: int = FAISS_TOP_K):
    """Embed the query and search the FAISS index."""
    vector = embed(query)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding norm is zero; cannot normalize query vector.")
    vector = vector / norm

    if vector.shape[0] != index.d:
        raise ValueError(
            f"Embedding dimension {vector.shape[0]} != index dimension {index.d}. "
            "Ensure the same embedding model was used to build the index."
        )

    scores, idxs = index.search(np.expand_dims(vector, axis=0), top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        cid = chunk_ids[idx]
        results.append(
            {
                "chunk_id": cid,
                "score": float(score),
                "text": None,
                "page_number": None,
                "source": "faiss",
            }
        )
    return results


def search_bm25(
    bm25: BM25Okapi, chunk_ids: List[str], query: str, top_k: int = BM25_TOP_K
):
    """Search BM25 index."""
    if hasattr(bm25, "corpus_size") and bm25.corpus_size != len(chunk_ids):
        raise ValueError(
            f"BM25 index corpus size {bm25.corpus_size} does not match "
            f"{len(chunk_ids)} chunk ids."
        )

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        cid = chunk_ids[int(idx)]
        results.append(
            {
                "chunk_id": cid,
                "score": float(scores[int(idx)]),
                "text": None,
                "page_number": None,
                "source": "bm25",
            }
        )
    return results


def hydrate_metadata(
    results: List[Dict[str, object]], cached_metadata: Optional[Dict[str, Dict]]
):
    """Attach text/page metadata from DB, falling back to cached BM25 metadata."""
    if not results:
        return results

    ids = [r["chunk_id"] for r in results]
    db_meta = fetch_chunk_texts(ids)
    cached_metadata = cached_metadata or {}

    for r in results:
        cid = r["chunk_id"]
        meta = db_meta.get(cid) or cached_metadata.get(cid) or {}
        r["text"] = meta.get("text")
        r["page_number"] = meta.get("page_number")
    return results


def merge_results(
    faiss_results: List[Dict[str, object]],
    bm25_results: List[Dict[str, object]],
    limit: int = HYBRID_LIMIT,
):
    """Combine FAISS and BM25 results, removing duplicates and preserving order."""
    combined: List[Dict[str, object]] = []
    seen = set()

    for res in faiss_results + bm25_results:
        cid = res["chunk_id"]
        if cid in seen:
            continue
        combined.append(res)
        seen.add(cid)
        if len(combined) >= limit:
            break
    return combined


def search(
    index,
    chunk_ids: List[str],
    query: str,
    top_k: int = FAISS_TOP_K,
    cached_metadata: Optional[Dict[str, Dict]] = None,
):
    """
    Search a FAISS index and hydrate results with metadata.

    - Embeds and normalizes the query once via `embed`.
    - Returns the top_k results sorted by score (as produced by FAISS).
    - Attaches `text` and `page_number` using `fetch_chunk_texts` or cached metadata.
    """
    faiss_results = search_faiss(index, chunk_ids, query, top_k=top_k)
    return hydrate_metadata(faiss_results, cached_metadata)


def pick_best_answer(query: str, results: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Ask the chat model to pick the best answer from the retrieved chunks.
    Returns a payload with the answer and the chunk id the model picked.
    """
    if not results:
        return {"answer": "No relevant chunks found.", "chunk_id": None}

    formatted_chunks = []
    for r in results:
        snippet = (r.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        page_info = r.get("page_number")
        page_label = f"page {page_info}" if page_info else "page unknown"
        source_label = r.get("source", "faiss")
        formatted_chunks.append(
            f"- id: {r['chunk_id']} ({source_label}, {page_label})\n"
            f"  score: {r['score']:.4f}\n  text: {snippet}"
        )

    prompt = (
        "You are selecting the best answer from retrieved chunks.\n"
        "Respond concisely to the user's question using the most relevant chunk.\n"
        "If nothing fits, say so.\n"
        "Reply ONLY in JSON with keys 'answer' and 'chunk_id'.\n\n"
        f"Question: {query}\n\n"
        "Retrieved chunks (ordered by similarity):\n"
        + "\n".join(formatted_chunks)
        + "\n\nExample JSON: {\"answer\": \"...\", \"chunk_id\": \"chunk_xxxx\"}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {"answer": content, "chunk_id": None}

    return {
        "answer": parsed.get("answer", content),
        "chunk_id": parsed.get("chunk_id"),
    }


def _safe_log_retrieval(query: str, results: List[Dict[str, object]]) -> Optional[str]:
    """Persist retrieval event if repository and DB are available."""
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
                "retrieved_chunk_ids": [r["chunk_id"] for r in results],
                "top_k": len(results),
                "scores": [r.get("score") for r in results],
            }
        )
        return event_id
    except Exception:
        return None


def _safe_log_chat(
    session_id: str, query: str, answer: str, retrieval_event_id: Optional[str]
):
    """Persist chat history for the session if repository and DB are available."""
    if not ChatHistoryRepository:
        return
    try:
        repo = ChatHistoryRepository()
        repo.add_message(
            {
                "message_id": str(uuid4()),
                "session_id": session_id,
                "role": "user",
                "content": query,
                "retrieval_event_id": retrieval_event_id,
            }
        )
        repo.add_message(
            {
                "message_id": str(uuid4()),
                "session_id": session_id,
                "role": "assistant",
                "content": answer,
                "retrieval_event_id": retrieval_event_id,
            }
        )
    except Exception:
        return


def main():
    session_id = str(uuid4())
    try:
        faiss_index, faiss_chunk_ids = load_faiss_index()
        bm25_index, bm25_chunk_ids, bm25_metadata = load_bm25_index()
    except Exception as exc:
        print(f"Failed to load index: {exc}")
        print("If the BM25 files are missing, run `python ingestion/build_bm25.py` first.")
        sys.exit(1)

    print("\nType a query and press Enter. Empty line or Ctrl+C to quit.")

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            print("Goodbye.")
            break

        try:
            faiss_results = search_faiss(
                faiss_index, faiss_chunk_ids, query, top_k=FAISS_TOP_K
            )
            bm25_results = search_bm25(
                bm25_index, bm25_chunk_ids, query, top_k=BM25_TOP_K
            )
            results = merge_results(faiss_results, bm25_results, limit=HYBRID_LIMIT)
            results = hydrate_metadata(results, bm25_metadata)
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        if not results:
            print("No results returned.")
            continue

        print("\nTop matches (hybrid FAISS + BM25):")
        for rank, res in enumerate(results, start=1):
            snippet = ""
            if res["text"]:
                snippet = res["text"].strip().replace("\n", " ")
                snippet = (snippet[:180] + "...") if len(snippet) > 180 else snippet
            page_str = (
                f" (page {res['page_number']})" if res.get("page_number") else ""
            )
            source_str = res.get("source", "faiss")
            print(
                f"{rank}. {res['chunk_id']}{page_str}  "
                f"[{source_str}] score={res['score']:.4f}"
            )
            if snippet:
                print(f"    {snippet}")

        answer_payload = pick_best_answer(query, results)
        answer = answer_payload.get("answer")
        chosen_chunk_id = answer_payload.get("chunk_id")
        chosen_page = None
        if chosen_chunk_id:
            for res in results:
                if res["chunk_id"] == chosen_chunk_id:
                    chosen_page = res.get("page_number")
                    break
        elif results:
            chosen_chunk_id = results[0]["chunk_id"]
            chosen_page = results[0].get("page_number")

        print("\nLLM-chosen answer:\n", answer)
        page_label = f"p.{chosen_page}" if chosen_page else "p.unknown"
        chunk_label = chosen_chunk_id or "unknown_chunk"
        print(f"[{answer}] [{page_label}] [{chunk_label}]")

        event_id = _safe_log_retrieval(query, results)
        _safe_log_chat(session_id, query, answer, event_id)


if __name__ == "__main__":
    main()
