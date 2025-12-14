import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from config.settings import OPENAI_API_KEY

# Default locations and parameters for FAISS retrieval
FAISS_INDEX_DIR = Path("data/faiss_index")
FAISS_INDEX_NAME = "faiss_hnsw"
FAISS_TOP_K = 5
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)


def load_faiss_index(
    index_name: str = FAISS_INDEX_NAME, index_dir: Optional[Path] = None
) -> Tuple[faiss.Index, List[str]]:
    """Load FAISS index file and the parallel chunk_id list."""
    index_dir = Path(index_dir) if index_dir else FAISS_INDEX_DIR
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


def search_faiss(
    index: faiss.Index, chunk_ids: List[str], query: str, top_k: int = FAISS_TOP_K
):
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


def search(
    index: faiss.Index,
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
