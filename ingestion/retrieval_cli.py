"""
Simple FAISS smoke test.

How it works:
- Loads a saved FAISS index and its chunk_id mapping from data/faiss_index.
- Takes user input, embeds it with OpenAI, and runs a similarity search.
- Prints the top matches (chunk_id, score) and tries to show chunk text if a
  PostgreSQL connection is available.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from openai import OpenAI

from config.settings import OPENAI_API_KEY

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
INDEX_DIR = Path("data/faiss_index")
INDEX_NAME = "faiss_hnsw"  # change to "faiss_flat" if you prefer
TOP_K = 5
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)


def load_faiss_index(index_name: str = INDEX_NAME):
    """Load FAISS index file and the parallel chunk_id list."""
    index_path = INDEX_DIR / f"{index_name}.bin"
    ids_path = INDEX_DIR / f"{index_name}_chunk_ids.json"

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


def search(index, chunk_ids: List[str], query: str, top_k: int = TOP_K):
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
    found_ids = [chunk_ids[i] for i in idxs[0] if i != -1]
    metadata = fetch_chunk_texts(found_ids)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        cid = chunk_ids[idx]
        chunk_meta = metadata.get(cid, {})
        results.append(
            {
                "chunk_id": cid,
                "score": float(score),
                "text": chunk_meta.get("text"),
                "page_number": chunk_meta.get("page_number"),
            }
        )
    return results


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
        formatted_chunks.append(
            f"- id: {r['chunk_id']} ({page_label})\n  score: {r['score']:.4f}\n  text: {snippet}"
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


def main():
    try:
        index, chunk_ids = load_faiss_index()
    except Exception as exc:
        print(f"Failed to load index: {exc}")
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
            results = search(index, chunk_ids, query)
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        if not results:
            print("No results returned.")
            continue

        print("\nTop matches:")
        for rank, res in enumerate(results, start=1):
            snippet = ""
            if res["text"]:
                snippet = res["text"].strip().replace("\n", " ")
                snippet = (snippet[:180] + "...") if len(snippet) > 180 else snippet
            page_str = (
                f" (page {res['page_number']})" if res.get("page_number") else ""
            )
            print(f"{rank}. {res['chunk_id']}{page_str}  score={res['score']:.4f}")
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


if __name__ == "__main__":
    main()
