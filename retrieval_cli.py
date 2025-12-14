"""
Hybrid retrieval smoke test (FAISS + BM25).

How it works:
- Loads saved FAISS and BM25 indexes plus their chunk_id mappings.
- Takes user input, embeds it with OpenAI, and queries both indexes.
- Merges top results (5 FAISS + 5 BM25) and asks the chat model to pick the best
  answer using the retrieved chunk text when available.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from config.settings import OPENAI_API_KEY
from storage.bm25_loader import BM25_TOP_K, load_bm25_index, search_bm25
from storage.faiss_loader import (
    FAISS_INDEX_DIR,
    FAISS_TOP_K,
    hydrate_metadata,
    load_faiss_index,
    search as faiss_search,
    search_faiss,
)
from storage.memory_manager import (
    fetch_recent_messages,
    persist_messages,
    resolve_session_id,
    trim_history,
)

try:
    from storage.repositories import (
        RetrievalEventRepository,
    )
except Exception:
    RetrievalEventRepository = None  # type: ignore

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
INDEX_DIR = FAISS_INDEX_DIR  # kept for compatibility with existing callers/tests
HYBRID_LIMIT = 10
CHAT_MODEL = "gpt-4o-mini"
HISTORY_DAYS = 10

client = OpenAI(api_key=OPENAI_API_KEY)


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
    """Proxy to the FAISS loader search to keep backward compatibility."""
    return faiss_search(index, chunk_ids, query, top_k=top_k, cached_metadata=cached_metadata)


def _format_history(history: List[Dict[str, str]]) -> str:
    """Format chat history for prompts (expects newest-first ordering)."""
    return "\n".join(
        f"- {m.get('role', 'unknown')}: {m.get('content', '').strip()}"
        for m in history
        if m.get("content")
    )


def rewrite_query_with_history(
    query: str, history: List[Dict[str, str]], max_chars: int = 3000
) -> str:
    """
    Rewrite the user query so it is standalone for retrieval (coreference resolution).
    Falls back to the original query on any failure.
    """
    trimmed_history = trim_history(history, max_chars=max_chars)
    formatted_history = _format_history(trimmed_history)
    prompt = (
        "Rewrite the user's last message so it is fully self-contained for retrieval.\n"
        "Resolve pronouns and references using the conversation history.\n"
        "Return ONLY the rewritten query."
        f"\n\nConversation (newest first):\n{formatted_history or '- none'}"
        f"\n\nUser message:\n{query}"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You turn follow-ups into standalone questions for search.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or query
    except Exception:
        return query


def build_history_memory_chunk(
    query: str, history: List[Dict[str, str]], max_chars: int = 3000
) -> Optional[Dict[str, object]]:
    """
    Summarize relevant facts from chat history into a synthetic chunk (episodic memory).
    Returns None if nothing is relevant or if the model fails.
    """
    trimmed_history = trim_history(history, max_chars=max_chars)
    formatted_history = _format_history(trimmed_history)
    prompt = (
        "Given the conversation, extract the facts needed to answer the new question.\n"
        "If no facts are relevant, reply with 'NONE'. Keep the summary under 120 words."
        f"\n\nConversation (newest first):\n{formatted_history or '- none'}"
        f"\n\nQuestion:\n{query}"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Summarize only the relevant facts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        if content.upper().startswith("NONE"):
            return None
        return {
            "chunk_id": "chat_history_memory",
            "score": 1.0,
            "text": content,
            "page_number": None,
            "source": "chat_history",
        }
    except Exception:
        return None


def pick_best_answer(
    query: str,
    results: List[Dict[str, str]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
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

    history = history or []
    history = trim_history(history)
    formatted_history = "\n".join(
        f"- {m.get('role', 'unknown')}: {m.get('content', '').strip()}"
        for m in history
        if m.get("content")
    )

    prompt = (
        "You are selecting the best answer from retrieved chunks.\n"
        "Respond concisely to the user's question using the most relevant chunk.\n"
        "If nothing fits, say so.\n"
        "Reply ONLY in JSON with keys 'answer' and 'chunk_id'.\n"
        "Conversation history is provided (newest first) to give context.\n\n"
        f"Conversation history:\n{formatted_history or '- none'}\n\n"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid retrieval CLI with memory recall.")
    parser.add_argument(
        "--session-id",
        type=str,
        default=str(uuid4()),
        help="Session identifier used to load/save chat history in the database.",
    )
    return parser.parse_args()


def _safe_log_retrieval(
    query: str, results: List[Dict[str, object]], original_query: Optional[str] = None
) -> Optional[str]:
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
                "original_query": original_query,
            }
        )
        return event_id
    except Exception:
        return None


def _safe_log_chat(
    session_id: str, query: str, answer: str, retrieval_event_id: Optional[str]
):
    """Persist chat history for the session if repository and DB are available."""
    messages = [
        {
            "message_id": str(uuid4()),
            "session_id": session_id,
            "role": "user",
            "content": query,
            "retrieval_event_id": retrieval_event_id,
        },
        {
            "message_id": str(uuid4()),
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
            "retrieval_event_id": retrieval_event_id,
        },
    ]
    persist_messages(session_id, messages)


def main():
    args = parse_args()
    session_id = resolve_session_id(args.session_id)
    session_history: List[Dict[str, str]] = []
    try:
        faiss_index, faiss_chunk_ids = load_faiss_index()
        bm25_index, bm25_chunk_ids, bm25_metadata = load_bm25_index()
    except Exception as exc:
        print(f"Failed to load index: {exc}")
        print("If the BM25 files are missing, run `python ingestion/build_bm25.py` first.")
        sys.exit(1)

    print(f"Session ID: {session_id} (reuse this to recall DB-backed history)")
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
            # Track the latest user message in memory for this session
            session_history.insert(0, {"role": "user", "content": query})

            db_history = fetch_recent_messages(session_id, days=HISTORY_DAYS)
            history_for_prompt = session_history + db_history

            standalone_query = rewrite_query_with_history(query, history_for_prompt)
            memory_chunk = build_history_memory_chunk(query, history_for_prompt)
            if standalone_query != query:
                print(f"\nRewritten for retrieval: {standalone_query}")

            faiss_results = search_faiss(
                faiss_index, faiss_chunk_ids, standalone_query, top_k=FAISS_TOP_K
            )
            bm25_results = search_bm25(
                bm25_index, bm25_chunk_ids, standalone_query, top_k=BM25_TOP_K
            )
            doc_results = merge_results(faiss_results, bm25_results, limit=HYBRID_LIMIT)
            doc_results = hydrate_metadata(doc_results, bm25_metadata)
            memory_results = [memory_chunk] if memory_chunk else []
        except Exception as exc:
            print(f"Search failed: {exc}")
            continue

        if doc_results:
            print("\nTop matches (hybrid FAISS + BM25):")
            for rank, res in enumerate(doc_results, start=1):
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
        else:
            print("No document results returned.")

        doc_answer_payload = pick_best_answer(query, doc_results, history=history_for_prompt)
        doc_answer = doc_answer_payload.get("answer")
        doc_chunk_id = doc_answer_payload.get("chunk_id")
        doc_page = None
        if doc_chunk_id:
            for res in doc_results:
                if res["chunk_id"] == doc_chunk_id:
                    doc_page = res.get("page_number")
                    break

        if memory_results:
            mem_answer_payload = pick_best_answer(query, memory_results, history=history_for_prompt)
            memory_answer = mem_answer_payload.get("answer")
            memory_chunk_id = mem_answer_payload.get("chunk_id") or "chat_history_memory"
        else:
            memory_answer = "No prior memory found for this session."
            memory_chunk_id = None

        print("\nDocument answer:\n", doc_answer)
        doc_page_label = f"p.{doc_page}" if doc_page else "p.unknown"
        doc_chunk_label = doc_chunk_id or "unknown_chunk"
        print(f"[{doc_answer}] [{doc_page_label}] [{doc_chunk_label}]")

        print("\nMemory answer:\n", memory_answer)
        mem_chunk_label = memory_chunk_id or "unknown_chunk"
        print(f"[{memory_answer}] [memory] [{mem_chunk_label}]")

        combined_answer = f"Documents: {doc_answer}\nMemory: {memory_answer}"

        # Track the assistant reply in memory for this session
        session_history.insert(0, {"role": "assistant", "content": combined_answer})

        event_id = _safe_log_retrieval(
            standalone_query, doc_results, original_query=query
        )
        _safe_log_chat(session_id, query, combined_answer, event_id)


if __name__ == "__main__":
    main()
