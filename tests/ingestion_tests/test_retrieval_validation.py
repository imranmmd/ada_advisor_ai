import json
import os
import types
import sys
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

# Ensure imports that rely on the OpenAI API key do not fail during tests
os.environ.setdefault("OPENAI_API_KEY", "test-key")

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from storage import faiss_loader
import retrieval_cli
from rag_core.retrievers.hybrid_retriever import HybridRetriever


class DummyChatResponse:
    def __init__(self, content: str):
        self.choices = [
            types.SimpleNamespace(message=types.SimpleNamespace(content=content))
        ]


def build_index(vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vectors, dtype=np.float32))
    return index


def recall_at_k(results, relevant_ids, k):
    retrieved_ids = [res["chunk_id"] for res in results[:k]]
    hits = set(retrieved_ids) & set(relevant_ids)
    return len(hits) / float(len(relevant_ids))


@pytest.fixture(autouse=True)
def stub_openai_client(monkeypatch):
    dummy_client = types.SimpleNamespace(
        embeddings=types.SimpleNamespace(
            create=lambda **kwargs: types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.0, 0.0])]
            )
        ),
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kwargs: DummyChatResponse(
                    '{"answer": "stub", "chunk_id": null}'
                )
            )
        ),
    )
    monkeypatch.setattr(faiss_loader, "client", dummy_client)
    monkeypatch.setattr(retrieval_cli, "client", dummy_client)


def test_load_faiss_index_reads_index_and_ids(tmp_path, monkeypatch):
    chunk_ids = ["chunk_a", "chunk_b"]
    index = build_index([[1.0, 0.0], [0.0, 1.0]])
    index_name = "custom_index"

    faiss.write_index(index, str(tmp_path / f"{index_name}.bin"))
    (tmp_path / f"{index_name}_chunk_ids.json").write_text(
        json.dumps(chunk_ids), encoding="utf-8"
    )

    loaded_index, loaded_ids = faiss_loader.load_faiss_index(
        index_name, index_dir=tmp_path
    )
    assert loaded_index.ntotal == len(chunk_ids)
    assert loaded_ids == chunk_ids


def test_search_returns_top_k_in_score_order(monkeypatch):
    chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
    vectors = [
        [1.0, 0.0],  # best match
        [0.8, 0.1],
        [0.1, 0.9],
    ]
    index = build_index(vectors)
    observed_queries = []

    def fake_embed(text):
        observed_queries.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(faiss_loader, "embed", fake_embed)
    monkeypatch.setattr(
        faiss_loader,
        "fetch_chunk_texts",
        lambda ids: {
            cid: {"text": f"text for {cid}", "page_number": 1} for cid in ids
        },
    )

    results = faiss_loader.search(index, chunk_ids, "best vector", top_k=2)

    assert observed_queries == ["best vector"]
    assert [r["chunk_id"] for r in results] == ["chunk_1", "chunk_2"]
    assert results[0]["score"] >= results[1]["score"]


def test_pick_best_answer_respects_chat_choice(monkeypatch):
    def fake_create(**kwargs):
        return DummyChatResponse('{"answer": "use chunk_2", "chunk_id": "chunk_2"}')

    monkeypatch.setattr(
        retrieval_cli,
        "client",
        types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        ),
    )

    results = [
        {"chunk_id": "chunk_1", "score": 0.9, "text": "first", "page_number": 1},
        {"chunk_id": "chunk_2", "score": 0.8, "text": "second", "page_number": 2},
    ]

    payload = retrieval_cli.pick_best_answer("Which chunk?", results)

    assert payload["chunk_id"] == "chunk_2"
    assert "chunk_2" in payload["answer"]


def test_recall_at_five_and_ten(monkeypatch):
    vectors = [
        [0.9, 0.1],
        [0.85, 0.1],
        [0.8, 0.1],
        [0.75, 0.1],
        [0.7, 0.1],
        [0.65, 0.1],
        [0.6, 0.1],  # relevant but appears after top-5
        [0.55, 0.1],
        [0.5, 0.1],
        [0.45, 0.1],
    ]
    chunk_ids = [f"chunk_{i}" for i in range(len(vectors))]
    index = build_index(vectors)

    monkeypatch.setattr(
        faiss_loader, "embed", lambda _: np.array([1.0, 0.0], dtype=np.float32)
    )
    monkeypatch.setattr(faiss_loader, "fetch_chunk_texts", lambda ids: {})

    results = faiss_loader.search(index, chunk_ids, "query", top_k=10)

    recall_5 = recall_at_k(results, {"chunk_6"}, k=5)
    recall_10 = recall_at_k(results, {"chunk_6"}, k=10)

    assert recall_5 == 0.0
    assert recall_10 == 1.0


def test_rewrite_query_with_history_uses_model(monkeypatch):
    def fake_create(**kwargs):
        return DummyChatResponse("rewritten question")

    monkeypatch.setattr(
        retrieval_cli,
        "client",
        types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        ),
    )

    history = [{"role": "assistant", "content": "You asked about circles."}]
    rewritten = retrieval_cli.rewrite_query_with_history(
        "What about that?", history
    )

    assert rewritten == "rewritten question"


def test_build_history_memory_chunk_handles_none(monkeypatch):
    def fake_create(**kwargs):
        return DummyChatResponse("NONE")

    monkeypatch.setattr(
        retrieval_cli,
        "client",
        types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        ),
    )

    chunk = retrieval_cli.build_history_memory_chunk(
        "Remind me", [{"role": "user", "content": "Earlier context"}]
    )

    assert chunk is None


def test_hybrid_retriever_prefetches_to_improve_recall():
    class DummyRetriever:
        def __init__(self, label: str):
            self.label = label
            self.calls = []
            # Only the tail item carries the strong score; it should appear
            # in the fused top-k once we over-fetch.
            self.pool = [
                {"chunk_id": f"{label}_{i}", "score": score, "text": None, "page_number": None}
                for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 10.0])
            ]

        def search(self, query: str, top_k: int):
            self.calls.append(top_k)
            return [dict(self.pool[i], source=self.label) for i in range(min(top_k, len(self.pool)))]

    semantic = DummyRetriever("faiss")
    bm25 = DummyRetriever("bm25")
    retriever = HybridRetriever(
        semantic=semantic,
        bm25=bm25,
        fusion="weighted",
        w_faiss=1.0,
        w_bm25=0.0,
        limit=3,
        prefetch_factor=2.0,
    )

    results = retriever.search("query")

    assert semantic.calls == [6]
    assert bm25.calls == [6]
    assert len(results) == 3
    # The high-scoring tail item should surface because we over-fetched.
    assert results[0]["chunk_id"] == "faiss_5"
