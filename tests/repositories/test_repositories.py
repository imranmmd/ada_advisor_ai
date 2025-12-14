import sys
from pathlib import Path
import types

import pytest

# Provide a lightweight psycopg2 stub so imports succeed without the dependency.
if "psycopg2" not in sys.modules:
    sys.modules["psycopg2"] = types.SimpleNamespace(connect=lambda *_, **__: None)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.repositories.document_repo import DocumentRepository
from storage.repositories.chunk_repo import ChunkRepository
from storage.repositories.retrieval_repo import RetrievalEventRepository
from storage.repositories.chat_history_repo import ChatHistoryRepository


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []
        self.closed = False

    def execute(self, sql, params=None):
        self.executed.append((_normalize_sql(sql), params))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self.cursor_obj = cursor
        self.closed = False
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.rollbacks += 1
        else:
            self.commits += 1
        self.close()
        return False


def test_document_upsert_inserts_and_returns_id():
    cursor = FakeCursor(rows=[("doc-1",)])
    conn = FakeConnection(cursor)
    repo = DocumentRepository(lambda: conn)

    doc_id = repo.upsert(
        {
            "doc_id": "doc-1",
            "title": "Doc",
            "file_name": "a.pdf",
            "file_path": "/tmp/a.pdf",
            "page_count": 2,
        }
    )

    assert doc_id == "doc-1"
    assert cursor.executed, "No SQL executed"
    sql, params = cursor.executed[0]
    assert "INSERT INTO documents" in sql
    assert params["doc_id"] == "doc-1"
    assert conn.commits == 1
    assert conn.closed


def test_document_get_returns_dict():
    row = ("doc-2", "Title", "f.pdf", "/tmp/f.pdf", 5, 1, None)
    cursor = FakeCursor(rows=[row])
    conn = FakeConnection(cursor)
    repo = DocumentRepository(lambda: conn)

    doc = repo.get("doc-2")

    assert doc["doc_id"] == "doc-2"
    assert doc["page_count"] == 5
    sql, params = cursor.executed[0]
    assert "FROM documents" in sql
    assert params == ("doc-2",)


def test_chunk_list_by_document_orders_and_maps_rows():
    rows = [
        ("c-1", "d-1", 0, 1, "h", "text-1", 50, "now"),
        ("c-2", "d-1", 1, 2, None, "text-2", 60, "now"),
    ]
    cursor = FakeCursor(rows=rows)
    conn = FakeConnection(cursor)
    repo = ChunkRepository(lambda: conn)

    chunks = repo.list_by_document("d-1")

    assert [c["chunk_id"] for c in chunks] == ["c-1", "c-2"]
    sql, params = cursor.executed[0]
    assert "FROM chunks" in sql and "ORDER BY" in sql
    assert params == ("d-1",)


def test_retrieval_event_log_event_saves_arrays():
    cursor = FakeCursor(rows=[("evt-1",)])
    conn = FakeConnection(cursor)
    repo = RetrievalEventRepository(lambda: conn)
    event = {
        "event_id": "evt-1",
        "query_text": "hello?",
        "retrieved_chunk_ids": ["c1", "c2"],
        "top_k": 5,
        "scores": [0.9, 0.8],
        "query_embedding": [0.1, 0.2],
    }

    event_id = repo.log_event(event)

    assert event_id == "evt-1"
    sql, params = cursor.executed[0]
    assert "INSERT INTO retrieval_events" in sql
    assert params["retrieved_chunk_ids"] == ["c1", "c2"]
    assert conn.commits == 1


def test_chat_history_recent_messages_limits_results():
    rows = [
        ("m-2", "s-1", "assistant", "second", None, "now"),
        ("m-1", "s-1", "user", "first", "evt-1", "earlier"),
    ]
    cursor = FakeCursor(rows=rows)
    conn = FakeConnection(cursor)
    repo = ChatHistoryRepository(lambda: conn)

    messages = repo.recent_messages("s-1", limit=2, days=10)

    assert [m["message_id"] for m in messages] == ["m-2", "m-1"]
    sql, params = cursor.executed[0]
    assert "FROM chat_history" in sql and "LIMIT" in sql and "created_at" in sql
    assert params == ("s-1", 10, 2)
