from typing import Callable, Dict, List, Optional

from storage.db.connection import get_connection


class RetrievalEventRepository:
    """Repository for the `retrieval_events` table."""

    def __init__(self, connection_factory: Callable = get_connection):
        self._connection_factory = connection_factory

    def log_event(self, event: Dict, conn=None) -> str:
        payload = {
            "event_id": event["event_id"],
            "query_text": event["query_text"],
            "query_embedding": event.get("query_embedding"),
            "retrieved_chunk_ids": event.get("retrieved_chunk_ids"),
            "top_k": event.get("top_k"),
            "scores": event.get("scores"),
        }

        sql = """
        INSERT INTO retrieval_events (
            event_id, query_text, query_embedding,
            retrieved_chunk_ids, top_k, scores
        )
        VALUES (
            %(event_id)s, %(query_text)s, %(query_embedding)s,
            %(retrieved_chunk_ids)s, %(top_k)s, %(scores)s
        )
        ON CONFLICT (event_id) DO UPDATE SET
            query_text = EXCLUDED.query_text,
            query_embedding = EXCLUDED.query_embedding,
            retrieved_chunk_ids = EXCLUDED.retrieved_chunk_ids,
            top_k = EXCLUDED.top_k,
            scores = EXCLUDED.scores
        RETURNING event_id;
        """

        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        return row[0] if row else payload["event_id"]

    def get(self, event_id: str, conn=None) -> Optional[Dict]:
        sql = """
        SELECT event_id, query_text, query_embedding,
               retrieved_chunk_ids, top_k, scores, created_at
        FROM retrieval_events
        WHERE event_id = %s;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (event_id,))
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (event_id,))
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_recent(self, limit: int = 20, conn=None) -> List[Dict]:
        sql = """
        SELECT event_id, query_text, query_embedding,
               retrieved_chunk_ids, top_k, scores, created_at
        FROM retrieval_events
        ORDER BY created_at DESC
        LIMIT %s;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return {
            "event_id": row[0],
            "query_text": row[1],
            "query_embedding": row[2],
            "retrieved_chunk_ids": row[3],
            "top_k": row[4],
            "scores": row[5],
            "created_at": row[6],
        }
