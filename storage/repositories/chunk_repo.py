from typing import Callable, Dict, List, Optional, Sequence

from storage.db.connection import get_connection


class ChunkRepository:
    """Repository for the `chunks` table."""

    def __init__(self, connection_factory: Callable = get_connection):
        self._connection_factory = connection_factory

    def upsert(self, chunk: Dict, conn=None) -> str:
        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "order_index": chunk.get("order_index"),
            "page_number": chunk.get("page_number"),
            "header": chunk.get("header"),
            "text": chunk["text"],
            "token_count": chunk.get("token_count"),
        }

        sql = """
        INSERT INTO chunks (
            chunk_id, doc_id, order_index, page_number,
            header, text, token_count
        )
        VALUES (
            %(chunk_id)s, %(doc_id)s, %(order_index)s, %(page_number)s,
            %(header)s, %(text)s, %(token_count)s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            order_index = EXCLUDED.order_index,
            page_number = EXCLUDED.page_number,
            header = EXCLUDED.header,
            text = EXCLUDED.text,
            token_count = EXCLUDED.token_count
        RETURNING chunk_id;
        """

        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        return row[0] if row else payload["chunk_id"]

    def get(self, chunk_id: str, conn=None) -> Optional[Dict]:
        sql = """
        SELECT chunk_id, doc_id, order_index, page_number,
               header, text, token_count, created_at
        FROM chunks
        WHERE chunk_id = %s;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (chunk_id,))
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (chunk_id,))
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_by_document(self, doc_id: str, conn=None) -> List[Dict]:
        sql = """
        SELECT chunk_id, doc_id, order_index, page_number,
               header, text, token_count, created_at
        FROM chunks
        WHERE doc_id = %s
        ORDER BY order_index ASC NULLS LAST, chunk_id;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (doc_id,))
                rows = cur.fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (doc_id,))
                rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def bulk_upsert(self, chunks: Sequence[Dict], conn=None) -> List[str]:
        inserted_ids = []
        for chunk in chunks:
            inserted_ids.append(self.upsert(chunk, conn=conn))
        return inserted_ids

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "order_index": row[2],
            "page_number": row[3],
            "header": row[4],
            "text": row[5],
            "token_count": row[6],
            "created_at": row[7],
        }
