from typing import Callable, Dict, List, Optional

from storage.db.connection import get_connection


class DocumentRepository:
    """
    Repository for the `documents` table.
    Uses a connection factory so it can be mocked in tests.
    """

    def __init__(self, connection_factory: Callable = get_connection):
        self._connection_factory = connection_factory

    def upsert(self, document: Dict, conn=None) -> str:
        """
        Insert or update a document.
        Returns the document id that was written.
        """
        payload = {
            "doc_id": document["doc_id"],
            "title": document.get("title"),
            "file_name": document.get("file_name"),
            "file_path": document.get("file_path"),
            "page_count": document.get("page_count"),
            "version": document.get("version", 1),
            "ingested_at": document.get("ingested_at"),
        }

        sql = """
        INSERT INTO documents (
            doc_id, title, file_name, file_path,
            page_count, version, ingested_at
        )
        VALUES (
            %(doc_id)s, %(title)s, %(file_name)s, %(file_path)s,
            %(page_count)s, %(version)s, %(ingested_at)s
        )
        ON CONFLICT (doc_id) DO UPDATE SET
            title = EXCLUDED.title,
            file_name = EXCLUDED.file_name,
            file_path = EXCLUDED.file_path,
            page_count = EXCLUDED.page_count,
            version = EXCLUDED.version,
            ingested_at = EXCLUDED.ingested_at
        RETURNING doc_id;
        """

        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        return row[0] if row else payload["doc_id"]

    def get(self, doc_id: str, conn=None) -> Optional[Dict]:
        sql = """
        SELECT doc_id, title, file_name, file_path,
               page_count, version, ingested_at
        FROM documents
        WHERE doc_id = %s;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (doc_id,))
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (doc_id,))
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def list_all(self, conn=None) -> List[Dict]:
        sql = """
        SELECT doc_id, title, file_name, file_path,
               page_count, version, ingested_at
        FROM documents
        ORDER BY ingested_at DESC NULLS LAST, doc_id;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return {
            "doc_id": row[0],
            "title": row[1],
            "file_name": row[2],
            "file_path": row[3],
            "page_count": row[4],
            "version": row[5],
            "ingested_at": row[6],
        }
