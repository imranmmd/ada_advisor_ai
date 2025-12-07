from typing import Callable, Dict, List, Optional

from storage.db.connection import get_connection


class ChatHistoryRepository:
    """Repository for the `chat_history` table."""

    def __init__(self, connection_factory: Callable = get_connection):
        self._connection_factory = connection_factory

    def add_message(self, message: Dict, conn=None) -> str:
        payload = {
            "message_id": message["message_id"],
            "session_id": message["session_id"],
            "role": message["role"],
            "content": message["content"],
            "retrieval_event_id": message.get("retrieval_event_id"),
        }

        sql = """
        INSERT INTO chat_history (
            message_id, session_id, role, content, retrieval_event_id
        )
        VALUES (
            %(message_id)s, %(session_id)s, %(role)s, %(content)s, %(retrieval_event_id)s
        )
        ON CONFLICT (message_id) DO UPDATE SET
            session_id = EXCLUDED.session_id,
            role = EXCLUDED.role,
            content = EXCLUDED.content,
            retrieval_event_id = EXCLUDED.retrieval_event_id
        RETURNING message_id;
        """

        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, payload)
                row = cur.fetchone()
        return row[0] if row else payload["message_id"]

    def recent_messages(self, session_id: str, limit: int = 20, conn=None) -> List[Dict]:
        sql = """
        SELECT message_id, session_id, role, content,
               retrieval_event_id, created_at
        FROM chat_history
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (session_id, limit))
                rows = cur.fetchall()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (session_id, limit))
                rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get(self, message_id: str, conn=None) -> Optional[Dict]:
        sql = """
        SELECT message_id, session_id, role, content,
               retrieval_event_id, created_at
        FROM chat_history
        WHERE message_id = %s;
        """
        if conn is None:
            with self._connection_factory() as conn, conn.cursor() as cur:
                cur.execute(sql, (message_id,))
                row = cur.fetchone()
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (message_id,))
                row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return {
            "message_id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "retrieval_event_id": row[4],
            "created_at": row[5],
        }
