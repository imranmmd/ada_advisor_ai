import os
import json
from storage.db.connection import get_connection

EMBEDDINGS_DIR = "data/embeddings/"  # folder with embedding_batch_*.json files


def _to_vector_literal(embedding):
    """
    Ensure embeddings are sent in a pgvector-friendly literal format.
    Accepts list/tuple or preformatted string.
    """
    if isinstance(embedding, str):
        # Strings may come from JSON dumps; ensure they represent a numeric list
        try:
            parsed = json.loads(embedding)
        except Exception:
            raise TypeError("Embedding string is not valid JSON.")
        return _to_vector_literal(parsed)

    if isinstance(embedding, (list, tuple)):
        sanitized = []
        for x in embedding:
            if not isinstance(x, (int, float)):
                raise TypeError("Embedding must contain only numbers.")
            sanitized.append(float(x))
        return "[" + ",".join(str(x) for x in sanitized) + "]"

    raise TypeError(f"Unsupported embedding type: {type(embedding)}")


def insert_row(cur, row):
    """Insert a single embedding row into the database."""
    cur.execute(
        """
        INSERT INTO chunk_embeddings (
            chunk_id, doc_id, order_index, page_number, header,
            token_count, text, embedding
        )
        VALUES (
            %(chunk_id)s, %(doc_id)s, %(order_index)s, %(page_number)s,
            %(header)s, %(token_count)s, %(text)s, %(embedding)s::vector
        )
        ON CONFLICT (chunk_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
        """,
        row
    )


def import_embeddings():
    """Import embeddings from JSON files into PostgreSQL."""
    files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith("embedding_batch_")]

    if not files:
        print("❌ No embedding JSON files found.")
        return

    files.sort()  # ensure correct order

    conn = get_connection()
    cur = conn.cursor()

    for filename in files:
        path = os.path.join(EMBEDDINGS_DIR, filename)
        print(f"\n→ Loading file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            batch = json.load(f)

        print(f"   {len(batch)} embeddings found")

        inserted = 0
        for record in batch:
            try:
                safe_record = {
                    **record,
                    "embedding": _to_vector_literal(record.get("embedding")),
                }
                insert_row(cur, safe_record)
                inserted += 1
            except Exception as exc:
                conn.rollback()
                print(f"   ⚠️ Skipping record {record.get('chunk_id')}: {exc}")
                cur = conn.cursor()

        conn.commit()
        print(f"   ✔ Imported {inserted} embeddings into PostgreSQL from {filename}")

    cur.close()
    conn.close()
    print("\n✅ All embedding batches imported successfully!")


if __name__ == "__main__":
    import_embeddings()
