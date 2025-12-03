import os
import json
from storage.db.connection import get_connection

EMBEDDINGS_DIR = "data/embeddings/"  # folder with embedding_batch_*.json files


def insert_row(cur, row):
    cur.execute(
        """
        INSERT INTO chunk_embeddings (
            chunk_id, doc_id, order_index, page_number, header,
            token_count, text, embedding
        )
        VALUES (
            %(chunk_id)s, %(doc_id)s, %(order_index)s, %(page_number)s,
            %(header)s, %(token_count)s, %(text)s, %(embedding)s
        )
        ON CONFLICT (chunk_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
        """,
        row
    )


def import_embeddings():
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

        for record in batch:
            insert_row(cur, record)

        conn.commit()
        print(f"   ✔ Imported {len(batch)} embeddings into PostgreSQL")

    cur.close()
    conn.close()
    print("\n✅ All embedding batches imported successfully!")


if __name__ == "__main__":
    import_embeddings()
