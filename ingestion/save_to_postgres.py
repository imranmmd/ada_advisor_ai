import json
import os
from storage.db.connection import get_connection
from storage.repositories import DocumentRepository, ChunkRepository


# ============================================================
# 4.4 Validate Metadata Integrity
# ============================================================
def validate_metadata(documents, chunks):
    """
    Checks:
    1. Every chunk.doc_id exists in documents
    2. Required fields exist
    3. Chunk order continuity
    """
    errors = []

    # 1. Collect doc_ids
    doc_ids = {d["doc_id"] for d in documents}

    for ch in chunks:
        if ch["doc_id"] not in doc_ids:
            errors.append(
                f"Chunk {ch['chunk_id']} references unknown doc_id {ch['doc_id']}."
            )

    # 2. Required fields
    doc_required = ["doc_id", "title", "file_name", "file_path", "page_count", "ingested_at"]
    chunk_required = ["chunk_id", "doc_id", "order_index", "text", "token_count"]

    for d in documents:
        for f in doc_required:
            if f not in d:
                errors.append(f"Document {d} missing field: {f}")

    for c in chunks:
        for f in chunk_required:
            if f not in c:
                errors.append(f"Chunk {c} missing field: {f}")

    # 3. Check order_index continuity for each doc_id
    from collections import defaultdict
    orders = defaultdict(list)

    for c in chunks:
        orders[c["doc_id"]].append(c["order_index"])

    for doc_id, arr in orders.items():
        arr_sorted = sorted(arr)
        for i in range(1, len(arr_sorted) + 1):
            if arr_sorted[i - 1] != i:
                errors.append(
                    f"Order index discontinuity in document '{doc_id}'. Expected continuous sequence."
                )

    return errors



# ============================================================
# Main Function: Save All Metadata
# ============================================================
def save_metadata_to_postgres(documents_path="documents.json", chunks_path="chunks.json"):
    # 1. Load JSONs
    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. Validate metadata
    errors = validate_metadata(documents, chunks)
    if errors:
        print("❌ Metadata integrity check failed:")
        for e in errors:
            print("   -", e)
        return

    # 3. Insert into Postgres
    conn = get_connection()
    doc_repo = DocumentRepository()
    chunk_repo = ChunkRepository()

    try:
        for d in documents:
            doc_repo.upsert(d, conn=conn)

        for c in chunks:
            chunk_repo.upsert(c, conn=conn)

        conn.commit()
        print("✅ Successfully saved metadata to Postgres.")

    except Exception as e:
        conn.rollback()
        print("❌ Error saving metadata:", e)

    finally:
        conn.close()



if __name__ == "__main__":
    default_docs = os.path.join("data", "metadata", "documents.json")
    default_chunks = os.path.join("data", "metadata", "chunks.json")
    save_metadata_to_postgres(default_docs, default_chunks)
