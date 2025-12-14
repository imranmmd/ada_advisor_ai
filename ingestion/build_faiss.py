import os
import sys
import json
import ast
from pathlib import Path

import numpy as np
import faiss

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.db.connection import get_connection


# ===============================================================
# |           LOAD EMBEDDINGS FROM POSTGRES DB                  |
# ===============================================================
def load_embeddings_from_db():
    """Load embeddings and their chunk IDs from PostgreSQL."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT chunk_id, embedding
        FROM chunk_embeddings
        ORDER BY chunk_id;
    """)

    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No embeddings found in chunk_embeddings; build embeddings first.")

    chunk_ids = []
    vectors = []

    for chunk_id, embedding in rows:
        chunk_ids.append(chunk_id)

        if isinstance(embedding, str):
            embedding = ast.literal_eval(embedding)

        vectors.append(np.array(embedding, dtype=np.float32))

    vectors = np.vstack(vectors)
    print(f"Loaded {vectors.shape[0]} vectors (dim = {vectors.shape[1]})")
    return chunk_ids, vectors


# ===============================================================
# |          BUILD FAISS INDEX - FlatIP                         |
# ===============================================================
def build_faiss_flatip(dim):
    """Build a FAISS IndexFlatIP index.
    """
    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    return index


# ===============================================================
# |          BUILD FAISS INDEX - HNSW                           |
# ===============================================================
def build_faiss_hnsw(dim, M=64, efConstruction=200):
    """Build a FAISS IndexHNSWFlat index.
    """
    print("Building FAISS IndexHNSWFlat...")
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = 128
    return index


# ===============================================================
# |                   ADD VECTORS TO INDEX                      |
# ===============================================================
def add_vectors(index, vectors):
    """Add vectors to the FAISS index with normalization.
    """
    print(f"Adding {vectors.shape[0]} vectors to index...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    index.add(normalized)
    print("Vectors added successfully.")


# ===============================================================
# |                 SAVE INDEX + CHUNK IDS                      |
# ===============================================================
def save_index(index, chunk_ids, file_name):
    """Save FAISS index and chunk_ids mapping to disk.
    """
    os.makedirs("data/faiss_index", exist_ok=True)

    index_path = f"data/faiss_index/{file_name}.bin"
    ids_path = f"data/faiss_index/{file_name}_chunk_ids.json"

    faiss.write_index(index, index_path)
    print(f"Saved FAISS index → {index_path}")

    with open(ids_path, "w") as f:
        json.dump(chunk_ids, f, indent=2)

    print(f"Saved chunk_ids mapping → {ids_path}")


# ===============================================================
# |                      MAIN PIPELINE                          |
# ===============================================================
def build_faiss_indexes():
    """Build FAISS indexes from embeddings stored in PostgreSQL.
    """
    print("Loading embeddings from PostgreSQL...")
    chunk_ids, vectors = load_embeddings_from_db()

    dim = vectors.shape[1]

    # ------------------------------
    # Build FlatIP index
    # ------------------------------
    flat_index = build_faiss_flatip(dim)
    add_vectors(flat_index, vectors)
    save_index(flat_index, chunk_ids, "faiss_flat")

    # ------------------------------
    # Build HNSW index
    # ------------------------------
    hnsw_index = build_faiss_hnsw(dim)
    add_vectors(hnsw_index, vectors)
    save_index(hnsw_index, chunk_ids, "faiss_hnsw")

    print("\n✅ All FAISS indexes built and saved successfully!")


# ===============================================================
# |                        ENTRY POINT                          |
# ===============================================================
if __name__ == "__main__":
    build_faiss_indexes()
