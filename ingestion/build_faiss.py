import os
import sys
import json
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import faiss

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.db.connection import get_connection


@dataclass(frozen=True)
class FaissConfig:
    """Immutable settings for building FAISS indexes."""

    output_dir: Path = Path("data/faiss_index")
    flat_name: str = "faiss_flat"
    hnsw_name: str = "faiss_hnsw"
    hnsw_m: int = 64
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128


class FaissBuilder:
    """Builds FAISS indexes from stored embeddings with configurable settings."""

    def __init__(
        self,
        config: FaissConfig = FaissConfig(),
        connection_factory: Callable[[], object] = get_connection,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory

    @staticmethod
    def _normalize_embedding(raw_embedding) -> np.ndarray:
        """
        Convert DB-returned embedding into a float32 numpy array.
        Handles pgvector (list), text, bytes/memoryview, or numpy inputs.
        """
        if isinstance(raw_embedding, np.ndarray):
            return raw_embedding.astype(np.float32)
        if isinstance(raw_embedding, (list, tuple)):
            return np.asarray(raw_embedding, dtype=np.float32)
        if isinstance(raw_embedding, (bytes, bytearray, memoryview)):
            return np.frombuffer(raw_embedding, dtype=np.float32)
        if isinstance(raw_embedding, str):
            parsed = ast.literal_eval(raw_embedding)
            return np.asarray(parsed, dtype=np.float32)
        raise TypeError(f"Unsupported embedding type from DB: {type(raw_embedding)}")

    def load_embeddings_from_db(self) -> Tuple[List[str], np.ndarray]:
        conn = self._connection_factory()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT chunk_id, embedding
            FROM chunk_embeddings
            WHERE embedding IS NOT NULL
            ORDER BY order_index ASC NULLS LAST, chunk_id;
            """
        )

        rows = cur.fetchall()
        conn.close()

        if not rows:
            raise RuntimeError("No embeddings found in chunk_embeddings; build embeddings first.")

        chunk_ids: List[str] = []
        vectors: List[np.ndarray] = []

        for chunk_id, embedding in rows:
            chunk_ids.append(chunk_id)
            vectors.append(self._normalize_embedding(embedding))

        matrix = np.vstack(vectors)
        print(f"Loaded {matrix.shape[0]} vectors (dim = {matrix.shape[1]})")
        return chunk_ids, matrix

    def build_faiss_flatip(self, dim: int) -> faiss.IndexFlatIP:
        print("Building FAISS IndexFlatIP...")
        return faiss.IndexFlatIP(dim)

    def build_faiss_hnsw(self, dim: int) -> faiss.IndexHNSWFlat:
        print("Building FAISS IndexHNSWFlat...")
        index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
        index.hnsw.efConstruction = self.config.hnsw_ef_construction
        index.hnsw.efSearch = self.config.hnsw_ef_search
        return index

    @staticmethod
    def add_vectors(index: faiss.Index, vectors: np.ndarray) -> None:
        """Normalize and add vectors to a FAISS index."""
        print(f"Adding {vectors.shape[0]} vectors to index...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        index.add(normalized)
        print("Vectors added successfully.")

    def save_index(self, index: faiss.Index, chunk_ids: List[str], file_name: str) -> None:
        """Save FAISS index and chunk_ids mapping to disk."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.config.output_dir / f"{file_name}.bin"
        ids_path = self.config.output_dir / f"{file_name}_chunk_ids.json"

        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index → {index_path}")

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(chunk_ids, f, indent=2)

        print(f"Saved chunk_ids mapping → {ids_path}")

    def build_all(self) -> None:
        """Build and persist FlatIP and HNSW indexes."""
        print("Loading embeddings from PostgreSQL...")
        chunk_ids, vectors = self.load_embeddings_from_db()
        dim = vectors.shape[1]

        flat_index = self.build_faiss_flatip(dim)
        self.add_vectors(flat_index, vectors)
        self.save_index(flat_index, chunk_ids, self.config.flat_name)

        hnsw_index = self.build_faiss_hnsw(dim)
        self.add_vectors(hnsw_index, vectors)
        self.save_index(hnsw_index, chunk_ids, self.config.hnsw_name)

        print("\n✅ All FAISS indexes built and saved successfully!")


if __name__ == "__main__":
    FaissBuilder().build_all()
