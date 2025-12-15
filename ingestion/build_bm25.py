import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from rank_bm25 import BM25Okapi

# Ensure project root is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from storage.db.connection import get_connection


@dataclass(frozen=True)
class BM25Config:
    output_dir: Path = Path("data/bm25_index")
    name: str = "bm25"
    k1: float = 1.5
    b: float = 0.75


class BM25Builder:
    """Builds and persists BM25 artifacts from database chunk text."""

    def __init__(
        self,
        config: BM25Config = BM25Config(),
        connection_factory: Callable[[], object] = get_connection,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory

    def load_chunks(self) -> Tuple[List[str], List[str], Dict[str, Dict]]:
        conn = self._connection_factory()
        cur = None

        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT chunk_id, text, page_number
                FROM chunk_embeddings
                WHERE text IS NOT NULL AND TRIM(text) <> ''
                ORDER BY order_index ASC NULLS LAST, chunk_id;
                """
            )
            rows = cur.fetchall()
        finally:
            try:
                if cur and hasattr(cur, "close"):
                    cur.close()
            except Exception:
                pass
            try:
                if hasattr(conn, "close"):
                    conn.close()
            except Exception:
                pass

        if not rows:
            raise RuntimeError("No chunk text found in chunk_embeddings; load data into the database first.")

        texts: List[str] = []
        chunk_ids: List[str] = []
        metadata: Dict[str, Dict] = {}

        for chunk_id, text, page_number in rows:
            cleaned_text = (text or "").strip()
            if not chunk_id or not cleaned_text:
                continue

            chunk_ids.append(chunk_id)
            texts.append(cleaned_text)
            metadata[chunk_id] = {"text": cleaned_text, "page_number": page_number}

        if not chunk_ids:
            raise ValueError("No valid chunks were found to index.")

        print(f"Loaded {len(chunk_ids)} chunks from database")
        return texts, chunk_ids, metadata

    @staticmethod
    def build_bm25_index(texts: List[str], k1: float, b: float) -> BM25Okapi:
        tokenized_docs = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
        print(f"Built BM25 index with {len(tokenized_docs)} documents")
        return bm25

    @staticmethod
    def save_index(
        bm25: BM25Okapi,
        chunk_ids: List[str],
        metadata: Dict[str, Dict],
        output_dir: Path,
        name: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / f"{name}.pkl"
        ids_path = output_dir / f"{name}_chunk_ids.json"
        metadata_path = output_dir / f"{name}_chunks.json"

        with index_path.open("wb") as f:
            pickle.dump(bm25, f)
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_ids, f, indent=2)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Saved BM25 index to {index_path}")
        print(f"Saved chunk ids to {ids_path}")
        print(f"Saved chunk metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build BM25 index from database chunk text.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BM25Config.output_dir,
        help="Where to write the BM25 index artifacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=BM25Config.name,
        help="Base filename for the index (e.g., bm25 -> bm25.pkl).",
    )
    parser.add_argument("--k1", type=float, default=BM25Config.k1, help="BM25 k1 hyperparameter.")
    parser.add_argument("--b", type=float, default=BM25Config.b, help="BM25 b hyperparameter.")
    return parser.parse_args()


def main():
    args = parse_args()
    builder = BM25Builder(
        BM25Config(
            output_dir=args.output_dir,
            name=args.name,
            k1=args.k1,
            b=args.b,
        )
    )
    print("Loading chunks from database...")
    texts, chunk_ids, metadata = builder.load_chunks()

    print("Building BM25 index...")
    bm25 = builder.build_bm25_index(texts, k1=builder.config.k1, b=builder.config.b)

    print("Saving index artifacts...")
    builder.save_index(bm25, chunk_ids, metadata, builder.config.output_dir, builder.config.name)

    print("Done.")


if __name__ == "__main__":
    main()
