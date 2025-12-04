import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

EMBEDDING_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/bm25_index")
INDEX_NAME = "bm25"


def load_chunks(embedding_dir: Path) -> Tuple[List[str], List[str], Dict[str, Dict]]:
    """
    Load chunk text and ids from embedding batch JSON files.
    Returns raw texts, chunk_ids, and lightweight metadata for fallback lookup.
    """
    batch_files = sorted(embedding_dir.glob("embedding_batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No embedding batches found in {embedding_dir}")

    texts: List[str] = []
    chunk_ids: List[str] = []
    metadata: Dict[str, Dict] = {}

    for batch_file in batch_files:
        with batch_file.open("r", encoding="utf-8") as f:
            batch = json.load(f)

        if not isinstance(batch, list):
            raise ValueError(f"Unexpected structure in {batch_file}, expected a list")

        for item in batch:
            chunk_id = item.get("chunk_id")
            text = (item.get("text") or "").strip()
            page_number = item.get("page_number")
            if not chunk_id or not text:
                continue

            chunk_ids.append(chunk_id)
            texts.append(text)
            metadata[chunk_id] = {"text": text, "page_number": page_number}

        print(f"Loaded {len(batch)} chunks from {batch_file.name}")

    if not chunk_ids:
        raise ValueError("No valid chunks were found to index.")

    print(f"Total chunks loaded: {len(chunk_ids)}")
    return texts, chunk_ids, metadata


def build_bm25_index(texts: List[str], k1: float, b: float) -> BM25Okapi:
    """Tokenize text and build BM25 index."""
    tokenized_docs = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    print(f"Built BM25 index with {len(tokenized_docs)} documents")
    return bm25


def save_index(
    bm25: BM25Okapi,
    chunk_ids: List[str],
    metadata: Dict[str, Dict],
    output_dir: Path,
    name: str,
) -> None:
    """Persist BM25 index, chunk ids, and lightweight metadata."""
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
    parser = argparse.ArgumentParser(description="Build BM25 index from embedding batches.")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=EMBEDDING_DIR,
        help="Directory containing embedding_batch_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the BM25 index artifacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=INDEX_NAME,
        help="Base filename for the index (e.g., bm25 -> bm25.pkl).",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 hyperparameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b hyperparameter.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading chunks...")
    texts, chunk_ids, metadata = load_chunks(args.embedding_dir)

    print("Building BM25 index...")
    bm25 = build_bm25_index(texts, k1=args.k1, b=args.b)

    print("Saving index artifacts...")
    save_index(bm25, chunk_ids, metadata, args.output_dir, args.name)

    print("Done.")


if __name__ == "__main__":
    main()
