import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

# Default locations and parameters for BM25 retrieval
BM25_INDEX_DIR = Path("data/bm25_index")
BM25_INDEX_NAME = "bm25"
BM25_TOP_K = 5


def load_bm25_index(
    index_name: str = BM25_INDEX_NAME, index_dir: Optional[Path] = None
) -> Tuple[BM25Okapi, List[str], Dict[str, Dict]]:
    """Load BM25 pickle plus chunk ids and cached metadata."""
    index_dir = Path(index_dir) if index_dir else BM25_INDEX_DIR
    index_path = index_dir / f"{index_name}.pkl"
    ids_path = index_dir / f"{index_name}_chunk_ids.json"
    metadata_path = index_dir / f"{index_name}_chunks.json"

    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing BM25 index or id file. Expected {index_path} and {ids_path}"
        )

    with index_path.open("rb") as f:
        bm25 = pickle.load(f)
    with ids_path.open("r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    metadata: Dict[str, Dict] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    print(f"Loaded BM25 index {index_name} with {len(chunk_ids)} documents")
    return bm25, chunk_ids, metadata


def search_bm25(
    bm25: BM25Okapi, chunk_ids: List[str], query: str, top_k: int = BM25_TOP_K
):
    """Search BM25 index."""
    if hasattr(bm25, "corpus_size") and bm25.corpus_size != len(chunk_ids):
        raise ValueError(
            f"BM25 index corpus size {bm25.corpus_size} does not match "
            f"{len(chunk_ids)} chunk ids."
        )

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        cid = chunk_ids[int(idx)]
        results.append(
            {
                "chunk_id": cid,
                "score": float(scores[int(idx)]),
                "text": None,
                "page_number": None,
                "source": "bm25",
            }
        )
    return results
