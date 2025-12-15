import os
import json
from dataclasses import dataclass
from typing import List, Protocol

from openai import OpenAI

from config.settings import OPENAI_API_KEY

# ----------------------------------------------------------
# |                     CONFIGURATION                      |
# ----------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    chunks_path: str = "data/metadata/chunks.json"
    embeddings_dir: str = "data/embeddings/"
    model: str = "text-embedding-3-large"
    batch_size: int = 64  # safe for 221k tokens


class EmbeddingClient(Protocol):
    """Minimal interface for an embedding provider."""

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:  # pragma: no cover - protocol
        ...


class OpenAIEmbeddingClient:
    """Thin adapter over OpenAI for dependency inversion and testing."""

    def __init__(self, api_key: str = OPENAI_API_KEY) -> None:
        self._client = OpenAI(api_key=api_key)

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

def _load_chunks(path: str):
    """Load chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _batch(iterable, batch_size):
    """Yield successive batches from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield i, iterable[i : i + batch_size]


class EmbeddingPipeline:
    """Embeds chunk text and writes batches as descriptive objects."""

    def __init__(
        self,
        config: EmbeddingConfig = EmbeddingConfig(),
        client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config
        self._client = client or OpenAIEmbeddingClient()

    def create_embeddings(self) -> None:
        chunks = _load_chunks(self.config.chunks_path)
        total = len(chunks)
        print(f"Loaded {total} chunks from metadata file.")

        os.makedirs(self.config.embeddings_dir, exist_ok=True)
        print("Starting batch embedding...")

        batch_counter = 1
        for start_index, chunk_batch in _batch(chunks, self.config.batch_size):
            text_batch = [ch["text"] for ch in chunk_batch]
            print(f"→ Embedding batch {batch_counter} ({start_index} – {start_index + len(chunk_batch) - 1})")

            embeddings = self._client.embed(self.config.model, text_batch)
            merged = [
                {
                    "chunk_id": meta["chunk_id"],
                    "doc_id": meta["doc_id"],
                    "order_index": meta["order_index"],
                    "page_number": meta["page_number"],
                    "header": meta["header"],
                    "token_count": meta["token_count"],
                    "text": meta["text"],
                    "embedding": emb,
                }
                for emb, meta in zip(embeddings, chunk_batch)
            ]

            batch_file = os.path.join(
                self.config.embeddings_dir, f"embedding_batch_{batch_counter:03d}.json"
            )

            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)

            print(f"   ✔ Saved: {batch_file}")
            batch_counter += 1

        print("\n✅ All embeddings saved in batches!")
        print(f"Total batches: {batch_counter - 1}")
        print(f"Output directory: {self.config.embeddings_dir}")


if __name__ == "__main__":
    EmbeddingPipeline().create_embeddings()
