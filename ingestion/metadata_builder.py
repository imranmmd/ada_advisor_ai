import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback if tiktoken is unavailable
    tiktoken = None

from ingestion.models import ChunkMetadata, DocumentMetadata

@dataclass(frozen=True)
class MetadataConfig:
    cleaned_text_dir: str = "data/cleaned_text"
    chunks_dir: str = "data/chunks"
    metadata_dir: str = "data/metadata"
    document_metadata_path: str = os.path.join("data/metadata", "documents.json")
    chunk_metadata_path: str = os.path.join("data/metadata", "chunks.json")

PAGE_MARKER_PATTERN = re.compile(
    r"^===\s*PAGE\s+(\d+)\s*===\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def slugify(value: str) -> str:
    """Create a filesystem/ID friendly slug."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "untitled"


def title_from_filename(base_name: str) -> str:
    """Derive a human-friendly title from a filename stem."""
    spaced = re.sub(r"[_-]+", " ", base_name).strip()
    return spaced.title() if spaced else base_name


def count_pages_from_text(text: str) -> int:
    """Count pages based on normalized page markers inserted during cleaning."""
    matches = [int(m.group(1)) for m in PAGE_MARKER_PATTERN.finditer(text)]
    if matches:
        return max(matches)
    return 1 if text.strip() else 0


def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken when available; fallback to whitespace tokens."""
    if tiktoken is None:
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class MetadataBuilder:
    """Builds document and chunk metadata as descriptive objects."""

    def __init__(self, config: MetadataConfig = MetadataConfig()) -> None:
        self.config = config

    def build_documents(self, cleaned_dir: str | None = None) -> List[DocumentMetadata]:
        cleaned_dir = cleaned_dir or self.config.cleaned_text_dir
        documents: List[DocumentMetadata] = []
        if not os.path.isdir(cleaned_dir):
            print(f"⚠️ Cleaned text directory not found: {cleaned_dir}")
            return documents

        for filename in sorted(os.listdir(cleaned_dir)):
            if not filename.lower().endswith(".txt"):
                continue

            file_path = os.path.join(cleaned_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            base = os.path.splitext(filename)[0]
            slug = slugify(base)
            documents.append(
                DocumentMetadata(
                    doc_id=f"doc_{slug}",
                    title=title_from_filename(base),
                    file_name=filename,
                    file_path=file_path,
                    page_count=count_pages_from_text(text),
                    version=1,
                    ingested_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                )
            )

        return documents

    def build_chunks(self, chunks_dir: str | None = None) -> List[ChunkMetadata]:
        chunks_dir = chunks_dir or self.config.chunks_dir
        chunks_metadata: List[ChunkMetadata] = []

        if not os.path.isdir(chunks_dir):
            print(f"⚠️ Chunk directory not found: {chunks_dir}")
            return chunks_metadata

        for filename in sorted(os.listdir(chunks_dir)):
            if not filename.endswith("_chunks.json"):
                continue

            chunk_file_path = os.path.join(chunks_dir, filename)
            with open(chunk_file_path, "r", encoding="utf-8") as f:
                chunk_payload = json.load(f)

            base = re.sub(r"_chunks$", "", os.path.splitext(filename)[0])
            slug = slugify(base)
            doc_id = f"doc_{slug}"

            for idx, chunk in enumerate(chunk_payload.get("chunks", []), start=1):
                chunk_text = chunk.get("text", "")
                token_cnt = chunk.get("token_count") or token_count(chunk_text)
                order_index = chunk.get("order_index", idx)

                chunks_metadata.append(
                    ChunkMetadata(
                        chunk_id=f"chunk_{slug}_{idx:04d}",
                        doc_id=doc_id,
                        order_index=order_index,
                        page_number=chunk.get("page_number"),
                        header=chunk.get("header"),
                        text=chunk_text,
                        token_count=token_cnt,
                    )
                )

        return chunks_metadata

    @staticmethod
    def write_json(path: str, payload) -> None:
        """Write the given payload as JSON to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    builder = MetadataBuilder()
    documents = builder.build_documents()
    chunk_metadata = builder.build_chunks()

    MetadataBuilder.write_json(
        builder.config.document_metadata_path, [doc.to_dict() for doc in documents]
    )
    MetadataBuilder.write_json(
        builder.config.chunk_metadata_path, [chunk.to_dict() for chunk in chunk_metadata]
    )

    print(f"✅ Saved {len(documents)} documents → {builder.config.document_metadata_path}")
    print(f"✅ Saved {len(chunk_metadata)} chunks → {builder.config.chunk_metadata_path}")


if __name__ == "__main__":
    main()
