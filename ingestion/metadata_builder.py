import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback if tiktoken is unavailable
    tiktoken = None

CLEANED_TEXT_DIR = "data/cleaned_text"
CHUNKS_DIR = "data/chunks"
METADATA_DIR = "data/metadata"
DOCUMENT_METADATA_PATH = os.path.join(METADATA_DIR, "documents.json")
CHUNK_METADATA_PATH = os.path.join(METADATA_DIR, "chunks.json")

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


def build_document_metadata(cleaned_dir: str = CLEANED_TEXT_DIR) -> List[Dict[str, Any]]:
    documents = []

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

        documents.append({
            "doc_id": f"doc_{slug}",
            "title": title_from_filename(base),
            "file_name": filename,
            "file_path": file_path,
            "page_count": count_pages_from_text(text),
            "version": 1,
            "ingested_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        })

    return documents


def build_chunk_metadata(chunks_dir: str = CHUNKS_DIR) -> List[Dict[str, Any]]:
    chunks_metadata = []

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

            chunks_metadata.append({
                "chunk_id": f"chunk_{slug}_{idx:04d}",
                "doc_id": doc_id,
                "order_index": order_index,
                "page_number": chunk.get("page_number"),
                "header": chunk.get("header"),
                "text": chunk_text,
                "token_count": token_cnt,
            })

    return chunks_metadata


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    documents = build_document_metadata()
    chunk_metadata = build_chunk_metadata()

    write_json(DOCUMENT_METADATA_PATH, documents)
    write_json(CHUNK_METADATA_PATH, chunk_metadata)

    print(f"✅ Saved {len(documents)} documents → {DOCUMENT_METADATA_PATH}")
    print(f"✅ Saved {len(chunk_metadata)} chunks → {CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    main()
