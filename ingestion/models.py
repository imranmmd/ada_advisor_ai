from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ChunkPayload:
    chunk_id: int
    header: Optional[str]
    page_number: Optional[int]
    text: str
    token_count: int
    order_index: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "header": self.header,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count,
            "order_index": self.order_index,
        }


@dataclass(frozen=True)
class ChunkFile:
    source_file: str
    source_file_name: str
    chunks: List[ChunkPayload]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "source_file_name": self.source_file_name,
            "total_chunks": self.total_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }


@dataclass(frozen=True)
class DocumentMetadata:
    doc_id: str
    title: str
    file_name: str
    file_path: str
    page_count: int
    version: int
    ingested_at: str

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page_count": self.page_count,
            "version": self.version,
            "ingested_at": self.ingested_at,
        }


@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    order_index: int
    page_number: Optional[int]
    header: Optional[str]
    token_count: int
    text: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "order_index": self.order_index,
            "page_number": self.page_number,
            "header": self.header,
            "token_count": self.token_count,
            "text": self.text,
        }


__all__ = ["ChunkPayload", "ChunkFile", "DocumentMetadata", "ChunkMetadata"]
