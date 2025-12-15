import os
import re
import json
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None

import nltk
from nltk.data import find as nltk_find

from ingestion.models import ChunkFile, ChunkPayload


@dataclass(frozen=True)
class ChunkingConfig:
    """Immutable settings for semantic chunking."""

    max_tokens: int = 350
    model: str = "gpt-4o-mini"
    output_dir: str = "data/chunks"
    token_counter: Callable[[str, str], int] = None  # type: ignore

# -----------------------------------------------------------
# |                   Utility: token counter                |
# -----------------------------------------------------------
def count_tokens(text, model="gpt-4o-mini"):
    """Count the number of tokens in a given text for a specific model."""
    if tiktoken is None:
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# -----------------------------------------------------------
# |               Detect headers and subheaders             |
# -----------------------------------------------------------
HEADER_PATTERN = re.compile(
    r"""
    ^(\s*
        (
            ([A-Z][A-Z ]{2,})                 
            |(\d+(\.\d+)*\s+)                 
            |([A-Z]\.)                        
        )
    )
    """,
    re.VERBOSE
)

def is_header(line: str) -> bool:
    """Detect if a line is a header based on formatting patterns."""
    line = line.strip()
    if not line:
        return False
    return bool(HEADER_PATTERN.match(line))


# -----------------------------------------------------------
# |                 Detect page number markers              |
# -----------------------------------------------------------
PAGE_PATTERN = re.compile(
    r"""(?ix)                              # ignore case, verbose
    (?:                                    # common formats:
        page[:\-\| ]*\s*(\d+)              #   "Page 2", "PAGE: 3", "page|4"
        |                                  #
        \[page[:\-\| ]*\s*(\d+)\]          #   "[PAGE:5]"
        |                                  #
        ={2,}\s*page\s*(\d+)\s*={2,}       #   "=== PAGE 6 ==="
        |                                  #
        (\d+)\s*\|\s*p\s*a\s*g\s*e         #   "7 | P a g e"
    )
    """
)

def detect_page(line: str):
    """Detect page number from a line, if any."""
    normalized = line.strip()
    if not normalized:
        return None

    # Remove internal whitespace so that "P a g e" can be matched
    collapsed = re.sub(r"\s+", "", normalized)

    m = PAGE_PATTERN.search(collapsed)
    if m:
        for g in m.groups():
            if g:
                return int(g)

    # Fallback: lines that are just a small integer (many PDFs export page
    # numbers as bare digits). Keep it conservative to avoid list items.
    if re.fullmatch(r"\d{1,4}", collapsed):
        return int(collapsed)

    return None


# -----------------------------------------------------------
# |                       Split paragraphs                  |
# -----------------------------------------------------------
def paragraph_split(text: str):
    """Split text into paragraphs."""
    return [p.strip() for p in text.split("\n") if p.strip()]


# -----------------------------------------------------------
# |                            Sentence split               |
# -----------------------------------------------------------
def split_sentences(paragraph):
    """Split a paragraph into sentences using NLTK, with fallback."""
    try:
        nltk_find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            # Fallback to naive split if download is unavailable (e.g., offline)
            return re.split(r"(?<=[.!?])\s+", paragraph)
    return nltk.sent_tokenize(paragraph)


# -----------------------------------------------------------
# |      MAIN: SEMANTIC CHUNKER TO JSON (with page numbers) |
# -----------------------------------------------------------
class SemanticChunker:
    """Class-based semantic chunker to support reuse and testing."""

    def __init__(
        self,
        config: ChunkingConfig = ChunkingConfig(),
        token_counter: Callable[[str, str], int] | None = None,
    ) -> None:
        self.config = config
        self._count_tokens = token_counter or config.token_counter or count_tokens

    def chunk_file(self, file_path: str) -> ChunkFile:
        paragraphs = paragraph_split(self._read_file(file_path))

        chunks: List[ChunkPayload] = []
        current_chunk: List[str] = []
        current_header: Optional[str] = None
        current_page = 1
        chunk_index = 1

        def append_chunk(parts: List[str]) -> None:
            nonlocal chunk_index
            if not parts:
                return
            full_chunk_text = "\n".join(parts)
            payload = ChunkPayload(
                chunk_id=chunk_index,
                header=current_header,
                page_number=current_page,
                text=full_chunk_text,
                token_count=self._count_tokens(full_chunk_text, self.config.model),
                order_index=chunk_index,
            )
            chunks.append(payload)
            chunk_index += 1

        for para in paragraphs:
            maybe_page = detect_page(para)
            if maybe_page is not None:
                if current_chunk:
                    append_chunk(current_chunk)
                    current_chunk = [current_header] if current_header else []
                current_page = maybe_page
                continue

            if is_header(para):
                if current_chunk:
                    append_chunk(current_chunk)
                    current_chunk = []
                current_header = para
                current_chunk.append(para)
                continue

            para_tokens = self._count_tokens(para, self.config.model)
            if para_tokens > self.config.max_tokens:
                sentences = split_sentences(para)
                temp: List[str] = []
                for s in sentences:
                    if self._count_tokens(" ".join(temp + [s]), self.config.model) <= self.config.max_tokens:
                        temp.append(s)
                    else:
                        append_chunk(current_chunk + [" ".join(temp)])
                        current_chunk = [current_header] if current_header else []
                        temp = [s]
                if temp:
                    append_chunk(current_chunk + [" ".join(temp)])
                current_chunk = [current_header] if current_header else []
                continue

            if self._count_tokens("\n".join(current_chunk + [para]), self.config.model) <= self.config.max_tokens:
                current_chunk.append(para)
            else:
                append_chunk(current_chunk)
                current_chunk = [current_header, para] if current_header else [para]

        if current_chunk:
            append_chunk(current_chunk)

        return ChunkFile(
            source_file=file_path,
            source_file_name=os.path.basename(file_path),
            chunks=chunks,
        )

    def write_json(self, chunk_file: ChunkFile) -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(chunk_file.source_file))[0]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name)
        json_output_path = os.path.join(self.config.output_dir, f"{safe_name}_chunks.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(chunk_file.to_dict(), f, indent=4, ensure_ascii=False)
        return json_output_path

    @staticmethod
    def _read_file(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def semantic_chunker_to_json_with_pages(file_path: str, max_tokens=350, model="gpt-4o-mini", output_dir="data/chunks"):
    """
    Chunk a text document into semantic chunks with page numbers and headers.
    Saves the chunks as a JSON file in the specified output directory.
    """
    chunker = SemanticChunker(ChunkingConfig(max_tokens=max_tokens, model=model, output_dir=output_dir))
    chunk_file = chunker.chunk_file(file_path)
    json_output_path = chunker.write_json(chunk_file)
    print(f"✨ JSON saved → {json_output_path}")
    print(f"✨ Total Chunks: {chunk_file.total_chunks}\n")
    return chunk_file.to_dict()



if __name__ == "__main__":
    txt_folder = "data/cleaned_text"

    for filename in os.listdir(txt_folder):
        if filename.lower().endswith(".txt"):
            txt_path = os.path.join(txt_folder, filename)

            print(f"\n📄 Chunking: {txt_path}")
            semantic_chunker_to_json_with_pages(txt_path)
