import os
import re
import json

try:
    import tiktoken
except ImportError:
    tiktoken = None

import nltk
from nltk.data import find as nltk_find

# -----------------------------------------------------------
# Utility: token counter
# -----------------------------------------------------------
def count_tokens(text, model="gpt-4o-mini"):
    if tiktoken is None:
        return len(text.split())
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# -----------------------------------------------------------
# Detect headers and subheaders
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
    line = line.strip()
    if not line:
        return False
    return bool(HEADER_PATTERN.match(line))


# -----------------------------------------------------------
# Detect page number markers
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
# Split paragraphs
# -----------------------------------------------------------
def paragraph_split(text: str):
    return [p.strip() for p in text.split("\n") if p.strip()]


# -----------------------------------------------------------
# Sentence split
# -----------------------------------------------------------
def split_sentences(paragraph):
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
# MAIN: SEMANTIC CHUNKER TO JSON (with page numbers)
# -----------------------------------------------------------
def semantic_chunker_to_json_with_pages(file_path: str, max_tokens=350, model="gpt-4o-mini", output_dir="data/chunks"):

    def append_chunk(parts):
        """Append a chunk built from the given parts and advance the index."""
        nonlocal chunk_index
        if not parts:
            return

        full_chunk_text = "\n".join(parts)
        chunks.append({
            "chunk_id": chunk_index,
            "header": current_header,
            "page_number": current_page,
            "text": full_chunk_text,
            "token_count": count_tokens(full_chunk_text, model),
            "order_index": chunk_index
        })
        chunk_index += 1

    # OUTPUT DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name)
    json_output_path = os.path.join(output_dir, f"{safe_name}_chunks.json")

    # READ TEXT
    with open(file_path, "r", encoding="utf-8") as f:
        document = f.read()

    paragraphs = paragraph_split(document)

    chunks = []
    current_chunk = []
    current_header = None
    current_page = 1  # default page
    chunk_index = 1

    for para in paragraphs:

        # Detect page markers → update current_page
        maybe_page = detect_page(para)
        if maybe_page is not None:
            # Close the current chunk before switching to the next page
            if current_chunk:
                append_chunk(current_chunk)
                current_chunk = [current_header] if current_header else []
            current_page = maybe_page
            continue

        # Detect header
        if is_header(para):
            if current_chunk:
                append_chunk(current_chunk)
                current_chunk = []

            current_header = para
            current_chunk.append(para)
            continue

        # Paragraph too large → split by sentences
        if count_tokens(para, model) > max_tokens:
            sentences = split_sentences(para)
            temp = []

            for s in sentences:
                if count_tokens(" ".join(temp + [s]), model) <= max_tokens:
                    temp.append(s)
                else:
                    append_chunk(current_chunk + [" ".join(temp)])
                    current_chunk = [current_header] if current_header else []
                    temp = [s]

            if temp:
                append_chunk(current_chunk + [" ".join(temp)])

            current_chunk = [current_header] if current_header else []
            continue

        # Normal paragraph
        if count_tokens("\n".join(current_chunk + [para]), model) <= max_tokens:
            current_chunk.append(para)
        else:
            # flush
            append_chunk(current_chunk)
            current_chunk = [current_header, para] if current_header else [para]

    # Final flush
    if current_chunk:
        append_chunk(current_chunk)

    # ----------------------------------------------------------
    # SAVE JSON FILE
    # ----------------------------------------------------------
    final_json = {
        "source_file": file_path,
        "source_file_name": os.path.basename(file_path),
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print(f"✨ JSON saved → {json_output_path}")
    print(f"✨ Total Chunks: {len(chunks)}\n")

    return final_json



if __name__ == "__main__":
    txt_folder = "data/cleaned_text"

    for filename in os.listdir(txt_folder):
        if filename.lower().endswith(".txt"):
            txt_path = os.path.join(txt_folder, filename)

            print(f"\n📄 Chunking: {txt_path}")
            semantic_chunker_to_json_with_pages(txt_path)
