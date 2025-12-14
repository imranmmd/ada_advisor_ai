import os
import json
from openai import OpenAI
from config.settings import OPENAI_API_KEY
# ----------------------------------------------------------
# |                     CONFIGURATION                      |
# ----------------------------------------------------------
CHUNKS_PATH = "data/metadata/chunks.json"
EMBEDDINGS_DIR = "data/embeddings/"

MODEL = "text-embedding-3-large"
BATCH_SIZE = 64  # safe for 221k tokens

client = OpenAI()
client.api_key = OPENAI_API_KEY

# ----------------------------------------------------------
# |                       LOAD CHUNKS                      |
# ----------------------------------------------------------
def load_chunks():
    """Load chunks from JSON file."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------
# |                BATCH SPLITTER                          |
# ----------------------------------------------------------
def batch(iterable, batch_size):
    """Yield successive batches from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield i, iterable[i : i + batch_size]

# ----------------------------------------------------------
# |                     MAIN PIPELINE                      |
# ----------------------------------------------------------
def create_embeddings():
    """Create embeddings for document chunks and save in batches."""
    chunks = load_chunks()
    total = len(chunks)

    print(f"Loaded {total} chunks from metadata file.")

    # ensure directory exists
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print("Starting batch embedding...")

    batch_counter = 1

    for start_index, chunk_batch in batch(chunks, BATCH_SIZE):

        # --------------------------
        # 1. Prepare text inputs
        # --------------------------
        text_batch = [ch["text"] for ch in chunk_batch]

        print(f"→ Embedding batch {batch_counter} "
              f"({start_index} – {start_index + len(chunk_batch) - 1})")

        # --------------------------
        # 2. Call OpenAI Embeddings
        # --------------------------
        response = client.embeddings.create(
            model=MODEL,
            input=text_batch
        )

        embeddings = [item.embedding for item in response.data]

        # --------------------------
        # 3. Merge with metadata
        # --------------------------
        merged = []

        for emb, meta in zip(embeddings, chunk_batch):
            merged.append({
                "chunk_id": meta["chunk_id"],
                "doc_id": meta["doc_id"],
                "order_index": meta["order_index"],
                "page_number": meta["page_number"],
                "header": meta["header"],
                "token_count": meta["token_count"],
                "text": meta["text"],
                "embedding": emb
            })

        # --------------------------
        # 4. Write batch file
        # --------------------------
        batch_file = os.path.join(
            EMBEDDINGS_DIR,
            f"embedding_batch_{batch_counter:03d}.json"
        )

        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        print(f"   ✔ Saved: {batch_file}")

        batch_counter += 1

    print("\n✅ All embeddings saved in batches!")
    print(f"Total batches: {batch_counter - 1}")
    print(f"Output directory: {EMBEDDINGS_DIR}")


# ----------------------------------------------------------
# |                   ENTRY POINT                          |
# ----------------------------------------------------------
if __name__ == "__main__":
    create_embeddings()
