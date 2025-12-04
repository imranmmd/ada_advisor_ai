import os
import json
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
import time


# ===============================================================
# 1. LOAD EMBEDDING BATCHES FROM JSON
# ===============================================================
def load_embedding_batches(batch_files):
    """
    Load multiple embedding batch JSON files
    batch_files: list of file paths to JSON files
    Returns: combined chunks and chunk_ids
    """
    all_chunks = []
    chunk_ids = []
    
    for batch_file in batch_files:
        print(f"Loading batch from {batch_file}...")
        
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        # Assuming structure: {"chunks": [{"chunk_id": "...", "text": "..."}]}
        if isinstance(batch_data, dict) and 'chunks' in batch_data:
            chunks = batch_data['chunks']
        elif isinstance(batch_data, list):
            chunks = batch_data
        else:
            raise ValueError(f"Unexpected JSON structure in {batch_file}")
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', chunk.get('id', f'chunk_{len(chunk_ids)}'))
            text = chunk.get('text', '')
            
            all_chunks.append(chunk)
            chunk_ids.append(chunk_id)
        
        print(f"  Loaded {len(chunks)} chunks from {batch_file}")
    
    print(f"\nTotal chunks loaded: {len(chunk_ids)}\n")
    
    return all_chunks, chunk_ids


# ===============================================================
# 2. BUILD BM25 INDEX
# ===============================================================
def build_bm25_index(all_chunks, k1=1.5, b=0.75):
    """
    Build BM25 index from text chunks
    k1, b: BM25 hyperparameters
    """
    print("Building BM25 index...")
    
    # Tokenize documents
    tokenized_docs = []
    for chunk in all_chunks:
        text = chunk.get('text', '').lower()
        tokens = text.split()  # Simple whitespace tokenization
        tokenized_docs.append(tokens)
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    
    print(f"✅ BM25 index built with {len(tokenized_docs)} documents\n")
    return bm25, tokenized_docs


# ===============================================================
# 3. SEARCH WITH BM25
# ===============================================================
def search_bm25(bm25, tokenized_docs, query, top_k=5):
    """
    Search using BM25
    Returns: list of (doc_index, score) tuples
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k results
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(int(idx), float(scores[idx])) for idx in top_indices]
    
    return results


# ===============================================================
# 4. TEST RETRIEVAL
# ===============================================================
def test_retrieval(all_chunks, bm25, tokenized_docs):
    """
    Test BM25 retrieval with interactive mode
    """
    print("="*70)
    print("BM25 SEARCH - INTERACTIVE MODE")
    print("="*70)
    print("Type your queries to search (type 'quit' to exit):\n")
    
    while True:
        user_query = input("📝 Enter query: ").strip()
        
        if user_query.lower() == 'quit':
            print("Exiting...")
            break
        
        if not user_query:
            print("⚠️  Query cannot be empty. Try again.\n")
            continue
        
        print(f"\n{'─'*70}")
        print(f"Query: '{user_query}'")
        print(f"{'─'*70}")
        
        bm25_results = search_bm25(bm25, tokenized_docs, user_query, top_k=5)
        
        if not bm25_results or bm25_results[0][1] == 0:
            print("❌ No relevant results found.")
        else:
            for rank, (idx, score) in enumerate(bm25_results, 1):
                chunk = all_chunks[idx]
                text = chunk.get('text', '')[:150]
                if len(chunk.get('text', '')) > 150:
                    text += "..."
                print(f"  {rank}. [Score: {score:.4f}] {text}")
        
        print()


# ===============================================================
# 5. SAVE BM25 INDEX
# ===============================================================
def save_bm25_index(bm25, chunk_ids, file_name="bm25"):
    """
    Save BM25 index and chunk IDs (similar to FAISS structure)
    """
    os.makedirs("data/bm25_index", exist_ok=True)
    
    # Save BM25 (using pickle)
    import pickle
    index_path = f"data/bm25_index/{file_name}.pkl"
    with open(index_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"✅ Saved BM25 index → {index_path}")
    
    # Save chunk IDs mapping
    ids_path = f"data/bm25_index/{file_name}_chunk_ids.json"
    with open(ids_path, 'w') as f:
        json.dump(chunk_ids, f, indent=2)
    print(f"✅ Saved chunk_ids mapping → {ids_path}")


# ===============================================================
# 6. MAIN PIPELINE
# ===============================================================
def main():
    print("\n" + "="*70)
    print("BM25 INDEX BUILDER & TESTER")
    print("="*70 + "\n")
    
    # Specify your embedding batch files here
    batch_files = [
        r"BM25\embedding_batch_001.json",  # Replace with your file paths
        r"BM25\embedding_batch_002.json"
    ]
    
    # Check if files exist
    missing_files = [f for f in batch_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("Please ensure the embedding batch JSON files exist.")
        return
    
    # Load embeddings from batches
    all_chunks, chunk_ids = load_embedding_batches(batch_files)
    
    # Build BM25 index
    bm25, tokenized_docs = build_bm25_index(all_chunks)
    
    # Build BM25 index
    bm25, tokenized_docs = build_bm25_index(all_chunks)
    
    # Save index
    save_bm25_index(bm25, chunk_ids, file_name="bm25")
    
    print("\n" + "="*70)
    print("✅ BM25 index built and saved successfully!")
    print("="*70 + "\n")
    
    # Test retrieval
    test_retrieval(all_chunks, bm25, tokenized_docs)


# ===============================================================
# ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    main()