-- ================================
-- DOCUMENTS TABLE
-- ================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    file_name TEXT,
    file_path TEXT,
    page_count INTEGER,
    version INTEGER DEFAULT 1,
    ingested_at TIMESTAMPTZ
);

-- ================================
-- CHUNKS TABLE
-- ================================
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    order_index INTEGER,
    page_number INTEGER,
    header TEXT,
    text TEXT NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id        VARCHAR PRIMARY KEY,
    doc_id          VARCHAR,
    order_index     INT,
    page_number     INT,
    header          TEXT,
    token_count     INT,
    text            TEXT,
    embedding       vector(3072)        -- if using pgvector
);