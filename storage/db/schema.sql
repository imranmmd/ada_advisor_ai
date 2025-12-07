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
    chunk_id        TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    doc_id          TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
    order_index     INT,
    page_number     INT,
    header          TEXT,
    token_count     INT,
    text            TEXT,
    embedding       vector(3072)        -- if using pgvector
);
-- ================================
-- RETRIEVAL EVENTS TABLE
-- ================================
CREATE TABLE IF NOT EXISTS retrieval_events (
    event_id        TEXT PRIMARY KEY,
    query_text      TEXT NOT NULL,
    query_embedding vector(3072),                 -- optional: store query embedding
    retrieved_chunk_ids TEXT[],                   -- array of chunk_ids in ranked order
    top_k           INTEGER,
    scores          FLOAT8[],                     -- cosine similarity scores
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
-- ================================
-- CHAT HISTORY TABLE
-- ================================
CREATE TABLE IF NOT EXISTS chat_history (
    message_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,                 -- group messages into a session
    role            TEXT CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    retrieval_event_id TEXT REFERENCES retrieval_events(event_id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);