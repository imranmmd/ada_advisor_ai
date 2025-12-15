# ada_advisor_ai

# AI Student Advisor

## Overview
The **AI Student Advisor** is a Retrieval-Augmented Generation (RAG) system that answers student questions using institutional documents (syllabi, regulations, lecture materials). The system prioritizes **traceability, modularity, and correctness** by strictly separating ingestion, storage, retrieval, and interface concerns.

The solution combines **semantic retrieval (FAISS)**, **lexical retrieval (BM25)**, and **LLM-based generation**, producing grounded answers with controllable context and citations.

---

## High-Level Architecture
The system is organized into four independent stages:

1. **Ingestion** – preprocess and index documents  
2. **Storage** – persist structured data and indexes  
3. **RAG Core** – retrieve, fuse, and generate answers  
4. **Interface** – expose the system via a Telegram bot  

This separation ensures that changes in one stage do not propagate to others.

---

## Component Responsibilities

### 1. Ingestion Layer (`ingestion/`)
**Purpose:** Offline preprocessing of knowledge sources.

Responsibilities:
- Extract and clean text from PDFs, Word, and HTML
- Perform semantic chunking with headers and page numbers
- Generate structured metadata
- Persist documents and chunks to PostgreSQL
- Generate and store embeddings
- Build FAISS and BM25 indexes

This layer contains **no runtime query logic**.

---

### 2. Storage Layer (`storage/`)
**Purpose:** Centralized persistence and retrieval.

Responsibilities:
- PostgreSQL schema initialization and connection handling
- Repository-based data access (`DocumentRepository`, `ChunkRepository`, etc.)
- Loading FAISS and BM25 indexes from disk
- Managing conversation history and memory trimming

Business logic and ranking decisions are intentionally excluded.

---

### 3. RAG Core (`rag_core/`)
**Purpose:** Core reasoning and retrieval pipeline.

Responsibilities:
- Generate query embeddings via a model-agnostic interface
- Retrieve relevant chunks using:
  - Semantic retrieval (FAISS)
  - Lexical retrieval (BM25)
  - Hybrid fusion (weighted / RRF)
- Orchestrate prompt construction and LLM calls
- Manage caching of embeddings and prompts
- Format citations and source references

All decision-making logic resides here.

---

### 4. Interface Layer (`bot/`)
**Purpose:** User interaction only.

Responsibilities:
- Handle Telegram updates (polling or webhook)
- Parse and normalize user messages
- Call the RAG pipeline
- Format responses for Telegram (Markdown / HTML)
- Handle errors, fallbacks, and logging

The interface layer does not perform retrieval or generation.

---

## Data Flow Summary
1. Documents are ingested, cleaned, chunked, and embedded.
2. Chunks and metadata are stored in PostgreSQL.
3. FAISS and BM25 indexes are built and loaded.
4. A user query triggers hybrid retrieval.
5. Retrieved context is injected into an LLM prompt.
6. A grounded answer is returned to the user.

---

## SOLID Design Principles

### Single Responsibility Principle (SRP)
Each class has exactly one responsibility:
- Repositories handle persistence only
- Retrievers handle ranking only
- The orchestrator coordinates the pipeline only

---

### Open–Closed Principle (OCP)
The system is extensible without modification:
- New retrievers can be added without changing the orchestrator
- Embedding models can be swapped via abstraction

---

### Liskov Substitution Principle (LSP)
All retrievers follow a common retrieval contract, allowing interchangeable use within the hybrid retriever.

---

### Interface Segregation Principle (ISP)
Interfaces are minimal and role-specific:
- Repositories expose only database operations
- Retrievers expose only retrieval behavior

No component depends on unused methods.

---

### Dependency Inversion Principle (DIP)
High-level logic depends on abstractions:
- The RAG orchestrator depends on retriever interfaces, not FAISS or BM25 directly
- Storage access is isolated behind repositories

This enables testing, mocking, and future replacement.

---

### How to run?
- Set environment: `OPENAI_API_KEY` (existing RAG flow) and `TELEGRAM_BOT_TOKEN`.
- Optional: `TELEGRAM_WEBHOOK_URL`/`TELEGRAM_WEBHOOK_PATH` to enable webhooks (defaults to polling), `TELEGRAM_ADMIN_CHAT_ID` for error alerts, `TELEGRAM_RAG_TOP_K` to override retrieval depth.
- Run locally with polling: `python -m rag_core.bot.main`.
- For webhooks: set the webhook URL envs, expose the chosen port, and start the bot (`PORT`/`TELEGRAM_WEBHOOK_PORT` default to 8443).
