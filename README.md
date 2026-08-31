# PDF Q&A System

A local PDF Q&A system built with FastAPI, ChromaDB, and OpenAI-compatible LLMs.

## PDF Chosen

- **Document:** `Data/` directory (scans all PDFs recursively)
- **Size:** ~47 pages
- **Why:** Realistic use case for Q&A over organizational reports with structured content.

## Architecture

| Component     | Choice                                      | Reason |
|---------------|---------------------------------------------|--------|
| Embeddings    | `sentence-transformers/all-mpnet-base-v2`  | High-quality dense embeddings, open-source. |
| Vector Store  | ChromaDB                                    | Persistent storage, easy LangChain integration, lightweight. |
| LLM           | `openai/gpt-oss-120b`                       | OpenAI-compatible local/server model. |
| API           | FastAPI                                     | Fast, async-capable, automatic OpenAPI docs. |

## Files

```
src/
  __init__.py
  config.py     - Settings (paths, models, constants)
  llm.py       - LLM client + query expansion
  retriever.py - Hybrid BM25 + dense retrieval (RRF)
  reranker.py  - BAAI/bge-reranker-base
  utils.py     - Helper functions
  ingest.py   - PDF extraction + chunking

main.py         - FastAPI entry
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Add OPENAI_API_KEY or configure local LLM endpoint in .env

# Ingest PDFs
python -m src.ingest

# Start server
python main.py
```

## API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the goal of the annual report?"}'
```

**Request:** `{"question": "..."}`  
**Response:** `{"answer": "...", "sources": [...]}`

## System Design

### Chunking Strategy

- **Method:** RecursiveCharacterTextSplitter
- **Chunk size:** 500 characters
- **Overlap:** 80 characters
- **Separators:** `["\n\n", "\n", " ", ""]` (preserve paragraphs, then lines, then sentences)

**Rationale:** 500 chars balances context coverage with prompt token limits. 80 overlap reduces boundary cuts.

### Vector Store Choice

ChromaDB selected because:
- Persistent on-disk storage (survives restarts)
- Simple setup, no external service needed
- Adequate for small datasets (<1000 chunks)

For larger datasets: consider Weaviate (HNSW indexing) or Pinecone (managed, scalable).

### Hybrid Retrieval

- **Dense retrieval:** `all-mpnet-base-v2` embeddings + MMR search
- **Sparse retrieval:** BM25Okapi for exact keyword matching
- **Fusion:** Reciprocal Rank Fusion (RRF) with `k=60`
- **Parameters:** `RETRIEVAL_K=35`, `RETRIEVAL_FETCH_K=120`, `lambda_mult=0.5`

### Reranking

- **Model:** BAAI/bge-reranker-base
- **Top-K:** 12
- **Truncate:** 512 characters

### Query Expansion

For queries >4 words, generates 5 sub-queries to improve recall.

## Optimizations Applied

1. Global caching: Retriever and LLM client initialized once
2. Hybrid retrieval: BM25 + dense embeddings fused with RRF
3. MMR retrieval: Balances relevance vs diversity
4. Reranking: BGE reranker for final top-12 results
5. Query expansion: For queries >4 words, generates 5 sub-queries

## Future Improvements

1. HNSW indexing — For >10k vectors, switch Chroma to HNSW
2. Async endpoints — Use FastAPI async for concurrent requests
3. Query caching — Cache LLM responses for repeated questions
4. Cold start — Preload embeddings at startup

## Honest Evaluation

### Q&A Test Set

1. What is the goal of the annual report?
2. How many submissions were received in 2023?
3. What are the main partnerships mentioned?
4. What sustainability or outreach efforts are described?
5. What metrics are reported for usage growth?
6. What fiscal year does the report cover?
7. Which sections describe community engagement or diversity?
8. What funding or operational challenges are referenced?

### What Breaks

| Issue                              | Cause                                | Fix |
|------------------------------------|--------------------------------------|-----|
| Questions 6-8 return unrelated answers | Content not in retrieved context    | Increase chunk size, add more chunks (k>35). |
| Query expansion adds latency       | Extra LLM call per query             | Cache expansions. |
| Reranking slow on long docs        | Truncation at 512 chars              | Increase truncation limit. |

### Root Cause Analysis

1. Coverage gaps: Some sections (governance, diversity) may be sparse in PDF
2. Retrieval recall: k=35 may miss low-ranking relevant chunks
3. Chunk boundaries: 500 chars may split tables mid-row

### What I'd Fix First

1. Increase fetch_k to 200 — Better recall for edge cases
2. Add section-aware chunking — Use PDF headings as chunk boundaries
3. Hybrid search — Combine keyword (BM25) with semantic search
