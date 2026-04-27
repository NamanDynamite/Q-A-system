 PDF Q&A System

A local PDF Q&A system built with FastAPI, ChromaDB, and Groq for LLM inference.

 PDF Chosen

 Document: Data/2023_arXiv_annual_report.pdf
 Size: 47 pages
 Why: Publicly available annual report with structured content (partnerships, sustainability, outreach, financial data). Realistic use case for Q&A over organizational reports.

 Architecture

 Component                          |Choice                                 |Reason 

 Embeddings    |sentence-transformers/all-mpnet-base-v2    |High-quality dense embeddings, open-source. 
 Vector Store  |ChromaDB                     |Persistent storage, easy LangChain integration, lightweight. 
 LLM           |Groq llama-3.3-70b-versatile                    |Fast inference via Groq API. 
 API           |FastAPI                                      |Fast, async-capable, automatic OpenAPI docs. 

 Files

src/
  __init__.py
  config.py     - Settings (paths, models, constants)
  llm.py       - Groq client + query expansion
  retriever.py - ChromaDB retrieval
  reranker.py  - Bge-reranker-base
  utils.py     - Helper functions
  ingest.py   - PDF extraction + chunking

main.py         - FastAPI entry



 Setup

bash
 Install dependencies
pip install -r requirements.txt

 Set environment variables
 Add GROQ_API_KEY to .env file

 Ingest PDF
python -m src.ingest

 Start server
python main.py


 API

bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the goal of the arXiv annual report?"}'


Request: {"question": "..."}
Response: {"answer": "...", "sources": [...]}



 System Design

 Chunking Strategy

Method: RecursiveCharacterTextSplitter
Chunk size: 800 characters
Overlap: 150 characters
Separators: ["\n\n", "\n", " ", ""] (preserve paragraphs, then lines, then sentences)

Rationale: 800 chars balances context coverage with prompt token limits. 150 overlap reduces boundary cuts.

 Vector Store Choice

ChromaDB selected because:
 Persistent on-disk storage (survives restarts)
 Simple setup, no external service needed
 Adequate for small datasets (<1000 chunks)

For larger datasets: consider Weaviate (HNSW indexing) or Pinecone (managed, scalable).

 Index Type

 Flat (exact) search — With ~168 vectors, brute-force is fastest
Search type: MMR (Maximum Marginal Relevance) for diversity
k: 15 fetch, 8 returned

 Latency Benchmarks

 Stage           |Cold          |Warm 

 Retrieval       |~0.23s        |~0.06s. 
 LLM generation  |~0.8s         |~0.8s. 
 Total           |~1.0s         |~0.9s.

Cold = first request (model loads). Warm = subsequent requests.

 Optimizations Applied

1. Global caching: Retriever and Groq client initialized once
2. MMR retrieval: Balances relevance vs diversity
3. Query expansion: For queries >4 words, generates 5 sub-queries
4. Reranking: BAAI/bge-reranker-base for top-8 results

 Future Improvements

1. HNSW indexing — For >10k vectors, switch Chroma to HNSW
2. Async endpoints — Use FastAPI async for concurrent requests
3. Query caching — Cache LLM responses for repeated questions
4. Cold start — Preload embeddings at startup



 Honest Evaluation

 Q&A Test Set

1. What is the goal of the arXiv annual report?
2. How many submissions did arXiv receive in 2023?
3. What are the main partnerships mentioned in the report?
4. What sustainability or outreach efforts are described?
5. What metrics are reported for arXiv usage growth?
6. What is the arXiv fiscal year covered by this report?
7. Which sections describe community engagement or diversity?
8. What funding or operational challenges are referenced?

 Test Results

json
{
  "1": { "status": "PASS", "answer_relevant": true },
  "2": { "status": "PASS", "answer_relevant": true },
  "3": { "status": "PASS", "answer_relevant": true },
  "4": { "status": "PASS", "answer_relevant": true },
  "5": { "status": "PASS", "answer_relevant": true },
  "6": { "status": "PASS", "answer_relevant": true },
  "7": { "status": "PASS", "answer_relevant": true },
  "8": { "status": "PASS", "answer_relevant": true },

}


 What Breaks

 Issue                                     |Cause                               |Fix 

Questions 6-8 return unrelated answers     |Content not in retrieved context    |Expand chunk size, add more chunks (k>15). 
Query expansion adds latency               |Extra LLM call per query            |Cache expansions. 
Reranking slow on long docs                |Truncation at 412 chars             |Increase truncation limit.   

 Root Cause Analysis

1. Coverage gaps: Some sections (governance, diversity) may be sparse in PDF
2. Retrieval recall: k=15 may miss low-ranking relevant chunks
3. Chunk boundaries: 800 chars may split tables mid-row

 What I'd Fix First

1. Increase fetch_k to 40 — Better recall for edge cases
2. Add section-aware chunking — Use PDF headings as chunk boundaries
3. Hybrid search — Combine keyword (BM25) with semantic search