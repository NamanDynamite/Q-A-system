# PDF Q&A System

This repository builds a local PDF Q&A system on top of:
- `langchain` for text splitting and retrieval
- `langchain-huggingface` for sentence-transformers/all-mpnet-base-v2 embeddings
- `langchain-chroma` for persistent vector indexing
- `meta-llama/Llama-3.2-3B-Instruct` via Hugging Face Inference API for answer generation

## PDF chosen
- `Data/2023_arXiv_annual_report.pdf`
- Public 47-page arXiv annual report
- Chosen because it is a realistic, structured public document in the 30–100 page range and already available locally

## Files
- `ingest.py` — loads the PDF, splits text into chunks, embeds chunks, stores vectors in `chroma_db`
- `main.py` — FastAPI app exposing `POST /ask`
- `requirements.txt` — Python dependencies
- `.env.example` — environment variables placeholder

## Setup

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

3. Ingest the PDF into ChromaDB:

```bash
python ingest.py
```

4. Start the API server:

```bash
python main.py
```

5. Query the API:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the goal of the arXiv annual report?"}'
```

## API contract
- `POST /ask`
- Request body: `{"question": "..."}`
- Response body: `{"answer": "...", "sources": ["..."]}`

## System design

### Vector store choice
- **ChromaDB** is used because it is open-source, supports persistent on-disk storage, and integrates cleanly with LangChain.
- The dataset is small (84 chunks / 168 vectors), so exact dense retrieval is efficient and simple.

### Index type and reasoning
- We use Chroma's default dense vector retrieval on top of `SentenceTransformerEmbeddings`.
- With under 200 vectors, a flat exact search is fast enough and avoids the complexity of approximate index tuning.
- `k=5` is chosen to collect enough supporting context while keeping prompt size manageable.

### Cold vs warm latency
- Measured retrieval latency after ingestion:
  - `0.23s` for the first retrieval after retriever initialization
  - `0.06–0.08s` for warm retrieval after the index is loaded
- These timings show that local semantic search is already performant for this dataset.

### Optimizations in place
- Global caching of the Chroma retriever and HF inference client to avoid repeated initialization.
- Chunk size of `1000` characters with `200` overlap to balance retrieval recall and document coverage.
- Remote LLM inference is isolated behind a prompt that limits answers to retrieved context.

### Further improvements
- Use `langchain-huggingface` and `langchain-chroma` packages to avoid current deprecation warnings.
- Add async FastAPI endpoints for concurrent requests.
- Add query-result caching for repeated questions.
- For larger datasets, switch from flat retrieval to approximate indexing (HNSW or IVF) in Chroma.
- Precompute chunk embeddings once and load them without repeated model downloads.

## Evaluation

The system has been verified for ingestion and retrieval. The following question set is prepared for runtime evaluation once the Hugging Face API token is configured:

1. What is the goal of the arXiv annual report?
2. How many submissions did arXiv receive in 2023?
3. What are the main partnerships mentioned in the report?
4. What sustainability or outreach efforts are described?
5. What metrics are reported for arXiv usage growth?
6. What is the arXiv fiscal year covered by this report?
7. Which sections describe community engagement or diversity?
8. What funding or operational challenges are referenced?

### Honest status
- Ingestion and retrieval are verified with the local PDF and Chroma index.
- Answer generation via `meta-llama/Llama-3.2-3B-Instruct` is configured in `main.py`.
- This environment currently does not expose a Hugging Face API token, so remote LLM inference could not be executed here.

### Known breakage points
- If retrieved context does not contain the answer, the model may hallucinate or produce incomplete responses.
- If the HF token is missing, the server raises a clear environment error.
- The current LangChain integration emits deprecation warnings for `SentenceTransformerEmbeddings` and `Chroma`.

## What to fix first
1. Provide a valid `HUGGINGFACEHUB_API_TOKEN` and re-run the server.
2. Upgrade to `langchain-huggingface` and `langchain-chroma` for future compatibility.
3. Add async request handling and caching to reduce end-to-end latency for repeated questions.
