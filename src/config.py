import os
from pathlib import Path

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

RETRIEVAL_K = 15
RETRIEVAL_FETCH_K = 40
RETRIEVAL_LAMBDA = 0.3

RERANK_TOP_K = 8
RERANK_TRUNCATE = 512

QUERY_EXPAND_MIN_WORDS = 4
QUERY_EXPAND_COUNT = 5

PDF_PATH = "Data/2023_arXiv_annual_report.pdf"