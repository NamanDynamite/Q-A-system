from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
CHROMA_PATH = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

RETRIEVAL_K = 35
RETRIEVAL_FETCH_K = 120
RETRIEVAL_LAMBDA = 0.5

RERANK_TOP_K = 12
RERANK_TRUNCATE = 512

QUERY_EXPAND_MIN_WORDS = 4
QUERY_EXPAND_COUNT = 5


MIN_CHUNK_LENGTH = 30  
MIN_CHUNK_LENGTH_FALLBACK = 20  
