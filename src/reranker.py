from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from .config import RERANK_TOP_K, RERANK_TRUNCATE


_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def rerank(query: str, docs: list[Document], top_k: int = RERANK_TOP_K) -> list[Document]:
    if not docs:
        return []
    
    reranker = get_reranker()
    pairs = [[query, doc.page_content[:RERANK_TRUNCATE]] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]