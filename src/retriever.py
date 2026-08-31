import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import (
    CHROMA_PATH,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_LAMBDA,
    MIN_CHUNK_LENGTH,
    MIN_CHUNK_LENGTH_FALLBACK,
)


_retriever = None
_vectorstore = None
_bm25_index = None
_bm25_docs = None


def reset_retriever():
    global _retriever, _vectorstore, _bm25_index, _bm25_docs
    _retriever = None
    _vectorstore = None
    _bm25_index = None
    _bm25_docs = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    return _vectorstore


def get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = get_vectorstore()
        _retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_FETCH_K,
                "lambda_mult": RETRIEVAL_LAMBDA
            }
        )
    return _retriever


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _load_corpus(vectorstore: Chroma) -> tuple[list[list[str]], list[Document]]:
    collection = vectorstore._collection
    result = collection.get()
    texts = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    doc_ids = result.get("ids", [])

    docs = []
    for i, text in enumerate(texts):
        meta = metadatas[i] if i < len(metadatas) else {}
        docs.append(Document(page_content=text, metadata=meta, id=doc_ids[i]))

    tokenized = [_tokenize(doc.page_content) for doc in docs]
    return tokenized, docs


def _get_bm25() -> tuple[BM25Okapi | None, list[Document]]:
    global _bm25_index, _bm25_docs
    if _bm25_index is None:
        vectorstore = get_vectorstore()
        tokenized, docs = _load_corpus(vectorstore)
        if not docs:
            _bm25_index = None
            _bm25_docs = []
        else:
            _bm25_index = BM25Okapi(tokenized)
            _bm25_docs = docs
    return _bm25_index, _bm25_docs


def _bm25_search(query: str, top_k: int) -> list[Document]:
    bm25, docs = _get_bm25()
    if bm25 is None or not docs:
        return []
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [docs[i] for i in top_indices if scores[i] > 0]


def _dense_search(query: str) -> list[Document]:
    retriever = get_retriever()
    return retriever.invoke(query)


def _rrf_fuse(
    bm25_docs: list[Document],
    dense_docs: list[Document],
    k: int = 60
) -> list[Document]:
    doc_scores: dict[tuple[str, int], float] = {}
    doc_map: dict[tuple[str, int], Document] = {}

    def _doc_key(doc: Document, fallback_rank: int) -> tuple[str, int]:
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", fallback_rank)
        if not isinstance(chunk_id, int):
            chunk_id = fallback_rank
        return (source, chunk_id)

    for rank, doc in enumerate(bm25_docs):
        key = _doc_key(doc, rank)
        doc_map[key] = doc
        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc in enumerate(dense_docs):
        key = _doc_key(doc, rank)
        if key not in doc_map:
            doc_map[key] = doc
        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    ranked_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in ranked_keys]


def filter_chunks(docs: list[Document], min_length: int | None = None) -> list[Document]:
    if min_length is None:
        min_length = MIN_CHUNK_LENGTH
    return [doc for doc in docs if len(doc.page_content.strip()) >= min_length]


def retrieve(query: str) -> list[Document]:
    bm25_docs = _bm25_search(query, top_k=RETRIEVAL_K)
    dense_docs = _dense_search(query)

    fused = _rrf_fuse(bm25_docs, dense_docs)

    filtered = filter_chunks(fused, min_length=MIN_CHUNK_LENGTH)
    if not filtered and fused:
        return filter_chunks(fused, min_length=MIN_CHUNK_LENGTH_FALLBACK)
    return filtered
