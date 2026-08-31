from langchain_core.documents import Document

from .config import QUERY_EXPAND_MIN_WORDS


def should_expand(query: str) -> bool:
    return len(query.split()) > QUERY_EXPAND_MIN_WORDS


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for d in docs:
        text = d.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(d)
    return unique_docs


def build_context(docs: list[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def build_sources(docs: list[Document]) -> list[dict]:
    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]