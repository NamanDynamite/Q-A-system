from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import CHROMA_PATH, EMBEDDING_MODEL, RETRIEVAL_K, RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA


_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        _retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_FETCH_K,
                "lambda_mult": RETRIEVAL_LAMBDA
            }
        )
    return _retriever


def retrieve(query: str) -> list[Document]:
    retriever = get_retriever()
    return retriever.invoke(query)