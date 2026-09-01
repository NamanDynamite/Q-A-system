import os
import shutil
from pathlib import Path

import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL
from .retriever import reset_retriever


def get_pdf_paths(data_dir: str | os.PathLike[str] = DATA_DIR) -> list[str]:
    pdf_dir = Path(data_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(str(path) for path in pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
    return pdf_paths


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def create_documents(text: str, source: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"source": source, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    return documents


def ingest():
    pdf_paths = get_pdf_paths()

    if os.path.exists(CHROMA_PATH):
        print(f"Clearing existing ChromaDB at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    all_documents: list[Document] = []
    for pdf_path in pdf_paths:
        print(f"Extracting text from {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from {pdf_path}")

        print(f"Creating documents from {pdf_path}")
        documents = create_documents(text, source=pdf_path)
        print(f"Created {len(documents)} chunks from {pdf_path}")
        all_documents.extend(documents)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating ChromaDB at {CHROMA_PATH}")
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )

    try:
        count = vectorstore._collection.count()
    except Exception:
        count = len(all_documents)

    reset_retriever()

    print(f"Indexed {count} vectors")
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()