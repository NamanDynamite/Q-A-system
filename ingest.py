import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

PDF_PATH = "Data/2023_arXiv_annual_report.pdf"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from all pages of the PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def create_documents(text: str, source: str) -> list[Document]:
    """Split text into chunks and create Document objects."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
    """Main ingestion pipeline."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"Extracting text from {PDF_PATH}...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(text)} characters")

    print("Creating documents...")
    documents = create_documents(text, source=PDF_PATH)
    print(f"Created {len(documents)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating ChromaDB at {CHROMA_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = len(documents)

    print(f"Indexed {count} vectors")
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest()