from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import time
from pathlib import Path
import dotenv
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

env_path = Path(__file__).resolve().parent / ".env"
print(f"[DEBUG] Loading .env from {env_path}")
print(f"[DEBUG] Existing HUGGINGFACEHUB_API_TOKEN before load: {repr(os.environ.get('HUGGINGFACEHUB_API_TOKEN'))}")
dotenv.load_dotenv(dotenv_path=env_path, override=True)
print(f"[DEBUG] Environment file loaded: exists={env_path.exists()}")
print(f"[DEBUG] HUGGINGFACEHUB_API_TOKEN after load: {repr(os.environ.get('HUGGINGFACEHUB_API_TOKEN'))}")

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN_ENV_VARS = ["HUGGINGFACEHUB_API_TOKEN"]

app = FastAPI()
retriever = None
hf_client = None

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: List[str]


def get_hf_token() -> str:
    for name in HF_TOKEN_ENV_VARS:
        token = os.environ.get(name)
        print(f"[DEBUG] Checking env var {name}: {'FOUND' if token else 'MISSING'}")
        if token:
            print(f"[DEBUG] Using Hugging Face token from {name}, length={len(token)}, repr={repr(token)}")
            return token
    raise EnvironmentError(
        "Set HUGGINGFACEHUB_API_TOKEN to use the Hugging Face Inference API."
    )


def get_retriever():
    global retriever
    if retriever is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def get_hf_client():
    global hf_client
    if hf_client is None:
        token = get_hf_token()
        print(f"[DEBUG] Creating InferenceClient for model {LLM_MODEL}")
        hf_client = InferenceClient(model=LLM_MODEL, token=token)
    return hf_client

@app.get("/")
def root():
    return {"status": "ok", "message": "PDF Q&A System ready"}

@app.post("/ask", response_model=Answer)
def ask(question: Question):
    retriever = get_retriever()
    retrieval_start = time.time()
    docs = retriever.invoke(question.question)
    retrieval_time = time.time() - retrieval_start

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found for this question.")

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [
        f"{doc.metadata.get('source', 'document')} chunk {doc.metadata.get('chunk_id', i)}"
        for i, doc in enumerate(docs)
    ]

    prompt = (
        "You are an expert assistant. Use only the context below to answer the question. "
        "If the answer cannot be found in the context, say that it is not present.\n\n"
        f"Context:\n{context}\n\nQuestion: {question.question}\n\nAnswer:"
    )

    client = get_hf_client()
    generation_start = time.time()
    try:
        print(f"[DEBUG] Sending chat_completion request to {LLM_MODEL}")
        result = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            max_tokens=220,
            temperature=0.0,
        )
        generation_time = time.time() - generation_start
        print(f"[DEBUG] Received response: {result}")
    except Exception as err:
        print(f"[DEBUG] chat_completion error: {repr(err)}")
        raise HTTPException(status_code=500, detail=f"LLM inference failed: {err}")

    if isinstance(result, dict) and "choices" in result:
        answer = result["choices"][0]["message"]["content"].strip()
    else:
        answer = str(result).strip()

    if answer.startswith(prompt):
        answer = answer[len(prompt) :].strip()
    answer = answer.split("Answer:")[-1].strip()

    print(
        f"retrieval={retrieval_time:.3f}s generation={generation_time:.3f}s "
        f"source_count={len(sources)}"
    )
    return Answer(answer=answer, sources=sources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")