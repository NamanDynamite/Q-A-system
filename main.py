from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from pathlib import Path
import dotenv

from src.llm import chat, expand_query
from src.retriever import retrieve
from src.reranker import rerank
from src.utils import should_expand, deduplicate_docs, build_context, build_sources

dotenv.load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


app = FastAPI()


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


QA_PROMPT = (
    "You are a document question-answering system.\n\n" 
    "RULES:\n" 
    "1. Use ONLY the provided context.\n" 
    "2. Extract ALL relevant entities when the question asks for lists (e.g., partnerships, organizations, stakeholders).\n"
    "3. Combine information across multiple context chunks if needed.\n" 
    "4. If the answer is not explicitly stated, infer from the context by combining related ideas.\n" 
    "5. Prefer high-level summaries when the question asks for purpose, goal, or intent.\n" 
    "6. DO NOT say 'not available' if a reasonable answer can be constructed from the context.\n" 
    "7. Only say 'The information is not available.' if absolutely nothing relevant exists.\n\n" 
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG system ready"}


@app.post("/ask", response_model=Answer)
def ask(question: Question):
    retrieval_start = time.time()

    if should_expand(question.question):
        expanded_queries = expand_query(question.question)
    else:
        expanded_queries = [question.question]

    all_docs = []
    for q in expanded_queries:
        all_docs.extend(retrieve(q))

    docs = deduplicate_docs(all_docs)

    retrieval_time = time.time() - retrieval_start

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    docs = rerank(question.question, docs)

    context = build_context(docs)
    sources = build_sources(docs)

    prompt = QA_PROMPT.format(context=context, question=question.question)

    generation_start = time.time()
    try:
        answer = chat(prompt)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"LLM error: {err}")

    generation_time = time.time() - generation_start
    answer = answer.split("Answer:")[-1].strip()

    print(f"retrieval={retrieval_time:.3f}s | generation={generation_time:.3f}s | docs={len(docs)}")

    return Answer(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)