import os
import json
from groq import Groq
from .config import LLM_MODEL


def get_groq_token() -> str:
    token = os.environ.get("GROQ_API_KEY")
    if not token:
        raise EnvironmentError("Set GROQ_API_KEY")
    return token


_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=get_groq_token())
    return _groq_client


def chat(prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
    client = get_groq_client()
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result.choices[0].message.content.strip()


def expand_query(query: str) -> list[str]:
    prompt = f"""Generate 5 diverse search queries for the question below.

Question: {query}

Return ONLY a JSON array of strings."""

    try:
        text = chat(prompt, max_tokens=150, temperature=0.3)
        queries = json.loads(text)
        if isinstance(queries, list):
            return [query] + queries
    except Exception:
        pass
    return [query]