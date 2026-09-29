import os

from dotenv import load_dotenv
from openai import OpenAI

from .hybrid_search import rrf_search
from .search_utils import DEFAULT_SEARCH_LIMIT, RRFSearchResult

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def rag(query: str) -> tuple[RRFSearchResult, str]:
    docs: RRFSearchResult = rrf_search(query, limit=DEFAULT_SEARCH_LIMIT)
    prompt = f"""
    You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    content = (response.choices[0].message.content or "").strip()

    return docs, content
