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


def summarize(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> tuple[RRFSearchResult, str]:
    results: RRFSearchResult = rrf_search(query, limit=limit)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    content = (response.choices[0].message.content or "").strip()

    return results, content
