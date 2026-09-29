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
    results: RRFSearchResult = rrf_search(query, limit=DEFAULT_SEARCH_LIMIT)
    prompt = f"""
    You are a RAG agent for Webflyx, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {results}

    Answer:
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    content = (response.choices[0].message.content or "").strip()

    return results, content


def summarize(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> tuple[RRFSearchResult, str]:
    results: RRFSearchResult = rrf_search(query, limit=limit)
    prompt = f"""
Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    content = (response.choices[0].message.content or "").strip()

    return results, content


def summarize_with_citations(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> tuple[RRFSearchResult, str]:
    results: RRFSearchResult = rrf_search(query, limit=limit)
    prompt = f"""
Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:
"""
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    content = (response.choices[0].message.content or "").strip()

    return results, content


def question_answering(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> tuple[RRFSearchResult, str]:
    results: RRFSearchResult = rrf_search(query, limit=limit)

    prompt = f"""
Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {query}

Documents:
{results}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:
"""
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    content = (response.choices[0].message.content or "").strip()

    return results, content
