import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    EvaluationSummary,
    QueryEvaluationResult,
    SearchResult,
    load_golden_dataset,
    load_movies,
)
from .semantic_search import SemanticSearch

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = DEFAULT_SEARCH_LIMIT
):
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k if k > 0 else 0


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = DEFAULT_SEARCH_LIMIT
):
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs) if relevant_docs else 0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def llm_evaluate_results(query: str, results: list[SearchResult]) -> list[int]:
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not found. Skipping LLM evaluation.")
        return [0] * len(results)

    formatted_results = []
    for i, result in enumerate(results, start=1):
        formatted_results.append(f"{i}. {result.get('title', '')}")

    prompt = f"""
    Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]
    """

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    ranking_text = (response.choices[0].message.content or "").strip()
    scores = json.loads(ranking_text)

    if len(scores) == len(results):
        return list(map(int, scores))

    raise ValueError(
        f"LLM response parsing error. Expected {len(results)} scores, but got {len(scores)}. Response: {scores}"
    )


def evaluate_command(limit: int = DEFAULT_SEARCH_LIMIT) -> EvaluationSummary:
    movies = load_movies()
    golden_dataset = load_golden_dataset()
    test_cases = golden_dataset["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    results_by_query: dict[str, QueryEvaluationResult] = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_docs = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, k=limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, k=limit)
        f1 = f1_score(precision, recall)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

        total_precision += precision

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }
