import os
from typing import Literal

from .keyword_search import InvertedIndex
from .query_enhancement import enhance_query
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    CombinedScoreData,
    Movie,
    RRFScoreData,
    RRFSearchResult,
    SearchResult,
    WeightedSearchResult,
    format_search_result,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_rrf_results(
            bm25_results, semantic_results, k
        )  # a new function, similar in spirit to combine_search_results
        return combined[:limit]


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_search_results(
    results: list[SearchResult],
) -> list[tuple[SearchResult, float]]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])
    normalized_scores = normalize(scores)
    return [(result, normalized_scores[i]) for i, result in enumerate(results)]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1 / (k + rank)


def combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    alpha: float = DEFAULT_ALPHA,
) -> list[SearchResult]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores: dict[int, CombinedScoreData] = {}

    for result, normalized_score in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        combined_scores[doc_id]["bm25_score"] = max(
            combined_scores[doc_id]["bm25_score"], normalized_score
        )

    for result, normalized_score in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        combined_scores[doc_id]["semantic_score"] = max(
            combined_scores[doc_id]["semantic_score"], normalized_score
        )

    hybrid_results: list[SearchResult] = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    return hybrid_results


def combine_rrf_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    k: int = RRF_K,
) -> list[SearchResult]:
    combined_scores: dict[int, RRFScoreData] = {}

    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        combined_scores[doc_id]["bm25_rank"] = rank
        combined_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        combined_scores[doc_id]["semantic_rank"] = rank
        combined_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    rrf_results: list[SearchResult] = []
    for doc_id, data in combined_scores.items():
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    rrf_results.sort(key=lambda x: x["score"], reverse=True)
    return rrf_results


def weighted_search(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> WeightedSearchResult:
    documents = load_movies()
    hybrid_search_instance = HybridSearch(documents=documents)
    results = hybrid_search_instance.weighted_search(
        query=query, alpha=alpha, limit=limit
    )

    original_query = query

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search(
    query: str,
    k: int = RRF_K,
    enhance: Literal["spell", "rewrite"] | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> RRFSearchResult:
    documents = load_movies()
    hybrid_search_instance = HybridSearch(documents=documents)

    enhanced_query = enhance_query(query, enhance)
    results = hybrid_search_instance.rrf_search(query=enhanced_query, k=k, limit=limit)

    return {
        "original_query": query,
        "query": enhanced_query,
        "enhancement_method": enhance,
        "k": k,
        "results": results,
    }
