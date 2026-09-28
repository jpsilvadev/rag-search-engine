import argparse

from lib.evaluation import llm_evaluate_results
from lib.hybrid_search import RRF_K, normalize, rrf_search, weighted_search
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted hybrid search"
    )
    weighted_search_parser.add_argument(
        "query", type=str, help="Query string for the weighted hybrid search"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform a Reciprocal Rank Fusion (RRF) hybrid search"
    )
    rrf_search_parser.add_argument(
        "query", type=str, help="Query string for the RRF hybrid search"
    )
    rrf_search_parser.add_argument(
        "--k",
        type=int,
        nargs="?",
        default=RRF_K,
        help=f"RRF parameter k (default={RRF_K})",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to use",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Apply LLM evaluation to the search results",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            print(normalize(args.scores))
        case "weighted-search":
            result = weighted_search(
                query=args.query,
                alpha=args.alpha,
                limit=args.limit,
            )

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )

            for i, res in enumerate(result["results"], start=1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata")
                if (
                    metadata is not None
                    and "bm25_score" in metadata
                    and "semantic_score" in metadata
                ):
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            result = rrf_search(
                query=args.query,
                k=args.k,
                enhance=args.enhance,
                rerank_method=args.rerank_method,
                limit=args.limit,
            )

            if result["enhancement_method"]:
                print(
                    f"Enhanced query ({result['enhancement_method']}): '{result['original_query']}' -> '{result['query']}'"
                )

            if result["reranked"]:
                print(
                    f"Re-ranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )
            print(
                f"RRF Hybrid Search Results for '{result['query']}' (k={result['k']}):"
            )
            for i, res in enumerate(result["results"], start=1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Re-rank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Re-rank Rank: {res.get('batch_rank', 0)}")
                if "crossencoder_score" in res:
                    print(
                        f"   Cross Encoder Score: {res.get('crossencoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.4f}")
                metadata = res.get("metadata")
                if metadata is not None:
                    bm25_rank = metadata.get("bm25_rank")
                    semantic_rank = metadata.get("semantic_rank")
                    bm25_label = f"#{bm25_rank + 1}" if bm25_rank is not None else "-"
                    semantic_label = (
                        f"#{semantic_rank + 1}" if semantic_rank is not None else "-"
                    )
                    print(
                        f"   BM25 rank: {bm25_label}, Semantic rank: {semantic_label}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

            if args.evaluate:
                print("LLM Evaluation (0-3 relevance scale):")
                llm_scores = llm_evaluate_results(
                    query=args.query, results=result["results"]
                )

                for i, (res, score) in enumerate(
                    zip(result["results"], llm_scores), start=1
                ):
                    print(f"{i}. {res['title']}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
