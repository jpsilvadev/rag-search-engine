import argparse

from lib.augmented_generation import rag, summarize, summarize_with_citations
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform text summarization"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit the number of documents to summarize",
    )

    citations_parser = subparsers.add_parser("citations", help="RAG with citations")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit the number of documents to summarize",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            docs, content = rag(query)
            print("Search Results:")
            for res in docs.get("results", []):
                print(f"- {res.get('title', '')}")
            print("\nRAG Response:")
            print(content)
            print()
        case "summarize":
            query = args.query
            limit = args.limit
            results, content = summarize(query, limit=limit)
            print("Search Results:")
            for res in results.get("results", []):
                print(f"- {res.get('title', '')}")
            print("\nLLM Summary:")
            print(content)
            print()
        case "citations":
            query = args.query
            limit = args.limit
            results, content = summarize_with_citations(query, limit=limit)
            print("Search Results:")
            for res in results.get("results", []):
                print(f"- {res.get('title', '')}")
            print("\nLLM Answer:")
            print(content)
            print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
