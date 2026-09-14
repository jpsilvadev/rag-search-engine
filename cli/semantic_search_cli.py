import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
)
from lib.semantic_search import (
    chunk_text,
    embed_query_text,
    embed_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify embedding model loaded")
    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for all documents"
    )

    embed_parser = subparsers.add_parser("embed_text", help="Embed a given text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Embed a given query text"
    )
    embed_query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Search for a given query")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit number of search results",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given document")
    chunk_parser.add_argument("text", type=str, help="Document text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="Size of each chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=DEFAULT_CHUNK_OVERLAP,
        help="Size of overlap in each chunk",
    )

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
