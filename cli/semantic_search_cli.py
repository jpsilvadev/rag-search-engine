import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
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

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
