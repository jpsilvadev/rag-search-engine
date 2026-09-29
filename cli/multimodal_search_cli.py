import argparse

from lib.multimodal_search import (
    DEFAULT_SEARCH_LIMIT,
    image_search,
    verify_image_embedding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_image_embedding_parser.add_argument(
        "image_path", type=str, help="Path to the image to verify"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search using images"
    )
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to the image to search"
    )
    image_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of search results to return",
    )
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search(args.image_path, limit=args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']} (similarity: {res['score']:.4f})")
                print(f"  {res['document']}[TRUNCATED...]")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
