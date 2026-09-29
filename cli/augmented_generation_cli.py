import argparse

from lib.augmented_generation import rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
