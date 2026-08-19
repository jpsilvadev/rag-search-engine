import argparse

from lib.keyword_search import build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Search movies by using keywords"
    )
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Returns term frequencies for a given ID and term"
    )
    tf_parser.add_argument(
        "doc_id", type=int, help="Document id to compute term frequency"
    )
    tf_parser.add_argument("term", type=str, help="Keyword to get frequency for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for idx, movie in enumerate(results, start=1):
                print(f"{idx}. {movie.get('title', 'Unknown')}")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document ID '{args.doc_id}': {tf}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
