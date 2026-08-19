import argparse

from lib.keyword_search import build_command, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser(
        "search", help="Search movies by using keywords"
    )
    search_parser.add_argument("query", type=str, help="Search query")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
