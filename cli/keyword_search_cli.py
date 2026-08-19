import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)


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
        "doc_id", type=int, help="Document ID to compute term frequency"
    )
    tf_parser.add_argument("term", type=str, help="Keyword to get TF for")

    idf_parser = subparsers.add_parser(
        "idf", help="Returns the inverse document frequency for a given term"
    )
    idf_parser.add_argument("term", type=str, help="Keyword to get IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Returns TF-IDF for a given document ID and term"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document id to compute TF-IDF")
    tfidf_parser.add_argument("term", type=str, help="Keyword to get TF-IDF for")

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
            print(
                f"Term frequency of '{args.term}' in document ID '{args.doc_id}': {tf}"
            )
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term} in document '{args.doc_id}': {tfidf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
