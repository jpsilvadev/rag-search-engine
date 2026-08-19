from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    Movie,
    load_movies,
    load_stop_words,
    preprocess_text,
)

STOP_WORDS = load_stop_words()


def search_command(query, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    movies = load_movies()
    results = []

    query_tokens = tokenize(query)

    for movie in movies:
        title_tokens = tokenize(movie["title"])
        # match at least one query token with any title token
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def tokenize(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    # tokenization: word-based
    valid_tokens = [token.strip() for token in tokens if token]

    # filter out stop words
    filtered_tokens = [token for token in valid_tokens if token not in STOP_WORDS]

    stemmer = PorterStemmer()

    # stemming
    stemmed_words = list(map(stemmer.stem, filtered_tokens))
    return stemmed_words


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(
        q_token in t_token for q_token in query_tokens for t_token in title_tokens
    )
