import string

from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies


def search_command(query, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    movies = load_movies()
    results = []
    query = tokenize(query)

    for movie in movies:
        title_tokens = tokenize(movie["title"])
        # match at least one query token with any title token
        if has_matching_token(query, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> str:
    # case sensitivity
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    # tokenization: word-based
    return [token.strip() for token in tokens if token]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(
        q_token in t_token for q_token in query_tokens for t_token in title_tokens
    )
