import os
import pickle

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    load_movies,
    load_stop_words,
    preprocess_text,
)

STOP_WORDS = load_stop_words()


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        term_doc_ids = self.index.get(term, set())
        return sorted(term_doc_ids)

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        with open(os.path.join(CACHE_PATH, "index.pkl"), "wb") as handler:
            pickle.dump(self.index, handler)

        with open(os.path.join(CACHE_PATH, "docmap.pkl"), "wb") as handler:
            pickle.dump(self.docmap, handler)


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


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

    # hardcoded for testing purposes
    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


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
