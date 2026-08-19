import math
import os
import pickle
from collections import Counter

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
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        with open(self.index_path, "wb") as handler:
            pickle.dump(self.index, handler)

        with open(self.docmap_path, "wb") as handler:
            pickle.dump(self.docmap, handler)

        with open(self.term_frequencies_path, "wb") as handler:
            pickle.dump(self.term_frequencies, handler)

    def load(self) -> None:
        with open(self.index_path, "rb") as handler:
            self.index = pickle.load(handler)
        with open(self.docmap_path, "rb") as handler:
            self.docmap = pickle.load(handler)

        with open(self.term_frequencies_path, "rb") as handler:
            self.term_frequencies = pickle.load(handler)

    def get_documents(self, term: str) -> list[int]:
        term_doc_ids = self.index.get(term, set())
        return sorted(term_doc_ids)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id] = Counter(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def search_command(query, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    index = InvertedIndex()
    index.load()
    query_tokens = tokenize(query)

    seen = set()
    results = []
    for token in query_tokens:
        matching_doc_ids = index.get_documents(token)

        for doc_id in matching_doc_ids:
            # avoid unnecessary lookups
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(index.docmap.get(doc_id))
            if len(results) >= limit:
                break
    return results


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    tf = index.get_tf(doc_id, token)
    return tf


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    idf = index.get_idf(token)
    return idf


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


def tokenize_single_term(text: str) -> str:
    token = tokenize(text)
    if len(token) != 1:
        raise ValueError("Have more than 1 token after processing")
    return token[0]
