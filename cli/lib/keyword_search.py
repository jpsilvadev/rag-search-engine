import math
import os
import pickle
from collections import Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    SearchResult,
    format_search_result,
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
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

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
        with open(self.doc_lengths_path, "wb") as handler:
            pickle.dump(self.doc_lengths, handler)

    def load(self) -> None:
        with open(self.index_path, "rb") as handler:
            self.index = pickle.load(handler)
        with open(self.docmap_path, "rb") as handler:
            self.docmap = pickle.load(handler)
        with open(self.term_frequencies_path, "rb") as handler:
            self.term_frequencies = pickle.load(handler)
        with open(self.doc_lengths_path, "rb") as handler:
            self.doc_lengths = pickle.load(handler)

    def get_documents(self, term: str) -> list[int]:
        term_doc_ids = self.index.get(term, set())
        return sorted(term_doc_ids)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.doc_lengths[doc_id] = len(tokens)

        self.term_frequencies[doc_id] = Counter(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        return avg_doc_length

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        # length normalization prevents larger documents from dominating in term searches
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        # saturation -> (tf * (k1 + 1)) / (tf + k1)
        # saturation prevents a term from dominating due to many repetitions
        # many many more repetitions does not necessarily equal more relevance
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def get_bm25idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def bm25(self, doc_id: int, term: str) -> float:
        bm25tf = self.get_bm25tf(doc_id, term)
        bm25idf = self.get_bm25idf(term)
        return bm25tf * bm25idf

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        tokens = tokenize(query)
        scores: dict[int, float] = {}

        # term-at-a-time (TAAT) implementation
        # instead of document-at-a-time (DAAT)
        for token in tokens:
            doc_ids = self.index.get(token, set())
            if not doc_ids:
                continue
            for doc_id in doc_ids:
                bm25 = self.bm25(doc_id, token)
                if doc_id not in scores:
                    scores[doc_id] = bm25
                else:
                    scores[doc_id] += bm25

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        search_results: list[SearchResult] = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            search_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=scores[doc_id],
            )
            search_results.append(search_result)
        return search_results


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


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    tfidf = index.get_tfidf(doc_id, token)
    return tfidf


def bm25tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    bm25tf = index.get_bm25tf(doc_id, token, k1, b)
    return bm25tf


def bm25idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    token = tokenize_single_term(term)
    bm25idf = index.get_bm25idf(token)
    return bm25idf


def bm25search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[SearchResult]:
    index = InvertedIndex()
    index.load()
    search_results = index.bm25_search(query, limit)
    return search_results


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
