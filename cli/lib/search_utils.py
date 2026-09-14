import json
import os
import string
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    id: int
    title: str
    document: str  # NOTE:going to use Movie["description"] temporarily
    score: float


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


# consts
DEFAULT_SEARCH_LIMIT = 5

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4


BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")


# helpers
def preprocess_text(text: str) -> str:
    # case sensitivity
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        stop_words = [preprocess_text(word) for word in f.read().splitlines()]
        return stop_words


def format_search_result(
    doc_id: int, title: str, document: str, score: float
) -> SearchResult:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": score,
    }
