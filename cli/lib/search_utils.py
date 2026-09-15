import json
import os
import string
from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float


class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    metadata: ChunkMetadata | None
    score: float


# consts

SCORE_PRECISION = 4
DEFAULT_SEARCH_LIMIT = 5

DEFAULT_CHUNK_SIZE = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 1


BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")


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
    doc_id: int,
    title: str,
    document: str,
    score: float,
    metadata: ChunkMetadata | None = None,
) -> SearchResult:
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata,
    }
