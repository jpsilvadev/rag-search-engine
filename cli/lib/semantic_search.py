import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import (
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    SemanticSearchResult,
    load_movies,
)


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: NDArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def verify(self) -> None:
        verify_model()

    def generate_embedding(self, text: str) -> NDArray[Any]:
        if not text or str.isspace(text):
            raise ValueError("text cannot be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[Movie]) -> NDArray[Any]:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents

        docs = []
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                raise ValueError("document must have an 'id' field")
            self.document_map[doc_id] = doc
            doc_repr = f"{doc['title']}: {doc['description']}"
            docs.append(doc_repr)
        self.embeddings = self.model.encode(docs, show_progress_bar=True)
        np.save(file=MOVIE_EMBEDDINGS_PATH, arr=self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> NDArray[Any]:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents

        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                raise ValueError("document must have an 'id' field")
            self.document_map[doc_id] = doc

        if os.path.exists(path=MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(file=MOVIE_EMBEDDINGS_PATH)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[SemanticSearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        q_embed = self.generate_embedding(query)

        similarities: list[tuple[float, Movie]] = []
        for i, embed in enumerate(self.embeddings):
            similarity_score = cosine_similarity(q_embed, embed)
            similarities.append((similarity_score, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)
        similarities = similarities[:limit]

        results: list[SemanticSearchResult] = []
        for score, doc in similarities:
            res: SemanticSearchResult = {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            results.append(res)
        return results


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def verify_embeddings() -> None:
    search_instance = SemanticSearch()
    documents: list[Movie] = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def semantic_search(query: str, limit: int = 5):
    search_instance = SemanticSearch()
    documents: list[Movie] = load_movies()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)
    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['title']} (score: {result['score']}.4f)\n  {result['description']}\n"
        )


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
