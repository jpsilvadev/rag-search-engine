import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import MOVIE_EMBEDDINGS_PATH, Movie, load_movies


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
