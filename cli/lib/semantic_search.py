import json
import os
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    ChunkMetadata,
    ChunkScore,
    Movie,
    SearchResult,
    SemanticSearchResult,
    format_search_result,
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

        self.document_map = {}
        docs = []
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                raise ValueError("document must have an 'id' field")
            self.document_map[doc_id] = doc
            doc_repr = f"{doc['title']}: {doc['description']}"
            docs.append(doc_repr)
        self.embeddings = self.model.encode(docs, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(file=MOVIE_EMBEDDINGS_PATH, arr=self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> NDArray[Any]:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents

        self.document_map = {}
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def search_chunks(self, query: str, limit: int = 10) -> list[SearchResult]:
        if self.chunk_embeddings is None or self.chunk_embeddings.size == 0:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        if self.chunk_metadata is None:
            raise ValueError(
                "No chunk metadata loaded. Call `load_or_create_chunk_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        q_embed = self.generate_embedding(query)

        chunk_scores: list[ChunkScore] = []
        for i, chunk_embed in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(q_embed, chunk_embed)
            chunk_score: ChunkScore = {
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": similarity_score,
            }
            chunk_scores.append(chunk_score)

        idx_to_bestchunk: dict[int, ChunkScore] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score.get("movie_idx", None)
            if (
                movie_idx not in idx_to_bestchunk
                or chunk_score["score"] > idx_to_bestchunk[movie_idx]["score"]
            ):
                idx_to_bestchunk[movie_idx] = chunk_score

        sorted_chunk_scores = sorted(
            idx_to_bestchunk.values(), key=lambda x: x["score"], reverse=True
        )
        filtered_chunk_scores = sorted_chunk_scores[:limit]

        results: list[SearchResult] = []
        for i, chunk in enumerate(filtered_chunk_scores):
            movie = self.documents[chunk.get("movie_idx")]
            search_result: SearchResult = format_search_result(
                doc_id=movie["id"],
                title=movie["title"],
                document=movie["description"],
                score=chunk.get("score"),
                metadata=self.chunk_metadata[chunk.get("chunk_idx")],
            )
            results.append(search_result)
        return results

    def build_chunk_embeddings(self, documents: list[Movie]) -> NDArray[Any]:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                raise ValueError("document must have an 'id' field")
            self.document_map[doc_id] = doc

        chunks: list = []
        chunk_metadata: list[ChunkMetadata] = []

        for i, doc in enumerate(documents):
            if not doc.get("description"):
                continue
            doc_chunks = semantic_chunking(
                text=doc.get("description"),
                chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )
            chunks.extend(doc_chunks)

            for j, chunk in enumerate(doc_chunks):
                doc_metadata: ChunkMetadata = {
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(doc_chunks),
                }
                chunk_metadata.append(doc_metadata)

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, mode="w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(chunk_metadata),
                },
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> NDArray[Any]:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                raise ValueError("document must have an 'id' field")
            self.document_map[doc_id] = doc

        if (os.path.exists(CHUNK_EMBEDDINGS_PATH)) and (
            os.path.exists(CHUNK_METADATA_PATH)
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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


def embed_chunks() -> None:
    chunked_search_instance = ChunkedSemanticSearch()
    documents: list[Movie] = load_movies()
    embeddings = chunked_search_instance.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_search(query: str, limit: int = 5):
    search_instance = SemanticSearch()
    documents: list[Movie] = load_movies()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)
    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['title']} (score: {result['score']}.4f)\n  {result['description']}\n"
        )


def search_chunked(query: str, limit: int = 5):
    chunked_search_instance = ChunkedSemanticSearch()
    documents: list[Movie] = load_movies()
    chunked_search_instance.load_or_create_chunk_embeddings(documents)
    results = chunked_search_instance.search_chunks(query, limit)
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result.get('title')} (score: {result.get('score'):.4f})")
        print(f"   {result.get('document')}...")


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words: list[str] = text.split()
    chunks: list[str] = []
    num_words = len(words)
    start = 0
    while start < num_words:
        chunk_words = words[start : start + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks


def semantic_chunking(
    text: str,
    chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    stripped_text = text.strip()
    if not stripped_text:
        return []
    sentences = re.split(pattern=r"(?<=[.!?])\s+", string=stripped_text)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [stripped_text]

    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    num_sentences = len(sentences)
    start = 0
    while start < num_sentences:
        chunk_sentences = sentences[start : start + chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        start += chunk_size - overlap
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def semantically_chunk_text(
    text: str,
    chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunking(text, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")
