from typing import Any

from numpy import dtype, ndarray
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    Movie,
    SearchResult,
    format_search_result,
    load_movies,
)
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self, model_name: str = "clip-ViT-B-32", documents: list[Movie] | None = None
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        self.texts = [
            f"{doc['title']}: {doc['description']}" for doc in (self.documents)
        ]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> ndarray[tuple[Any, ...], dtype[Any]]:
        img = Image.open(image_path)
        return self.model.encode([img])[0]

    def search_with_image(
        self, image_path: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        image_embedding = self.embed_image(image_path)

        similarities = []
        for i, doc in enumerate(self.documents):
            similarity = cosine_similarity(self.text_embeddings[i], image_embedding)
            similarities.append((i, similarity))

        top_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i][1], reverse=True
        )
        results = [
            format_search_result(
                doc_id=self.documents[i]["id"],
                title=self.documents[i]["title"],
                document=self.documents[i]["description"],
                score=similarities[i][1],
            )
            for i in top_indices
        ]
        return results[:limit]


def verify_image_embedding(image_path: str) -> None:
    multimodal_search_instance = MultimodalSearch()
    embedding = multimodal_search_instance.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search(
    image_path: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[SearchResult]:
    documents = load_movies()
    multimodal_search_instance = MultimodalSearch(documents=documents)
    return multimodal_search_instance.search_with_image(image_path, limit)
