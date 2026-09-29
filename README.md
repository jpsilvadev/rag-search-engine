# RAG Search Engine

A retrieval and RAG pipeline built from scratch, with every stage exposed as a CLI command:

- **Keyword search:** inverted index, TF-IDF and BM25
- **Semantic search:** sentence embeddings, fixed-size and semantic chunking
- **Hybrid search:** weighted score fusion and Reciprocal Rank Fusion (RRF)
- **Query enhancement:** LLM spell correction, rewriting and expansion
- **Reranking:** LLM (individual and batch) and cross-encoder
- **Evaluation:** precision@k, recall@k and F1 against a golden dataset, plus LLM relevance scoring
- **Retrieval-augmented generation:** answers, summaries and cited answers built from the search results
- **Multimodal search:** image-to-text search with CLIP, and query rewriting from an image

The repo includes a demo dataset of ~5,000 movies (`data/movies.json`), which all the examples below use.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The first run downloads the embedding models from Hugging Face (`all-MiniLM-L6-v2`, `clip-ViT-B-32`, `cross-encoder/ms-marco-TinyBERT-L2-v2`). Indexes and embeddings are cached in `cache/`, and are built on first use if missing.

### OpenRouter API key

The LLM features (hybrid search, RAG, evaluation, image description) call OpenRouter. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys) and put it in a `.env` file in the project root:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

The keyword, semantic and multimodal CLIs work without a key.

> [!NOTE]
> All LLM calls use the `openrouter/free` model, which routes each request to whichever free model is available. Because of that, a request may hang, fail with an error, or return an unrelated reply such as `User Safety: safe` (a safety classifier's label) instead of a real answer. Image input (`describe_image_cli.py`) is hit the hardest. If that happens, re-run the command, or add credits to your OpenRouter account and switch `model` to a paid model.

## Usage

Run every command from the project root with `uv run cli/<name>.py <command> [args]`. Add `-h` to any CLI or command to see all of its options.

### Keyword search: `keyword_search_cli.py`

| Command                           | Description                               |
| --------------------------------- | ----------------------------------------- |
| `build`                           | Build the inverted index (run this first) |
| `search <query>`                  | Basic keyword search                      |
| `tf <doc_id> <term>`              | Term frequency                            |
| `idf <term>`                      | Inverse document frequency                |
| `tfidf <doc_id> <term>`           | TF-IDF score                              |
| `bm25tf <doc_id> <term> [k1] [b]` | BM25 TF score                             |
| `bm25idf <term>`                  | BM25 IDF score                            |
| `bm25search <query> [limit]`      | Full BM25 search                          |

```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25search "space adventure" 10
```

### Semantic search: `semantic_search_cli.py`

| Command                                                    | Description                             |
| ---------------------------------------------------------- | --------------------------------------- |
| `verify`                                                   | Check that the embedding model loads    |
| `verify_embeddings`                                        | Build or load embeddings for all movies |
| `embed_text <text>` / `embed_query <query>`                | Embed a single text or query            |
| `search <query> [--limit N]`                               | Semantic search over whole documents    |
| `chunk <text> [--chunk-size N] [--overlap N]`              | Split text into fixed-size chunks       |
| `semantic_chunk <text> [--max-chunk-size N] [--overlap N]` | Split text into chunks by sentence      |
| `embed_chunks`                                             | Build chunk embeddings for all movies   |
| `search_chunked <query> [--limit N]`                       | Semantic search over chunks             |

```bash
uv run cli/semantic_search_cli.py search "a bear lost in London" --limit 5
```

### Hybrid search: `hybrid_search_cli.py`

| Command                                                                                                                                     | Description                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `normalize <scores...>`                                                                                                                     | Min-max normalize a list of scores                                                              |
| `weighted-search <query> [--alpha A] [--limit N]`                                                                                           | Weighted BM25 + semantic (`alpha=1` is all BM25)                                                |
| `rrf-search <query> [--k K] [--enhance spell\|rewrite\|expand] [--rerank-method individual\|batch\|cross_encoder] [--evaluate] [--limit N]` | Reciprocal Rank Fusion, with optional LLM query enhancement, reranking and relevance evaluation |

```bash
uv run cli/hybrid_search_cli.py weighted-search "dinosaur theme park" --alpha 0.3
uv run cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance rewrite --rerank-method cross_encoder
```

### RAG: `augmented_generation_cli.py`

| Command                         | Description                          |
| ------------------------------- | ------------------------------------ |
| `rag <query>`                   | Search, then generate an answer      |
| `summarize <query> [--limit N]` | Summarize the search results         |
| `citations <query> [--limit N]` | Answer with citations to the results |
| `question <query> [--limit N]`  | Answer a question from the results   |

```bash
uv run cli/augmented_generation_cli.py question "Which movies feature talking animals?"
```

### Evaluation: `evaluation_cli.py`

Computes precision@k, recall@k and F1 against `data/golden_dataset.json`.

```bash
uv run cli/evaluation_cli.py --limit 5
```

### Multimodal search

`multimodal_search_cli.py` searches movies with an image, using CLIP embeddings:

```bash
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg --limit 5
```

`describe_image_cli.py` rewrites a text query using an image:

```bash
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear movie"
```

## License

See [LICENSE](LICENSE).
