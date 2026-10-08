"""
Tool 1: Vector search over an agronomy knowledge base.

Embeddings come from the Gemini `embed_content` API (`GeminiEmbeddingFunction`)
when GEMINI_API_KEY is set. Without a key — e.g. running the tool layer
offline, as in tests/test_tool_dispatch.py — it falls back automatically to
a local TF-IDF + TruncatedSVD pipeline (`LocalEmbeddingFunction`) fit once on
the knowledge corpus, so the project still runs end-to-end with zero network
dependency. ChromaDB storage, query, and the tool schema are identical either
way; only the embedding source changes.
"""
from __future__ import annotations

import glob
import os
from typing import List, Dict

import chromadb
import joblib
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings

from app import config

_EMBEDDER_PATH = os.path.join(config.CHROMA_PERSIST_DIR, "local_embedder.joblib")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Wraps the Gemini `embed_content` API as a ChromaDB embedding function.
    Used for both indexing and querying with task_type="RETRIEVAL_DOCUMENT",
    which is the simpler of the two valid setups (Gemini also supports a
    dedicated "RETRIEVAL_QUERY" type for the query side, which would give
    marginally better retrieval — a reasonable next iteration).
    """

    def __init__(self):
        from google import genai

        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._model = config.GEMINI_EMBEDDING_MODEL

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        from google.genai import types

        response = self._client.models.embed_content(
            model=self._model,
            contents=list(input),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [list(e.values) for e in response.embeddings]

    def ensure_ready(self, corpus: List[str]) -> None:
        pass  # nothing to fit — the Gemini endpoint is stateless per call


class LocalEmbeddingFunction(EmbeddingFunction):
    """
    Offline fallback used automatically when GEMINI_API_KEY is not set:
    TF-IDF -> TruncatedSVD, fit once on the agronomy corpus. Deterministic
    and dependency-free — keeps the tool layer testable (see
    tests/test_tool_dispatch.py) without requiring an API key.
    """

    def __init__(self):
        from sklearn.decomposition import TruncatedSVD  # local import: only needed for fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._TruncatedSVD = TruncatedSVD
        self._TfidfVectorizer = TfidfVectorizer
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        if os.path.exists(_EMBEDDER_PATH):
            bundle = joblib.load(_EMBEDDER_PATH)
            self.vectorizer = bundle["vectorizer"]
            self.svd = bundle["svd"]

    def fit(self, corpus: List[str], n_components: int = 128):
        n_components = min(n_components, max(2, len(corpus) - 1))
        self.vectorizer = self._TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = self.vectorizer.fit_transform(corpus)
        self.svd = self._TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd}, _EMBEDDER_PATH)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("LocalEmbeddingFunction must be fit() before use.")
        tfidf = self.vectorizer.transform(list(input))
        vecs = self.svd.transform(tfidf)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).tolist()

    def ensure_ready(self, corpus: List[str]) -> None:
        if self.vectorizer is None:
            self.fit(corpus)


_embedding_fn = None  # GeminiEmbeddingFunction | LocalEmbeddingFunction, chosen lazily


def _get_embedding_fn():
    """
    Picks the Gemini embedding function when GEMINI_API_KEY is configured,
    else falls back to the local TF-IDF/SVD embedder — chosen once and
    cached for the process lifetime.
    """
    global _embedding_fn
    if _embedding_fn is None:
        if config.GEMINI_API_KEY:
            _embedding_fn = GeminiEmbeddingFunction()
        else:
            _embedding_fn = LocalEmbeddingFunction()
    return _embedding_fn


def _get_client() -> chromadb.ClientAPI:
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def _load_corpus():
    doc_paths = sorted(glob.glob(str(config.KNOWLEDGE_DOCS_DIR / "*.txt")))
    ids, texts, metadatas = [], [], []
    for path in doc_paths:
        topic = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            ids.append(f"{topic}-{i}")
            texts.append(chunk)
            metadatas.append({"topic": topic, "chunk_index": i})
    return ids, texts, metadatas


def build_index(force: bool = False) -> int:
    """
    Prepares the embedder (fits the local TF-IDF/SVD fallback if no Gemini
    key is set; no-op for the Gemini embedder) and ingests every .txt file
    in data/agronomy_knowledge/ into ChromaDB, chunked at the paragraph
    level. Returns the number of chunks indexed.
    """
    client = _get_client()

    if force:
        try:
            client.delete_collection(config.CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        if os.path.exists(_EMBEDDER_PATH):
            os.remove(_EMBEDDER_PATH)
        global _embedding_fn
        _embedding_fn = None

    ids, texts, metadatas = _load_corpus()
    if not texts:
        return 0

    embedder = _get_embedding_fn()
    embedder.ensure_ready(texts)

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME, embedding_function=embedder
    )

    if collection.count() > 0 and not force:
        return collection.count()

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return collection.count()


def _get_collection():
    client = _get_client()
    embedder = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME, embedding_function=embedder
    )
    if collection.count() == 0:
        build_index()
        collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME, embedding_function=_get_embedding_fn()
        )
    return collection


def vector_search(query: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieves the top_k most relevant agronomy knowledge chunks for a query.

    This is exposed to the LLM as a callable tool: `search_agronomy_knowledge`.
    """
    collection = _get_collection()
    if collection.count() == 0:
        build_index()
        collection = _get_collection()

    results = collection.query(query_texts=[query], n_results=top_k)

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "topic": meta.get("topic"),
                "text": doc,
                "relevance_score": round(1 - dist, 4) if dist is not None else None,
            }
        )
    return hits


# --- Tool schema exposed to the Gemini function-calling loop ---
TOOL_SCHEMA = {
    "name": "search_agronomy_knowledge",
    "description": (
        "Semantic search over a curated agronomy knowledge base covering crop "
        "diseases, pest management, fertilizer guidance, irrigation practices, "
        "and soil health. Use this when the farmer's question needs domain "
        "knowledge rather than a live number (weather, soil carbon)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural-language description of what to look up, e.g. 'yellowing wheat leaves nitrogen deficiency'.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of knowledge chunks to retrieve. Defaults to 3.",
            },
        },
        "required": ["query"],
    },
}
