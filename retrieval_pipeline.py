import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from document_loader import chunk_documents, load_documents
from hybrid_search import bm25_search, merge_results, vector_search
from query_optimizer import optimize_query
from reranker import rerank_chunks


logger = logging.getLogger(__name__)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def create_embeddings(
    texts: list[str],
    vectorizer: TfidfVectorizer | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, TfidfVectorizer]:
    if fit:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        matrix = vectorizer.fit_transform(texts)
    elif vectorizer is not None:
        matrix = vectorizer.transform(texts)
    else:
        raise ValueError("Pass a vectorizer or set fit=True.")

    return normalize(matrix.toarray().astype(np.float32)), vectorizer


class AdvancedRetrievalPipeline:
    """Advanced RAG retrieval: query optimization -> hybrid search -> reranking."""

    schema_version = "advanced_v1"

    def __init__(
        self,
        data_dir: Path,
        chroma_dir: Path,
        vectorizer_path: Path,
        collection_name: str,
    ) -> None:
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.vectorizer_path = vectorizer_path
        self.collection_name = collection_name

    def clear_cache(self) -> None:
        self._load_store_cached.cache_clear()
        self._load_all_chunks_cached.cache_clear()

    def build_vector_store(self) -> dict[str, int]:
        self.chroma_dir.mkdir(exist_ok=True)
        documents = load_documents(self.data_dir)
        chunks = chunk_documents(documents)
        embeddings, vectorizer = create_embeddings([chunk["text"] for chunk in chunks], fit=True)

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=self.collection_name,
            metadata={"schema_version": self.schema_version},
        )

        collection.add(
            ids=[str(chunk["id"]) for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk["id"],
                    "source": chunk["source"],
                    "filename": chunk["filename"],
                    "page": chunk["page"] or "",
                    "text_hash": chunk["text_hash"],
                }
                for chunk in chunks
            ],
            embeddings=embeddings.tolist(),
        )

        with self.vectorizer_path.open("wb") as file:
            pickle.dump(vectorizer, file)

        self.clear_cache()
        logger.info("Indexed %s chunks from %s document pages/files", len(chunks), len(documents))
        return {"documents": len(documents), "chunks": len(chunks)}

    def vector_store_exists(self) -> bool:
        if not self.chroma_dir.exists() or not self.vectorizer_path.exists():
            return False
        try:
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            collection = client.get_collection(name=self.collection_name)
            metadata = collection.metadata or {}
            return metadata.get("schema_version") == self.schema_version
        except Exception:
            return False

    def ensure_vector_store(self) -> None:
        if not self.vector_store_exists():
            self.build_vector_store()

    @lru_cache(maxsize=1)
    def _load_store_cached(self):
        if not self.chroma_dir.exists() or not self.vectorizer_path.exists():
            raise RuntimeError("ChromaDB vector store not found. Build the index first.")

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collection = client.get_collection(name=self.collection_name)
        with self.vectorizer_path.open("rb") as file:
            vectorizer = pickle.load(file)
        return collection, vectorizer

    @lru_cache(maxsize=1)
    def _load_all_chunks_cached(self) -> tuple[dict[str, Any], ...]:
        collection, _ = self._load_store_cached()
        result = collection.get(include=["documents", "metadatas"])
        chunks = []
        for text, metadata in zip(result["documents"], result["metadatas"]):
            chunks.append(
                {
                    "id": int(metadata["chunk_id"]),
                    "text": text,
                    "source": metadata["source"],
                    "filename": metadata.get("filename", metadata["source"]),
                    "page": metadata["page"] or None,
                    "distance": None,
                }
            )
        return tuple(chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[dict[str, Any]]:
        collection, vectorizer = self._load_store_cached()
        optimized_query = optimize_query(question)
        query_vectors, _ = create_embeddings([optimized_query], vectorizer=vectorizer)
        candidate_count = max(top_k * 4, 12)

        vector_results = vector_search(collection, query_vectors[0], top_n=candidate_count)
        keyword_results = bm25_search(
            optimized_query,
            list(self._load_all_chunks_cached()),
            top_n=candidate_count,
        )
        merged = merge_results(vector_results, keyword_results)
        final_chunks = rerank_chunks(optimized_query, merged, top_k=top_k)

        logger.info(
            "Advanced retrieval: question=%r optimized=%r vector=%s keyword=%s final=%s",
            question,
            optimized_query,
            len(vector_results),
            len(keyword_results),
            len(final_chunks),
        )
        return final_chunks
