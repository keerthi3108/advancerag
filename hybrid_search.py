import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def bm25_search(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Small BM25 implementation to avoid an extra dependency."""
    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return []

    tokenized_docs = [tokenize(chunk["text"]) for chunk in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)

    doc_frequency: dict[str, int] = defaultdict(int)
    for tokens in tokenized_docs:
        for term in set(tokens):
            doc_frequency[term] += 1

    scores: list[tuple[float, dict[str, Any]]] = []
    total_docs = len(chunks)

    for chunk, tokens, doc_length in zip(chunks, tokenized_docs, doc_lengths):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in frequencies:
                continue
            idf = math.log(1 + (total_docs - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5))
            tf = frequencies[term]
            denominator = tf + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1))
            score += idf * (tf * (k1 + 1)) / denominator
        if score > 0:
            item = dict(chunk)
            item["keyword_score"] = score
            scores.append((score, item))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scores[:top_n]]


def vector_search(
    collection,
    query_embedding: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    query_result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict[str, Any]] = []
    for text, metadata, distance in zip(
        query_result["documents"][0],
        query_result["metadatas"][0],
        query_result["distances"][0],
    ):
        item = {
            "id": int(metadata["chunk_id"]),
            "text": text,
            "source": metadata["source"],
            "filename": metadata.get("filename", metadata["source"]),
            "page": metadata["page"] or None,
            "distance": float(distance),
            "vector_score": 1 / (1 + float(distance)),
        }
        results.append(item)

    return results


def merge_results(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    vector_weight: float = 0.65,
    keyword_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """Deduplicate chunks and combine vector + keyword scores."""
    vector_scores = _min_max_normalize(
        {str(item["id"]): item.get("vector_score", 0.0) for item in vector_results}
    )
    keyword_scores = _min_max_normalize(
        {str(item["id"]): item.get("keyword_score", 0.0) for item in keyword_results}
    )

    merged: dict[int, dict[str, Any]] = {}
    for item in vector_results + keyword_results:
        chunk_id = int(item["id"])
        merged.setdefault(chunk_id, {}).update(item)

    for chunk_id, item in merged.items():
        key = str(chunk_id)
        item["hybrid_score"] = (
            vector_weight * vector_scores.get(key, 0.0)
            + keyword_weight * keyword_scores.get(key, 0.0)
        )

    return sorted(merged.values(), key=lambda item: item["hybrid_score"], reverse=True)
