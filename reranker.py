from typing import Any

from hybrid_search import tokenize


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rerank with lexical coverage plus hybrid score, then keep best chunks."""
    query_terms = set(tokenize(query))
    reranked = []

    for chunk in chunks:
        chunk_terms = set(tokenize(chunk["text"]))
        coverage = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        score = 0.75 * chunk.get("hybrid_score", 0.0) + 0.25 * coverage
        item = dict(chunk)
        item["rerank_score"] = score
        reranked.append(item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]
