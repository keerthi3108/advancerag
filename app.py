import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

LANGSMITH_ENABLED = (
    os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    and bool(os.getenv("LANGSMITH_API_KEY"))
)

try:
    if not LANGSMITH_ENABLED:
        raise ImportError
    from langsmith import traceable
except Exception:
    def traceable(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator


DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
VECTORIZER_PATH = CHROMA_DIR / "vectorizer.pkl"
COLLECTION_NAME = "naive_rag_documents"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a simple naive RAG assistant.
Answer only from the provided context.
If the answer is available, explain it in clear student-friendly language.
If the answer is not available in the context, say: "I could not find that in the uploaded document."
Do not mention chunk numbers in the answer because the app shows sources separately.
Do not add extra notes outside the answer."""


def groq_client() -> OpenAI:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env before asking questions."
        )
    return OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)


@traceable(name="load_documents")
def load_documents(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []

    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        {
                            "text": text,
                            "source": path.name,
                            "page": page_number,
                        }
                    )
        elif path.suffix.lower() in {".txt", ".md"}:
            documents.append(
                {
                    "text": path.read_text(encoding="utf-8"),
                    "source": path.name,
                    "page": None,
                }
            )

    if not documents:
        raise RuntimeError(f"No readable documents found in {data_dir}")

    return documents


@traceable(name="chunk_documents")
def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_id = 1

    for document in documents:
        text = " ".join(document["text"].split())
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": document["source"],
                        "page": document["page"],
                    }
                )
                chunk_id += 1
            start += chunk_size - overlap

    return chunks


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


@traceable(name="create_embeddings")
def create_embeddings(
    texts: list[str],
    vectorizer: TfidfVectorizer | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, TfidfVectorizer]:
    if fit:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
    elif vectorizer is not None:
        matrix = vectorizer.transform(texts)
    else:
        raise ValueError("Pass a vectorizer or set fit=True.")

    return normalize(matrix.toarray().astype(np.float32)), vectorizer


@traceable(name="build_vector_store")
def build_vector_store() -> None:
    CHROMA_DIR.mkdir(exist_ok=True)
    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings, vectorizer = create_embeddings([chunk["text"] for chunk in chunks], fit=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=[str(chunk["id"]) for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "chunk_id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"] or "",
            }
            for chunk in chunks
        ],
        embeddings=embeddings.tolist(),
    )

    with VECTORIZER_PATH.open("wb") as file:
        pickle.dump(vectorizer, file)

    print(f"Indexed {len(chunks)} chunks from {len(documents)} document pages/files.")
    print(f"Saved ChromaDB vector store to {CHROMA_DIR}")


def load_vector_store():
    if not CHROMA_DIR.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError("ChromaDB vector store not found. Run: .\\run.bat --index")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    with VECTORIZER_PATH.open("rb") as file:
        vectorizer = pickle.load(file)
    return collection, vectorizer


def vector_store_exists() -> bool:
    if not CHROMA_DIR.exists() or not VECTORIZER_PATH.exists():
        return False

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        client.get_collection(name=COLLECTION_NAME)
        return True
    except Exception:
        return False


def ensure_vector_store() -> None:
    if vector_store_exists():
        return
    build_vector_store()


@traceable(name="retrieve")
def retrieve(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    collection, vectorizer = load_vector_store()
    query_vectors, _ = create_embeddings([question], vectorizer=vectorizer)
    query_result = collection.query(
        query_embeddings=query_vectors.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict[str, Any]] = []
    for chunk_id, text, metadata, distance in zip(
        query_result["ids"][0],
        query_result["documents"][0],
        query_result["metadatas"][0],
        query_result["distances"][0],
    ):
        results.append(
            {
                "id": int(metadata["chunk_id"]),
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"] or None,
                "distance": float(distance),
            }
        )

    return results


def format_context(chunks: list[dict[str, Any]]) -> str:
    context_parts = []
    for chunk in chunks:
        page = f", page {chunk['page']}" if chunk.get("page") else ""
        context_parts.append(
            f"[chunk {chunk['id']} | {chunk['source']}{page}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


@traceable(name="answer_question")
def answer_question(question: str, top_k: int = 4) -> tuple[str, list[dict[str, Any]]]:
    client = groq_client()
    relevant_chunks = retrieve(question, top_k=top_k)
    context = format_context(relevant_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content or "", relevant_chunks


def print_sources(chunks: list[dict[str, Any]]) -> None:
    print("\nSources")
    for chunk in chunks:
        page = f", page {chunk['page']}" if chunk.get("page") else ""
        print(
            f"- chunk {chunk['id']}: {chunk['source']}{page}, distance={chunk['distance']:.3f}"
        )


def interactive_loop(top_k: int) -> None:
    print("Naive RAG is ready. Type a question, or 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer, sources = answer_question(question, top_k=top_k)
        print(f"\nAnswer:\n{answer}")
        print_sources(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple naive RAG with LangSmith tracing.")
    parser.add_argument("--index", action="store_true", help="Load documents and build the vector store.")
    parser.add_argument("--ask", help="Ask one question against the vector store.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    if args.index:
        build_vector_store()

    if args.ask:
        answer, sources = answer_question(args.ask, top_k=args.top_k)
        print(f"\nAnswer:\n{answer}")
        print_sources(sources)
    elif not args.index:
        interactive_loop(top_k=args.top_k)


if __name__ == "__main__":
    main()
