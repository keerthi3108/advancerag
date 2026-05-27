import argparse
import logging
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from retrieval_pipeline import AdvancedRetrievalPipeline

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
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6500"))

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

retrieval_pipeline = AdvancedRetrievalPipeline(
    data_dir=DATA_DIR,
    chroma_dir=CHROMA_DIR,
    vectorizer_path=VECTORIZER_PATH,
    collection_name=COLLECTION_NAME,
)

SYSTEM_PROMPT = """You are an advanced RAG assistant.
Answer only from the provided context.
If the answer is available, explain it in clear student-friendly language.
If the answer is not available in the context, say: "I could not find that in the uploaded document."
Do not mention chunk numbers in the answer because the app shows sources separately.
When useful, synthesize information across multiple uploaded documents.
Be concise, accurate, and avoid guessing."""


def groq_client() -> OpenAI:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to .env before asking questions."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


@traceable(name="build_vector_store")
def build_vector_store() -> None:
    stats = retrieval_pipeline.build_vector_store()
    print(f"Indexed {stats['chunks']} chunks from {stats['documents']} document pages/files.")
    print(f"Saved ChromaDB vector store to {CHROMA_DIR}")


def load_vector_store():
    return retrieval_pipeline._load_store_cached()


def vector_store_exists() -> bool:
    return retrieval_pipeline.vector_store_exists()


def ensure_vector_store() -> None:
    retrieval_pipeline.ensure_vector_store()


@traceable(name="retrieve")
def retrieve(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    return retrieval_pipeline.retrieve(question, top_k=top_k)


def format_context(chunks: list[dict[str, Any]]) -> str:
    context_parts = []
    total_chars = 0
    for chunk in chunks:
        page = f", page {chunk['page']}" if chunk.get("page") else ""
        block = f"[source: {chunk['source']}{page}]\n{chunk['text']}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        total_chars += len(block)
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
