import hashlib
from pathlib import Path
from typing import Any

from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_documents(data_dir: Path) -> list[dict[str, Any]]:
    """Load supported files and keep filename/page metadata for citations."""
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
                            "filename": path.name,
                            "page": page_number,
                        }
                    )
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(
                    {
                        "text": text,
                        "source": path.name,
                        "filename": path.name,
                        "page": None,
                    }
                )

    if not documents:
        raise RuntimeError(f"No readable documents found in {data_dir}")

    return documents


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 700,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """Create overlapping chunks and remove exact duplicates across files/pages."""
    chunks: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    chunk_id = 1

    for document in documents:
        text = " ".join(document["text"].split())
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            text_hash = hashlib.sha256(chunk_text.lower().encode("utf-8")).hexdigest()

            if chunk_text and text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": document["source"],
                        "filename": document["filename"],
                        "page": document["page"],
                        "text_hash": text_hash,
                    }
                )
                chunk_id += 1

            start += chunk_size - overlap

    return chunks
