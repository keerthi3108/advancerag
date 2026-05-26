from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import DATA_DIR, ROOT, answer_question, build_vector_store, ensure_vector_store


app = FastAPI(title="Naive RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    try:
        ensure_vector_store()
    except Exception as exc:
        print(f"ChromaDB was not built on startup: {exc}")

FRONTEND_DIST = ROOT / "frontend" / "dist"

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


def serialize_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source["id"],
        "source": source["source"],
        "page": source["page"],
        "distance": source["distance"],
        "text": source["text"],
    }


def list_documents() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(exist_ok=True)
    documents = []
    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".txt", ".md"}:
            documents.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "type": path.suffix.lower().lstrip("."),
                }
            )
    return documents


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    return {"documents": list_documents()}


@app.post("/api/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    DATA_DIR.mkdir(exist_ok=True)
    allowed_suffixes = {".pdf", ".txt", ".md"}
    saved = []

    for file in files:
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed_suffixes:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {filename}. Use PDF, TXT, or MD.",
            )

        target = DATA_DIR / filename
        target.write_bytes(await file.read())
        saved.append(filename)

    return {"saved": saved, "documents": list_documents()}


@app.post("/api/index")
def index_documents() -> dict[str, Any]:
    try:
        build_vector_store()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"message": "ChromaDB index is ready.", "documents": list_documents()}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, sources = answer_question(question, top_k=request.top_k)
    except Exception as exc:
        message = str(exc)
        if "invalid_api_key" in message or "Invalid API Key" in message:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Groq rejected the API key. Update GROQ_API_KEY in the "
                    "backend hosting environment and redeploy."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": answer,
        "sources": [serialize_source(source) for source in sources],
    }


@app.get("/")
def serve_frontend():
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.exists():
        return JSONResponse(
            {
                "status": "ok",
                "service": "Naive RAG backend",
                "message": "Backend is running. Use /api/health for health checks.",
            }
        )
    return FileResponse(index_file)


@app.get("/{path:path}")
def serve_frontend_routes(path: str) -> FileResponse:
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found.")
    return serve_frontend()
