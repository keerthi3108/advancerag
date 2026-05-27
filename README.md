# Advanced Hybrid RAG with LangSmith

This project started as a simple naive RAG pipeline and now includes an advanced retrieval layer:

`Documents -> Load Data -> Chunking -> TF-IDF Embeddings -> ChromaDB -> Query Optimization -> Hybrid Search -> Reranking -> Compact Context -> Groq LLM -> Final Answer`

The current document is:

- `data/Artificial Intelligence Act - Wikipedia.pdf`

## Setup

Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Then fill in:

- `GROQ_API_KEY`
- `LANGSMITH_API_KEY`, if you want traces in LangSmith

LangSmith tracing is enabled through these environment variables:

```text
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Simple Naive RAG
```

For deployed frontend builds, set this frontend environment variable to the backend URL:

```text
VITE_API_BASE=https://your-backend-url
```

## Run

Start the full React + Python app with one command:

```powershell
.\run_app.bat
```

Then open:

```text
http://127.0.0.1:8000
```

This builds the React frontend and serves it from the Python backend.

Check whether both are running:

```powershell
.\check_app.bat
```

Build the vector store:

```powershell
.\run.bat --index
```

Ask a question:

```powershell
.\run.bat --ask "What is the Artificial Intelligence Act?"
```

Or start interactive mode:

```powershell
.\run.bat
```

Optional: start the Python backend only:

```powershell
.\run_backend.bat
```

Optional: start the React frontend preview separately:

```powershell
.\run_frontend.bat
```

Then open:

```text
http://127.0.0.1:5173
```

The React frontend builds and then serves the app locally. It lets you upload documents, rebuild ChromaDB, ask questions in a chat screen, and view readable source evidence.

If you are running the frontend for the first time, install its dependencies once:

```powershell
cd frontend
npm install
```

## How It Works

1. Reads PDFs, `.txt`, and `.md` files from `data/`.
2. Splits text into smaller overlapping chunks and removes duplicate chunks.
3. Creates simple local TF-IDF embeddings.
4. Saves embeddings and metadata in ChromaDB under `chroma_db/`.
5. Optimizes the user query with lightweight intent-preserving expansion.
6. Runs hybrid retrieval:
   - Chroma vector similarity search
   - BM25-style keyword search
7. Merges and deduplicates results.
8. Reranks chunks using hybrid score plus query term coverage.
9. Sends only the best compact context to Groq.
10. Uses a strict prompt: answer only from context, otherwise say `I could not find that in the uploaded document.`
11. Sends traces to LangSmith when LangSmith environment variables are configured.

## Notes

The backend is still beginner-friendly, but the retrieval architecture is now closer to production RAG. It avoids extra LLM calls during retrieval, supports multiple uploaded documents, stores source/page metadata, caches Chroma/vectorizer reads, and limits context before sending it to Groq.

## Advanced RAG Files

- `document_loader.py` - loads PDFs/TXT/MD files and chunks them with metadata.
- `query_optimizer.py` - rewrites/expands queries without calling an LLM.
- `hybrid_search.py` - combines Chroma vector search with BM25-style keyword search.
- `reranker.py` - reranks merged chunks before generation.
- `retrieval_pipeline.py` - orchestrates indexing, caching, retrieval, merging, and reranking.

## GitHub Push

From the project folder:

```powershell
git init
git add .
git commit -m "Build naive RAG chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

Do not push `.env`, `chroma_db/`, `vector_store/`, `frontend/node_modules/`, or `frontend/dist/`. They are already ignored.

## Vercel Deploy

This repository includes `vercel.json` for deploying the React frontend from `frontend/`.

In Vercel:

1. Import the GitHub repository.
2. Select the account/team: `keerthis-projects-6ce39bff`.
3. Keep the root directory as the repository root.
4. Vercel will use `vercel.json`:
   - Build command: `cd frontend && npm install && npm run build`
   - Output directory: `frontend/dist`
5. Add environment variable:

```text
VITE_API_BASE=https://your-backend-url
```

Important: Vercel should host the React frontend. The Python FastAPI + ChromaDB backend should be hosted separately because it needs a persistent backend process and local ChromaDB storage.
