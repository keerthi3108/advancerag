import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Bot,
  Database,
  FileText,
  Layers3,
  Loader2,
  Trash2,
  Send,
  Upload,
  User,
} from "lucide-react";

const RAW_API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

function compactText(text, limit = 260) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (clean.length <= limit) return clean;
  return `${clean.slice(0, limit).replace(/\s+\S*$/, "")}...`;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function SourceList({ sources }) {
  if (!sources?.length) return null;

  return (
    <details className="sources">
      <summary>Sources used</summary>
      <div className="source-list">
        {sources.map((source, index) => (
          <article className="source-card" key={`${source.id}-${index}`}>
            <div>
              <strong>{source.source}</strong>
              <span>{source.page ? `Page ${source.page}` : "Document"}</span>
            </div>
            <p>{compactText(source.text)}</p>
          </article>
        ))}
      </div>
    </details>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";
  const Icon = isUser ? User : Bot;

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="avatar">
        <Icon size={18} />
      </div>
      <div className="bubble">
        <p>{message.content}</p>
        <SourceList sources={message.sources} />
      </div>
    </div>
  );
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Advanced RAG is ready. Upload documents, build ChromaDB, then ask questions across your indexed knowledge base.",
    },
  ]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const chatEndRef = useRef(null);

  const documentCount = useMemo(() => documents.length, [documents]);

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_BASE}/api/documents`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch {
      setStatus("Backend is not reachable. Check VITE_API_BASE in deployment settings.");
    }
  }

  async function uploadFiles() {
    if (!files.length) return;
    setBusy(true);
    setStatus("Uploading documents...");

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed. Check VITE_API_BASE.");
      setDocuments(data.documents || []);
      setFiles([]);
      setStatus("Documents uploaded and ChromaDB was rebuilt.");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function rebuildIndex() {
    setBusy(true);
    setStatus("Building ChromaDB index...");

    try {
      const response = await fetch(`${API_BASE}/api/index`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Indexing failed. Check VITE_API_BASE.");
      setDocuments(data.documents || []);
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteDocument(name) {
    const confirmed = window.confirm(`Delete ${name} and rebuild the index?`);
    if (!confirmed) return;

    setBusy(true);
    setStatus(`Deleting ${name}...`);

    try {
      const response = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Delete failed.");
      setDocuments(data.documents || []);
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    setQuestion("");
    setBusy(true);
    setStatus("Retrieving from ChromaDB and asking Groq...");
    setMessages((current) => [...current, { role: "user", content: trimmed }]);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, top_k: topK }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Chat failed. Check VITE_API_BASE.");
      setMessages((current) => [
        ...current,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
      setStatus("");
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: error.message },
      ]);
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Database size={24} />
          </div>
          <div>
            <h1>Advanced RAG</h1>
            <p>Advanced Hybrid Retrieval</p>
          </div>
        </div>

        <section className="pipeline-card">
          <div>
            <span>Query Rewrite</span>
            <strong>Hybrid Search</strong>
          </div>
          <div>
            <span>BM25 + Vector</span>
            <strong>Reranked Context</strong>
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            <Upload size={18} />
            <h2>Documents</h2>
          </div>
          <label className="dropzone">
            <input
              type="file"
              accept=".pdf,.txt,.md"
              multiple
              onChange={(event) => setFiles(Array.from(event.target.files || []))}
            />
            <Upload size={22} />
            <span>{files.length ? `${files.length} file(s) selected` : "Choose PDF, TXT, or MD"}</span>
          </label>
          <button disabled={!files.length || busy} onClick={uploadFiles}>
            Upload
          </button>
        </section>

        <section className="panel">
          <div className="panel-title">
            <Layers3 size={18} />
            <h2>Advanced Index</h2>
          </div>
          <button disabled={busy} onClick={rebuildIndex}>
            {busy ? <Loader2 className="spin" size={17} /> : <Database size={17} />}
            Build ChromaDB
          </button>
          <label className="range-label">
            Retrieved chunks: <strong>{topK}</strong>
            <input
              type="range"
              min="1"
              max="10"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
        </section>

        <section className="panel documents">
          <div className="panel-title">
            <FileText size={18} />
            <h2>Loaded Files</h2>
          </div>
          {documentCount ? (
            documents.map((document) => (
              <div className="doc-row" key={document.name}>
                <FileText size={16} />
                <div>
                  <strong>{document.name}</strong>
                  <span>{formatSize(document.size)}</span>
                </div>
                <button
                  className="icon-button danger"
                  disabled={busy}
                  onClick={() => deleteDocument(document.name)}
                  title={`Delete ${document.name}`}
                  type="button"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))
          ) : (
            <p className="muted">No documents found.</p>
          )}
        </section>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">Advanced RAG Workspace</p>
            <h2>Ask across uploaded documents</h2>
            <p className="header-copy">
              Optimized query, hybrid retrieval, reranking, and compact context before Groq answers.
            </p>
          </div>
          <div className="status-pill">{documentCount} document(s)</div>
        </header>

        <div className="chat-window">
          {messages.map((message, index) => (
            <Message message={message} key={index} />
          ))}
          {busy && (
            <div className="message message-assistant">
              <div className="avatar">
                <Bot size={18} />
              </div>
              <div className="bubble typing">
                <Loader2 className="spin" size={18} />
                Working...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {status && <div className="status-bar">{status}</div>}

        <form className="composer" onSubmit={askQuestion}>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask something from the document..."
          />
          <button disabled={busy || !question.trim()} type="submit">
            {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </section>
    </main>
  );
}
