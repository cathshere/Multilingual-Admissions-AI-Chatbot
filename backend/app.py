"""
KEAM Admissions Chatbot — Backend API
---------------------------------------
Flask wrapper around the LangGraph RAG pipeline (planner -> retriever -> answerer)
originally prototyped in a Colab notebook. Serves a single POST /api/chat endpoint
consumed by the frontend in ../frontend/index.html.

Setup:
    1. pip install -r requirements.txt
    2. Put your prospectus PDF at backend/data/Prospectus2026.pdf
       (or set PDF_PATH env var to point elsewhere)
    3. Set GROQ_API_KEY as an environment variable
       (get one free at https://console.groq.com)
    4. python app.py
       -> serves on http://localhost:5000

First run will build a FAISS/TF-IDF index and cache it to backend/KEAM_faiss.pkl.
Subsequent runs load instantly from cache. Delete the cache file to force a rebuild
after replacing the PDF.
"""
from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
import textwrap
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from langdetect import detect
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from langgraph.graph import StateGraph, END

# ─── Configuration ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
PDF_PATH   = Path(os.environ.get("PDF_PATH", BASE_DIR / "data" / "Prospectus2026.pdf"))
CACHE_FILE = Path(os.environ.get("CACHE_FILE", BASE_DIR / "KEAM_faiss.pkl"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 350))   # words per chunk
CHUNK_STEP = int(os.environ.get("CHUNK_STEP", 150))   # stride
EMBED_DIM  = int(os.environ.get("EMBED_DIM", 128))    # LSA dimensions
TOP_K      = int(os.environ.get("TOP_K", 5))          # passages per query

GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

LANGUAGE_NAMES = {
    "en": "English", "ml": "Malayalam", "hi": "Hindi",
    "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "fr": "French", "de": "German", "es": "Spanish",
    "zh-cn": "Chinese", "ar": "Arabic", "ja": "Japanese",
}

# ─── Prompts (unchanged from the notebook) ─────────────────────────────
PLANNER_SYS = textwrap.dedent("""
    You are a routing assistant for a KEAM admissions chatbot.
    The user may write in any language.
    Decide whether the question needs searching the KEAM 2026 Prospectus
    for specific details (action = "retrieve") or can be answered from
    general knowledge (action = "answer_direct").

    Rules:
    - Admission dates, fees, eligibility, courses, documents, reservations,
      exam schedules, hostel info -> retrieve
    - Greetings, general knowledge, off-topic questions -> answer_direct

    Reply with ONLY one word: retrieve   OR   answer_direct
""").strip()

ANSWER_SYS = textwrap.dedent("""
    You are a helpful admissions assistant for KEAM (Kerala Engineering Admission Management).  You answer questions STRICTLY based on the
    provided context excerpts from the KEAM 2026 Admissions Prospectus.

    CRITICAL RULE: You MUST reply in {lang_name} ({lang_code}).
    If the user wrote in Malayalam, reply in Malayalam.
    If in Hindi, reply in Hindi. Match the user's language exactly.

    Guidelines:
    - Be concise and factual.
    - Quote page numbers when helpful (e.g., "as per page 12 of the prospectus").
    - If the context does not contain enough information, say so clearly in {lang_name}.
    - Do NOT make up fees, dates, or eligibility criteria.
""").strip()

FALLBACK_SYS = textwrap.dedent("""
    You are a helpful assistant for KEAM admissions.
    Reply in {lang_name} ({lang_code}).
    If the question is off-topic, politely redirect to KEAM admissions.
""").strip()


# ─── State schema ───────────────────────────────────────────────────────
class ChatState(BaseModel):
    query:     str
    lang_code: str = "en"
    lang_name: str = "English"
    action:    Optional[str] = None
    docs:      Optional[list] = None
    answer:    Optional[str] = None


# ─── PDF -> chunks ──────────────────────────────────────────────────────
def extract_pdf_text(path: Path) -> list[dict]:
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(pages, size=CHUNK_SIZE, step=CHUNK_STEP):
    chunks, meta = [], []
    for page_info in pages:
        words = page_info["text"].split()
        page_num = page_info["page"]
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + size])
            if len(chunk.strip()) < 60:
                continue
            chunks.append(chunk)
            meta.append({"page": page_num, "offset": i})
    return chunks, meta


# ─── Vector store: TF-IDF + LSA -> FAISS cosine index ──────────────────
class FAISSStore:
    def __init__(self, chunks, meta, cache=CACHE_FILE, dim=EMBED_DIM):
        if cache.exists():
            with open(cache, "rb") as f:
                data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.svd = data["svd"]
            self.dim = data["dim"]
            self.chunks = data["chunks"]
            self.meta = data["meta"]
            vecs = data["vectors"]
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vecs)
        else:
            self._build(chunks, meta, cache, dim)

    def _build(self, chunks, meta, cache, dim):
        self.chunks = chunks
        self.meta = meta
        self.vectorizer = TfidfVectorizer(
            max_features=20_000, ngram_range=(1, 1), sublinear_tf=True
        )
        X_sp = self.vectorizer.fit_transform(chunks)
        real_dim = min(dim, X_sp.shape[1] - 1, X_sp.shape[0] - 1)
        self.dim = real_dim
        self.svd = TruncatedSVD(n_components=real_dim, random_state=42)
        X_dense = self.svd.fit_transform(X_sp).astype(np.float32)
        faiss.normalize_L2(X_dense)
        self.index = faiss.IndexFlatIP(real_dim)
        self.index.add(X_dense)
        with open(cache, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer, "svd": self.svd,
                "dim": self.dim, "chunks": self.chunks,
                "meta": self.meta, "vectors": X_dense,
            }, f)

    def _embed(self, text):
        x = self.svd.transform(self.vectorizer.transform([text])).astype(np.float32)
        faiss.normalize_L2(x)
        return x

    def search(self, query, k=TOP_K):
        scores, idxs = self.index.search(self._embed(query), k)
        return [
            {"text": self.chunks[i], "score": float(scores[0][r]), "page": self.meta[i]["page"]}
            for r, i in enumerate(idxs[0]) if i != -1
        ]


# ─── Groq LLM wrapper ───────────────────────────────────────────────────
class GroqLLM:
    def __init__(self, model=GROQ_MODEL, api_key=GROQ_API_KEY):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and set it as an environment variable before starting the server."
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, system, user, max_tokens=700):
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


def detect_language(text: str):
    try:
        code = detect(text)
    except Exception:
        code = "en"
    name = LANGUAGE_NAMES.get(code, code.upper())
    return code, name


# ─── Lazy global init (index + llm built on first request) ────────────
_vector_store: Optional[FAISSStore] = None
_llm: Optional[GroqLLM] = None
_graph_app = None
_init_error: Optional[str] = None


def _build_graph():
    graph = StateGraph(ChatState)

    def node_detect_language(state: ChatState) -> ChatState:
        code, name = detect_language(state.query)
        state.lang_code, state.lang_name = code, name
        return state

    def node_planner(state: ChatState) -> ChatState:
        routing_prompt = (
            f"User question (may be in {state.lang_name}): {state.query}\n\n"
            "Route this: reply ONLY with 'retrieve' or 'answer_direct'."
        )
        decision = _llm.chat(PLANNER_SYS, routing_prompt, max_tokens=8)
        state.action = "retrieve" if "retrieve" in decision.lower() else "answer_direct"
        return state

    def node_retrieve(state: ChatState) -> ChatState:
        state.docs = _vector_store.search(state.query, k=TOP_K)
        return state

    def node_answer(state: ChatState) -> ChatState:
        sys_prompt = (
            ANSWER_SYS.format(lang_name=state.lang_name, lang_code=state.lang_code)
            if state.docs else
            FALLBACK_SYS.format(lang_name=state.lang_name, lang_code=state.lang_code)
        )
        if state.docs:
            passages = "\n\n".join(
                f"[Page {d['page']} | relevance {d['score']:.2f}]\n{d['text']}"
                for d in state.docs
            )
            user_msg = (
                f"Context from KEAM 2026 Prospectus:\n{passages}\n\n"
                f"Question ({state.lang_name}): {state.query}"
            )
        else:
            user_msg = state.query
        state.answer = _llm.chat(sys_prompt, user_msg, max_tokens=700)
        return state

    graph.add_node("detect_language", node_detect_language)
    graph.add_node("planner", node_planner)
    graph.add_node("retriever", node_retrieve)
    graph.add_node("answer", node_answer)

    graph.set_entry_point("detect_language")
    graph.add_edge("detect_language", "planner")
    graph.add_conditional_edges(
        "planner", lambda s: s.action,
        {"retrieve": "retriever", "answer_direct": "answer"},
    )
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def _ensure_initialized():
    """Build the index + LLM + graph once, lazily, on first request."""
    global _vector_store, _llm, _graph_app, _init_error
    if _graph_app is not None or _init_error is not None:
        return

    try:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it as an environment variable and restart the server."
            )
        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"Prospectus PDF not found at {PDF_PATH}. Place your PDF there or set PDF_PATH."
            )

        _llm = GroqLLM()

        if CACHE_FILE.exists():
            _vector_store = FAISSStore([], [])
        else:
            pages = extract_pdf_text(PDF_PATH)
            chunks, meta = chunk_pages(pages)
            _vector_store = FAISSStore(chunks, meta)

        _graph_app = _build_graph()
    except Exception as e:
        _init_error = str(e)


# ─── Flask app ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
FRONTEND_DIR = os.path.join(BASE_DIR.parent, "frontend")

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/api/health", methods=["GET"])
def health():
    _ensure_initialized()
    return jsonify({
        "ready": _graph_app is not None,
        "error": _init_error,
        "model": GROQ_MODEL,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    _ensure_initialized()
    if _init_error:
        return jsonify({"error": _init_error}), 503

    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' in request body."}), 400

    try:
        result = _graph_app.invoke({"query": query})
        docs = result.get("docs") or []
        sources = [
            {"page": d["page"], "score": round(d["score"], 3)}
            for d in docs
        ]
        return jsonify({
            "answer": result["answer"],
            "lang_code": result["lang_code"],
            "lang_name": result["lang_name"],
            "action": result["action"],
            "sources": sources,
        })
    except Exception as e:
        return jsonify({"error": f"Something went wrong answering that: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)