# KEAM Admissions Desk

A multilingual admissions chatbot for KEAM, built from the original Colab notebook
(LangGraph planner → FAISS/TF-IDF retriever → Groq LLM answerer), now split into:

- **backend/** — Flask API that runs the pipeline
- **frontend/** — a single-file HTML/CSS/JS chat UI ("posh dossier" design: deep ink-teal,
  kasavu gold, and vermillion accents; language is shown as a wax-seal-style badge, and
  every answer carries a small ticket-tag showing whether it came from the prospectus or
  general knowledge)

## 1. Backend setup

```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Put your prospectus PDF here:
#   backend/data/Prospectus2026.pdf
# (or set PDF_PATH to point elsewhere)

export GROQ_API_KEY="your_groq_key_here"   # https://console.groq.com

python app.py
# -> running on http://localhost:5000
```

The first request builds a TF-IDF/LSA + FAISS index from the PDF and caches it to
`backend/KEAM_faiss.pkl`. Delete that file to force a rebuild after swapping the PDF.

### Environment variables

| Variable       | Default                          | Purpose                          |
|----------------|-----------------------------------|-----------------------------------|
| `GROQ_API_KEY` | *(required)*                      | Groq API key                      |
| `PDF_PATH`     | `backend/data/Prospectus2026.pdf` | Path to the prospectus PDF        |
| `CACHE_FILE`   | `backend/KEAM_faiss.pkl`         | Where the vector index is cached  |
| `GROQ_MODEL`   | `llama-3.3-70b-versatile`         | Groq model name                   |
| `TOP_K`        | `5`                                | Passages retrieved per query      |
| `PORT`         | `5000`                            | Flask server port                 |

## 2. Frontend setup

The frontend is a single static HTML file — no build step.

```bash
cd frontend
python -m http.server 8000
# open http://localhost:8000
```

If your backend runs anywhere other than `http://localhost:5000`, set it before the page
loads by adding this above the `<script>` tag in `index.html`:

```html
<script>window.KEAM_API_BASE = "https://your-backend-url";</script>
```

## 3. Using it

Open the frontend in a browser. The status chip in the header shows whether the backend
is reachable and configured. Ask a question in any language (English, Malayalam, Hindi,
Tamil, etc.) — the assistant detects the language, decides whether it needs to search the
prospectus, and replies in the same language, citing page numbers when it used the
prospectus.

## Notes on the port from the notebook

- Removed Colab-specific bits (`google.colab.userdata`, notebook `display()`/mermaid
  graph rendering).
- `GROQ_API_KEY` now comes from a standard environment variable.
- The vector store and LLM are initialized lazily on first request, and initialization
  errors (missing PDF, missing key) are surfaced through `/api/health` and `/api/chat`
  instead of crashing the process, so the frontend can show a helpful status.
- The `/api/chat` response includes the routing `action` (`retrieve` vs `answer_direct`)
  and the source pages used, which the frontend renders as a small provenance tag under
  each answer.