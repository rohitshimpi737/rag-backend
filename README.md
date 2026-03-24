# Prabhupada RAG

## Project Structure
```
prabhupada_rag/
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── retrieval_service.py
│   ├── generation_service.py
│   └── requirements.txt
├── data/
│   ├── chromadb/
│   ├── chunks/
│   ├── cleaned/
│   ├── raw/
│   └── scraped/
├── scraper/
│   ├── config.py
│   ├── scraper.py
│   ├── parser.py
│   └── utils.py
└── legacy/
  ├── run_scraper.py
  ├── clean.py
  ├── chunk.py
  ├── embed.py
  ├── retrieve.py
  ├── generate.py
  ├── app.py
  └── requirements-scraper.txt
```

## Setup
```bash
pip install -r backend/requirements.txt
```

## Legacy Scraper Usage
```bash
# Scrape all three books
python legacy/run_scraper.py --books all

# Scrape specific book
python legacy/run_scraper.py --books bg
python legacy/run_scraper.py --books iso
python legacy/run_scraper.py --books noi

# Scrape with custom delay (seconds between requests)
python legacy/run_scraper.py --books all --delay 2.0

# Resume from where you left off (skips already-saved raw HTML)
python legacy/run_scraper.py --books all --resume
```

For scraper-only dependencies:

```bash
pip install -r legacy/requirements-scraper.txt
```

## Output
Each book produces a JSON file in `data/scraped/` with this structure per record:
```json
{
  "id": "bg_2_47",
  "book": "Bhagavad Gita As It Is",
  "book_code": "bg",
  "division_1": 2,
  "division_2": 47,
  "division_3": null,
  "reference": "BG 2.47",
  "verse_sanskrit": "...",
  "word_for_word": "...",
  "translation": "...",
  "purport": "..."
}
```

## FastAPI Backend (Separated from UI)

This project now includes a dedicated FastAPI backend so your UI and backend can evolve independently while preserving the same retrieval/generation behavior.

### Backend files

- `backend/main.py` — FastAPI app and routes
- `backend/schemas.py` — API contracts (request/response models)
- `backend/retrieval_service.py` — vector retrieval + reranking
- `backend/generation_service.py` — grounded prompt building + answer generation
- `backend/embeddings.py` — query embedding providers
- `backend/llm_service.py` — LLM provider routing
- `backend/vector_store.py` — ChromaDB access
- `backend/config.py` — provider and runtime config
- `backend/requirements.txt` — backend-only dependencies

### Backend setup

```bash
pip install -r backend/requirements.txt
```

### Start backend server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Endpoints

- `GET /health`
  - Returns backend status, active embedding provider, active LLM, and vector collection size.

- `POST /retrieve`
  - Runs retrieval only (vector search + rerank).
  - Body:

```json
{
  "query": "What is the nature of the soul?",
  "top_n": 5,
  "k": 20,
  "book_filter": ["bg", "iso"]
}
```

- `POST /ask`
  - Runs full pipeline: retrieval + grounded answer generation.
  - Body:

```json
{
  "query": "How should one perform devotional service?",
  "top_n": 5,
  "k": 20,
  "book_filter": ["bg", "noi"]
}
```

### Result consistency

The backend is fully standalone from the Streamlit/UI layer, but follows the same pipeline logic as your current implementation:

- query embedding
- vector search in ChromaDB
- cross-encoder reranking
- grounded LLM generation with citations

So query behavior remains aligned with your current system logic.

### Environment variables

- `ACTIVE_EMBEDDING_PROVIDER` = `baai` | `gemini` | `openai` (default: `baai`)
- `ACTIVE_LLM_PROVIDER` = `groq` | `gemini` | `openai` | `anthropic` (default: `groq`)
- `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` as needed

If you switch providers from defaults (`baai` + `groq`), install the corresponding SDKs as needed:

- Gemini: `google-generativeai`
- OpenAI: `openai`
- Anthropic: `anthropic`
