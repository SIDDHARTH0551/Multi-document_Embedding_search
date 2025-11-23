README.md (summary)
# What this project provides

- Load a folder of text files (100-200 docs)

- Clean & compute SHA256 per document

- Generate embeddings using sentence-transformers/all-MiniLM-L6-v2

- Cache embeddings in a local SQLite DB (re-use unchanged docs)

- Build a FAISS index (or fall back to NumPy cosine similarity)

- FastAPI /search endpoint that returns top_k results with ranking explanation

# How caching works

- Each doc is hashed with SHA256 of its cleaned text.

- The SQLite cache stores: doc_id, embedding (JSON array), hash, updated_at, filename, length.

- On embedding generation: if hash matches stored value, cached embedding is reused; otherwise embedding is recomputed and cache updated.

# How to run

- Create a virtualenv and install requirements: pip install -r requirements.txt.

- Prepare data/docs/ with .txt files.

- Generate embeddings / populate cache & build index:

- python -m src.embedder --docs_dir data/docs --cache_path data/cache.sqlite --batch_size 32

# Start API server:

- uvicorn src.api:app --reload --port 8000

- Search using POST /search with JSON body: { "query": "quantum physics basics", "top_k": 5 }

# Design choices

- Sentence Transformers chosen for local, reproducible embeddings.

- SQLite chosen for robust, small-footprint persistent cache.

- FAISS preferred for fast search; NumPy fallback available.

- Modular structure: embedder / cache manager / search engine / api.# Multi-document_Embedding-search-Engine
