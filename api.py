"""FastAPI server exposing /search endpoint. This module loads cache, builds index,
and serves query requests.
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np


from src.cache_manager import CacheManager
from src.search_engine import SearchEngine
from sentence_transformers import SentenceTransformer


app = FastAPI()


# Load cache and build index on startup
CACHE_PATH = 'data/cache.sqlite'
cache = CacheManager(CACHE_PATH)
all_docs = cache.all()
if not all_docs:
    print('Warning: cache empty. Run embedding generation first.')


embeddings = [d['embedding'] for d in all_docs]
doc_ids = [d['doc_id'] for d in all_docs]
search_engine = None
if embeddings:
    search_engine = SearchEngine(embeddings=embeddings, doc_ids=doc_ids)


model = SentenceTransformer('all-MiniLM-L6-v2')


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def read_root():
    return {"message": "Hello"}


@app.post('/search')
def search(req: SearchRequest):
    if not search_engine:
        return {"results": []}
    q = req.query
    q_emb = model.encode([q], convert_to_numpy=True)[0]
    results = search_engine.search(q_emb, top_k=req.top_k)
    return {"results": [{"doc_id": doc_id, "score": score} for doc_id, score in results]}   