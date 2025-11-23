"""Builds a vector index from cached embeddings and performs search.
Falls back to numpy cosine similarity if faiss is not present.
"""
import numpy as np
from typing import List, Tuple


try:
    import faiss
    _have_faiss = True
except Exception:
    _have_faiss = False


class SearchEngine:
    def __init__(self, embeddings: List[List[float]], doc_ids: List[str]):
        self.embeddings = np.array(embeddings, dtype='float32')
        self.doc_ids = list(doc_ids)
        self.dim = self.embeddings.shape[1]
        self.index = None
        if _have_faiss:
            self._build_faiss()
        else:
            # pre-normalize
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.normed = self.embeddings / norms


    def _build_faiss(self):
        # use Inner Product on normalized vectors to get cosine similarity
        # normalize embeddings
        emb = self.embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)


    def search(self, query_emb: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        q = np.array(query_emb, dtype='float32')
        if _have_faiss and self.index is not None:
            # normalize
            q = q / (np.linalg.norm(q) + 1e-12)
            D, I = self.index.search(q.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                results.append((self.doc_ids[int(idx)], float(score)))
                return results
        else:
            # cosine via numpy
            qnorm = q / (np.linalg.norm(q) + 1e-12)
            sims = (self.normed @ qnorm).astype(float)
            top_idx = np.argsort(-sims)[:top_k]
            return [(self.doc_ids[int(i)], float(sims[int(i)])) for i in top_idx]