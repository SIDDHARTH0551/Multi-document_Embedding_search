# src/embedder.py
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.cache_manager import CacheManager


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Minimal cleaning: lowercase, remove HTML tags, collapse whitespace."""
    import re

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # remove html tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_docs(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return numpy array of embeddings for list of texts."""
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings


def main(docs_dir: str, cache_path: str, batch_size: int = 32):
    docs = list(Path(docs_dir).glob("*.txt"))
    if not docs:
        print("No .txt files found in", docs_dir)
        return

    cache = CacheManager(cache_path)
    embedder = Embedder()

    to_embed = []  # tuples (doc_id, filename, text, hash, length)
    for f in docs:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
            continue

        cleaned = clean_text(text)
        h = compute_sha256(cleaned)
        doc_id = f.stem
        length = len(cleaned.split())
        cached = cache.get(doc_id)
        if cached and cached.get("hash") == h:
            # reuse cached embedding
            continue
        to_embed.append((doc_id, f.name, cleaned, h, length))

    if not to_embed:
        print("All documents are up-to-date in cache.")
        return

    # batch embed
    texts = [t[2] for t in to_embed]
    embeddings = embedder.embed_docs(texts, batch_size=batch_size)

    # write back to cache
    for i, (doc_id, filename, cleaned, h, length) in enumerate(to_embed):
        emb = embeddings[i].astype(float).tolist()
        cache.upsert(
            doc_id=doc_id, embedding=emb, hash_val=h, filename=filename, length=length
        )

    print(f"Updated cache with {len(to_embed)} documents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate & cache embeddings for docs")
    parser.add_argument("--docs_dir", required=True, help="Folder containing .txt files")
    parser.add_argument(
        "--cache_path", required=True, help="Path to sqlite cache file (e.g. data/cache.sqlite)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()
    main(args.docs_dir, args.cache_path, args.batch_size)
