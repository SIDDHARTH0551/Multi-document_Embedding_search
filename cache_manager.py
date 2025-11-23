# cache_manager.py
"""Simple SQLite-backed cache manager for storing embeddings."""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict


class CacheManager:
    def __init__(self, db_path: str = "data/cache.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database and table."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                embedding TEXT,
                hash TEXT,
                length INTEGER,
                updated_at TEXT
            )
            """
        )

        conn.commit()
        conn.close()

    def upsert(
        self,
        doc_id: str,
        embedding: List[float],
        hash_val: str,
        filename: str,
        length: int,
    ):
        """Insert or update an embedding record."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        emb_json = json.dumps(embedding)
        now = datetime.utcnow().isoformat()

        cur.execute(
            """
            INSERT OR REPLACE INTO embeddings 
            (doc_id, filename, embedding, hash, length, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, emb_json, hash_val, length, now),
        )

        conn.commit()
        conn.close()

    def get(self, doc_id: str) -> Optional[Dict]:
        """Fetch a single document embedding by ID."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT doc_id, filename, embedding, hash, length, updated_at
            FROM embeddings WHERE doc_id = ?
            """,
            (doc_id,),
        )

        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        doc_id, filename, emb_json, hash_val, length, updated_at = row

        return {
            "doc_id": doc_id,
            "filename": filename,
            "embedding": json.loads(emb_json),
            "hash": hash_val,
            "length": length,
            "updated_at": updated_at,
        }

    def all(self) -> List[Dict]:
        """Return all cached embedding records."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT doc_id, filename, embedding, hash, length, updated_at 
            FROM embeddings
            """
        )

        rows = cur.fetchall()
        conn.close()

        results = []
        for row in rows:
            doc_id, filename, emb_json, hash_val, length, updated_at = row
            results.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "embedding": json.loads(emb_json),
                    "hash": hash_val,
                    "length": length,
                    "updated_at": updated_at,
                }
            )

        return results
