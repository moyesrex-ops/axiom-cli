"""ChromaDB-based local vector store for semantic memory search."""

import time
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    """Local ChromaDB vector store for agent memory.

    Stores embeddings of conversations, facts, and decisions
    for semantic retrieval. Uses ChromaDB's built-in sentence-transformers
    embedding function for zero-config operation.

    Data stored at: memory/chroma/ (relative to project root)
    """

    def __init__(self, persist_dir: str = "memory/chroma"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy init ChromaDB — only when first needed."""
        if self._initialized:
            return
        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Use default embedding function (all-MiniLM-L6-v2)
            self._collection = self._client.get_or_create_collection(
                name="axiom_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.debug("ChromaDB initialized at %s with %d entries",
                        self.persist_dir, self._collection.count())
        except Exception as e:
            logger.warning("ChromaDB init failed: %s — memory will be degraded", e)
            self._initialized = False

    @property
    def count(self) -> int:
        """Number of entries in the vector store."""
        if not self._initialized:
            self._ensure_initialized()
        if self._collection is None:
            return 0
        return self._collection.count()

    def add(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a text entry to the vector store.

        Args:
            text: The text content to embed and store.
            metadata: Optional metadata dict (session_id, role, topic, etc.)
            doc_id: Optional custom ID. Auto-generated if not provided.

        Returns:
            The document ID assigned to this entry.
        """
        self._ensure_initialized()
        if self._collection is None:
            return ""

        if doc_id is None:
            doc_id = f"mem_{int(time.time() * 1000)}_{self.count}"

        meta = metadata or {}
        meta.setdefault("timestamp", time.time())

        # ChromaDB metadata values must be str, int, float, or bool
        clean_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)

        try:
            self._collection.add(
                documents=[text],
                metadatas=[clean_meta],
                ids=[doc_id],
            )
        except Exception as e:
            logger.warning("Failed to add to vector store: %s", e)

        return doc_id

    def search(
        self,
        query: str,
        k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across stored memories.

        Args:
            query: Natural language search query.
            k: Number of results to return.
            where: Optional ChromaDB where filter (e.g., {"role": "user"}).

        Returns:
            List of dicts with keys: id, text, metadata, distance
        """
        self._ensure_initialized()
        if self._collection is None or self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count()),
                where=where,
            )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

        entries = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
                entries.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                })

        return entries

    def delete(self, doc_id: str) -> bool:
        """Delete a memory entry by ID."""
        self._ensure_initialized()
        if self._collection is None:
            return False
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
