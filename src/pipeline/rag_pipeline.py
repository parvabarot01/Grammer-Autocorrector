"""Retrieval-augmented grammar correction pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represent a retrieved chunk from the knowledge base."""

    text: str
    score: float
    source: str
    chunk_id: int


class GrammarRAGPipeline:
    """Build and query a grammar-rule knowledge base for prompt augmentation."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "data/vector_store",
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize retrieval settings and local state."""

        self.embedding_model = embedding_model
        self.vector_store_path = Path(vector_store_path)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        self._index_backend = "uninitialized"
        self._encoder: Any = None
        self._faiss: Any = None
        self._index: Any = None
        self._embeddings: Optional[np.ndarray] = None
        self._chunks: List[Dict[str, Any]] = []
        self._documents: List[str] = []

    def build_knowledge_base(self, documents: List[str]) -> None:
        """Split documents, embed chunks, and build a search index.

        Args:
            documents: Source documents or grammar rules to index.
        """

        chunk_records = self._chunk_documents(documents)
        if not chunk_records:
            raise ValueError(
                "Cannot build a knowledge base from an empty document list."
            )

        embeddings = self._encode_texts([record["text"] for record in chunk_records])
        self._index_backend, self._index = self._build_index(embeddings)
        self._embeddings = embeddings
        self._chunks = chunk_records
        self._documents = list(documents)
        self._persist_knowledge_base()
        LOGGER.info(
            "Built grammar knowledge base with %d chunks using %s backend.",
            len(self._chunks),
            self._index_backend,
        )

    def load_knowledge_base(self) -> None:
        """Load a persisted knowledge base from disk."""

        metadata_path = self.vector_store_path / "metadata.json"
        chunks_path = self.vector_store_path / "chunks.json"
        if not metadata_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Knowledge base artifacts are missing from {self.vector_store_path}."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._index_backend = str(metadata["backend"])
        self._chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._documents = [chunk["text"] for chunk in self._chunks]

        if self._index_backend == "faiss":
            faiss = self._import_faiss()
            self._faiss = faiss
            self._index = faiss.read_index(str(self.vector_store_path / "index.faiss"))
            self._embeddings = np.load(self.vector_store_path / "embeddings.npy")
        else:
            self._index = None
            self._embeddings = np.load(self.vector_store_path / "embeddings.npy")

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query string.

        Args:
            query: User query or sentence to match against indexed knowledge.
            top_k: Optional override for the number of retrieved chunks.

        Returns:
            List[RetrievedChunk]: Ranked chunks ordered by ascending distance.
        """

        self._ensure_knowledge_base_loaded()
        limit = top_k or self.top_k
        query_embedding = self._encode_texts([query])[0]
        distances, indices = self._search_index(query_embedding, limit)

        results: List[RetrievedChunk] = []
        for distance, index in zip(distances, indices):
            if index < 0 or index >= len(self._chunks):
                continue
            chunk = self._chunks[index]
            results.append(
                RetrievedChunk(
                    text=str(chunk["text"]),
                    score=float(distance),
                    source=str(chunk["source"]),
                    chunk_id=int(chunk["chunk_id"]),
                )
            )
        return results

    def augment_prompt(self, query: str, template: str = None) -> str:
        """Augment a correction prompt with retrieved supporting context.

        Args:
            query: Input sentence requiring grammar correction.
            template: Optional prompt template containing `{input}` and
                `{context}` placeholders.

        Returns:
            str: Augmented prompt with contextual grammar rules.
        """

        retrieved = self.retrieve(query)
        context_lines = [
            f"- [{chunk.source}#{chunk.chunk_id}] {chunk.text}" for chunk in retrieved
        ]
        context_block = (
            "\n".join(context_lines) if context_lines else "- No context found."
        )
        prompt_template = template or (
            "You are a grammar correction assistant.\n"
            "Relevant grammar guidance:\n{context}\n\n"
            "Input sentence:\n{input}\n\n"
            "Return only the corrected sentence."
        )
        return prompt_template.format(input=query, context=context_block)

    def rag_correct(self, text: str, llm_fn: Callable) -> str:
        """Run retrieval-augmented prompt construction and call an LLM function.

        Args:
            text: Input sentence to correct.
            llm_fn: Callable that accepts a prompt string and returns model output.

        Returns:
            str: Post-processed corrected text.
        """

        prompt = self.augment_prompt(text)
        response = llm_fn(prompt)
        return str(response).strip()

    def add_grammar_rules(self, rules: List[str]) -> None:
        """Append grammar rules and rebuild the knowledge base.

        Args:
            rules: New rule strings to index.
        """

        self._ensure_knowledge_base_loaded(allow_uninitialized=True)
        documents = list(self._documents) + list(rules)
        self.build_knowledge_base(documents)

    def get_relevant_rules(self, text: str) -> List[str]:
        """Retrieve rule texts relevant to an input sentence.

        Args:
            text: Input sentence to match against the rule index.

        Returns:
            List[str]: Retrieved rule strings.
        """

        return [chunk.text for chunk in self.retrieve(text)]

    def _chunk_documents(self, documents: Iterable[str]) -> List[Dict[str, Any]]:
        """Chunk source documents into overlapping character windows."""

        chunk_records: List[Dict[str, Any]] = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunk_id = 0
        for document_index, document in enumerate(documents):
            normalized = str(document).strip()
            if not normalized:
                continue
            if len(normalized) <= self.chunk_size:
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "text": normalized,
                        "source": f"document_{document_index}",
                    }
                )
                chunk_id += 1
                continue

            for start in range(0, len(normalized), step):
                chunk_text = normalized[start : start + self.chunk_size].strip()
                if not chunk_text:
                    continue
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "source": f"document_{document_index}",
                    }
                )
                chunk_id += 1
                if start + self.chunk_size >= len(normalized):
                    break
        return chunk_records

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence-transformers or a deterministic fallback."""

        if self._encoder is None:
            self._encoder = self._build_encoder()
        embeddings = self._encoder.encode(texts)
        return np.asarray(embeddings, dtype=np.float32)

    def _build_encoder(self) -> Any:
        """Build the embedding encoder with a fallback strategy."""

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            LOGGER.warning(
                "sentence-transformers is unavailable; using a hashing fallback "
                "encoder."
            )
            return _HashingSentenceEncoder()

        return SentenceTransformer(self.embedding_model)

    def _build_index(self, embeddings: np.ndarray) -> tuple[str, Any]:
        """Build a FAISS index when available, otherwise use a numpy fallback."""

        try:
            faiss = self._import_faiss()
        except ImportError:
            return "numpy", None

        index = faiss.IndexFlatL2(int(embeddings.shape[1]))
        index.add(embeddings)
        self._faiss = faiss
        return "faiss", index

    def _persist_knowledge_base(self) -> None:
        """Persist indexed chunks and vectors to disk."""

        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        chunks_path = self.vector_store_path / "chunks.json"
        metadata_path = self.vector_store_path / "metadata.json"
        chunks_path.write_text(json.dumps(self._chunks, indent=2), encoding="utf-8")

        if self._embeddings is None:
            raise RuntimeError("Embeddings are missing; cannot persist knowledge base.")
        np.save(self.vector_store_path / "embeddings.npy", self._embeddings)

        if self._index_backend == "faiss" and self._index is not None:
            self._faiss.write_index(
                self._index, str(self.vector_store_path / "index.faiss")
            )

        metadata = {
            "backend": self._index_backend,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunk_count": len(self._chunks),
            "vector_dimension": int(self._embeddings.shape[1]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _search_index(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[List[float], List[int]]:
        """Search the active index backend for nearest chunks."""

        if self._embeddings is None:
            raise RuntimeError("Knowledge base embeddings are not loaded.")

        limit = min(top_k, len(self._chunks))
        if limit <= 0:
            return [], []

        if self._index_backend == "faiss" and self._index is not None:
            distances, indices = self._index.search(
                query_embedding.reshape(1, -1),
                limit,
            )
            return distances[0].tolist(), indices[0].tolist()

        distances = np.linalg.norm(self._embeddings - query_embedding, axis=1)
        ranked_indices = np.argsort(distances)[:limit]
        return distances[ranked_indices].tolist(), ranked_indices.tolist()

    def _ensure_knowledge_base_loaded(self, allow_uninitialized: bool = False) -> None:
        """Load the persisted knowledge base if nothing is currently in memory."""

        if self._chunks and self._embeddings is not None:
            return
        metadata_path = self.vector_store_path / "metadata.json"
        chunks_path = self.vector_store_path / "chunks.json"
        if metadata_path.exists() and chunks_path.exists():
            self.load_knowledge_base()
            return
        if allow_uninitialized:
            return
        raise RuntimeError("Knowledge base has not been built yet.")

    def _import_faiss(self) -> Any:
        """Import faiss when available."""

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISS-backed retrieval. "
                "Install it with `pip install faiss-cpu`."
            ) from exc
        return faiss


class _HashingSentenceEncoder:
    """Deterministic embedding fallback used when sentence-transformers is absent."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into hashed bag-of-words vectors."""

        vectors = [self._encode_single(text) for text in texts]
        return np.asarray(vectors, dtype=np.float32)

    def _encode_single(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimension
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
