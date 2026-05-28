# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import os
import logging
from typing import Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.services.cache_engine import SemanticCacheEngine
from src.services.rerank_engine import LocalReranker

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K_VECTOR_DB = int(os.getenv("TOP_K_VECTOR_DB", "20"))
TOP_K_RERANKER = int(os.getenv("TOP_K_RERANKER", "3"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "ms-marco-MiniLM-L-6-v2")


class MockVectorDB:
    def __init__(self):
        self._documents: list[dict[str, Any]] = []

    def ingest(self, documents: list[dict[str, Any]]):
        self._documents.extend(documents)
        logger.info("VectorDB: %d documentos ingeridos (total=%d)", len(documents), len(self._documents))

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[dict[str, Any]]:
        if not self._documents:
            return []
        results = self._documents[:top_k]
        logger.info("VectorDB: query retornou %d chunks", len(results))
        return results


class MockLLM:
    def generate(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            f"[Relevancia: {c['score']:.4f}]\n{c['text']}" for c in context_chunks
        )
        response = (
            f"## Resposta\n\n"
            f"**Pergunta:** {query}\n\n"
            f"**Contexto ({len(context_chunks)} chunks reordenados):**\n\n"
            f"{context_text}\n\n"
            f"---\n"
            f"*Resposta baseada nos {len(context_chunks)} documentos mais relevantes.*"
        )
        return response


class RAGPipeline:
    def __init__(self):
        logger.info("Inicializando RAGPipeline...")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._cache = SemanticCacheEngine()
        self._reranker = LocalReranker()
        self._vector_db = MockVectorDB()
        self._llm = MockLLM()
        logger.info("RAGPipeline inicializado (modelo=%s, reranker=%s)", EMBEDDING_MODEL, RERANKER_MODEL_NAME)

    def _generate_embedding(self, text: str) -> list[float]:
        emb = self._embedder.encode(text, normalize_embeddings=True)
        return emb.tolist()

    def ingest(self, documents: list[dict[str, Any]]):
        self._vector_db.ingest(documents)

    def query(self, question: str) -> dict[str, Any]:
        query_vector = self._generate_embedding(question)

        cached = self._cache.check_cache(query_vector)
        if cached is not None:
            return {
                "question": question,
                "response": cached["response"],
                "source": "cache",
                "cache_score": cached["score"],
                "num_chunks_used": 0,
                "llm_called": False,
            }

        raw_chunks = self._vector_db.search(query_vector, top_k=TOP_K_VECTOR_DB)

        reranked = self._reranker.rerank(question, raw_chunks)

        response_text = self._llm.generate(question, reranked)

        self._cache.store_cache(question, response_text, query_vector)

        return {
            "question": question,
            "response": response_text,
            "source": "llm",
            "cache_score": None,
            "num_chunks_used": len(reranked),
            "llm_called": True,
            "chunks": reranked,
        }
