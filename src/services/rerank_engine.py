# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import os
import logging
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL_NAME", "ms-marco-MiniLM-L-6-v2")
TOP_K_RERANKER = int(os.getenv("TOP_K_RERANKER", "3"))


class LocalReranker:
    def __init__(self):
        self._ranker = None
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=RERANKER_MODEL)
            logger.info("FlashRank inicializado (modelo=%s)", RERANKER_MODEL)
        except Exception as exc:
            logger.warning("FlashRank indisponivel (%s). Usando passthrough.", exc)

    @property
    def available(self) -> bool:
        return self._ranker is not None

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.available or not chunks:
            if chunks:
                logger.info("FlashRank indisponivel — retornando top %d chunks", min(len(chunks), TOP_K_RERANKER))
            return self._passthrough(chunks)

        from flashrank import RerankRequest

        passages = []
        for idx, chunk in enumerate(chunks):
            passages.append({
                "id": idx,
                "text": chunk.get("text", chunk.get("content", chunk.get("page_content", ""))),
                "metadata": chunk.get("metadata", {}),
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

        top_results = results[:TOP_K_RERANKER]

        enriched = []
        for item in top_results:
            enriched.append({
                "id": item.get("id"),
                "text": item.get("text"),
                "score": round(float(item.get("score", 0.0)), 6),
                "metadata": item.get("metadata", {}),
            })

        logger.info(
            "Rerank: %d chunks -> top %d (modelo=%s)",
            len(chunks), len(enriched), RERANKER_MODEL,
        )
        return enriched

    def _passthrough(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        top_k = min(len(chunks), TOP_K_RERANKER)
        results = []
        for idx in range(top_k):
            chunk = chunks[idx]
            results.append({
                "id": idx,
                "text": chunk.get("text", chunk.get("content", chunk.get("page_content", ""))),
                "score": 1.0 / (idx + 1),
                "metadata": chunk.get("metadata", {}),
            })
        return results
