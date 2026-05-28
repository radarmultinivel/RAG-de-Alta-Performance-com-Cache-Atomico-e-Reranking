# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import os
import json
import logging
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from src.config.redis_client import init_redis, INDEX_NAME, DOC_PREFIX

load_dotenv()

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.96"))


class SemanticCacheEngine:
    def __init__(self):
        self._redis = init_redis()

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def check_cache(self, query_vector: list[float]) -> Optional[dict]:
        if not self.available:
            return None

        vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        try:
            results = self._redis.execute_command(
                "FT.SEARCH",
                INDEX_NAME,
                f"*=>[KNN 1 @vector_data $vec AS score]",
                "PARAMS", "2", "vec", vector_bytes,
                "RETURN", "3", "question", "response", "score",
                "SORTBY", "score",
                "DIALECT", "2",
            )
        except Exception as exc:
            logger.warning("Erro na busca VSS Redis: %s", exc)
            return None

        if not results or isinstance(results, (int, str)) or results[0] == 0:
            return None

        record = results[2]
        entry = {}

        if isinstance(record, list):
            for i in range(0, len(record), 2):
                key = record[i].decode("utf-8") if isinstance(record[i], bytes) else str(record[i])
                val = record[i + 1]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                entry[key] = val
        elif isinstance(record, bytes):
            try:
                entry = json.loads(record.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Falha ao decodificar resultado do cache.")
                return None

        raw_score = entry.get("score", "1.0")
        try:
            distance = float(raw_score)
        except (ValueError, TypeError):
            distance = 1.0

        similarity = 1.0 - distance

        if similarity >= SEMANTIC_CACHE_THRESHOLD:
            logger.info("CACHE HIT | score=%.4f (threshold=%.4f)", similarity, SEMANTIC_CACHE_THRESHOLD)
            return {
                "question": entry.get("question", ""),
                "response": self._sanitize_response(entry.get("response", "")),
                "score": round(similarity, 4),
                "source": "cache",
            }

        logger.info("CACHE MISS | score=%.4f < threshold=%.4f", similarity, SEMANTIC_CACHE_THRESHOLD)
        return None

    def store_cache(self, question: str, response: str, query_vector: list[float]) -> bool:
        if not self.available:
            return False

        safe_response = self._sanitize_response(response)
        vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        cache_key = f"{DOC_PREFIX}{abs(hash(question))}"

        try:
            self._redis.hset(cache_key, mapping={
                "vector_data": vector_bytes,
                "question": question,
                "response": safe_response,
            })
            logger.info("Resposta armazenada em cache (chave=%s)", cache_key)
            return True
        except Exception as exc:
            logger.warning("Erro ao armazenar cache: %s", exc)
            return False

    def clear_cache(self) -> bool:
        if not self.available:
            return False
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{DOC_PREFIX}*")
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info("Cache semantico limpo.")
            return True
        except Exception as exc:
            logger.warning("Erro ao limpar cache: %s", exc)
            return False
