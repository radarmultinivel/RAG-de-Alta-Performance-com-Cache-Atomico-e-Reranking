# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import pytest
from src.services.cache_engine import SemanticCacheEngine


class TestFailsafeCache:
    def test_fail_open_quando_redis_indisponivel(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:19999")
        engine = SemanticCacheEngine()

        assert engine.available is False

        query_vector = [0.1] * 384
        result = engine.check_cache(query_vector)
        assert result is None

        stored = engine.store_cache("pergunta", "resposta", query_vector)
        assert stored is False
