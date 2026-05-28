# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import pytest
from src.services.cache_engine import SemanticCacheEngine


class TestCacheSemantico:
    def test_cache_hit_semantico(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        engine = SemanticCacheEngine()

        if not engine.available:
            pytest.skip("Redis nao disponivel.")

        query_vec_a = [0.1] * 384
        query_vec_b = [0.1001] * 384

        engine.store_cache(
            question="Como redefinir a senha do SAP?",
            response="Passo a passo para resetar a senha do SAP no portal corporativo.",
            query_vector=query_vec_a,
        )

        result = engine.check_cache(query_vec_b)

        assert result is not None
        assert result["source"] == "cache"
        assert result["score"] >= 0.95
        assert "SAP" in result["response"]
        assert result["question"] is not None

        engine.clear_cache()

    def test_cache_miss_por_diferenca_semantica(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        engine = SemanticCacheEngine()

        if not engine.available:
            pytest.skip("Redis nao disponivel.")

        vec_sap = [0.1] * 384
        vec_rh = [0.9] * 384

        engine.store_cache(
            question="Como redefinir a senha do SAP?",
            response="Procedimento SAP.",
            query_vector=vec_sap,
        )

        result = engine.check_cache(vec_rh)

        assert result is None

        engine.clear_cache()
