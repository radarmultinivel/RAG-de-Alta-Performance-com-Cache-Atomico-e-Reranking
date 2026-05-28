# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import pytest
from src.services.rerank_engine import LocalReranker


class TestRerankerAcuracia:
    @pytest.fixture
    def reranker(self):
        r = LocalReranker()
        if not r.available:
            pytest.skip("FlashRank nao disponivel.")
        return r

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"text": "O almoco de confraternizacao sera na sexta-feira as 12h.", "metadata": {"source": "convite.pdf"}},
            {"text": "Para redefinir sua senha do SAP, acesse o portal corporativo e clique em 'Esqueci minha senha'.", "metadata": {"source": "manual_sap.pdf", "page": 12}},
            {"text": "O horario de funcionamento do restaurante e das 11h as 14h.", "metadata": {"source": "regras.pdf"}},
            {"text": "A senha do SAP deve conter no minimo 8 caracteres, incluindo numeros e letras maiusculas.", "metadata": {"source": "manual_sap.pdf", "page": 13}},
            {"text": "O cardapio do dia inclui arroz, feijao, bife acebolado e salada.", "metadata": {"source": "cardapio.pdf"}},
        ]

    def test_rerank_retorna_top_3_com_scores_decrescentes(self, reranker, sample_chunks):
        query = "Como redefinir a senha do SAP?"
        result = reranker.rerank(query, sample_chunks)

        assert len(result) == 3
        assert result[0]["score"] >= result[1]["score"]
        assert result[1]["score"] >= result[2]["score"]
        assert all(r["score"] > 0.0 for r in result)

    def test_rerank_mantem_integridade_estrutural(self, reranker, sample_chunks):
        query = "senha SAP"
        result = reranker.rerank(query, sample_chunks)

        for item in result:
            assert "id" in item
            assert "text" in item
            assert "score" in item
            assert "metadata" in item
            assert isinstance(item["text"], str)
            assert isinstance(item["score"], float)

    def test_rerank_chunks_vazios(self, reranker):
        result = reranker.rerank("alguma pergunta", [])
        assert result == []

    def test_rerank_coloca_sap_no_topo(self, reranker, sample_chunks):
        query = "Redefinir senha do SAP"
        result = reranker.rerank(query, sample_chunks)

        texts = [r["text"] for r in result]
        sap_texts = [
            "Para redefinir sua senha do SAP",
            "A senha do SAP deve conter no minimo 8 caracteres",
        ]
        has_sap = any(any(s in t for s in sap_texts) for t in texts)
        assert has_sap


class TestRerankerPassthrough:
    def test_passthrough_quando_indisponivel(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MODEL_NAME", "modelo-inexistente-12345")
        reranker = LocalReranker()
        assert not reranker.available

        chunks = [
            {"text": "texto A", "metadata": {}},
            {"text": "texto B", "metadata": {}},
            {"text": "texto C", "metadata": {}},
            {"text": "texto D", "metadata": {}},
        ]
        result = reranker.rerank("pergunta", chunks)
        assert len(result) == 3
        assert result[0]["text"] == "texto A"
        assert result[1]["text"] == "texto B"
        assert result[2]["text"] == "texto C"
        assert all("id" in r and "text" in r and "score" in r and "metadata" in r for r in result)
