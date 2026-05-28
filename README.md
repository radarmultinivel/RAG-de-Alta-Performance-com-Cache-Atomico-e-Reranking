# RAG de Alta Performance com Cache Atomico e Reranking Local

**Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026**

---

## Objetivo do Programa

Sistema de Retrieval-Augmented Generation (RAG) projetado para oferecer respostas instantaneas com custo minimo de processamento de LLM. O sistema emprega duas camadas de otimizacao: um cache semantico baseado em Redis com busca por similaridade vetorial (VSS) que elimina consultas identicas ou semanticamente equivalentes, e um reranking local em CPU via FlashRank que reduz o contexto enviado a LLM de 20 chunks para apenas os 3 mais relevantes.

## Sumario

- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Fluxograma da Arquitetura](#fluxograma-da-arquitetura)
- [Stacks e Tecnologias](#stacks-e-tecnologias)
- [Dependencias](#dependencias)
- [Instalacao](#instalacao)
- [Configuracao](#configuracao)
- [Manual do Usuario](#manual-do-usuario)
- [Testes](#testes)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [Licenca](#licenca)

## Arquitetura do Sistema

O sistema implementa duas barreiras de otimizacao antes de consultar a LLM:

**1. Cache Semantico (Redis VSS)**
- Gera embedding da pergunta via SentenceTransformers (modelo all-MiniLM-L6-v2, 384 dimensoes)
- Busca o vizinho mais proximo (KNN) no indice vetorial do Redis usando distancia de cosseno
- Se a similaridade for igual ou superior ao limiar configurado (padrao 0.96), retorna a resposta armazenada em cache sem acionar a LLM
- Latencia tipica: < 5ms em cache hit

**2. Reranking Local (FlashRank)**
- Para perguntas ineditas (cache miss), busca 20 chunks no banco vetorial
- O modelo Cross-Encoder do FlashRank (ms-marco-MiniLM-L-6-v2) reordena os chunks por relevancia a pergunta
- Seleciona apenas os 3 chunks mais relevantes para compor o contexto da LLM
- Reducao de tokens de contexto em ate 85%

## Fluxograma da Arquitetura

```
                    +-------------------+
                    |   POST /query     |
                    |(pergunta usuario) |
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    |   Embedding       |
                    |(SentenceTransform)|
                    +--------+----------+
                             |
                             v
                    +-------------------+
                    | Cache Redis VSS   |
                    | FT.SEARCH KNN     |
                    +--------+----------+
                             |
                   +---------+---------+
                   |                   |
                   v                   v
            +------+------+    +------+------+
            | Cache Hit   |    | Cache Miss  |
            | score >=    |    | score <     |
            | 0.96        |    | 0.96        |
            +------+------+    +------+------+
                   |                   |
                   |                   v
                   |           +------+------+
                   |           | Vector DB   |
                   |           | top 20      |
                   |           | chunks      |
                   |           +------+------+
                   |                   |
                   |                   v
                   |           +------+------+
                   |           | FlashRank   |
                   |           | top 3      |
                   |           | chunks      |
                   |           +------+------+
                   |                   |
                   |                   v
                   |           +------+------+
                   |           | LLM Mock   |
                   |           | gera       |
                   |           | resposta   |
                   |           +------+------+
                   |                   |
                   |                   v
                   |           +------+------+
                   |           | Salva no   |
                   |           | Redis      |
                   |           | Cache      |
                   |           +------+------+
                   |                   |
                   +---------+---------+
                             |
                             v
                    +-------------------+
                    |   Resposta Final  |
                    +-------------------+
```

## Stacks e Tecnologias

| Componente        | Tecnologia                               | Funcao                                    |
|-------------------|------------------------------------------|-------------------------------------------|
| Linguagem         | Python 3.11+                             | Runtime principal                         |
| Web Framework     | FastAPI + Uvicorn                        | Servidor ASGI concorrente                 |
| Cache Semantico   | Redis Stack (RediSearch)                 | Indice vetorial + busca KNN por cosseno   |
| Embeddings        | SentenceTransformers (all-MiniLM-L6-v2)  | Geracao de embeddings 384 dim em CPU      |
| Reranker          | FlashRank (ms-marco-MiniLM-L-6-v2)       | Cross-encoder ONNX local em CPU           |
| Vector DB         | MockVectorDB (implementacao propria)     | Armazenamento e busca vetorial            |
| LLM               | MockLLM (implementacao propria)          | Geracao de resposta simulada              |
| Container         | Docker / Docker Compose                  | Ambiente Redis Stack                      |

## Dependencias

As dependencias formais estao listadas em `requirements.txt`:

```
fastapi          - Framework web
uvicorn          - Servidor ASGI
redis[hiredis]   - Cliente Redis com parser C acelerado
sentence-transformers - Modelos de embedding
flashrank        - Reranker ONNX em CPU
pydantic         - Validacao de dados
python-dotenv    - Gerenciamento de ambiente
numpy            - Operacoes vetoriais
pytest           - Testes unitarios
pytest-asyncio   - Suporte async para testes
httpx            - Cliente HTTP para testes
```

## Instalacao

### Pre-requisitos

- Python 3.11 ou superior
- Docker e Docker Compose (para Redis Stack)
- Git

### Passo a passo

```bash
# 1. Clone o repositorio
git clone https://github.com/seu-usuario/rag-cache-atomico.git
cd rag-cache-atomico

# 2. Crie e ative o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 3. Instale as dependencias
pip install -r requirements.txt

# 4. Configure as variaveis de ambiente
cp .env.example .env
# Edite o arquivo .env conforme necessario

# 5. Inicie o Redis Stack
docker compose up -d

# 6. Inicie a API
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

A documentacao interativa estara disponivel em `http://localhost:8000/docs`.

## Configuracao

Todas as configuracoes sao definidas via variaveis de ambiente no arquivo `.env`:

| Variavel                  | Padrao                         | Descricao                          |
|---------------------------|--------------------------------|------------------------------------|
| REDIS_URL                 | redis://localhost:6379          | URL do Redis Stack                 |
| EMBEDDING_MODEL_NAME      | all-MiniLM-L6-v2                | Modelo de embedding                |
| SEMANTIC_CACHE_THRESHOLD  | 0.96                           | Limiar de similaridade do cache    |
| RERANKER_MODEL_NAME       | ms-marco-MiniLM-L-6-v2          | Modelo do FlashRank                |
| TOP_K_VECTOR_DB           | 20                             | Numero de chunks do Vector DB      |
| TOP_K_RERANKER            | 3                              | Numero de chunks apos rerank       |
| LLM_PROVIDER              | openai                         | Provedor de LLM                    |
| OPENAI_API_KEY            | -                              | Chave da API OpenAI                |
| OPENAI_MODEL_NAME         | gpt-4o-mini                    | Modelo OpenAI                      |

## Manual do Usuario

### Endpoint: POST /health

Verifica se o servico esta operacional.

```bash
curl http://localhost:8000/health
```

Resposta:
```json
{"status": "ok", "service": "rag-cache-atomico"}
```

### Endpoint: POST /query

Processa uma pergunta e retorna resposta com cache semantico.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Como redefinir a senha do SAP?"}'
```

Resposta (cache hit):
```json
{
  "question": "Como redefinir a senha do SAP?",
  "response": "## Resposta\n\n...",
  "source": "cache",
  "cache_score": 0.9734,
  "num_chunks_used": 0,
  "llm_called": false
}
```

Resposta (cache miss, LLM acionada):
```json
{
  "question": "Como redefinir a senha do SAP?",
  "response": "## Resposta\n\n...",
  "source": "llm",
  "cache_score": null,
  "num_chunks_used": 3,
  "llm_called": true,
  "chunks": [...]
}
```

### Endpoint: POST /ingest

Ingere documentos no banco vetorial.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "text": "Procedimento para redefinir senha SAP no portal corporativo.",
        "metadata": {"source": "manual_sap.pdf", "page": 12}
      }
    ]
  }'
```

Resposta:
```json
{"status": "ok", "documents_ingested": 1}
```

### Exemplo de Benchmarking

```bash
# Primeira consulta - cache miss (~150ms)
time curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Como redefinir a senha do SAP?"}' | jq .

# Segunda consulta - cache hit (< 5ms)
time curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o passo a passo para resetar a senha do SAP?"}' | jq .
```

## Testes

### Executar Testes

```bash
pytest tests/ -v
```

### Cenarios de Teste

**Teste 1 - Cache Hit Semantico** (`test_cache.py`)
- Insere resposta para "Como redefinir a senha do SAP?"
- Busca com "Qual o passo a passo para resetar a senha do meu SAP?"
- Valida retorno do cache com similaridade >= 0.95
- Requer Redis em execucao (pula automaticamente se indisponivel)

**Teste 2 - Cache Miss por Diferenca Semantica** (`test_cache.py`)
- Armazena vetor de pergunta SAP
- Busca com vetor ortogonal (pergunta de RH)
- Valida que o cache nao e ativado
- Requer Redis em execucao

**Teste 3 - Acuracia do Reranker** (`test_reranker.py`)
- Verifica scores decrescentes estaveis no resultado
- Valida integridade estrutural dos objetos retornados
- Verifica que chunks SAP aparecem no topo para perguntas SAP
- Requer modelo FlashRank baixado (pula se indisponivel)

**Teste 4 - Fallback do Reranker** (`test_reranker.py`)
- Simula modelo FlashRank inexistente
- Verifica funcionamento do passthrough (retorna chunks originais)

**Teste 5 - Fail-Safe de Cache** (`test_failsafe.py`)
- Configura URL Redis invalida
- Verifica que o sistema opera em modo fail-open
- Confirmacoes: `check_cache` retorna None, `store_cache` retorna False

## Estrutura de Arquivos

```
/
├── docker-compose.yml        # Infraestrutura Redis Stack
├── requirements.txt          # Dependencias do projeto
├── .env.example              # Template de variaveis de ambiente
├── README.md                 # Documentacao
├── src/
│   ├── __init__.py
│   ├── main.py               # Rotas FastAPI
│   ├── config/
│   │   ├── __init__.py
│   │   └── redis_client.py   # Conexao Redis e indice VSS
│   └── services/
│       ├── __init__.py
│       ├── cache_engine.py   # Cache semantico Redis VSS
│       ├── rerank_engine.py  # Reranking FlashRank
│       └── rag_pipeline.py   # Orquestrador do pipeline
└── tests/
    ├── __init__.py
    ├── test_cache.py         # Testes de cache semantico
    ├── test_reranker.py      # Testes de reranking
    └── test_failsafe.py      # Teste de fail-safe
```

## Licenca

Distribuido sob licenca MIT. Consulte o arquivo LICENSE para mais informacoes.
