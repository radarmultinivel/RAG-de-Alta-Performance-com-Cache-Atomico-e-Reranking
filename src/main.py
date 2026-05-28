# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.services.rag_pipeline import RAGPipeline

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

pipeline: RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Inicializando pipeline RAG...")
    pipeline = RAGPipeline()
    logger.info("Pipeline RAG pronto.")
    yield
    logger.info("Pipeline RAG encerrado.")


app = FastAPI(
    title="RAG Cache Atomico + Reranking Local",
    description="Motor RAG com cache semantico Redis VSS e reranking local FlashRank em CPU.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Como redefinir a senha do SAP?"
            }
        }


class QueryResponse(BaseModel):
    question: str
    response: str
    source: str
    cache_score: float | None = None
    num_chunks_used: int = 0
    llm_called: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Como redefinir a senha do SAP?",
                "response": "## Resposta\n\nPara redefinir a senha...",
                "source": "cache",
                "cache_score": 0.9734,
                "num_chunks_used": 0,
                "llm_called": False,
            }
        }


class IngestRequest(BaseModel):
    documents: list[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {"text": "Procedimento para redefinir senha SAP...", "metadata": {"source": "manual_sap.pdf", "page": 12}}
                ]
            }
        }


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "documents_ingested": 1,
            }
        }


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "rag-cache-atomico"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta vazia.")
    try:
        result = pipeline.query(request.question.strip())
        return QueryResponse(**result)
    except Exception as exc:
        logger.exception("Erro ao processar query")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="Nenhum documento.")
    try:
        pipeline.ingest(request.documents)
        return IngestResponse(status="ok", documents_ingested=len(request.documents))
    except Exception as exc:
        logger.exception("Erro ao ingerir documentos")
        raise HTTPException(status_code=500, detail=str(exc))
