# Desenvolvido por L. A. Leandro - Sao Jose dos Campos - SP - 28/05/2026

import os
import logging
from typing import Optional

import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

VECTOR_DIM = 384
INDEX_NAME = "idx:semantic_cache"
DOC_PREFIX = "semantic_cache:"
DISTANCE_METRIC = "COSINE"

_pool: Optional[redis.ConnectionPool] = None
_client: Optional[redis.Redis] = None


def _get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def get_redis_client() -> redis.Redis:
    global _pool, _client
    if _client is not None:
        try:
            _client.ping()
            return _client
        except Exception:
            logger.warning("Conexao Redis perdida. Reconectando...")
            _client = None
            _pool = None
    url = _get_redis_url()
    _pool = redis.ConnectionPool.from_url(
        url,
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=3,
        retry_on_timeout=True,
        max_connections=20,
    )
    _client = redis.Redis(connection_pool=_pool)
    try:
        _client.ping()
        logger.info("Conectado ao Redis em %s", url)
    except Exception as exc:
        logger.warning("Redis indisponivel em %s: %s", url, exc)
    return _client


def create_vss_index(client: redis.Redis) -> bool:
    try:
        client.execute_command("FT.INFO", INDEX_NAME)
        logger.info("Indice VSS '%s' ja existe.", INDEX_NAME)
        return True
    except redis.ResponseError:
        pass
    try:
        client.execute_command("FT.CREATE", INDEX_NAME, "ON", "HASH",
                               "PREFIX", "1", DOC_PREFIX,
                               "SCHEMA",
                               "vector_data", "VECTOR", "FLAT", "6",
                               "TYPE", "FLOAT32", "DIM", str(VECTOR_DIM),
                               "DISTANCE_METRIC", DISTANCE_METRIC,
                               "question", "TEXT", "weight", "1.0",
                               "response", "TEXT", "weight", "0.5")
        logger.info("Indice VSS '%s' criado.", INDEX_NAME)
        return True
    except redis.ResponseError as exc:
        logger.error("Falha ao criar indice VSS: %s", exc)
        return False


def init_redis() -> Optional[redis.Redis]:
    client = get_redis_client()
    try:
        client.ping()
    except Exception:
        logger.warning("Redis nao disponivel — operando sem cache.")
        return None
    create_vss_index(client)
    return client
