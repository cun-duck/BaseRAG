from sentence_transformers import SentenceTransformer
from core.caching import embedding_cache
from config import app_config
from utils.logger import logger

embedder = SentenceTransformer(app_config.embedding_model)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    logger.info("Generating embeddings...")
    return embedder.encode(texts, show_progress_bar=True)