import logging
from config import app_config

logging.basicConfig(
    level=app_config.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("RAGSystem")