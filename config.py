from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    cache_dir: str = "./embeddings_cache"
    chunking_method: str = "semantic"
    semantic_threshold: float = 0.85
    chunk_size: int = 1024
    chunk_overlap: int = 128
    supported_formats: list = field(
        default_factory=lambda: ['.pdf', '.docx', '.txt', '.md']
    )
    log_level: str = "INFO"
    hf_api_key: str = field(default_factory(lambda: os.getenv("HF_API_KEY"))

    def __post_init__(self):
        if not self.hf_api_key.startswith("hf_"):
            raise ValueError(
                "Invalid Hugging Face API Key format. Please check your .env file\n"
                "Format token harus: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            )

app_config = AppConfig()