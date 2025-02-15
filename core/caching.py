from diskcache import Cache
from hashlib import sha256
import os
from config import app_config

class EmbeddingCache:
    def __init__(self):
        os.makedirs(app_config.cache_dir, exist_ok=True)
        self.cache = Cache(app_config.cache_dir)
        
    def get_key(self, content: str) -> str:
        return sha256(content.encode()).hexdigest()
        
    def __contains__(self, key: str) -> bool:
        return key in self.cache
        
    def __getitem__(self, key: str):
        return self.cache[key]
        
    def __setitem__(self, key: str, value):
        self.cache[key] = value

embedding_cache = EmbeddingCache()