from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from config import app_config

class SemanticChunker:
    def __init__(self):
        self.embedder = SentenceTransformer(app_config.embedding_model)
        
    def semantic_split(self, texts: List[str]) -> List[str]:
        chunks = []
        current_chunk = []
        
        embeddings = self.embedder.encode(texts)
        similarities = np.dot(embeddings, embeddings.T)
        
        for i in range(len(texts)):
            if not current_chunk:
                current_chunk.append(texts[i])
                continue
                
            similarity = similarities[i-1, i]
            if similarity < app_config.semantic_threshold:
                chunks.append("\n".join(current_chunk))
                current_chunk = [texts[i]]
            else:
                current_chunk.append(texts[i])
                
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

class HybridTextSplitter:
    def __init__(self):
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app_config.chunk_size,
            chunk_overlap=app_config.chunk_overlap
        )
        self.semantic_splitter = SemanticChunker()

    def split_text(self, text: str) -> List[str]:
        if app_config.chunking_method == "semantic":
            return self.semantic_splitter.semantic_split([text])
        return self.recursive_splitter.split_text(text)