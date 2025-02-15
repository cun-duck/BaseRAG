import hashlib
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document
import logging

logger = logging.getLogger("RAGSystem")

class DocumentHandler:
    def __init__(self):
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader
        }
    
    def get_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {str(e)}")
            raise
    
    def load_document(self, file_path: str) -> List[Document]:
        try:
            ext = Path(file_path).suffix.lower()
            if ext not in self.loaders:
                raise ValueError(f"Unsupported file format: {ext}")
            
            loader = self.loaders[ext](file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Document loading error: {str(e)}")
            raise
    
    def sanitize_content(self, content: str) -> str:
        try:
            return content.encode('utf-8', 'replace').decode('utf-8')
        except UnicodeDecodeError as ude:
            logger.warning(f"Unicode decode error: {str(ude)}")
            return content.encode('utf-8', 'ignore').decode('utf-8')
        except Exception as e:
            logger.error(f"Content sanitization error: {str(e)}")
            return content