from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from pathlib import Path
from core.chunking import HybridTextSplitter
from utils.logger import logger
from config import app_config

LOADERS = {
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.txt': TextLoader,
    '.md': UnstructuredMarkdownLoader
}

def process_uploaded_file(file_path: str) -> list[str]:
    ext = Path(file_path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file format: {ext}")
    
    loader = LOADERS[ext](file_path)
    docs = loader.load()
    
    
    if ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    else:
        raw_text = " ".join([doc.page_content for doc in docs])
    
    return raw_text