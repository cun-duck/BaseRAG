# 🚀 BaseRAG: Advanced Retrieval-Augmented Generation Framework

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Gradio Version](https://img.shields.io/badge/gradio-3.0%2B-orange)

**BaseRAG** is a production-ready framework for building and evaluating Retrieval-Augmented Generation (RAG) systems with integrated performance analytics. Designed for developers and researchers, it combines state-of-the-art language models with modular document processing and real-time metrics visualization.



## 🔥 Key Features

- **Multi-LLM Support**
  - DeepSeek, OpenAI, Gemini, and Claude integration
  - Easy model switching via unified interface

- **Smart Document Processing**
  - Dynamic chunking with configurable sizes (256-2048 tokens)
  - Semantic caching for efficient retrieval
  - File format support (PDF, TXT, DOCX)

- **Real-Time Analytics Dashboard**
  - Processing time breakdowns
  - Precision@k and Recall@k metrics
  - Token consumption tracking

- **Enterprise-Grade Architecture**
  - Modular service components
  - Async-ready pipeline
  - Production error handling

## 🛠️ Technologies

- **Backbone**: PyTorch, Hugging Face Transformers
- **Vector DB**: FAISS (On-device)
- **UI**: Gradio
- **Analytics**: Matplotlib, Pandas

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
-  API Key

## 🧩 Project Structure
Copy
BaseRAG/
├── core/               # RAG Engine Components
│   ├── retrieval.py    # Hybrid search algorithms
│   ├── chunking.py     # Adaptive document splitting
│   └── caching.py      # Semantic cache layer
│
├── models/             # LLM Integration
│   ├── llm.py          # Unified model interface
│   └── embedding.py    # Vectorization pipelines
│
├── utils/              # Infrastructure
│   ├── logger.py       # Production logging
│   └── error_handling.py # Fault tolerance
│
├── app.py              # Main application
├── config.py           # Runtime configuration
└── requirements.txt    # Dependency manifest
