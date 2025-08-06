# 📘 recommendme: A Minimal Library for Dataset Embedding & Search

**recommendme** is a lightweight Python library to:
- Load question/problem datasets from CSV or JSON.
- Extract structured fields like title, description, tags, and difficulty.
- Generate dense vector embeddings using [SentenceTransformers]
- Index them using [FAISS] for efficient similarity search.

---

## 🔧 Features

- Easy conversion from raw CSV to structured JSON.
- Supports flexible field mapping and defaults.
- Embedding generation via BAAI/bge-small-en (or any other SBERT model).
- FAISS-based similarity indexing and retrieval.
---

## 📦 Installation

```bash
pip install -U sentence-transformers faiss-cpu pandas numpy
