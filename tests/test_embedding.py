import os
import json
import tempfile
import shutil
import numpy as np
import pytest

from recommendme.embeddings import Embedding
sample_data = [
    {
        "title": "Q1",
        "desc": "What is 2+2?",
        "tags": ["math", "easy"],
        "difficulty": "Easy"
    },
    {
        "title": "Q2",
        "desc": "Explain gravity.",
        "tags": ["physics", "medium"],
        "difficulty": "Medium"
    }
]

@pytest.fixture(scope="module")
def temp_json_file():
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "sample.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)
    yield file_path
    shutil.rmtree(temp_dir)


def test_generate_embeddings_creates_faiss_and_metadata(temp_json_file):
    embedder = Embedding(temp_json_file)
    embedder.generate_embeddings()

    faiss_path = f"embeddings/sample.json_faiss_index.bin"
    metadata_path = f"embeddings/sample.json_id_map.json"

    assert os.path.exists(faiss_path)
    assert os.path.exists(metadata_path)

    # Clean up
    os.remove(faiss_path)
    os.remove(metadata_path)


def test_get_faiss_embeddings_returns_index(temp_json_file):
    embedder = Embedding(temp_json_file)
    embedder.generate_embeddings()
    index = embedder.get_faiss_embeddings()
    assert isinstance(index.ntotal, int)
    assert index.ntotal == 2

    # Clean up
    os.remove(embedder.output_faiss_bin)
    os.remove(f"embeddings/{embedder.filename}_id_map.json")


def test_get_metadata_returns_data(temp_json_file):
    embedder = Embedding(temp_json_file)
    embedder.generate_embeddings()
    metadata = embedder.get_metadata()
    assert isinstance(metadata, list)
    assert metadata[0]["title"] == "Q1"

    # Clean up
    os.remove(embedder.output_faiss_bin)
    os.remove(f"embeddings/{embedder.filename}_id_map.json")
