import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os


class SearchEngine:
    def __init__(self, index_path, metadata_path, model_name="BAAI/bge-small-en", threshold=0.3):
        """
        index_path: Path to FAISS index .bin file
        metadata_path: Path to metadata JSON file
        model_name: SentenceTransformer model
        threshold: Distance threshold to filter irrelevant results
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def encode_query(self, query):
        return self.model.encode([query], convert_to_numpy=True).astype("float32")

    def search(self, query, top_k=10, exclude_id=None):
        query_vector = self.encode_query(query)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= self.threshold:
                continue  
            if exclude_id is not None and self.metadata[idx]["id"] == exclude_id:
                continue  
            item = self.metadata[idx].copy()
            item["score"] = float(dist)
            results.append(item)

        return results
