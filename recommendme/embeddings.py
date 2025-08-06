import os 
import json 
import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en"

class Embedding:
    def __init__(self, path, model=DEFAULT_MODEL):
        self.model_name = model 
        self.path = path 
        self.filename = os.path.splitext(os.path.basename(path))[0]
        self.output_faiss_bin = f"embeddings/{self.filename}_faiss_index.bin"

    def generate_embeddings(self):
        """
        Generates sentence embeddings and saves:
        - FAISS binary index
        - Metadata JSON file
        """
        with open(self.path, 'r') as f:
            data = json.load(f)

        model = SentenceTransformer(self.model_name)

        texts = [
            f"{p['title']}. {p['desc']} Difficulty: {p['difficulty']}. Tags: {', '.join(p['tags'])}"
            for p in data
        ]

        embeddings = np.array(model.encode(texts, normalize_embeddings=True)).astype("float32")

        os.makedirs("embeddings", exist_ok=True)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        # Save metadata
        with open(f"embeddings/{self.filename}_id_map.json", "w") as f:
            json.dump(data, f, indent=2)

        # Save FAISS index
        faiss.write_index(index, f"embeddings/{self.filename}_faiss_index.bin")

    def get_faiss_embeddings(self):
        if not os.path.exists(self.output_faiss_bin):
            raise ValueError("FAISS index file does not exist. Generate embeddings first.")
        return faiss.read_index(self.output_faiss_bin)


    def get_metadata(self):
        if not os.path.exists(f"embeddings/{self.filename}_id_map.json") :
            raise ValueError("Metadata file does not exists")
        with open(f"embeddings/{self.filename}_id_map.json", 'r') as f:
            data = json.load(f)

        return data 