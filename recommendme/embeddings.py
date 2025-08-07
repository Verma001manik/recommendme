import os
import json
import time
import faiss
import numpy as np
import pandas as pd
import torch 
from sentence_transformers import SentenceTransformer
from utils import clear_memory

DEFAULT_MODEL = "BAAI/bge-small-en"
device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedding:
    """
    A class to generate, manage, and save sentence embeddings using SentenceTransformers and FAISS.

    Supports standard and chunked processing for large datasets.
    """

    def __init__(self, path, model=DEFAULT_MODEL, text_template=None):
        """
        Initialize the embedding generator.

        Args:
            path (str): Path to the input JSON file.
            model (str): HuggingFace model name for SentenceTransformer.
            text_template (str, optional): Custom format string for text generation.
        """
        self.model_name = model
        self.path = path
        self.filename = os.path.splitext(os.path.basename(path))[0]
        self.output_faiss_bin = f"embeddings/{self.filename}_faiss_index.bin"
        self.output_metadata = f"embeddings/{self.filename}_id_map.json"
        self.text_template = text_template

        # Defaults for large dataset processing
        self.CHUNK_SIZE = 50000
        self.BATCH_SIZE = 64

    def get_chunk_size(self):
        """
        Returns the current chunk size used for chunked embedding generation.
        """
        return self.CHUNK_SIZE

    def set_chunk_size(self, cs):
        """
        Set the chunk size for chunked embedding generation.

        Args:
            cs (int): Chunk size to set. Must be a positive integer.
        """
        if not isinstance(cs, int) or cs < 1:
            raise ValueError("Chunk size must be a positive integer")
        self.CHUNK_SIZE = cs

    def get_batch_size(self):
        """
        Returns the current batch size used during model encoding.
        """
        return self.BATCH_SIZE

    def set_batch_size(self, bs):
        """
        Set the batch size for embedding generation.

        Args:
            bs (int): Batch size. Must be a positive integer.
        """
        if not isinstance(bs, int) or bs < 1:
            raise ValueError("Batch size must be a positive integer")
        self.BATCH_SIZE = bs

    def generate_embeddings(self, save=True):
        """
        Generates embeddings for the input JSON file and builds a FAISS index.

        Args:
            save (bool): If True, saves FAISS index and metadata to disk.
        """
        model = SentenceTransformer(self.model_name).to(device)

        df = self.load_and_prepare_df()
        texts = self.create_text_to_embed(df)

        embeddings = np.array(
            model.encode(texts, normalize_embeddings=True, device=device)
        ).astype("float32")

        os.makedirs("embeddings", exist_ok=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        if save:
            faiss.write_index(index, self.output_faiss_bin)
            df.to_json(self.output_metadata, orient='records', indent=2)

        del df
        clear_memory()

    def generate_embeddings_chunked(self, save=True):
        """
        Generates embeddings in chunks to handle large datasets efficiently (CUDA only).

        Args:
            save (bool): If True, saves FAISS index and metadata to disk.

        Raises:
            ValueError: If CUDA is not available.
        """
        if device != 'cuda':
            raise ValueError("CUDA is not available")

        model = SentenceTransformer(self.model_name).to(device)

        df = self.load_and_prepare_df()
        texts = self.create_text_to_embed(df)

        start_time = time.time()
        all_embeddings = []
        total_chunks = (len(texts) + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

        for i in range(0, len(texts), self.CHUNK_SIZE):
            chunk_num = i // self.CHUNK_SIZE + 1
            chunk_texts = texts[i:i + self.CHUNK_SIZE]
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_texts)} texts)")

            chunk_embeddings = model.encode(
                chunk_texts,
                batch_size=self.BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=device
            )

            all_embeddings.append(chunk_embeddings)

            del chunk_texts, chunk_embeddings
            clear_memory()

            elapsed = time.time() - start_time
            avg_time = elapsed / chunk_num
            estimated_remaining = avg_time * (total_chunks - chunk_num)

            print(f"Chunk {chunk_num} completed. Elapsed: {elapsed:.1f}s, "
                  f"Estimated remaining: {estimated_remaining:.1f}s")

        embeddings = np.vstack(all_embeddings).astype('float32')
        clear_memory()

        print(f"Total embedding generation took: {time.time() - start_time:.2f} seconds")
        print(f"Embeddings shape: {embeddings.shape}")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        batch_size_faiss = 10000
        for i in range(0, len(embeddings), batch_size_faiss):
            batch = embeddings[i:i + batch_size_faiss]
            index.add(batch)
            if i % 100000 == 0:
                print(f"Added {min(i + batch_size_faiss, len(embeddings))}/{len(embeddings)} embeddings to index")

        if save:
            faiss.write_index(index, self.output_faiss_bin)
            df.to_json(self.output_metadata, orient='records', indent=2)

        clear_memory()

    def load_and_prepare_df(self):
        """
        Loads the JSON file and cleans missing values.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            df = pd.read_json(self.path, lines=True)
        except:
            df = pd.read_json(self.path)

        df = df.fillna('')
        return df

    def create_text_to_embed(self, df):
        """
        Converts rows of the DataFrame into text strings for embedding.

        Args:
            df (pd.DataFrame): Input DataFrame with 'title' and 'desc' fields.

        Returns:
            List[str]: List of formatted strings for embedding.
        """
        if self.text_template:
            return [self.text_template.format_map(row.to_dict()) for _, row in df.iterrows()]
        else:
            df['title'] = df['title'].astype(str)
            df['desc'] = df['desc'].astype(str)
            return ("title: " + df['title'] + ". desc: " + df['desc']).tolist()

    def get_faiss_embeddings(self):
        """
        Loads the FAISS index from disk.

        Returns:
            faiss.Index: FAISS index object.

        Raises:
            ValueError: If FAISS index file does not exist.
        """
        if not os.path.exists(self.output_faiss_bin):
            raise ValueError("FAISS index file does not exist. Generate embeddings first.")
        return faiss.read_index(self.output_faiss_bin)

    def get_metadata(self):
        """
        Loads metadata from disk (JSON).

        Returns:
            List[Dict]: List of metadata entries.

        Raises:
            ValueError: If metadata file does not exist.
        """
        if not os.path.exists(self.output_metadata):
            raise ValueError("Metadata file does not exist.")
        with open(self.output_metadata, 'r') as f:
            return json.load(f)


class DefaultDict(dict):
    """
    A dictionary that returns an empty string for missing keys.
    Useful for text template rendering.
    """
    def __missing__(self, key):
        return ""
