import faiss
import numpy as np
import os


def build_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def save_index(index, filename):
    index_dir = "indexes"
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, filename)
    faiss.write_index(index, path)
    return path


def load_index(path):
    return faiss.read_index(path)
