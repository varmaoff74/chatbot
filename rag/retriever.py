import numpy as np

def retrieve(index, query_embedding, chunks, k=20):
    scores, indices = index.search(np.array([query_embedding]).astype("float32"), k)

    return [chunks[i] for i in indices[0]]