from sentence_transformers import SentenceTransformer

# load once (important for latency)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str, batch_size=128):
    return model.encode(text,batch_size=batch_size, normalize_embeddings=True)