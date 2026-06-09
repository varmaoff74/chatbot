from .reranker import rerank
from .generator import generate_answer
from .embeddings import get_embedding
from .hybrid_search import hybrid_search


def query_pipeline(index, question, chunks, history_text=None, vector_k=20, bm25_k=20, fused_k=10, top_k=3):
    if chunks is None:
        raise ValueError("query_pipeline requires a list of chunks")

    # print(chunks)

    retrieved = hybrid_search(
        index,
        question,
        chunks,
        vector_k=vector_k,
        bm25_k=bm25_k,
        fused_k=fused_k,
        rrk=60,
    )

    ranked = rerank(question, retrieved, top_k=top_k)
    # print(ranked)
    return generate_answer(question, ranked, history_text=history_text)


