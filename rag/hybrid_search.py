import math
import re
from collections import Counter, defaultdict
import numpy as np

from .embeddings import get_embedding


def tokenize_text(text):
    return re.findall(r"\w+", text.lower())


def build_bm25(corpus_tokens):
    N = len(corpus_tokens)
    doc_freq = defaultdict(int)
    doc_term_freq = []
    doc_lengths = []

    for tokens in corpus_tokens:
        term_freq = Counter(tokens)
        doc_term_freq.append(term_freq)
        doc_lengths.append(len(tokens))
        for term in term_freq:
            doc_freq[term] += 1

    avgdl = sum(doc_lengths) / N if N else 0.0
    return {
        "N": N,
        "doc_freq": doc_freq,
        "doc_term_freq": doc_term_freq,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
    }

# k1 → controls importance of repeated words in documents
# k2 → controls importance of repeated words in the query
# b  → controls document length normalization
def bm25_scores(question, corpus_tokens, bm25_data, k1=1.5, b=0.75, k2=100):
    query_tokens = tokenize_text(question)
    if not query_tokens:
        return [0.0] * bm25_data["N"]

    q_freq = Counter(query_tokens)
    N = bm25_data["N"]
    avgdl = bm25_data["avgdl"]
    doc_freq = bm25_data["doc_freq"]
    doc_term_freq = bm25_data["doc_term_freq"]
    doc_lengths = bm25_data["doc_lengths"]

    scores = [0.0] * N
    for term, qf in q_freq.items():
        df = doc_freq.get(term, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        q_weight = ((k2 + 1) * qf) / (k2 + qf)
        for idx in range(N):
            tf = doc_term_freq[idx].get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * doc_lengths[idx] / avgdl)
            scores[idx] += idf * q_weight * ((tf * (k1 + 1)) / denom)

    return scores


def reciprocal_rank_fusion(rankings, k=60):
    fused_scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused_scores[idx] += 1.0 / (k + rank)

    sorted_inds = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in sorted_inds]


def hybrid_search(index, question, chunks, vector_k=20, bm25_k=20, fused_k=10, rrk=60):
    if not chunks:
        return []

    query_embedding = get_embedding(question)
    query_vec = np.array([query_embedding]).astype("float32")
    _, vector_indices = index.search(query_vec, vector_k)
    vector_indices = [int(idx) for idx in vector_indices[0] if idx >= 0 and idx < len(chunks)]

    corpus_tokens = [tokenize_text(chunk) for chunk in chunks]
    bm25_data = build_bm25(corpus_tokens)
    bm25_scores_list = bm25_scores(question, corpus_tokens, bm25_data)
    bm25_indices = sorted(range(len(bm25_scores_list)), key=lambda i: bm25_scores_list[i], reverse=True)[:bm25_k]

    fused_indices = reciprocal_rank_fusion([vector_indices, bm25_indices], k=rrk)
    fused_indices = [idx for idx in fused_indices if idx >= 0 and idx < len(chunks)]
    fused_indices = fused_indices[:fused_k]

    return [chunks[idx] for idx in fused_indices]
