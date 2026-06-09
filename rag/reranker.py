from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question, chunks, top_k=3):
    pairs = [(question, chunk) for chunk in chunks]
    # pairs = []
    # for chunk in chunks:
    #     print(chunk, end="\n\n\n")
    #     pairs.append((question, chunk))

    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    for item in ranked[0][:3]:
        print(item)
        print("\n\n\n")

    return [c[0] for c in ranked[:top_k]]