import os
import pickle
from .loader import load_pdf
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=150)

# parents = parent_splitter.split_documents(text)

# # print(parents)

# child_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)

# children = []

# for parent_id, parent in enumerate(parents):
#     child_chunks = child_splitter.split_documents(parent)
#     for child in child_chunks:
#         children.append((parent_id, child))
#     # print(parent_id, end="\n")






# default chunking parameters
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_OVERLAP = 150


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def recursive_chunking(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def save_chunks(chunks, chunk_path):
    directory = os.path.dirname(chunk_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    return chunk_path


def load_chunks(chunk_path):
    if chunk_path.lower().endswith(".pkl"):
        with open(chunk_path, "rb") as f:
            return pickle.load(f)
    if chunk_path.lower().endswith(".pdf"):
        return load_pdf(chunk_path)

    with open(chunk_path, "r", encoding="utf-8") as f:
        return f.read()