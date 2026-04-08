import numpy as np
from sentence_transformers import SentenceTransformer

# Load once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Simple cache
_embedding_cache = {}

def safe_embedding(text: str):
    try:
        if not text or not text.strip():
            return np.zeros(384)  # MiniLM dimension

        if text in _embedding_cache:
            return _embedding_cache[text]

        emb = model.encode(text)
        emb = np.array(emb)

        _embedding_cache[text] = emb
        return emb

    except Exception as e:
        print("❌ Embedding failed:", e)
        return np.zeros(384)


def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))