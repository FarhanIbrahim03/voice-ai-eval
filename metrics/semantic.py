from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

@dataclass
class SemanticResult:
    score: float

@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def compute_semantic_similarity(
    reference: str,
    hypothesis: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> SemanticResult:
    if not reference.strip() or not hypothesis.strip():
        return SemanticResult(score=0.0)

    model = _load_model(model_name)

    embeddings = model.encode(
        [reference, hypothesis],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    score = float(np.dot(embeddings[0], embeddings[1]))
    score = max(0.0, min(1.0, score))

    return SemanticResult(score=round(score, 6))