"""
Utilitaires vectoriels partagés (embeddings, similarité cosinus).
Utilisés par le retrieval graphe/vectoriel et la génération de réponse.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rag_config import E5_QUERY_PREFIX, E5_PASSAGE_PREFIX, is_e5_model


def encode_query(model, text: str, model_name: str = "") -> np.ndarray:
    """Encode une question (préfixe query: pour E5)."""
    q = f"{E5_QUERY_PREFIX}{text}" if is_e5_model(model_name) else text
    vec = model.encode([q], normalize_embeddings=True)
    return np.array(vec, dtype="float32").reshape(-1)


def encode_passages(model, texts: Sequence[str], model_name: str = "") -> np.ndarray:
    """Encode des passages (préfixe passage: pour E5)."""
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    if is_e5_model(model_name):
        texts = [f"{E5_PASSAGE_PREFIX}{t}" for t in texts]
    vecs = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")


def cosine_scores(query_vec: np.ndarray, passage_vecs: np.ndarray) -> np.ndarray:
    """Similarité cosinus (vecteurs déjà normalisés → produit scalaire)."""
    if passage_vecs.size == 0:
        return np.array([], dtype="float32")
    q = query_vec.reshape(1, -1)
    return (passage_vecs @ q.T).reshape(-1)


def top_k_indices(scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0 or k <= 0:
        return []
    k = min(k, scores.size)
    if k == scores.size:
        order = np.argsort(-scores)
    else:
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
    return order.tolist()


def rank_by_embedding(
    model,
    query: str,
    candidates: Sequence[str],
    model_name: str = "",
    top_k: Optional[int] = None,
    min_score: float = 0.0,
) -> List[Tuple[int, float, str]]:
    """Classe des candidats textuels par similarité vectorielle avec la question."""
    cleaned = [(i, (c or "").strip()) for i, c in enumerate(candidates) if (c or "").strip()]
    if not cleaned or model is None:
        return []
    indices, texts = zip(*cleaned)
    q_vec = encode_query(model, query, model_name)
    p_vecs = encode_passages(model, texts, model_name)
    scores = cosine_scores(q_vec, p_vecs)
    ranked = sorted(
        ((int(indices[i]), float(scores[i]), texts[i]) for i in range(len(texts)) if scores[i] >= min_score),
        key=lambda x: x[1],
        reverse=True,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked
