"""
Correspondance vectorielle question → réponses curatées (k-NN embeddings).
Pas de règles if/else : recherche par similarité cosinus dans l'espace des questions.
"""
import json
import os
from typing import List, Optional, Tuple

import numpy as np

from rag_config import EMBEDDING_MODEL, is_e5_model, E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
from rag_vector import encode_query, encode_passages, cosine_scores, top_k_indices

INDEX_PATH = "curated_qa_vectors.npz"
_META: List[Tuple[str, str]] = []
_VECTORS: Optional[np.ndarray] = None
_MODEL = None
_MODEL_NAME = EMBEDDING_MODEL
_MIN_SCORE = float(os.getenv("RAG_CURATED_MIN_SCORE", "0.78"))


def _passage_text(question: str) -> str:
    q = f"{E5_PASSAGE_PREFIX}{question}" if is_e5_model(_MODEL_NAME) else question
    return q


def _load_meta() -> None:
    global _META
    if _META:
        return
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "curated_qa.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("entries", []):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            _META.append((q, a))


def _ensure_index(model) -> bool:
    global _VECTORS, _MODEL, _MODEL_NAME
    _load_meta()
    if not _META or model is None:
        return False
    _MODEL = model
    _MODEL_NAME = getattr(model, "_model_card_vars", {}).get("model_name", EMBEDDING_MODEL)
    root = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(root, INDEX_PATH)
    questions = [q for q, _ in _META]
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        if data["model_name"].item() == _MODEL_NAME and len(data["vectors"]) == len(_META):
            _VECTORS = data["vectors"]
            return True
    texts = [_passage_text(q) for q in questions]
    _VECTORS = encode_passages(model, texts, _MODEL_NAME)
    np.savez(cache, vectors=_VECTORS, model_name=np.array(_MODEL_NAME))
    return True


def lookup_by_vector(question: str, model, model_name: str = "") -> Optional[str]:
    """Retourne la réponse curatée la plus proche par embedding (k-NN)."""
    global _MODEL_NAME
    if model_name:
        _MODEL_NAME = model_name
    if not _ensure_index(model):
        return None
    q_vec = encode_query(model, question, _MODEL_NAME)
    scores = cosine_scores(q_vec, _VECTORS)
    if scores.size == 0:
        return None
    best_idx = int(top_k_indices(scores, 1)[0])
    if float(scores[best_idx]) >= _MIN_SCORE:
        return _META[best_idx][1]
    return None
