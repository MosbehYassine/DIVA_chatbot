"""Maximal Marginal Relevance diversification for retrieved passages."""
import hashlib
from typing import Callable, Dict, List

import numpy as np


def _fallback_deduplicate(candidates: List[Dict], top_k: int) -> List[Dict]:
    seen = set()
    output = []
    for candidate in candidates:
        text = " ".join(str(candidate.get("text") or "").lower().split())
        source = str(candidate.get("source") or "")
        key = (source, hashlib.sha1(text.encode("utf-8")).hexdigest())
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(dict(candidate))
        if len(output) >= top_k:
            break
    return output


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def maximal_marginal_relevance(
    query_embedding,
    candidates,
    embedding_fn: Callable,
    top_k: int = 20,
    lambda_mult: float = 0.7,
):
    """Select relevant but non-redundant candidates while preserving metadata."""
    if not candidates or top_k <= 0:
        return []
    pool = []
    seen_exact = set()
    for candidate in candidates:
        text = " ".join(str(candidate.get("text") or "").lower().split())
        source = str(candidate.get("source") or "")
        key = (source, hashlib.sha1(text.encode("utf-8")).hexdigest())
        if not text or key in seen_exact:
            continue
        seen_exact.add(key)
        pool.append(dict(candidate))
    if len(pool) <= 1:
        return pool[:top_k]

    try:
        candidate_embeddings = []
        missing_indices = []
        for idx, candidate in enumerate(pool):
            existing = candidate.get("embedding")
            if existing is None:
                candidate_embeddings.append(None)
                missing_indices.append(idx)
            else:
                candidate_embeddings.append(np.asarray(existing, dtype="float32").reshape(-1))

        if missing_indices:
            texts = [str(pool[idx].get("text") or "") for idx in missing_indices]
            computed = np.asarray(embedding_fn(texts), dtype="float32")
            if computed.ndim == 1:
                computed = computed.reshape(1, -1)
            for row, idx in enumerate(missing_indices):
                candidate_embeddings[idx] = computed[row]

        matrix = np.vstack(candidate_embeddings).astype("float32")
        matrix = _normalize_rows(matrix)
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        query = _normalize_rows(query)[0]
        relevance = matrix @ query

        selected = []
        remaining = list(range(len(pool)))
        while remaining and len(selected) < min(top_k, len(pool)):
            best_idx = None
            best_score = -float("inf")
            for idx in remaining:
                redundancy = 0.0
                if selected:
                    redundancy = max(float(matrix[idx] @ matrix[chosen]) for chosen in selected)
                score = lambda_mult * float(relevance[idx]) - (1.0 - lambda_mult) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        output = []
        for rank, idx in enumerate(selected):
            candidate = pool[idx]
            candidate["mmr_score"] = float(
                lambda_mult * relevance[idx]
                - (1.0 - lambda_mult)
                * max(
                    [float(matrix[idx] @ matrix[other]) for other in selected[:rank]] or [0.0]
                )
            )
            output.append(candidate)
        return output
    except Exception as exc:
        print(f"   MMR embeddings indisponibles; fallback deduplication ({exc})")
        return _fallback_deduplicate(pool, top_k)
