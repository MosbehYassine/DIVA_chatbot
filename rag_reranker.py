"""Cross-encoder reranking with two optional backends and a score fallback."""
import math
import os
from typing import Dict, List, Optional

from rag_config import (
    RAG_CROSS_ENCODER_MODEL,
    RAG_ENABLE_CROSS_ENCODER_RERANKER,
    RAG_FINAL_TOP_K,
    RAG_RERANKER_BATCH_SIZE,
    RAG_RERANKER_USE_FP16,
)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = RAG_CROSS_ENCODER_MODEL,
        enabled: bool = RAG_ENABLE_CROSS_ENCODER_RERANKER,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.model = None
        self.backend = "combined_score"
        self._load_attempted = False
        self.unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if self._load_attempted or not self.enabled:
            return
        self._load_attempted = True
        try:
            from FlagEmbedding import FlagReranker

            use_fp16 = RAG_RERANKER_USE_FP16
            try:
                import torch

                use_fp16 = use_fp16 and torch.cuda.is_available()
            except Exception:
                use_fp16 = False
            self.model = FlagReranker(
                self.model_name,
                use_fp16=use_fp16,
            )
            self.backend = "flag_embedding"
            print(f"   Cross-encoder FlagEmbedding charge: {self.model_name}")
            return
        except Exception as flag_exc:
            self.unavailable_reason = str(flag_exc)

        try:
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name,
                device="cpu",
            )
            self.backend = "sentence_transformers"
            print(f"   Cross-encoder SentenceTransformers charge: {self.model_name}")
            return
        except Exception as st_exc:
            self.model = None
            self.backend = "combined_score"
            self.unavailable_reason = (
                f"FlagEmbedding: {self.unavailable_reason}; "
                f"SentenceTransformers: {st_exc}"
            )
            print(
                "   Cross-encoder indisponible; fallback combined_score "
                f"({self.unavailable_reason})"
            )

    @staticmethod
    def _fallback_score(candidate: Dict) -> float:
        for key in (
            "combined_score",
            "rerank_score",
            "hybrid_score",
            "vector_score",
            "lexical_score",
            "score",
        ):
            try:
                return float(candidate.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _normalize_logit(score: float) -> float:
        if 0.0 <= score <= 1.0:
            return score
        score = max(-60.0, min(60.0, score))
        return 1.0 / (1.0 + math.exp(-score))

    def _score(self, pairs):
        if self.backend == "flag_embedding":
            try:
                return self.model.compute_score(
                    pairs,
                    batch_size=RAG_RERANKER_BATCH_SIZE,
                    normalize=True,
                )
            except TypeError:
                return self.model.compute_score(pairs)
        return self.model.predict(
            pairs,
            batch_size=RAG_RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = RAG_FINAL_TOP_K,
    ) -> List[Dict]:
        if not candidates:
            return []
        limit = len(candidates) if top_k is None else max(0, int(top_k))
        ranked = [dict(candidate) for candidate in candidates]
        for candidate in ranked:
            candidate["pre_cross_encoder_score"] = self._fallback_score(candidate)

        self._load()
        if self.model is None:
            for candidate in ranked:
                candidate["cross_encoder_score"] = candidate["pre_cross_encoder_score"]
                candidate["cross_encoder_fallback"] = True
                candidate["cross_encoder_backend"] = "combined_score"
            ranked.sort(key=lambda item: item["cross_encoder_score"], reverse=True)
            return ranked[:limit]

        pairs = [
            [query, str(candidate.get("compressed_text") or candidate.get("text") or "")[:8000]]
            for candidate in ranked
        ]
        try:
            scores = self._score(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if not isinstance(scores, (list, tuple)):
                scores = [scores]
            for candidate, score in zip(ranked, scores):
                candidate["cross_encoder_score"] = self._normalize_logit(float(score))
                candidate["cross_encoder_fallback"] = False
                candidate["cross_encoder_backend"] = self.backend
            for candidate in ranked[len(scores):]:
                candidate["cross_encoder_score"] = candidate["pre_cross_encoder_score"]
                candidate["cross_encoder_fallback"] = True
                candidate["cross_encoder_backend"] = "combined_score"
        except Exception as exc:
            self.unavailable_reason = str(exc)
            print(f"   Echec cross-encoder; fallback combined_score ({exc})")
            for candidate in ranked:
                candidate["cross_encoder_score"] = candidate["pre_cross_encoder_score"]
                candidate["cross_encoder_fallback"] = True
                candidate["cross_encoder_backend"] = "combined_score"

        ranked.sort(key=lambda item: item["cross_encoder_score"], reverse=True)
        return ranked[:limit]
