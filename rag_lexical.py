"""Optional BM25/TF-IDF retrieval with a dependency-free overlap fallback."""
import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List

import numpy as np

from rag_config import RAG_ENABLE_BM25, RAG_ENABLE_TFIDF


def normalize_lexical(text: str) -> str:
    value = unicodedata.normalize("NFD", (text or "").lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return " ".join(re.findall(r"[a-z0-9_.-]+", value))


def tokenize(text: str) -> List[str]:
    return [token for token in normalize_lexical(text).split() if len(token) > 1]


class LexicalRetriever:
    def __init__(self, chunks_metadata):
        self.metadata = [
            item for item in (chunks_metadata or [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        self.tokenized = [tokenize(str(item.get("text") or "")) for item in self.metadata]
        self.backend = "token_overlap"
        self.index = None
        self.matrix = None

        if RAG_ENABLE_BM25:
            try:
                from rank_bm25 import BM25Okapi

                self.index = BM25Okapi(self.tokenized)
                self.backend = "bm25"
                return
            except Exception as exc:
                print(f"Lexical BM25 indisponible; recherche d'un fallback ({exc})")

        if RAG_ENABLE_TFIDF or RAG_ENABLE_BM25:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self.index = TfidfVectorizer(
                    preprocessor=normalize_lexical,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                )
                self.matrix = self.index.fit_transform(
                    [str(item.get("text") or "") for item in self.metadata]
                )
                self.backend = "tfidf"
                return
            except Exception as exc:
                print(f"Lexical TF-IDF indisponible; fallback token overlap ({exc})")

    @property
    def enabled(self) -> bool:
        return bool(self.metadata) and (RAG_ENABLE_BM25 or RAG_ENABLE_TFIDF)

    def _overlap_scores(self, query_tokens: List[str]) -> np.ndarray:
        query_counts = Counter(query_tokens)
        scores = []
        for tokens in self.tokenized:
            counts = Counter(tokens)
            overlap = sum(min(query_counts[token], counts[token]) for token in query_counts)
            coverage = overlap / max(sum(query_counts.values()), 1)
            rarity = sum(1.0 / math.sqrt(1.0 + counts[token]) for token in query_counts if counts[token])
            scores.append(coverage + 0.05 * rarity)
        return np.asarray(scores, dtype="float32")

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        if not self.metadata or top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        if self.backend == "bm25":
            scores = np.asarray(self.index.get_scores(query_tokens), dtype="float32")
        elif self.backend == "tfidf":
            query_vector = self.index.transform([query])
            scores = np.asarray((self.matrix @ query_vector.T).toarray()).reshape(-1)
        else:
            scores = self._overlap_scores(query_tokens)

        order = np.argsort(-scores)[:min(top_k, len(scores))]
        max_score = float(np.max(scores)) if scores.size else 0.0
        results = []
        for idx in order:
            raw_score = float(scores[idx])
            if raw_score <= 0:
                continue
            item = self.metadata[int(idx)]
            score = raw_score / max_score if max_score > 0 else raw_score
            results.append(
                {
                    **item,
                    "chunk_id": item.get("chunk_id", int(idx)),
                    "document_id": item.get("document_id", f"Document_{idx}"),
                    "text": item.get("text", ""),
                    "source": item.get("source", "unknown"),
                    "lexical_score": float(score),
                    "score": float(score),
                    "retrieval_source": self.backend,
                    "method": self.backend,
                }
            )
        return results
