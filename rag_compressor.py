"""Extractive contextual compression that never invents passage text."""
from typing import Dict, List

from rag_answer import (
    extract_question_terms,
    normalize_for_match,
    score_sentence,
    split_sentences,
)
from rag_config import (
    ANSWER_LEXICAL_WEIGHT,
    ANSWER_VECTOR_WEIGHT,
    RAG_ENABLE_CONTEXT_COMPRESSION,
    RAG_MAX_COMPRESSED_SENTENCES,
)
from rag_vector import rank_by_embedding


def compress_context(
    query: str,
    candidates: List[Dict],
    max_sentences: int = RAG_MAX_COMPRESSED_SENTENCES,
    embedding_model=None,
    embedding_model_name: str = "",
) -> List[Dict]:
    if not candidates:
        return []

    question_norm = normalize_for_match(query)
    keywords = extract_question_terms(query)
    sentence_rows = []
    for candidate_index, candidate in enumerate(candidates):
        original_text = str(candidate.get("text") or "")
        sentences = split_sentences(original_text)
        if not sentences and original_text.strip():
            sentences = [original_text.strip()]
        for sentence_index, sentence in enumerate(sentences):
            lexical = min(1.0, score_sentence(question_norm, keywords, sentence))
            keyword_hits = sum(
                1 for keyword in keywords
                if keyword in normalize_for_match(sentence)
            ) / max(len(keywords), 1)
            sentence_rows.append(
                {
                    "candidate_index": candidate_index,
                    "sentence_index": sentence_index,
                    "sentence": sentence,
                    "lexical_score": lexical,
                    "keyword_score": keyword_hits,
                }
            )

    if not sentence_rows:
        return [dict(candidate) for candidate in candidates]

    vector_scores = {}
    if embedding_model is not None:
        try:
            sentences = [row["sentence"] for row in sentence_rows]
            for idx, score, _ in rank_by_embedding(
                embedding_model,
                query,
                sentences,
                model_name=embedding_model_name,
                top_k=len(sentences),
            ):
                vector_scores[idx] = score
        except Exception as exc:
            print(f"   Compression vectorielle indisponible; mode lexical ({exc})")

    for idx, row in enumerate(sentence_rows):
        vector = float(vector_scores.get(idx, 0.0))
        lexical = 0.75 * row["lexical_score"] + 0.25 * row["keyword_score"]
        if embedding_model is not None and vector_scores:
            row["compression_score"] = (
                ANSWER_VECTOR_WEIGHT * vector
                + ANSWER_LEXICAL_WEIGHT * lexical
            )
        else:
            row["compression_score"] = lexical

    selected = sorted(
        sentence_rows,
        key=lambda row: row["compression_score"],
        reverse=True,
    )[:max(1, int(max_sentences))]
    selected_by_candidate = {}
    for row in selected:
        selected_by_candidate.setdefault(row["candidate_index"], []).append(row)

    compressed = []
    for candidate_index, candidate in enumerate(candidates):
        item = dict(candidate)
        original_text = str(candidate.get("text") or "")
        item["original_text"] = original_text
        rows = selected_by_candidate.get(candidate_index, [])
        rows.sort(key=lambda row: row["sentence_index"])
        compressed_text = " ".join(row["sentence"] for row in rows).strip()
        if len(compressed_text) < 40:
            compressed_text = original_text
        item["compressed_text"] = compressed_text
        item["compressed_sentences"] = [row["sentence"] for row in rows]
        item["compression_score"] = max(
            [row["compression_score"] for row in rows] or [0.0]
        )
        item["citation"] = {
            "source": item.get("source"),
            "title": item.get("title"),
            "section": item.get("section"),
            "module": item.get("module"),
            "parent_id": item.get("parent_id"),
        }
        compressed.append(item)
    return compressed


class ContextualCompressor:
    def __init__(
        self,
        enabled: bool = RAG_ENABLE_CONTEXT_COMPRESSION,
        max_sentences: int = RAG_MAX_COMPRESSED_SENTENCES,
    ):
        self.enabled = enabled
        self.max_sentences = max(1, int(max_sentences))

    def compress(
        self,
        query: str,
        candidates: List[Dict],
        embedding_model=None,
        embedding_model_name: str = "",
    ) -> List[Dict]:
        if not self.enabled:
            return [dict(candidate) for candidate in candidates]
        return compress_context(
            query,
            candidates,
            max_sentences=self.max_sentences,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
        )
