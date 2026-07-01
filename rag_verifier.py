"""Verify that an answer is grounded in retrieved local documentation."""
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from rag_answer import is_not_found_answer, normalize_for_match, split_sentences
from rag_config import (
    RAG_ENABLE_ANSWER_VERIFIER,
    RAG_ENABLE_LLM_ANSWER_VERIFIER,
    RAG_QUERY_LLM_MODEL,
    RAG_VERIFIER_MIN_CONFIDENCE,
)
from rag_vector import rank_by_embedding


RAG_VERIFIER_API_VERSION = 2
__all__ = ["RAG_VERIFIER_API_VERSION", "is_not_found_answer", "verify_answer"]

STOPWORDS = {
    "avec", "dans", "pour", "sans", "sur", "une", "des", "les", "est",
    "sont", "que", "qui", "comment", "quel", "quelle", "the", "and", "for",
    "with", "from", "what", "how", "this", "that", "are", "is",
}

GENERIC_ANSWERS = (
    "consultez la documentation",
    "cela depend",
    "suivez les instructions",
    "contactez votre administrateur",
    "refer to the documentation",
    "it depends",
)


def _terms(text: str) -> List[str]:
    return [
        token for token in normalize_for_match(text or "").split()
        if len(token) > 2 and token not in STOPWORDS
    ]


def _ratio_in_text(terms: List[str], text: str) -> float:
    normalized = normalize_for_match(text or "")
    unique = list(dict.fromkeys(terms))
    return (
        sum(1 for term in unique if term in normalized) / len(unique)
        if unique else 0.0
    )


def _unsupported_claims(answer: str, context: str) -> List[str]:
    claims = split_sentences(answer) or ([answer] if answer else [])
    unsupported = []
    for claim in claims:
        claim_terms = _terms(claim)
        if claim_terms and _ratio_in_text(claim_terms, context) < 0.45:
            unsupported.append(claim.strip())
    return unsupported


def _llm_verification(
    question: str,
    answer: str,
    context: str,
) -> Optional[Dict]:
    if not RAG_ENABLE_LLM_ANSWER_VERIFIER or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=RAG_QUERY_LLM_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Judge whether the answer is fully supported by the supplied "
                        "local documentation. Return JSON: answer_supported, "
                        "confidence, reason, unsupported_claims."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nAnswer: {answer}\n"
                        f"Documentation: {context[:12000]}"
                    ),
                },
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        return {
            "answer_supported": bool(data.get("answer_supported", False)),
            "confidence": confidence,
            "needs_retry": (
                not bool(data.get("answer_supported", False))
                or confidence < RAG_VERIFIER_MIN_CONFIDENCE
            ),
            "reason": str(data.get("reason") or "LLM groundedness judgment."),
            "unsupported_claims": list(data.get("unsupported_claims") or []),
            "backend": "llm",
        }
    except Exception as exc:
        print(f"   LLM verifier unavailable; using heuristics ({exc})")
        return None


def verify_answer(
    question: str,
    answer: Optional[str] = None,
    contexts: Optional[List[Dict]] = None,
    embedding_model=None,
    embedding_model_name: str = "",
    enabled: bool = RAG_ENABLE_ANSWER_VERIFIER,
    context_results: Optional[List[Dict]] = None,
) -> Dict:
    """Return groundedness, confidence, retry advice, and unsupported claims.

    The legacy call ``verify_answer(answer, contexts)`` remains supported.
    """
    if isinstance(answer, list) and contexts is None:
        contexts = answer
        answer = question
        question = ""
    contexts = contexts if contexts is not None else (context_results or [])
    answer = str(answer or "")

    if not enabled:
        return {
            "answer_supported": True,
            "confidence": 1.0,
            "needs_retry": False,
            "reason": "Answer verification disabled.",
            "unsupported_claims": [],
            "backend": "disabled",
        }

    context_parts = []
    for result in contexts:
        if not isinstance(result, dict):
            continue
        compressed = str(result.get("compressed_text") or "").strip()
        text = str(result.get("text") or "").strip()
        parent_text = str(result.get("parent_text") or "").strip()
        if compressed:
            context_parts.append(compressed)
        if text and text != compressed:
            context_parts.append(text)
        if parent_text and parent_text not in {compressed, text}:
            context_parts.append(parent_text)
    context = " ".join(context_parts).strip()
    if not context:
        return {
            "answer_supported": False,
            "confidence": 0.0,
            "needs_retry": True,
            "reason": "No retrieved context was available for verification.",
            "unsupported_claims": split_sentences(answer) or ([answer] if answer else []),
            "backend": "heuristic",
        }

    question_support = _ratio_in_text(_terms(question), context)
    if not answer or is_not_found_answer(answer):
        weak_context = question_support < 0.20
        return {
            "answer_supported": weak_context,
            "confidence": 0.80 if weak_context else 0.25,
            "needs_retry": not weak_context,
            "reason": (
                "The not-found answer is consistent with weak retrieved context."
                if weak_context
                else "Retrieved context appears related; a broader retry may find support."
            ),
            "unsupported_claims": [],
            "backend": "heuristic",
        }

    llm_result = _llm_verification(question, answer, context)
    if llm_result is not None:
        return llm_result

    answer_norm = normalize_for_match(answer)
    context_norm = normalize_for_match(context)
    if answer_norm and answer_norm in context_norm:
        return {
            "answer_supported": True,
            "confidence": 0.95,
            "needs_retry": False,
            "reason": "The answer is directly extracted from the retrieved context.",
            "unsupported_claims": [],
            "backend": "heuristic",
        }

    answer_terms = _terms(answer)
    token_support = _ratio_in_text(answer_terms, context)
    question_answer_coverage = _ratio_in_text(_terms(question), answer)
    context_sentences = split_sentences(context) or [context]
    sequence_support = max(
        SequenceMatcher(
            None,
            answer_norm,
            normalize_for_match(sentence),
        ).ratio()
        for sentence in context_sentences
    )

    embedding_support = 0.0
    if embedding_model is not None:
        ranked = rank_by_embedding(
            embedding_model,
            answer,
            context_sentences,
            model_name=embedding_model_name,
            top_k=1,
        )
        if ranked:
            embedding_support = max(0.0, min(1.0, float(ranked[0][1])))

    generic = (
        len(answer_terms) < 4
        or any(phrase in normalize_for_match(answer) for phrase in GENERIC_ANSWERS)
    )
    unsupported = _unsupported_claims(answer, context)
    unsupported_ratio = len(unsupported) / max(
        len(split_sentences(answer) or [answer]),
        1,
    )
    confidence = (
        0.50 * token_support
        + 0.20 * sequence_support
        + 0.20 * question_answer_coverage
        + 0.10 * (embedding_support if embedding_model is not None else token_support)
        - 0.20 * unsupported_ratio
        - (0.15 if generic else 0.0)
    )
    confidence = max(0.0, min(1.0, confidence))
    supported = (
        confidence >= RAG_VERIFIER_MIN_CONFIDENCE
        and token_support >= 0.55
        and unsupported_ratio < 0.5
        and not generic
    )
    return {
        "answer_supported": supported,
        "confidence": confidence,
        "needs_retry": not supported or confidence < RAG_VERIFIER_MIN_CONFIDENCE,
        "reason": (
            f"token_support={token_support:.2f}, "
            f"question_coverage={question_answer_coverage:.2f}, "
            f"sentence_support={sequence_support:.2f}, "
            f"unsupported_claim_ratio={unsupported_ratio:.2f}"
        ),
        "unsupported_claims": unsupported,
        "backend": "heuristic",
    }
