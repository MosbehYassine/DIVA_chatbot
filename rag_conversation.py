"""Deterministic conversation reformulation for follow-up RAG questions."""
import re
import unicodedata
from typing import Dict, List


FOLLOW_UP_PATTERNS = (
    r"^(et|puis|ensuite|alors)\b",
    r"\b(le|la|les|lui|leur|cela|ça|ca|ceci|celui|celle|it|this|that|them)\b",
)

STOPWORDS = {
    "comment", "faire", "quel", "quelle", "quels", "quelles", "pourquoi",
    "dans", "avec", "sans", "depuis", "harmony", "divalto", "peut", "on",
    "creer", "créer", "une", "un", "des", "les", "la", "le",
    "the", "how", "what", "where", "with", "from", "into", "does",
}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").lower())
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _is_follow_up(question: str) -> bool:
    normalized = _normalize(question).strip()
    if any(re.search(pattern, normalized) for pattern in FOLLOW_UP_PATTERNS):
        return True
    return len(normalized.split()) <= 5 and normalized.startswith(
        ("comment ", "et ", "where ", "how ")
    )


def _topic_from_history(history: List[Dict]) -> str:
    if not history:
        return ""
    last = history[-1]
    candidates = [
        str(last.get("question") or last.get("user_message") or ""),
        str(last.get("answer") or last.get("assistant_answer") or ""),
    ]
    for candidate in candidates:
        words = re.findall(r"[A-Za-zÀ-ÿ0-9_.-]+", candidate)
        useful = [
            word for word in words
            if len(word) > 2 and _normalize(word) not in STOPWORDS
        ]
        if useful:
            return " ".join(useful[-4:])
    return ""


def reformulate_with_history(question: str, history: List[Dict]) -> str:
    """Turn a context-dependent follow-up into a standalone question."""
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    if not question or not history or not _is_follow_up(question):
        return question
    topic = _topic_from_history(history)
    if not topic:
        return question

    normalized = _normalize(question)
    if "imprim" in normalized:
        return f"Comment imprimer {topic} dans Harmony/Divalto ?"
    if normalized.startswith(("et ", "puis ", "ensuite ", "alors ")):
        question = re.sub(
            r"^(et|puis|ensuite|alors)\s+",
            "",
            question,
            flags=re.IGNORECASE,
        )
    question = re.sub(
        r"\b(le|la|les|lui|leur|cela|ça|ca|ceci|it|this|that|them)\b",
        topic,
        question,
        count=1,
        flags=re.IGNORECASE,
    )
    if topic.casefold() not in question.casefold():
        question = f"{question.rstrip(' ?')} concernant {topic}"
    return f"{question.rstrip(' ?')} dans Harmony/Divalto ?"


__all__ = ["reformulate_with_history"]
