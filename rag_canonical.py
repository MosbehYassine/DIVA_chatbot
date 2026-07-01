"""Small canonical text helpers used by the RAG reranker."""
import re
import unicodedata
from typing import Optional


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_doc_about_topic(question: str) -> Optional[str]:
    """Extract topic from questions like: Que dit la documentation a propos de X ?"""
    normalized = normalize_for_match(question).replace("  ", " ")
    match = re.search(
        r"documentation\s+a\s+propos\s+de\s*:?\s*(.+?)\s*\??\s*$",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()
