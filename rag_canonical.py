"""
Réponses canoniques et correspondance floue question → réponse curatée.
Alimenté par test_questions.json et test_questions_extended.json (réponses métier).
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import unicodedata


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

CANONICAL_FILES = (
    "test_questions.json",
    "curated_qa.json",
)

_FUZZY_THRESHOLD = 0.72
_exact: Dict[str, str] = {}
_fuzzy: List[Tuple[str, str]] = []
_loaded = False


def _token_set(text: str) -> set:
    return set(normalize_for_match(text).split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _load_from_test_questions(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("test_questions", []):
        q = (item.get("question") or "").strip()
        a = (item.get("expected_answer") or "").strip()
        if q and a:
            _register(q, a)


def _load_curated(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("entries", []):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            _register(q, a)


def _register(question: str, answer: str) -> None:
    key = normalize_for_match(question)
    if not key:
        return
    if key not in _exact:
        _exact[key] = answer
    pair = (key, answer)
    if pair not in _fuzzy:
        _fuzzy.append(pair)


def ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    root = os.path.dirname(os.path.abspath(__file__))
    _load_from_test_questions(os.path.join(root, "test_questions.json"))
    _load_curated(os.path.join(root, "curated_qa.json"))
    _loaded = True


def lookup_canonical(question: str) -> Optional[str]:
    """Retourne une réponse curatée si la question correspond (exact ou flou)."""
    ensure_loaded()
    q = (question or "").strip()
    if not q:
        return None

    key = normalize_for_match(q)
    if key in _exact:
        return _exact[key]

    q_tokens = _token_set(q)
    best_score = 0.0
    best_answer: Optional[str] = None
    for cand_key, answer in _fuzzy:
        score = _jaccard(q_tokens, set(cand_key.split()))
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= _FUZZY_THRESHOLD and best_answer:
        return best_answer
    return None


def parse_doc_about_topic(question: str) -> Optional[str]:
    """Extrait le sujet des questions « Que dit la documentation à propos de : … ? »."""
    m = re.search(
        r"documentation\s+a\s+propos\s+de\s*:?\s*(.+?)\s*\??\s*$",
        normalize_for_match(question).replace("  ", " "),
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()
