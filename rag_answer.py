"""
Génération de réponse extractive à partir du contexte RAG (sans LLM externe).
Uniquement par correspondance vectorielle + lexical sur le contexte récupéré.
"""
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from rag_config import ANSWER_LEXICAL_WEIGHT, ANSWER_VECTOR_WEIGHT, ANSWER_MIN_SENTENCE_SCORE
from rag_vector import rank_by_embedding

STOPWORDS_FR = {
    "dans", "pour", "avec", "sans", "sous", "entre", "vers", "chez",
    "comment", "quels", "quelles", "quel", "quelle", "quoi",
    "peut", "peuvent", "sont", "est", "etre", "avoir", "faire", "cette",
    "cela", "ceci", "comme", "mais", "donc", "alors", "aussi", "tres",
    "plus", "moins", "tout", "tous", "toute", "toutes", "une", "des",
    "les", "aux", "par", "sur", "que", "qui", "dont", "ou", "the",
    "documentation", "depuis", "ouvre", "ouvrir",
}


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_text(text: str) -> str:
    """Nettoie les artefacts d'extraction HTML."""
    if not text:
        return ""
    t = text.replace("\r", "\n")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"([a-zàâäéèêëïîôùûüç])([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜ])", r"\1 \2", t)
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    return t.strip()


def extract_question_terms(question: str) -> List[str]:
    norm = normalize_for_match(question)
    return [w for w in norm.split() if len(w) > 2 and w not in STOPWORDS_FR]


def split_sentences(text: str) -> List[str]:
    clean = clean_text(text.replace("\n", " "))
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}", clean)
    return [p.strip() for p in parts if len(p.strip()) > 35]


def _is_title_like(sentence: str) -> bool:
    """Filtre les titres / en-têtes HTML qui ressemblent à des réponses."""
    words = sentence.split()
    if len(words) < 8:
        return True
    lower = sentence.lower()
    content_markers = (
        " est ", " sont ", " peut ", " peuvent ", " permet ", " doit ", " doivent ",
        " pour ", " si ", " lors ", " via ", " par ", " avec ", " sans ", " dans ",
        " remplac ", " stock ", " assure ", " accessible ", " explique ", " couvre ",
    )
    if not any(m in lower for m in content_markers):
        return True
    return False


def _sentence_has_terms(q_terms: List[str], sentence: str, min_hits: int = 1) -> bool:
    if not q_terms:
        return True
    s_norm = normalize_for_match(sentence)
    hits = sum(1 for t in q_terms if t in s_norm)
    return hits >= min_hits


def score_sentence(question_norm: str, q_terms: List[str], sentence: str) -> float:
    s_norm = normalize_for_match(sentence)
    if not s_norm:
        return 0.0
    overlap = sum(1 for w in q_terms if w in s_norm)
    if overlap == 0:
        return 0.0
    score = overlap / max(len(q_terms), 1)
    for term in q_terms:
        if len(term) >= 5 and term in s_norm:
            score += 0.12
    q_words = question_norm.split()
    for i in range(len(q_words) - 1):
        bigram = f"{q_words[i]} {q_words[i + 1]}"
        if len(bigram) > 5 and bigram in s_norm:
            score += 0.2
    if len(sentence) < 35:
        score *= 0.55
    if len(sentence) > 400:
        score *= 0.9
    return score


def build_context_from_results(results: List[Dict], max_chunks: int = 5) -> str:
    """Assemble le texte des meilleurs chunks pour la génération."""
    parts = []
    seen = set()
    for r in results[:max_chunks]:
        text = clean_text(r.get("text", "") or "")
        if not text:
            continue
        key = text[:120]
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return "\n\n".join(parts)


def _best_sentences(
    question: str,
    context: str,
    max_sentences: int = 3,
    embedding_model=None,
    embedding_model_name: str = "",
) -> List[str]:
    """Sélectionne les phrases les plus pertinentes (vectoriel + lexical)."""
    question_norm = normalize_for_match(question)
    q_terms = extract_question_terms(question)
    if not q_terms:
        q_terms = [w for w in question_norm.split() if len(w) > 3]

    sentences = split_sentences(context)
    if not sentences:
        return []

    lexical_scores = {
        i: score_sentence(question_norm, q_terms, sent)
        for i, sent in enumerate(sentences)
    }

    vector_scores: Dict[int, float] = {}
    if embedding_model is not None:
        ranked = rank_by_embedding(
            embedding_model,
            question,
            sentences,
            model_name=embedding_model_name,
            top_k=max(max_sentences * 2, 5),
        )
        for idx, score, _ in ranked:
            vector_scores[idx] = score

    combined = []
    for i, sent in enumerate(sentences):
        lex = lexical_scores.get(i, 0.0)
        vec = vector_scores.get(i, 0.0)
        if embedding_model is not None:
            score = ANSWER_VECTOR_WEIGHT * vec + ANSWER_LEXICAL_WEIGHT * min(1.0, lex)
        else:
            score = lex
        if score >= ANSWER_MIN_SENTENCE_SCORE:
            combined.append((score, sent))

    combined.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for _, sent in combined[:max_sentences]:
        if sent not in picked:
            picked.append(sent)
    return picked


def _format_answer(sentence: str) -> str:
    sentence = (sentence or "").strip()
    if not sentence:
        return ""
    return sentence if sentence.endswith((".", "!", "?")) else sentence + "."


def _source_token_bonus(q_terms: List[str], source: str) -> float:
    if not source or not q_terms:
        return 0.0
    stem = os.path.splitext(os.path.basename(source))[0].lower()
    parts = set(re.split(r"[^a-z0-9]+", stem))
    parts = {p for p in parts if len(p) >= 4}
    if not parts:
        return 0.0
    hits = sum(1 for t in q_terms if t in parts or any(t in p or p in t for p in parts))
    return min(0.25, hits / max(len(q_terms), 1) * 0.25)


def _answer_from_results(
    question: str,
    results: List[Dict],
    embedding_model=None,
    embedding_model_name: str = "",
) -> str:
    """Parcourt tous les chunks rerankés et extrait la phrase la plus proche (vectoriel)."""
    question_norm = normalize_for_match(question)
    q_terms = extract_question_terms(question)
    if not q_terms:
        q_terms = [w for w in question_norm.split() if len(w) > 3]

    if not results:
        return ""

    best_sentence = ""
    best_score = 0.0

    for rank, result in enumerate(results[:20]):
        text = clean_text(result.get("text") or "")
        if not text:
            continue
        src_bonus = _source_token_bonus(q_terms, str(result.get("source") or ""))
        rank_decay = 0.04 / (1.0 + 0.05 * rank)

        sentences = [
            s for s in (split_sentences(text) or [text[:600]])
            if _sentence_has_terms(q_terms, s) and not _is_title_like(s)
        ]
        if not sentences:
            continue

        batch_vectors: Dict[int, float] = {}
        if embedding_model is not None:
            for idx, score, _ in rank_by_embedding(
                embedding_model,
                question,
                sentences,
                model_name=embedding_model_name,
                top_k=len(sentences),
            ):
                batch_vectors[idx] = score

        for i, sent in enumerate(sentences):
            if len(sent) < 55 or len(sent) > 520:
                continue
            lex = min(1.0, score_sentence(question_norm, q_terms, sent))
            if lex < 0.08:
                continue
            vec = batch_vectors.get(i, 0.0)
            if embedding_model is not None:
                score = ANSWER_VECTOR_WEIGHT * vec + ANSWER_LEXICAL_WEIGHT * lex + src_bonus + rank_decay
            else:
                score = lex + src_bonus + rank_decay
            if score > best_score:
                best_score = score
                best_sentence = sent

    if best_sentence and best_score >= ANSWER_MIN_SENTENCE_SCORE * 0.35:
        return _format_answer(best_sentence)

    return ""


def _answer_from_context(
    question: str,
    context: str,
    embedding_model=None,
    embedding_model_name: str = "",
) -> str:
    sentences = _best_sentences(
        question,
        context,
        max_sentences=3,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
    if sentences:
        answer = " ".join(sentences)
        return _format_answer(answer)

    if embedding_model is not None:
        ranked = rank_by_embedding(
            embedding_model,
            question,
            split_sentences(context) or [context],
            model_name=embedding_model_name,
            top_k=1,
            min_score=ANSWER_MIN_SENTENCE_SCORE * 0.5,
        )
        if ranked:
            return _format_answer(ranked[0][2])

    question_norm = normalize_for_match(question)
    q_terms = extract_question_terms(question)
    chunks = split_sentences(context) or [context]
    best, best_sc = "", 0.0
    for chunk in chunks:
        sc = score_sentence(question_norm, q_terms, chunk)
        if sc > best_sc:
            best_sc = sc
            best = chunk
    if best and best_sc > 0.08:
        return _format_answer(best)

    return context[:600].strip() + ("..." if len(context) > 600 else "")


def generate_answer_from_results(
    question: str,
    results: List[Dict],
    embedding_model=None,
    embedding_model_name: str = "",
) -> str:
    """Réponse par correspondance vectorielle sur les chunks rerankés (graphe + FAISS)."""
    question = (question or "").strip()
    if not question:
        return "Désolé, je n'ai pas trouvé d'information pertinente dans la documentation."
    if not results:
        return "Désolé, je n'ai pas trouvé d'information pertinente dans la documentation."

    answer = _answer_from_results(
        question,
        results,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
    if answer:
        return answer

    context = build_context_from_results(results, max_chunks=5)
    return _answer_from_context(
        question,
        context,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )


def generate_answer(
    question: str,
    context: str,
    embedding_model=None,
    embedding_model_name: str = "",
) -> str:
    """Extrait la réponse la plus pertinente via correspondance vectorielle sur le contexte."""
    question = (question or "").strip()
    if not question:
        return "Désolé, je n'ai pas trouvé d'information pertinente dans la documentation."

    context = clean_text((context or "").strip())
    if not context:
        return "Désolé, je n'ai pas trouvé d'information pertinente dans la documentation."

    return _answer_from_context(
        question,
        context,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
