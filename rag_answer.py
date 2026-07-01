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
    "details", "detail", "importants", "important", "presentes", "presente",
    "module", "page", "contenu", "principal",
}

NOT_FOUND_MARKERS = (
    "pas trouve",
    "pas ete trouve",
    "n a pas ete trouve",
    "non trouve",
    "aucune information",
    "information pertinente",
    "not found",
)


def normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def is_not_found_answer(answer: str) -> bool:
    """Return whether an answer is an explicit documentary abstention."""
    normalized = normalize_for_match(answer)
    return any(marker in normalized for marker in NOT_FOUND_MARKERS)


def fallback_verify_answer(answer: str, context_results: List[Dict], **_) -> Dict:
    """Lexical verifier used only when rag_verifier has an incompatible API."""
    if not answer or is_not_found_answer(answer):
        return {
            "answer_supported": False,
            "confidence": 0.0,
            "needs_retry": False,
            "reason": "Fallback verifier received a not-found answer.",
        }

    context = " ".join(
        str(result.get("text") or "")
        for result in (context_results or [])
    ).strip()
    if not context:
        return {
            "answer_supported": False,
            "confidence": 0.0,
            "needs_retry": False,
            "reason": "Fallback verifier received no context.",
        }

    answer_terms = [
        term for term in normalize_for_match(answer).split()
        if len(term) > 2
    ]
    context_norm = normalize_for_match(context)
    supported_terms = sum(1 for term in answer_terms if term in context_norm)
    confidence = supported_terms / max(len(answer_terms), 1)
    supported = confidence >= 0.55
    return {
        "answer_supported": supported,
        "confidence": confidence,
        "needs_retry": not supported or confidence < 0.62,
        "reason": f"Fallback lexical verifier: token_support={confidence:.2f}",
    }


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


def _extract_best_span(
    question_norm: str,
    q_terms: List[str],
    text: str,
    min_tokens: int = 30,
    max_tokens: int = 80,
    step: int = 10,
) -> Tuple[str, float]:
    tokens = [tok for tok in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)]
    if not tokens:
        return "", 0.0
    if len(tokens) <= max_tokens:
        span = " ".join(tokens)
        return span, score_sentence(question_norm, q_terms, span)
    best_score = 0.0
    best_span = ""
    for start in range(0, len(tokens) - min_tokens + 1, step):
        end = min(len(tokens), start + max_tokens)
        span = " ".join(tokens[start:end])
        score = score_sentence(question_norm, q_terms, span)
        if score > best_score:
            best_score = score
            best_span = span
    return best_span, best_score


def build_context_from_results(results: List[Dict], max_chunks: int = 5) -> str:
    """Assemble le texte des meilleurs chunks pour la génération."""
    parts = []
    seen = set()
    for r in results[:max_chunks]:
        text = clean_text(r.get("compressed_text") or r.get("text", "") or "")
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


def _is_page_overview_question(question: str) -> bool:
    normalized = normalize_for_match(question)
    markers = (
        "que couvre la page",
        "contenu principal",
    )
    return any(marker in normalized for marker in markers)


def _is_detail_question(question: str) -> bool:
    normalized = normalize_for_match(question)
    markers = (
        "details importants",
        "details important",
        "presente dans",
        "presentes dans",
    )
    return any(marker in normalized for marker in markers)


def _quoted_phrases(question: str) -> List[str]:
    phrases = []
    for pattern in (r'"([^"]+)"', r"«([^»]+)»", r"“([^”]+)”"):
        phrases.extend(match.strip() for match in re.findall(pattern, question or ""))
    return [phrase for phrase in phrases if phrase]


def _result_title_score(result: Dict, quoted_phrase: str, question: str) -> float:
    phrase_norm = normalize_for_match(quoted_phrase)
    if not phrase_norm:
        return 0.0

    title_norm = normalize_for_match(result.get("title") or "")
    section_norm = normalize_for_match(result.get("section") or "")
    source_norm = normalize_for_match(os.path.splitext(os.path.basename(str(result.get("source") or "")))[0])
    fields = [title_norm, section_norm, source_norm]

    score = 0.0
    if any(phrase_norm == field for field in fields if field):
        score = 1.0
    elif any(phrase_norm in field or field in phrase_norm for field in fields if field):
        score = 0.88
    else:
        phrase_terms = set(phrase_norm.split())
        best_overlap = 0.0
        for field in fields:
            field_terms = set(field.split())
            if not field_terms:
                continue
            overlap = len(phrase_terms & field_terms) / max(len(phrase_terms), 1)
            best_overlap = max(best_overlap, overlap)
        score = best_overlap

    module = normalize_for_match(str(result.get("module") or ""))
    question_norm = normalize_for_match(question)
    if module and module in question_norm:
        score += 0.08
    return min(score, 1.0)


def _lead_answer_from_matching_result(
    question: str,
    results: List[Dict],
    max_chars: int = 900,
) -> str:
    phrases = _quoted_phrases(question)
    if not phrases:
        return ""

    best_result = None
    best_score = 0.0
    for result in results[:20]:
        for phrase in phrases:
            score = _result_title_score(result, phrase, question)
            if score > best_score:
                best_score = score
                best_result = result

    if best_result is None or best_score < 0.68:
        return ""
    return _lead_answer_from_top_result([best_result], max_chars=max_chars)


def _lead_answer_from_top_result(results: List[Dict], max_chars: int = 900) -> str:
    if not results:
        return ""
    result = results[0]
    title = clean_text(result.get("title") or result.get("section") or "")
    text = clean_text(result.get("text") or result.get("compressed_text") or "")
    if not text:
        return ""
    sentences = split_sentences(text)
    lead = " ".join(sentences[:3]) if sentences else text[:max_chars]
    lead = lead[:max_chars].strip()
    if title and normalize_for_match(title) not in normalize_for_match(lead[:160]):
        lead = f"{title}. {lead}"
    return _format_answer(lead)


def _result_full_text(result: Dict) -> str:
    return clean_text(
        result.get("parent_text")
        or result.get("text")
        or result.get("compressed_text")
        or ""
    )


def _block_windows(text: str, max_chars: int = 1000) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        text = clean_text(text)
        if not text:
            return []
        return [text[:max_chars]]

    windows = []
    for start in range(len(sentences)):
        current = []
        current_len = 0
        for end in range(start, min(len(sentences), start + 4)):
            sentence = sentences[end]
            next_len = current_len + len(sentence) + 1
            if current and next_len > max_chars:
                break
            current.append(sentence)
            current_len = next_len
            block = " ".join(current).strip()
            if 70 <= len(block) <= max_chars:
                windows.append(block)
    return list(dict.fromkeys(windows))


def _informativeness_score(block: str) -> float:
    normalized = normalize_for_match(block)
    words = normalized.split()
    if not words:
        return 0.0
    markers = (
        "permet", "permettent", "doit", "doivent", "peut", "peuvent",
        "lorsque", "si", "afin", "pour", "exemple", "selectionnez",
        "cliquez", "saisissez", "indique", "definit", "utilise",
        "obligatoire", "maximum", "minimum", "fichier", "parametre",
        "option", "code", "champ", "droits", "utilisateur",
    )
    marker_hits = sum(1 for marker in markers if marker in normalized)
    digit_bonus = 0.08 if re.search(r"\d", block) else 0.0
    punctuation_bonus = min(0.16, block.count(".") * 0.03 + block.count(":") * 0.04)
    length_score = min(1.0, len(words) / 65)
    return min(1.0, 0.45 * length_score + 0.08 * marker_hits + digit_bonus + punctuation_bonus)


def _detail_answer_from_matching_result(
    question: str,
    results: List[Dict],
    embedding_model=None,
    embedding_model_name: str = "",
    max_chars: int = 950,
) -> str:
    phrases = _quoted_phrases(question)
    question_norm = normalize_for_match(question)
    q_terms = extract_question_terms(question)
    phrase_terms = set()
    for phrase in phrases:
        phrase_terms.update(normalize_for_match(phrase).split())
    q_terms = [
        term for term in q_terms
        if term not in phrase_terms and term not in {"harmony", "divalto"}
    ]

    candidates = []
    for rank, result in enumerate(results[:20]):
        title_score = 0.0
        if phrases:
            title_score = max(
                _result_title_score(result, phrase, question)
                for phrase in phrases
            )
        elif rank == 0:
            title_score = 0.70
        if title_score < 0.55 and rank > 5:
            continue

        text = _result_full_text(result)
        if not text:
            continue
        blocks = _block_windows(text, max_chars=max_chars)
        if not blocks:
            continue

        vector_scores: Dict[int, float] = {}
        if embedding_model is not None:
            ranked = rank_by_embedding(
                embedding_model,
                question,
                blocks,
                model_name=embedding_model_name,
                top_k=min(len(blocks), 12),
            )
            for idx, score, _ in ranked:
                vector_scores[idx] = score

        title_norm = normalize_for_match(result.get("title") or result.get("section") or "")
        source_bonus = _source_token_bonus(
            list(phrase_terms) or q_terms,
            str(result.get("source") or ""),
        )
        for block_index, block in enumerate(blocks):
            block_norm = normalize_for_match(block)
            if _is_title_like(block) and len(block) < 180:
                continue
            lex = score_sentence(question_norm, q_terms, block) if q_terms else 0.0
            info = _informativeness_score(block)
            vec = vector_scores.get(block_index, 0.0)
            intro_penalty = 0.0
            if block_index == 0 and title_norm and title_norm in block_norm[: max(80, len(title_norm) + 20)]:
                intro_penalty += 0.18
            if phrase_terms and len(set(block_norm.split()) & phrase_terms) / max(len(phrase_terms), 1) > 0.75:
                intro_penalty += 0.08
            if len(block) < 120:
                intro_penalty += 0.10

            score = (
                0.36 * title_score
                + 0.24 * info
                + 0.18 * min(1.0, lex)
                + 0.14 * vec
                + source_bonus
                + 0.04 / (1.0 + rank)
                - intro_penalty
            )
            candidates.append((score, block))

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    answer = candidates[0][1][:max_chars].strip()
    return _format_answer(answer)


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
        text = clean_text(result.get("compressed_text") or result.get("text") or "")
        if not text:
            continue
        src_bonus = _source_token_bonus(q_terms, str(result.get("source") or ""))
        rank_decay = 0.04 / (1.0 + 0.05 * rank)

        sentences = [
            s for s in (split_sentences(text) or [text[:600]])
            if _sentence_has_terms(q_terms, s) and not _is_title_like(s)
        ]
        if sentences:
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

        span_text, span_score = _extract_best_span(question_norm, q_terms, text)
        if span_text:
            window_score = min(1.0, span_score)
            if window_score >= 0.15:
                score = ANSWER_LEXICAL_WEIGHT * window_score + src_bonus + rank_decay
                if embedding_model is not None and q_terms:
                    score += 0.08
                if score > best_score:
                    best_score = score
                    best_sentence = span_text

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

    return ""


def generate_answer_from_results(
    question: str,
    results: List[Dict],
    embedding_model=None,
    embedding_model_name: str = "",
) -> str:
    """Réponse par correspondance vectorielle sur les chunks rerankés (graphe + FAISS)."""
    question = (question or "").strip()
    if not question:
        return "Information non trouvée dans la documentation locale fournie."
    if not results:
        return "Information non trouvée dans la documentation locale fournie."

    if _is_detail_question(question):
        detail_answer = _detail_answer_from_matching_result(
            question,
            results,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
        )
        if detail_answer:
            return detail_answer

    if _is_page_overview_question(question):
        matching_answer = _lead_answer_from_matching_result(question, results)
        if matching_answer:
            return matching_answer
        lead_answer = _lead_answer_from_top_result(results)
        if lead_answer:
            return lead_answer

    answer = _answer_from_results(
        question,
        results,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
    if answer:
        return answer

    context = build_context_from_results(results, max_chunks=5)
    answer = _answer_from_context(
        question,
        context,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
    return answer or "Information non trouvée dans la documentation locale fournie."


def source_references(results: List[Dict], max_sources: int = 5) -> List[Dict]:
    references = []
    seen = set()
    for result in results:
        source = str(result.get("source") or "").strip()
        title = str(result.get("title") or "").strip()
        section = str(result.get("section") or "").strip()
        key = (source, title, section)
        if not source or key in seen:
            continue
        seen.add(key)
        references.append(
            {
                "source": source,
                "filename": os.path.basename(source),
                "title": title,
                "section": section,
                "module": result.get("module", ""),
                "parent_id": result.get("parent_id"),
            }
        )
        if len(references) >= max_sources:
            break
    return references


def answer_with_sources(answer: str, results: List[Dict]) -> str:
    if is_not_found_answer(answer):
        return answer
    references = source_references(results)
    labels = []
    for reference in references:
        label = reference["filename"]
        if reference.get("section"):
            label += f" - {reference['section']}"
        labels.append(label)
    return f"{answer}\n\nSources: {'; '.join(labels)}" if labels else answer


def generate_answer(
    question: str,
    context: str,
    embedding_model=None,
    embedding_model_name: str = "",
) -> str:
    """Extrait la réponse la plus pertinente via correspondance vectorielle sur le contexte."""
    question = (question or "").strip()
    if not question:
        return "Information non trouvée dans la documentation locale fournie."

    context = clean_text((context or "").strip())
    if not context:
        return "Information non trouvée dans la documentation locale fournie."

    return _answer_from_context(
        question,
        context,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
