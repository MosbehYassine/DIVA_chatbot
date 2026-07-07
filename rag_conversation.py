"""Deterministic conversation reformulation for follow-up RAG questions."""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List


FOLLOW_UP_PATTERNS = (
    r"^(et|puis|ensuite|alors)\b",
    r"\b(dedans|y|cela|ca|ceci|celui|celle|ceux|celles)\b",
    r"\b(ce|cet|cette|ces)\s+(module|fonction|procedure|parametres?|fichiers?|chemins?|erreurs?)\b",
    r"\b(le|la|les|lui|leur|it|this|that|them)\b",
)

STOPWORDS = {
    "a", "au", "aux", "avec", "ca", "ce", "ceci", "cela", "ces", "cet",
    "cette", "comment", "dans", "de", "des", "du", "est", "et", "faire",
    "harmony", "la", "le", "les", "leur", "lui", "on", "peut", "pour",
    "pourquoi", "qu", "que", "quel", "quelle", "quelles", "quels", "quoi",
    "sans", "sert", "sont", "sur", "the", "this", "that", "them", "une",
    "un", "where", "what", "with",
}


@dataclass
class ConversationFocus:
    module: str = ""
    topic: str = ""


@dataclass
class ReformulationResult:
    question: str
    history_used: bool = False
    confidence: float = 0.0
    module: str = ""
    topic: str = ""


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def _clean_question(question: str) -> str:
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    return question


def _is_follow_up(question: str) -> bool:
    normalized = _normalize(question)
    if any(re.search(pattern, normalized) for pattern in FOLLOW_UP_PATTERNS):
        return True
    words = normalized.split()
    return len(words) <= 5 and normalized.startswith(("comment ", "et ", "how "))


def _strip_follow_up_prefix(question: str) -> str:
    return re.sub(
        r"^(et|puis|ensuite|alors)\s+",
        "",
        question,
        flags=re.IGNORECASE,
    ).strip()


def _first_match(patterns: List[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            value = (match.group(1) or "").strip(" .,:;!?\"'")
            if value:
                return value
    return ""


def _module_from_text(text: str) -> str:
    module = _first_match(
        [
            r"\bmodule\s+(?:de\s+|d[' ]|du\s+|des\s+)?([A-Za-z0-9_.-]+)",
            r"\bmodule\s+([^\s?.!,;:]+)",
        ],
        text,
    )
    if module and _normalize(module) not in STOPWORDS:
        return module
    return ""


def _topic_from_question(question: str) -> str:
    topic = _first_match(
        [
            r"\b(?:creer|créer|ouvrir|modifier|supprimer|imprimer|lancer|choisir|gerer|gérer|configurer)\s+(?:une?|des|les|la|le|l')?\s*([A-Za-z0-9_.-]+)",
            r"\b((?:chemins?|fichiers?|erreurs?|utilisateurs?|profils?|imprimantes?|impressions?|factures?|zooms?)(?:\s+Harmony|\s+Divalto)?)\b",
        ],
        question,
    )
    if topic and _normalize(topic) not in STOPWORDS:
        return topic
    return ""


def _topic_from_sources(turn: Dict) -> str:
    context = turn.get("rag_context") or {}
    for item in context.get("sources", []) or []:
        if isinstance(item, dict):
            label = str(
                item.get("title")
                or item.get("filename")
                or item.get("source")
                or ""
            )
        else:
            label = str(item or "")
        label = re.sub(r"\.[A-Za-z0-9]+$", "", label)
        label = re.sub(r"[_-]+", " ", label)
        words = [
            word for word in re.findall(r"[A-Za-z0-9]+", label)
            if len(word) > 2 and _normalize(word) not in STOPWORDS
        ]
        if words:
            return " ".join(words[:4])
    return ""


def _focus_from_history(history: List[Dict]) -> ConversationFocus:
    focus = ConversationFocus()
    for turn in reversed(history or []):
        question = str(turn.get("question") or turn.get("user_message") or "")
        answer = str(turn.get("answer") or turn.get("assistant_answer") or "")

        if not focus.module:
            focus.module = _module_from_text(question) or _module_from_text(answer)
        if not focus.topic:
            focus.topic = _topic_from_question(question) or _topic_from_sources(turn)
        if focus.module and focus.topic:
            break
    return focus


def _replace_context_pronouns(question: str, focus: ConversationFocus) -> str:
    module_phrase = f"le module {focus.module}" if focus.module else ""
    topic = focus.topic

    if topic:
        question = re.sub(
            r"^(comment\s+)?(le|la|les|l')\s+(modifier|supprimer|ouvrir|imprimer|configurer)\b(.*)$",
            lambda match: (
                f"{match.group(1) or 'Comment '}"
                f"{match.group(3)} {_with_default_article(topic)}"
                f"{match.group(4) or ''}"
            ),
            question,
            count=1,
            flags=re.IGNORECASE,
        )

    if module_phrase:
        question = re.sub(
            r"\b(dedans|y)\b",
            f"dans {module_phrase}",
            question,
            flags=re.IGNORECASE,
        )
        question = re.sub(
            r"\b(ce|cet|cette)\s+module\b",
            module_phrase,
            question,
            flags=re.IGNORECASE,
        )
        question = re.sub(
            r"\b(ces|ce|cet|cette)\s+(parametres?|fichiers?|chemins?|erreurs?|fonctions?|procedures?)\b",
            rf"les \2 de {module_phrase}",
            question,
            flags=re.IGNORECASE,
        )
        question = re.sub(
            r"\b(cela|ca|ceci|this|that)\b",
            module_phrase,
            question,
            count=1,
            flags=re.IGNORECASE,
        )
    elif topic:
        question = re.sub(
            r"\b(lui|leur|cela|ca|ceci|it|this|that|them)\b",
            topic,
            question,
            count=1,
            flags=re.IGNORECASE,
        )

    return question


def _needs_focus_append(question: str, focus: ConversationFocus) -> bool:
    normalized = _normalize(question)
    if focus.module and _normalize(focus.module) in normalized:
        return False
    if focus.topic and _normalize(focus.topic) in normalized:
        return False
    return bool(focus.module or focus.topic)


def _finish_question(question: str) -> str:
    question = re.sub(r"\s+", " ", question).strip(" .")
    question = re.sub(r"\bde le module\b", "du module", question, flags=re.IGNORECASE)
    question = re.sub(r"\s+\?", "?", question)
    if not question.endswith("?"):
        question += " ?"
    if question:
        question = question[0].upper() + question[1:]
    return question


def _with_default_article(topic: str) -> str:
    topic = str(topic or "").strip()
    if not topic:
        return topic
    if re.match(r"^(le|la|les|un|une|des|du|de la|de l')\b", topic, flags=re.IGNORECASE):
        return topic
    if _normalize(topic).startswith("module "):
        return topic
    first_word = _normalize(topic).split()[0] if _normalize(topic).split() else ""
    if first_word.endswith("s"):
        return f"les {topic}"
    return f"la {topic}"


def reformulate_with_history_info(question: str, history: List[Dict]) -> ReformulationResult:
    """Turn a follow-up into a standalone question and expose confidence."""
    original = _clean_question(question)
    if not original or not history or not _is_follow_up(original):
        return ReformulationResult(question=original)

    focus = _focus_from_history(history)
    if not focus.module and not focus.topic:
        return ReformulationResult(question=original)

    confidence = 0.55
    if focus.module:
        confidence += 0.20
    if focus.topic:
        confidence += 0.15

    normalized = _normalize(original)
    direct_print_request = bool(
        re.search(r"\b(comment\s+)?(l\s*)?imprimer\b", normalized)
        and "modele" not in normalized
        and "mise en page" not in normalized
    )
    if direct_print_request and focus.topic:
        rewritten = _finish_question(
            f"Comment imprimer {_with_default_article(focus.topic)} dans Harmony/Divalto"
        )
        return ReformulationResult(
            question=rewritten,
            history_used=rewritten != original,
            confidence=min(0.95, confidence + 0.05),
            module=focus.module,
            topic=focus.topic,
        )

    rewritten = _strip_follow_up_prefix(original)
    rewritten = _replace_context_pronouns(rewritten, focus)

    if _needs_focus_append(rewritten, focus):
        if focus.module:
            rewritten = f"{rewritten.rstrip(' ?')} dans le module {focus.module}"
        elif focus.topic:
            rewritten = f"{rewritten.rstrip(' ?')} concernant {focus.topic}"

    if "harmony" not in _normalize(rewritten) and "divalto" not in _normalize(rewritten):
        rewritten = f"{rewritten.rstrip(' ?')} dans Harmony/Divalto"

    rewritten = _finish_question(rewritten)
    return ReformulationResult(
        question=rewritten,
        history_used=rewritten != original,
        confidence=min(0.95, confidence),
        module=focus.module,
        topic=focus.topic,
    )


def reformulate_with_history(question: str, history: List[Dict]) -> str:
    """Backward-compatible helper returning only the standalone question."""
    return reformulate_with_history_info(question, history).question


__all__ = ["reformulate_with_history", "reformulate_with_history_info"]
