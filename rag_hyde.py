"""Retrieval-only HyDE generation with an offline deterministic fallback."""
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

from rag_config import (
    RAG_HYDE_MAX_SENTENCES,
    RAG_HYDE_USE_LLM,
    RAG_QUERY_LLM_MODEL,
)


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _is_french(text: str) -> bool:
    normalized = unicodedata.normalize("NFD", (text or "").lower())
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    markers = {
        "comment", "quel", "quelle", "pourquoi", "dans", "avec", "imprimer",
        "configurer", "utilisateur", "fichier", "chemin", "est", "sont",
    }
    return len(markers.intersection(re.findall(r"[a-z]+", normalized))) >= 1


def _limit_sentences(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        if sentence.strip()
    ]
    limit = max(3, min(6, RAG_HYDE_MAX_SENTENCES))
    return " ".join(sentences[:limit])


def _llm_hyde(question: str, query_metadata: Dict) -> Optional[str]:
    if not RAG_HYDE_USE_LLM or not os.getenv("OPENAI_API_KEY"):
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
                        "Return JSON with a 'document' field. Write 3 to 6 short "
                        "sentences in the question language. This is a hypothetical "
                        "Harmony/Divalto technical passage used only for retrieval."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Metadata: {json.dumps(query_metadata, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return _limit_sentences(str(data.get("document") or ""))
    except Exception as exc:
        print(f"   HyDE LLM unavailable; using offline fallback ({exc})")
        return None


def generate_hypothetical_document(
    question: str,
    query_metadata: Optional[Dict] = None,
) -> str:
    """Generate a short hypothetical passage used only as a vector query."""
    question = re.sub(r"\s+", " ", str(question or "")).strip()
    if not question:
        return ""
    metadata = query_metadata or {}
    llm_text = _llm_hyde(question, metadata)
    if llm_text:
        return llm_text

    domain = str(metadata.get("domain") or "Harmony/Divalto")
    keywords = [
        str(value) for value in metadata.get("keywords", []) if str(value).strip()
    ][:8]
    terms = ", ".join(keywords) or question.rstrip(" ?")
    if _is_french(question):
        return _limit_sentences(
            f"La documentation technique Harmony/Divalto décrit la procédure liée à : {question.rstrip(' ?')}. "
            f"Le domaine concerné est {domain} et les termes principaux sont {terms}. "
            "La configuration s'effectue à partir des paramètres, fichiers ou programmes correspondants. "
            "La procédure précise les prérequis, les étapes d'exécution et les contrôles à effectuer."
        )
    return _limit_sentences(
        f"The Harmony/Divalto technical documentation describes the procedure for: {question.rstrip(' ?')}. "
        f"The relevant domain is {domain}, with the main terms {terms}. "
        "Configuration uses the corresponding settings, files, or programs. "
        "The procedure identifies prerequisites, execution steps, and validation checks."
    )


def build_hyde_queries(
    question: str,
    query_metadata: Optional[Dict] = None,
) -> List[str]:
    """Return the real question and a non-empty retrieval-only HyDE passage."""
    return _dedupe(
        [question, generate_hypothetical_document(question, query_metadata)]
    )


__all__ = ["generate_hypothetical_document", "build_hyde_queries"]
