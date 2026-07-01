"""Grounded LLM answer formulation on top of extractive RAG evidence."""
import os
import re
from typing import Dict, List, Optional

from rag_answer import is_not_found_answer
from rag_config import (
    RAG_ANSWER_LLM_MODEL,
    RAG_ENABLE_LLM_ANSWER_GENERATION,
    RAG_LLM_ANSWER_MAX_CONTEXT_CHARS,
)


def _load_env_once():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def _client_and_model():
    _load_env_once()
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openai_key and not openrouter_key:
        return None, ""

    from openai import OpenAI

    if openai_key:
        return OpenAI(api_key=openai_key), RAG_ANSWER_LLM_MODEL

    model = RAG_ANSWER_LLM_MODEL
    if "/" not in model:
        model = f"openai/{model}"
    return (
        OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        ),
        model,
    )


def _context_from_results(results: List[Dict]) -> str:
    parts = []
    total = 0
    for index, result in enumerate(results or [], 1):
        source = result.get("source") or result.get("title") or "source inconnue"
        section = result.get("section") or ""
        text = result.get("compressed_text") or result.get("text") or ""
        text = re.sub(r"\s+", " ", str(text)).strip()
        if not text:
            continue
        block = f"[{index}] Source: {source}\nSection: {section}\nPassage: {text}"
        remaining = RAG_LLM_ANSWER_MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        block = block[:remaining]
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def formulate_answer_with_llm(
    question: str,
    extractive_answer: str,
    context_results: List[Dict],
    reasoning_plan: Optional[List[str]] = None,
    session_summary: str = "",
) -> str:
    """Rephrase an extractive answer naturally while keeping it source-grounded."""
    if not RAG_ENABLE_LLM_ANSWER_GENERATION:
        return extractive_answer
    if not extractive_answer or is_not_found_answer(extractive_answer):
        return extractive_answer

    context = _context_from_results(context_results)
    if not context:
        return extractive_answer

    try:
        client, model = _client_and_model()
        if client is None:
            return extractive_answer

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant RAG documentaire Harmony/Divalto. "
                        "Formule une reponse naturelle, concise et utile en francais. "
                        "Tu dois utiliser uniquement les informations presentes dans "
                        "la reponse extractive et les passages fournis. "
                        "N'ajoute aucun fait non supporte. "
                        "Ne mentionne pas ton raisonnement interne. "
                        "Si les passages ne supportent pas la reponse, retourne "
                        "exactement: Information non trouvee dans la documentation locale fournie."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question utilisateur:\n{question}\n\n"
                        f"Plan interne de lecture documentaire:\n"
                        f"{'; '.join(reasoning_plan or [])}\n\n"
                        f"Resume de session:\n{session_summary}\n\n"
                        f"Reponse extractive a reformuler:\n{extractive_answer}\n\n"
                        f"Passages documentaires autorises:\n{context}\n\n"
                        "Reponds directement a la question. "
                        "Ne copie pas un long chunk; structure la reponse si utile."
                    ),
                },
            ],
        )
        answer = (response.choices[0].message.content or "").strip()
        return answer or extractive_answer
    except Exception as exc:
        print(f"   LLM answer formulation unavailable; using extractive answer ({exc})")
        return extractive_answer


__all__ = ["formulate_answer_with_llm"]
