"""Query rewriting, soft SelfQuery metadata, and retrieval-only HyDE."""
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

from rag_config import (
    RAG_ENABLE_HYDE,
    RAG_ENABLE_QUERY_REWRITING,
    RAG_QUERY_LLM_MODEL,
    RAG_QUERY_VARIANTS_COUNT,
)
from rag_hyde import generate_hypothetical_document


DOMAIN_PATTERNS = (
    ("Xlog", ("xlog", "xlogf", "xlog1")),
    ("spooler", ("spooler", "spouleur")),
    ("chemin", ("chemin", "path", "xpath")),
    ("impression", ("impression", "imprimante", "printer")),
    ("utilisateur", ("utilisateur", "user", "ldap", "identification")),
    ("Harmony", ("harmony", "xlan")),
    ("Divalto", ("divalto",)),
)

ABBREVIATIONS = {
    "ldap": "annuaire LDAP utilisateurs",
    "xlan": "serveur reseau XLAN Harmony",
    "odbc": "connexion ODBC base de donnees",
    "sql": "base de donnees SQL",
    "pdf": "document PDF",
}


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", (text or "").lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return " ".join(re.findall(r"[a-z0-9_.-]+", value))


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", (value or "").strip())
        key = _normalize(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def extract_query_metadata(user_query: str) -> Dict:
    normalized = _normalize(user_query)
    domain = "unknown"
    matched_terms = []
    for candidate_domain, terms in DOMAIN_PATTERNS:
        hits = [term for term in terms if term in normalized]
        if hits:
            domain = candidate_domain
            matched_terms.extend(hits)
            break

    keywords = [
        token for token in normalized.split()
        if len(token) > 2 and token not in {
            "avec", "dans", "pour", "comment", "quel", "quelle", "quels",
            "quelles", "est", "sont", "une", "des", "les", "sur",
        }
    ]
    keywords = list(dict.fromkeys(matched_terms + keywords))[:12]
    filters = {}
    if domain != "unknown":
        filters["domain_hint"] = domain
    return {
        "domain": domain,
        "keywords": keywords,
        "filters": filters,
    }


def _offline_variants(query: str, count: int) -> List[str]:
    normalized = _normalize(query)
    metadata = extract_query_metadata(query)
    expanded_terms = [
        expansion for abbreviation, expansion in ABBREVIATIONS.items()
        if abbreviation in normalized
    ]
    domain = metadata["domain"]
    domain_hint = domain if domain != "unknown" else "Harmony Divalto"
    stripped = query.strip().rstrip(" ?")
    variants = [
        normalized,
        f"{stripped} documentation technique {domain_hint}",
        f"definition fonctionnement configuration {stripped} {domain_hint}",
        f"procedure fichiers programmes parametres {stripped} {' '.join(expanded_terms)}",
    ]
    return _dedupe(variants)[:count]


def _llm_variants(query: str, count: int, include_hyde: bool) -> Optional[Dict]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "Retourne un JSON pour rechercher uniquement dans une documentation "
            f"Harmony/Divalto. Fournis {count} reformulations dans 'variants'."
        )
        if include_hyde:
            prompt += (
                " Ajoute 'hyde', un passage hypothetique servant uniquement "
                "a produire un embedding de recherche."
            )
        response = client.chat.completions.create(
            model=RAG_QUERY_LLM_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Retourne uniquement du JSON."},
                {"role": "user", "content": f"{prompt}\nQuestion: {query}"},
            ],
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"   Transformation LLM indisponible; repli local ({exc})")
        return None


def _rewrite_query(user_query: str, enabled: bool) -> List[str]:
    query = (user_query or "").strip()
    if not query:
        return []
    if not enabled:
        return [query]

    count = max(0, RAG_QUERY_VARIANTS_COUNT)
    transformed = _llm_variants(query, count, include_hyde=False)
    variants = list((transformed or {}).get("variants") or [])
    if len(variants) < count:
        variants.extend(_offline_variants(query, count))
    return _dedupe([query] + variants)[:count + 1]


def rewrite_query(user_query: str) -> List[str]:
    """Return the original query followed by unique rewritten variants."""
    return _rewrite_query(user_query, RAG_ENABLE_QUERY_REWRITING)


def transform_query(
    query: str,
    enable_expansion: bool = RAG_ENABLE_QUERY_REWRITING,
    enable_hyde: bool = RAG_ENABLE_HYDE,
    variants_count: int = RAG_QUERY_VARIANTS_COUNT,
) -> Dict:
    """Backward-compatible transformation object used by HybridRAG."""
    rewritten = _rewrite_query(query, enable_expansion)
    rewritten = _dedupe(rewritten)[:max(1, int(variants_count) + 1)]
    metadata = extract_query_metadata(query)
    hyde_text = (
        generate_hypothetical_document(query, metadata)
        if enable_hyde else ""
    )
    return {
        "original_query": query,
        "variants": rewritten[1:],
        "query_variants": rewritten,
        "query_metadata": metadata,
        "hyde_text": hyde_text,
        "used_llm": False,
    }
