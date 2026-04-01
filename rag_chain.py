"""
=============================================================================
DIVALTO HARMONY — RAG CHAIN  v2.1  (Expert Edition)
=============================================================================
Pipeline complet :
  Question utilisateur
    → détection hors-scope                        (ÉTAPE 11a)
    → contextualisation multi-tour                (ÉTAPE 13)
    → embed (multilingual-e5-large)
    → ChromaDB top-20
    → cross-encoder reranker
    → fetch parent chunks
    → prompt métier structuré                     (ÉTAPE 9)
    → reasoning Chain-of-Thought                  (ÉTAPE 10)
    → gestion incertitude & fallback 3 niveaux    (ÉTAPE 11)
    → personnalisation par profil                 (ÉTAPE 14)
    → réponse + sources + confiance

ÉTAPES COUVERTES :
  9  — Prompt templates métier structurés
  10 — Reasoning Layer (CoT contrôlé, filtré avant affichage)
  11 — Fallback à 3 niveaux (uncertain / not_found / out_of_scope)
  12 — Gestion de session et mémoire contextuelle
  13 — Scénarios multi-tours (contextualisation automatique des questions)
  14 — Personnalisation par profil (consultant / utilisateur_clé / nouveau)

INSTALL :
    pip install chromadb sentence-transformers google-genai numpy openai

CONFIGURE :
    set GEMINI_API_KEY=AIzaSy...
    python rag_chain.py
=============================================================================
"""

import os
import re
import json
import uuid
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "divalto_harmony"
EMBED_MODEL       = "intfloat/multilingual-e5-large"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVE_TOP_K    = 20
RERANK_TOP_K      = 3

# Seuils de confiance (ÉTAPE 11)
MIN_SCORE         = 0.70   # sous ce seuil → fallback total (ne pas halluciner)
UNCERTAIN_SCORE   = 0.75   # entre 0.70 et 0.75 → réponse avec avertissement

LLM_PROVIDER      = "gemini"
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "sk-...")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_MODEL      = "gpt-4o"
GEMINI_MODEL      = "gemini-2.5-flash"

MAX_HISTORY_TURNS = 5
PARENTS_FILE      = "chunks_parents.json"

# ─── ÉTAPE 9 : SYSTEM PROMPT MÉTIER ──────────────────────────────────────────
#
# Le system prompt est le contrat entre nous et le LLM.
# Il définit : rôle, contraintes, format de réponse, raisonnement interne.
# Le reasoning (ÉTAPE 10) est intégré ici — défini une seule fois,
# appliqué à toute la conversation sans répétition à chaque tour.
#
SYSTEM_PROMPT = """Tu es HAL (Harmony Assistant Logiciel), l'assistant expert \
de la documentation technique Divalto Harmony.

═══════════════════════════════════════════════════
RÔLE ET CONTRAINTES STRICTES
═══════════════════════════════════════════════════
• Tu réponds UNIQUEMENT en te basant sur les extraits de documentation fournis.
• Tu ne dois JAMAIS inventer, déduire ou supposer des informations absentes des extraits.
• Si les extraits ne contiennent pas la réponse, dis-le clairement et honnêtement.
• Tu sers des administrateurs système, développeurs Diva et utilisateurs finaux.

═══════════════════════════════════════════════════
RAISONNEMENT INTERNE — ÉTAPE 10 (Chain-of-Thought)
═══════════════════════════════════════════════════
Avant chaque réponse, raisonne mentalement en 3 étapes SANS les afficher :

  1. ANALYSE : de quoi parle cette question ? Quel module Divalto ? \
Procédure, définition ou diagnostic ?
  2. LECTURE : quels extraits sont pertinents ? Les extraits répondent-ils \
complètement, partiellement ou pas du tout ?
  3. PLAN : quelle réponse directe ? Étapes nécessaires ? Source à citer ? \
Point de vigilance ?

Génère directement la réponse finale — ne montre JAMAIS le raisonnement.

═══════════════════════════════════════════════════
FORMAT DE RÉPONSE OBLIGATOIRE
═══════════════════════════════════════════════════
1. RÉPONSE DIRECTE : une phrase courte qui répond directement à la question.

2. DÉTAIL / PROCÉDURE : si des étapes sont nécessaires, liste numérotée :
   1. Première étape
   2. Deuxième étape
   ...

3. SOURCE : toujours terminer par :
   📄 Source : [Module] > [Titre du document]

4. REMARQUE (si pertinent) : pré-requis ou points de vigilance importants.

═══════════════════════════════════════════════════
RÈGLES DE LANGAGE
═══════════════════════════════════════════════════
• Toujours en français.
• Technique et précis, mais accessible.
• Concis : 3 à 15 lignes selon la complexité.

═══════════════════════════════════════════════════
MODULES DISPONIBLES
═══════════════════════════════════════════════════
Administration, Installation, Record SQL, Codes d'Erreurs, Interface Windows,
Xdiva, ymeg, ymeg2, zoom, yzoom, Xwin Écran, Xwin Texte, Xwin Imprimante,
Chemins Harmony, Réseaux, et bien d'autres modules Divalto Harmony."""

# ─── ÉTAPE 14 : PROFILS UTILISATEUR ──────────────────────────────────────────
#
# Chaque profil reçoit des instructions de style supplémentaires injectées
# dans le prompt — pas dans le texte de la réponse.
# Règle d'or : ne JAMAIS envelopper les extraits documentaires avec le profil.
#
PROFILE_INSTRUCTIONS = {
    "consultant": (
        "\nSTYLE : L'utilisateur est un consultant Divalto expérimenté. "
        "Sois concis, technique, sans explications basiques. "
        "Utilise la terminologie Divalto sans la définir."
    ),
    "utilisateur_cle": (
        "\nSTYLE : L'utilisateur est un utilisateur clé (référent métier). "
        "Explique les concepts techniques brièvement. "
        "Donne des exemples concrets si disponibles dans les extraits."
    ),
    "nouveau": (
        "\nSTYLE : L'utilisateur est nouveau sur Divalto Harmony. "
        "Explique chaque terme technique entre parenthèses. "
        "Sois pédagogique et encourage à consulter son administrateur si besoin."
    ),
}

# ─── ÉTAPE 11 : DÉTECTION HORS-SCOPE ─────────────────────────────────────────
#
# Uniquement les cas CLAIREMENT hors documentation Divalto.
# Règle : si un doute existe, laisser le RAG décider (MIN_SCORE s'en chargera).
#
OUT_OF_SCOPE_PATTERNS = [
    r'\b(météo|football|recette|cuisine|film|musique)\b',
    r'\b(chatgpt|claude ai|llm|intelligence artificielle générale)\b(?!.*divalto)',
    r'\b(prix|tarif|coût|combien coûte|acheter|commander)\b(?!.*paramètre|.*configuration)',
]

def _is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in OUT_OF_SCOPE_PATTERNS)

# ─── ÉTAPE 11 : DÉTECTION RÉPONSE VIDE DU LLM ────────────────────────────────
#
# Quand le LLM dit "je n'ai pas trouvé" mais que le score était au-dessus
# du seuil, on active quand même le fallback avec suggestions de reformulation.
#
EMPTY_RESPONSE_SIGNALS = [
    "ne contient pas", "pas d'information",
    "pas trouvé", "aucune information",
    "ne mentionne pas", "n'est pas mentionné",
]

def _response_is_empty(response: str) -> bool:
    r = response.lower()
    return any(signal in r for signal in EMPTY_RESPONSE_SIGNALS)

# ─── ÉTAPE 13 : MOTS-CLÉS DE QUESTIONS DE SUIVI ──────────────────────────────
#
# Détecte si une question est une suite de la précédente.
# Si oui, on enrichit la question avec le contexte de session.
#
FOLLOWUP_PATTERNS = [
    r"^(et |mais |aussi |alors |donc |sinon |autrement )",
    r"^(qu[' ]est-ce que|c['']est quoi|pourquoi|comment|quand|où|qui)\b.{0,20}\?$",
    r"\b(dans ce cas|dans cette situation|pour ça|pour cela|à ce sujet)\b",
    r"^(et si|qu'en est-il de|quid de|et pour|et avec)\b",
    r"^(même chose|pareil|idem)\b",
]

def _is_followup_question(question: str) -> bool:
    q = question.strip().lower()
    if len(q) < 25:
        return True
    return any(re.search(p, q) for p in FOLLOWUP_PATTERNS)

# ─── UTILITAIRES ──────────────────────────────────────────────────────────────

def sanitize_id(chunk_id: str) -> str:
    nfkd    = unicodedata.normalize('NFKD', chunk_id)
    ascii_s = nfkd.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_s)

def _strip_thinking(text: str) -> str:
    """ÉTAPE 10 : retire les blocs <thinking> si le modèle les inclut."""
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return text.strip()

# ─── ÉTAPE 12 : SESSION & MÉMOIRE CONTEXTUELLE ───────────────────────────────

@dataclass
class UserProfile:
    """ÉTAPE 14 : profil utilisateur pour personnalisation."""
    role: str = "nouveau"        # "consultant" | "utilisateur_cle" | "nouveau"
    language: str = "fr"
    verbosity: str = "medium"    # "short" | "medium" | "detailed"

@dataclass
class SessionState:
    """
    ÉTAPE 12 : état de session persistant entre les tours.
    Stocke le module courant, le sujet, les entités récentes et les sources.
    Utilisé par l'ÉTAPE 13 pour enrichir les questions de suivi.
    """
    session_id: str
    current_topic: Optional[str]   = None
    current_module: Optional[str]  = None
    last_entities: List[str]       = field(default_factory=list)
    last_sources: List[Dict]       = field(default_factory=list)
    profile: UserProfile           = field(default_factory=UserProfile)

# ─── ASSISTANT ────────────────────────────────────────────────────────────────

class DivaltoAssistant:
    """
    RAG pipeline v2.1 pour Divalto Harmony.
    Couvre les étapes 9 à 14 du cahier des charges.
    """

    def __init__(self, user_role: str = "nouveau"):
        print("Initializing Divalto Harmony Assistant v2.1...")

        # ÉTAPE 12 : initialisation de la session
        self.session = SessionState(
            session_id=str(uuid.uuid4()),
            profile=UserProfile(role=user_role),
        )
        self.history: List[Dict] = []   # historique des échanges pour le LLM

        print(f"  Loading embedding model: {EMBED_MODEL}")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        print(f"  Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        print(f"  Connecting to ChromaDB: {CHROMA_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)
        count = self.collection.count()
        print(f"  Collection '{COLLECTION_NAME}': {count} chunks ready")

        print(f"  Loading parent chunks from: {PARENTS_FILE}")
        with open(PARENTS_FILE, encoding="utf-8") as f:
            parents_list = json.load(f)
        self.parent_map = {sanitize_id(p["chunk_id"]): p for p in parents_list}
        for p in parents_list:
            self.parent_map[p["chunk_id"]] = p
        print(f"  Parent map: {len(self.parent_map)} entries")
        print(f"  User profile: {user_role}")

        self._init_llm()
        print("Assistant v2.1 ready.\n")

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Traite une question et retourne :
        {
            question, response, sources, chunks_used,
            retrieved, found, confidence, fallback_level
        }
        """
        if verbose:
            print(f"\nQuestion: {question}")

        # ── ÉTAPE 11a : détection hors-scope ──────────────────────────
        if _is_out_of_scope(question):
            return self._fallback_out_of_scope(question)

        # ── ÉTAPE 13 : enrichissement des questions de suivi ──────────
        enriched_question = self._contextualize_question(question)
        if verbose and enriched_question != question:
            print(f"  [Multi-tour] Question enrichie: {enriched_question[:80]}...")

        # ── Embed + retrieve ──────────────────────────────────────────
        query_embedding = self._embed_query(enriched_question)
        candidates      = self._retrieve(query_embedding, k=RETRIEVE_TOP_K)

        if verbose:
            print(f"  Retrieved {len(candidates)} candidates")

        # ── ÉTAPE 11b : aucun résultat ────────────────────────────────
        if not candidates:
            return self._fallback_not_found(question)

        top_score = candidates[0]["score"]
        if verbose:
            print(f"  Top score: {top_score:.3f}")

        # ── ÉTAPE 11c : score insuffisant ─────────────────────────────
        if top_score < MIN_SCORE:
            return self._fallback_not_found(question)

        # ── Rerank + contexte ─────────────────────────────────────────
        reranked       = self._rerank(enriched_question, candidates, RERANK_TOP_K)
        context_chunks = self._fetch_parents(reranked)
        sources        = self._build_sources(reranked)
        is_uncertain   = top_score < UNCERTAIN_SCORE

        # ── ÉTAPES 9+10+14 : génération ───────────────────────────────
        response = self._generate(question, context_chunks, is_uncertain)

        # ── ÉTAPE 11d : LLM dit "je n'ai pas trouvé" ──────────────────
        if _response_is_empty(response):
            return self._fallback_not_found(question)

        # ── ÉTAPE 12 : mise à jour de la session ──────────────────────
        self._update_session(question, sources)

        # ── Mise à jour de l'historique LLM ───────────────────────────
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": response})
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

        return {
            "question"      : question,
            "response"      : response,
            "sources"       : sources,
            "chunks_used"   : [c["content"] for c in context_chunks],
            "retrieved"     : len(candidates),
            "found"         : True,
            "confidence"    : "medium" if is_uncertain else "high",
            "fallback_level": 1 if is_uncertain else 0,
        }

    def reset_history(self):
        """Réinitialise l'historique et la session (nouveau sujet)."""
        self.history = []
        self.session.current_topic  = None
        self.session.current_module = None
        self.session.last_entities  = []
        self.session.last_sources   = []
        print("Session réinitialisée.")

    def set_profile(self, role: str):
        """ÉTAPE 14 : change le profil en cours de session."""
        if role not in PROFILE_INSTRUCTIONS:
            print(f"  Profil inconnu: {role}. Valeurs: consultant, utilisateur_cle, nouveau")
            return
        self.session.profile.role = role
        print(f"  Profil mis à jour : {role}")

    # ─────────────────────────────────────────────────────────────────
    # ÉTAPE 12 : MISE À JOUR DE LA SESSION
    # ─────────────────────────────────────────────────────────────────

    def _update_session(self, question: str, sources: list):
        """
        Met à jour le contexte de session après chaque réponse réussie.
        Garde le module courant, le sujet et les entités pour les questions suivantes.
        """
        if sources:
            self.session.current_module = sources[0].get("module")
            self.session.current_topic  = sources[0].get("doc_title")

        entities = []
        for s in sources[:3]:
            if s.get("doc_title"):
                entities.append(s["doc_title"])
            if s.get("module"):
                entities.append(s["module"])
        # Déduplique en préservant l'ordre
        self.session.last_entities = list(dict.fromkeys(entities))[:6]
        self.session.last_sources  = sources[:3]

    # ─────────────────────────────────────────────────────────────────
    # ÉTAPE 13 : CONTEXTUALISATION DES QUESTIONS DE SUIVI
    # ─────────────────────────────────────────────────────────────────

    def _contextualize_question(self, question: str) -> str:
        """
        Si la question est une suite (courte, ou commence par "et si / mais / aussi..."),
        on enrichit la requête d'embedding avec le contexte de la session courante.
        Ceci améliore le recall sur les questions de suivi sans modifier
        ce qui est affiché à l'utilisateur.
        """
        if not self.session.current_topic:
            return question
        if not _is_followup_question(question):
            return question

        prefix_parts = []
        if self.session.current_module:
            prefix_parts.append(f"Module: {self.session.current_module}")
        if self.session.current_topic:
            prefix_parts.append(f"Sujet: {self.session.current_topic}")
        if self.session.last_entities:
            prefix_parts.append(f"Contexte: {', '.join(self.session.last_entities[:4])}")

        if not prefix_parts:
            return question

        return " | ".join(prefix_parts) + f" | {question}"

    # ─────────────────────────────────────────────────────────────────
    # ÉTAPE 11 : FALLBACK HANDLERS
    # ─────────────────────────────────────────────────────────────────

    def _fallback_not_found(self, question: str) -> dict:
        """
        Niveau 2 — aucun résultat exploitable ou LLM vide.
        Propose des reformulations basées sur les mots-clés de la question.
        """
        words    = [w for w in question.split() if len(w) > 4][:4]
        keywords = " / ".join(words) if words else "votre terme de recherche"
        response = (
            "Je n'ai pas trouvé de documentation correspondant à votre question "
            "dans la base Divalto Harmony.\n\n"
            "**Suggestions pour reformuler :**\n"
            "1. Utilisez des termes techniques Divalto "
            "(nom de module, de fonction, ou de fichier .htm)\n"
            f"2. Essayez des synonymes pour : *{keywords}*\n"
            "3. Vérifiez l'orthographe des noms de composants Harmony\n\n"
            "**Besoin d'aide ?**\n"
            "Contactez votre administrateur Divalto ou consultez "
            "la documentation officielle."
        )
        return {
            "question": question, "response": response,
            "sources": [], "chunks_used": [], "retrieved": 0,
            "found": False, "confidence": "none", "fallback_level": 2,
        }

    def _fallback_out_of_scope(self, question: str) -> dict:
        """
        Niveau 3 — question clairement hors périmètre Divalto.
        Escalade vers un humain.
        """
        response = (
            "Cette question ne concerne pas la documentation Divalto Harmony "
            "et dépasse mon périmètre d'assistance.\n\n"
            "Je suis spécialisé sur Divalto Harmony : administration, installation, "
            "développement Diva, modules ymeg/yzoom/zoom, Record SQL, etc.\n\n"
            "**Pour cette demande :**\n"
            "• Contactez le support Divalto directement\n"
            "• Consultez votre référent technique interne"
        )
        return {
            "question": question, "response": response,
            "sources": [], "chunks_used": [], "retrieved": 0,
            "found": False, "confidence": "none", "fallback_level": 3,
        }

    # ─────────────────────────────────────────────────────────────────
    # RETRIEVAL (inchangé — validé à recall@3 = 90%)
    # ─────────────────────────────────────────────────────────────────

    def _embed_query(self, question: str) -> list:
        embedding = self.embed_model.encode(
            f"query: {question}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def _retrieve(self, query_embedding: list, k: int) -> list:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            candidates.append({
                "content"    : doc,
                "source_file": meta.get("source_file", ""),
                "module"     : meta.get("module", ""),
                "parent_id"  : meta.get("parent_id", ""),
                "doc_title"  : meta.get("doc_title", ""),
                "chunk_type" : meta.get("chunk_type", ""),
                "score"      : 1.0 - dist,
            })
        candidates.sort(key=lambda x: -x["score"])
        return candidates

    def _rerank(self, question: str, candidates: list, top_k: int) -> list:
        if not candidates:
            return []
        pairs     = [(question, c["content"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)
        be_scores = [c["score"] for c in candidates]
        be_min, be_max = min(be_scores), max(be_scores)
        be_range  = be_max - be_min if be_max > be_min else 1.0
        be_norm   = [(s - be_min) / be_range for s in be_scores]
        scored = [
            {**cand, "rerank_score": float(ce) + 0.15 * float(be)}
            for cand, ce, be in zip(candidates, ce_scores, be_norm)
        ]
        scored.sort(key=lambda x: -x["rerank_score"])
        return scored[:top_k]

    def _fetch_parents(self, reranked: list) -> list:
        context_chunks, seen_parents = [], set()
        for child in reranked:
            parent_id = child.get("parent_id", "")
            parent    = (self.parent_map.get(parent_id) or
                         self.parent_map.get(sanitize_id(parent_id)))
            if parent and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                context_chunks.append({
                    "content"    : parent["content"],
                    "source_file": parent.get("source_file", child["source_file"]),
                    "module"     : parent.get("module", child["module"]),
                    "doc_title"  : parent.get("doc_title", child["doc_title"]),
                })
            elif parent_id not in seen_parents:
                seen_parents.add(parent_id or child["source_file"])
                context_chunks.append({
                    "content"    : child["content"],
                    "source_file": child["source_file"],
                    "module"     : child["module"],
                    "doc_title"  : child["doc_title"],
                })
        return context_chunks

    def _build_sources(self, reranked: list) -> list:
        sources, seen = [], set()
        for c in reranked:
            if c["source_file"] not in seen:
                seen.add(c["source_file"])
                sources.append({
                    "module"     : c["module"],
                    "doc_title"  : c["doc_title"],
                    "source_file": c["source_file"],
                    "score"      : round(c["score"], 3),
                })
        return sources

    # ─────────────────────────────────────────────────────────────────
    # ÉTAPES 9 + 10 + 14 : PROMPT BUILDING
    # ─────────────────────────────────────────────────────────────────

    def _build_prompt_messages(self, question: str, context_chunks: list,
                                uncertain: bool = False) -> list:
        """
        Construit la liste de messages pour le LLM.

        Architecture :
          [0] system  → SYSTEM_PROMPT (rôle + CoT + format) + profil utilisateur
          [1..N] user/assistant → historique des échanges précédents
          [N+1] user  → extraits documentaires + question actuelle

        RÈGLE CRITIQUE (ÉTAPE 14) :
          Le profil est injecté dans le system prompt — PAS dans le message
          utilisateur. Envelopper les extraits avec le profil corrompt le prompt.
        """
        # ── ÉTAPE 14 : personnalisation dans le system prompt ─────────
        profile_instruction = PROFILE_INSTRUCTIONS.get(
            self.session.profile.role, ""
        )
        system_with_profile = SYSTEM_PROMPT + profile_instruction

        # ── Formatage des extraits documentaires ─────────────────────
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"--- Extrait {i} "
                f"[{chunk['module']} > {chunk['doc_title']}] ---\n"
                f"{chunk['content']}"
            )
        context_text = "\n\n".join(context_parts)

        # ── ÉTAPE 11 : avertissement si score incertain ───────────────
        uncertainty_warning = ""
        if uncertain:
            uncertainty_warning = (
                "\n⚠️  Les extraits ont un score de pertinence modéré. "
                "Si tu n'es pas certain, commence ta réponse par : "
                "\"Je ne suis pas certain, mais d'après la documentation...\"\n\n"
            )

        # ── Message utilisateur : extraits + question ─────────────────
        user_message = (
            f"Voici les extraits de documentation Divalto Harmony :\n\n"
            f"{context_text}\n\n"
            f"{'─' * 60}\n"
            f"{uncertainty_warning}"
            f"Question : {question}"
        )

        messages = [{"role": "system", "content": system_with_profile}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _generate(self, question: str, context_chunks: list,
                  uncertain: bool = False) -> str:
        messages = self._build_prompt_messages(question, context_chunks, uncertain)
        if LLM_PROVIDER == "openai":
            raw = self._call_openai(messages)
        elif LLM_PROVIDER == "gemini":
            raw = self._call_gemini(messages)
        else:
            raise ValueError(f"LLM_PROVIDER inconnu: {LLM_PROVIDER}")
        return _strip_thinking(raw)

    def _call_openai(self, messages: list) -> str:
        try:
            from openai import OpenAI
            client   = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages,
                temperature=0.1, max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            raise ImportError("pip install openai")
        except Exception as e:
            return f"[Erreur LLM OpenAI: {e}]"

    def _call_gemini(self, messages: list) -> str:
        """
        Format natif Gemini : system_instruction séparé + contents structurés.
        Gemini utilise "model" (pas "assistant") pour les tours précédents.
        """
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=GEMINI_API_KEY)

            system_instruction = messages[0]["content"]

            contents = []
            for msg in messages[1:]:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    )
                )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            return response.text.strip()
        except ImportError:
            raise ImportError("pip install google-genai")
        except Exception as e:
            return f"[Erreur LLM Gemini: {e}]"

    def _init_llm(self):
        if LLM_PROVIDER == "gemini":
            if not GEMINI_API_KEY:
                print("  WARNING: GEMINI_API_KEY non définie.")
                print("    export GEMINI_API_KEY=AIzaSy...")
            else:
                print(f"  LLM: Gemini {GEMINI_MODEL}")
        elif LLM_PROVIDER == "openai":
            if OPENAI_API_KEY.startswith("sk-..."):
                print("  WARNING: OPENAI_API_KEY non configurée.")
            else:
                print(f"  LLM: OpenAI {OPENAI_MODEL}")


# ─── INTERACTIVE CHAT ─────────────────────────────────────────────────────────

def chat():
    print("Profil utilisateur ?")
    print("  1. consultant      — expérimenté, réponses concises")
    print("  2. utilisateur_cle — référent métier, exemples concrets")
    print("  3. nouveau         — débutant, réponses pédagogiques")
    role_map = {"1": "consultant", "2": "utilisateur_cle", "3": "nouveau"}
    choice = input("Choix (1/2/3, défaut=3) : ").strip()
    role   = role_map.get(choice, "nouveau")

    assistant = DivaltoAssistant(user_role=role)

    print("=" * 65)
    print("  HAL — Assistant Divalto Harmony v2.1")
    print("  Tapez votre question en français.")
    print("  Commandes : 'quit' | 'reset' | 'profil <role>'")
    print("=" * 65)

    while True:
        try:
            question = input("\nVous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "quitter"):
            print("Au revoir.")
            break
        if question.lower() == "reset":
            assistant.reset_history()
            continue
        if question.lower().startswith("profil "):
            assistant.set_profile(question.split(" ", 1)[1].strip())
            continue

        result = assistant.ask(question, verbose=True)

        label = {
            "high"  : "✅ Confiance haute",
            "medium": "⚠️  Confiance modérée",
            "none"  : "❌ Non trouvé",
        }.get(result.get("confidence", "none"), "")

        print(f"\n{label}")
        print(f"\nHAL : {result['response']}")

        if result["sources"]:
            print("\nSources :")
            for s in result["sources"]:
                print(f"  • [{s['module']}] {s['doc_title']}  (score: {s['score']})")
        else:
            print(f"\n[Fallback niveau {result.get('fallback_level', 0)}]")


if __name__ == "__main__":
    chat()