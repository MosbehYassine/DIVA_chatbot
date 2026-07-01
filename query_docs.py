#!/usr/bin/env python3
"""
Script de requête pour un système RAG hybride (Graph RAG + Vector RAG)
Combine recherche par graphe (entités) et recherche vectorielle (similarité sémantique)
"""

import os
import sys
import pickle
import json
import re
import math
import unicodedata
from difflib import SequenceMatcher
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from datetime import datetime
from session_manager import SessionManager
from rag_config import (
    EMBEDDING_MODEL,
    is_e5_model,
    E5_QUERY_PREFIX,
    GRAPH_PATH,
    GRAPH_SOURCES_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    PARENTS_METADATA_PATH,
    INDEX_CONFIG_PATH,
    RETRIEVAL_POOL_SIZE,
    RERANK_LEXICAL_WEIGHT,
    RERANK_VECTOR_WEIGHT,
    RERANK_EMBEDDING_WEIGHT,
    SOURCE_MATCH_BOOST,
    DOC_ABOUT_BOOST,
    GRAPH_ENTITY_TOP_K,
    ENTITY_VECTOR_MIN_SCORE,
    RERANK_GRAPH_WEIGHT,
    ABSTENTION_THRESHOLD,
    RAG_ENABLE_BM25,
    RAG_ENABLE_CONTEXT_COMPRESSION,
    RAG_ENABLE_CROSS_ENCODER_RERANKER,
    RAG_ENABLE_HYDE,
    RAG_ENABLE_MMR,
    RAG_ENABLE_QUERY_REWRITING,
    RAG_ENABLE_SELF_QUERY,
    RAG_ENABLE_TFIDF,
    RAG_ENABLE_PARENT_CHILD_RETRIEVAL,
    RAG_ENABLE_CHAT_MEMORY,
    RAG_CHAT_HISTORY_TURNS,
    RAG_ENABLE_CONVERSATION_REFORMULATION,
    RAG_ENABLE_SESSION_SUMMARY,
    RAG_ENABLE_SESSION_SOURCE_BOOST,
    RAG_SESSION_SOURCE_BOOST,
    RAG_ENABLE_REASONING_PLAN,
    RAG_ENABLE_ENTITY_VECTOR_MATCH,
    RAG_ENABLE_EMBEDDING_RERANK,
    RAG_MAX_RETRY_COUNT,
    RAG_VERIFIER_MIN_CONFIDENCE,
    RAG_ENABLE_ANSWER_VERIFIER,
    RAG_BM25_TOP_K,
    RAG_VECTOR_TOP_K,
    RAG_GRAPH_TOP_K,
    RAG_FUSION_TOP_K,
    RAG_MMR_TOP_K,
    RAG_FINAL_TOP_K,
    RAG_CROSS_ENCODER_CANDIDATE_K,
    RAG_EXTRACTION_TOP_K,
    RAG_TITLE_EXACT_BOOST,
    RAG_TITLE_PARTIAL_BOOST,
    RAG_MMR_LAMBDA,
    RAG_MIN_ANSWER_CONFIDENCE,
)
from rag_canonical import parse_doc_about_topic
from rag_answer import (
    build_context_from_results,
    answer_with_sources,
    fallback_verify_answer,
    generate_answer_from_results,
    is_not_found_answer,
    source_references,
)
from rag_compressor import ContextualCompressor
from rag_lexical import LexicalRetriever
from rag_mmr import maximal_marginal_relevance
from rag_query_transform import extract_query_metadata, transform_query
from rag_conversation import reformulate_with_history
from rag_llm_answer import formulate_answer_with_llm
from rag_reranker import CrossEncoderReranker
try:
    from rag_verifier import verify_answer
except (ImportError, AttributeError) as exc:
    verify_answer = fallback_verify_answer
    print(
        "Avertissement: API rag_verifier incompatible; "
        f"utilisation du verificateur lexical de secours ({exc})"
    )
from rag_vector import encode_query, encode_passages, cosine_scores, top_k_indices

SESSIONS_FILE = "hybrid_rag_sessions.json"
SESSIONS_DB_FILE = "hybrid_rag_sessions.db"


def _sanitize_session_id(value: str) -> str:
    value = re.sub(r"\s+", "_", str(value or "").strip())
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "default_cli_session"


def choose_cli_session(sessions: SessionManager) -> str:
    """Ask the user which persisted session should be used by the CLI."""
    if not sessions.get_session_info("default_cli_session"):
        sessions.create_session(
            "default_cli_session",
            "Default interactive CLI session",
        )

    available = sessions.list_sessions()
    print("\nSessions disponibles:")
    for index, (session_id, info) in enumerate(available, 1):
        active = " *" if info.get("is_active") else ""
        turns = info.get("turns", 0)
        description = info.get("description") or session_id
        print(f"  {index}. {session_id}{active} ({turns} tours) - {description}")
    print("  N. Creer une nouvelle session")

    current = sessions.current_session() or "default_cli_session"
    while True:
        choice = input(f"\nChoisir une session [Entree = {current}]: ").strip()
        if not choice:
            selected = current
        elif choice.lower() in {"n", "new", "nouvelle", "nouveau"}:
            raw_id = input("ID de la nouvelle session: ").strip()
            session_id = _sanitize_session_id(raw_id)
            description = input("Description courte (optionnel): ").strip()
            if not sessions.get_session_info(session_id):
                sessions.create_session(session_id, description or session_id)
            selected = session_id
        elif choice.isdigit() and 1 <= int(choice) <= len(available):
            selected = available[int(choice) - 1][0]
        else:
            selected = _sanitize_session_id(choice)
            if not sessions.get_session_info(selected):
                create = input(
                    f"Session '{selected}' introuvable. La creer ? [o/N]: "
                ).strip().lower()
                if create not in {"o", "oui", "y", "yes"}:
                    continue
                sessions.create_session(selected, selected)

        if sessions.switch_session(selected):
            print(f"Session active: {sessions.current_session()}")
            return sessions.current_session()
        print(f"Impossible d'activer la session '{selected}'.")


class ConfidenceCalibrator:
    """Modèle de calibration (Platt Sigmoid / Isotonic) pour le score de rerank."""
    def __init__(self, model_path="calibration_model.json"):
        self.model_path = model_path
        self.method = "platt"  # "platt" ou "isotonic"
        self.platt_a = -10.0   # Paramètres heuristiques par défaut
        self.platt_b = 5.0
        self.isotonic_x = []
        self.isotonic_y = []
        self.load()

    def load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.method = data.get("method", "platt")
                self.platt_a = data.get("platt_a", self.platt_a)
                self.platt_b = data.get("platt_b", self.platt_b)
                self.isotonic_x = data.get("isotonic_x", [])
                self.isotonic_y = data.get("isotonic_y", [])
                print(f"Modèle de calibration chargé : {self.model_path} ({self.method})")
            except Exception as e:
                print(f"Erreur lors du chargement de la calibration : {e}")

    def calibrate(self, score: float) -> float:
        if self.method == "platt":
            try:
                val = self.platt_a * score + self.platt_b
                val = max(-100.0, min(100.0, val)) # Éviter overflow
                return 1.0 / (1.0 + math.exp(val))
            except Exception:
                return 0.0
        elif self.method == "isotonic":
            if not self.isotonic_x or not self.isotonic_y:
                return 0.0
            return float(np.interp(score, self.isotonic_x, self.isotonic_y))
        return 0.0


class HybridRAG:
    """Système RAG hybride combinant Graph RAG et Vector RAG"""
    
    def __init__(self, graph_path: str = GRAPH_PATH, 
                 faiss_path: str = FAISS_INDEX_PATH,
                 metadata_path: str = CHUNKS_METADATA_PATH,
                 parents_path: str = PARENTS_METADATA_PATH,
                 session_manager=None):
        self.graph = None
        self.faiss_index = None
        self.metadata = []
        self.parents_by_id = {}
        self.child_to_parent = {}
        self.embedding_model = None
        self.embedding_model_name = EMBEDDING_MODEL
        self.calibrator = ConfidenceCalibrator()
        self.cross_encoder = CrossEncoderReranker()
        self.compressor = ContextualCompressor()
        self.lexical_retriever = None
        self.session_manager = (
            session_manager
            if session_manager is not None
            else (
                SessionManager(max_history_turns=RAG_CHAT_HISTORY_TURNS)
                if RAG_ENABLE_CHAT_MEMORY else None
            )
        )
        
        self._entity_index = {}
        self._entity_nodes: List[str] = []
        self._entity_embeddings = np.zeros((0, 1), dtype="float32")
        self._load_graph(graph_path)
        self._load_vector_index(faiss_path, metadata_path, parents_path)
        print(
            "Parent-child retrieval: "
            + ("enabled" if RAG_ENABLE_PARENT_CHILD_RETRIEVAL else "disabled")
        )
        self.lexical_retriever = LexicalRetriever(self.metadata)
        print(f"Retrieval lexical: {self.lexical_retriever.backend}")
        self._load_embedding_model()
        self._build_entity_index()
        self._build_entity_embeddings()

    def _load_graph(self, graph_path):
        """Charge le graphe NetworkX (avec repli sur le fichier standard)."""
        path = graph_path
        if not os.path.exists(path):
            for candidate in (GRAPH_PATH, GRAPH_SOURCES_PATH):
                if os.path.exists(candidate):
                    path = candidate
                    break
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"Graphe chargé ({path}): {self.graph.number_of_nodes()} noeuds, {self.graph.number_of_edges()} relations")
        else:
            print(f"Graphe non trouvé: {graph_path}")

    def _load_vector_index(self, faiss_path, metadata_path, parents_path):
        """Charge l'index FAISS et les métadonnées."""
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            try:
                with open(faiss_path, 'rb') as f:
                    self.faiss_index = pickle.load(f)
            except Exception as exc:
                self.faiss_index = None
                print(f"Index FAISS indisponible; mode graphe uniquement ({exc})")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            if os.path.exists(parents_path):
                try:
                    with open(parents_path, "r", encoding="utf-8") as f:
                        parents = json.load(f)
                    self.parents_by_id = {
                        str(parent.get("parent_id")): parent
                        for parent in parents
                        if isinstance(parent, dict) and parent.get("parent_id") is not None
                    }
                    print(f"Parents charges: {len(self.parents_by_id)}")
                except Exception as exc:
                    print(f"Metadonnees parents illisibles ({exc}); repli compatible")
            self._build_parent_lookup()
            if self.faiss_index is not None:
                print(
                    f"Index FAISS charge: {len(self.metadata)} chunks indexes "
                    f"(dim={self.faiss_index.d})"
                )
            if self.metadata and isinstance(self.metadata[0], dict):
                index_model = self.metadata[0].get("embedding_model")
                if index_model:
                    self.embedding_model_name = index_model
                    print(f"Modèle utilisé à l'indexation: {index_model}")
        elif os.path.exists(INDEX_CONFIG_PATH):
            with open(INDEX_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("embedding_model"):
                self.embedding_model_name = cfg["embedding_model"]
                print(f"Modèle (index_config.json): {self.embedding_model_name}")
        else:
            print(f"Index vectoriel non trouvé: {faiss_path} / {metadata_path}")

    def _build_parent_lookup(self):
        """Build child-to-parent mappings, including legacy one-chunk indexes."""
        for idx, child in enumerate(self.metadata):
            if not isinstance(child, dict):
                continue
            parent_id = str(child.get("parent_id") or f"legacy_parent_{idx}")
            child_id = str(child.get("child_id") or child.get("chunk_id") or idx)
            document_id = str(child.get("document_id") or f"Document_{idx}")
            self.child_to_parent[child_id] = parent_id
            self.child_to_parent[document_id] = parent_id
            self.child_to_parent[str(idx)] = parent_id
            if parent_id not in self.parents_by_id:
                self.parents_by_id[parent_id] = {
                    "parent_id": parent_id,
                    "text": child.get("parent_text") or child.get("text", ""),
                    "source": child.get("source", "unknown"),
                    "original_file_path": child.get(
                        "original_file_path",
                        child.get("source", "unknown"),
                    ),
                    "title": child.get("title", ""),
                    "section": child.get("section", ""),
                    "module": child.get("module", ""),
                    "chunk_strategy": child.get("chunk_strategy", "legacy"),
                }

    def _load_embedding_model(self):
        """Charge le modèle d'embeddings aligné sur l'index."""
        model_name = self.embedding_model_name
        print(f"Chargement du modèle d'embeddings: {model_name}...")
        try:
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(model_name)
            get_dim = getattr(
                self.embedding_model,
                "get_embedding_dimension",
                self.embedding_model.get_sentence_embedding_dimension,
            )
            model_dim = get_dim()
            print(f"Modèle chargé (dim={model_dim})")
            if self.faiss_index and model_dim != self.faiss_index.d:
                print(
                    f"\n⚠️  INCOMPATIBILITÉ: index FAISS en {self.faiss_index.d}D "
                    f"mais modèle en {model_dim}D.\n"
                    f"   Relancez: python ingest_docs.py\n"
                )
                self.embedding_model = None
        except Exception as exc:
            print(f"Erreur chargement du modèle d'embeddings: {exc}")
            self.embedding_model = None

    def _build_entity_index(self):
        """Index inversé terme → entités (évite le scan complet du graphe)."""
        if not self.graph:
            return
        index = {}
        for node in self.graph.nodes():
            if not isinstance(node, str) or node.startswith("Document_") or len(node) < 3:
                continue
            node_data = self.graph.nodes[node] if node in self.graph else {}
            tokens = {self._normalize_text(node)}
            if isinstance(node_data, dict):
                for alias in node_data.get("aliases") or []:
                    if isinstance(alias, str) and alias:
                        tokens.add(self._normalize_text(alias))
            for token in tokens:
                for word in token.split():
                    if len(word) >= 3:
                        index.setdefault(word, set()).add(node)
        self._entity_index = {k: list(v) for k, v in index.items()}

    def _build_entity_embeddings(self):
        """Pré-calcule les embeddings des entités du graphe pour la correspondance vectorielle."""
        if (
            not RAG_ENABLE_ENTITY_VECTOR_MATCH
            or not self.graph
            or not self.embedding_model
        ):
            return
        labels = []
        nodes = []
        for node in self.graph.nodes():
            if not isinstance(node, str) or node.startswith("Document_") or len(node) < 3:
                continue
            node_data = self.graph.nodes[node] if node in self.graph else {}
            parts = [node]
            if isinstance(node_data, dict):
                for alias in node_data.get("aliases") or []:
                    if isinstance(alias, str) and alias.strip():
                        parts.append(alias.strip())
            labels.append(" | ".join(parts))
            nodes.append(node)
        if not labels:
            return
        self._entity_nodes = nodes
        self._entity_embeddings = encode_passages(
            self.embedding_model, labels, self.embedding_model_name
        )
        print(f"Embeddings entités: {len(nodes)} noeuds indexés")

    def _entities_from_query_vector(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Entités du graphe les plus proches sémantiquement de la question."""
        if (
            not RAG_ENABLE_ENTITY_VECTOR_MATCH
            or self._entity_embeddings.size == 0
            or not self.embedding_model
        ):
            return []
        k = top_k or GRAPH_ENTITY_TOP_K
        q_vec = encode_query(self.embedding_model, query, self.embedding_model_name)
        scores = cosine_scores(q_vec, self._entity_embeddings)
        indices = top_k_indices(scores, k)
        found = []
        for idx in indices:
            score = float(scores[idx])
            if score >= ENTITY_VECTOR_MIN_SCORE:
                found.append((self._entity_nodes[idx], score))
        return found

    def _chunk_embedding_score(self, query: str, text: str) -> float:
        """Similarité vectorielle question ↔ passage (re-ranking)."""
        if not self.embedding_model or not text:
            return 0.0
        q_vec = encode_query(self.embedding_model, query, self.embedding_model_name)
        p_vec = encode_passages(self.embedding_model, [text], self.embedding_model_name)
        scores = cosine_scores(q_vec, p_vec)
        return float(scores[0]) if scores.size else 0.0

    @staticmethod
    def _source_tokens(path: str) -> set:
        if not path:
            return set()
        base = os.path.basename(path).lower()
        stem = os.path.splitext(base)[0]
        parts = re.split(r"[^a-z0-9]+", stem)
        return {p for p in parts if len(p) >= 4}

    def _topic_source_boost(self, query: str, source: str) -> float:
        topic = parse_doc_about_topic(query)
        if not topic:
            return 0.0
        topic_tokens = set(topic.split())
        src_tokens = self._source_tokens(source)
        if not topic_tokens or not src_tokens:
            return 0.0
        overlap = len(topic_tokens & src_tokens) / max(len(topic_tokens), 1)
        if overlap >= 0.5:
            return DOC_ABOUT_BOOST
        joined = "".join(topic_tokens)
        stem = "".join(sorted(src_tokens))
        if joined and joined in stem:
            return DOC_ABOUT_BOOST * 0.85
        return overlap * DOC_ABOUT_BOOST

    def _chunk_text_by_id(self, chunk_id) -> str:
        try:
            idx = int(str(chunk_id).replace("Document_", ""))
            if 0 <= idx < len(self.metadata):
                return self.metadata[idx].get("text", "") or ""
        except (TypeError, ValueError):
            pass
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return normalized

    @staticmethod
    def _match_norm(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    @staticmethod
    def _page_overview_query(query: str) -> bool:
        normalized = HybridRAG._match_norm(query)
        return any(
            marker in normalized
            for marker in (
                "que couvre la page",
                "contenu principal",
                "details importants",
                "details important",
                "presente dans",
                "presentes dans",
            )
        )

    @staticmethod
    def _module_hint(query: str) -> str:
        match = re.search(
            r"\bmodule\s+([A-Za-zÀ-ÿ0-9_.-]+)",
            query or "",
            re.IGNORECASE,
        )
        return HybridRAG._match_norm(match.group(1)) if match else ""

    def _title_match_score_for_query(self, query: str, result: Dict) -> float:
        phrases = self._quoted_phrases(query)
        if not phrases:
            return float(result.get("title_match_score", 0.0) or 0.0)

        fields = [
            result.get("title") or "",
            result.get("section") or "",
            os.path.splitext(os.path.basename(str(result.get("source") or "")))[0],
            result.get("source") or "",
        ]
        field_norms = [self._match_norm(value) for value in fields if value]
        best = float(result.get("title_match_score", 0.0) or 0.0)
        for phrase in phrases:
            phrase_norm = self._match_norm(phrase)
            if not phrase_norm:
                continue
            phrase_terms = set(phrase_norm.split())
            for field_norm in field_norms:
                if not field_norm:
                    continue
                if phrase_norm == field_norm:
                    score = 1.0
                elif phrase_norm in field_norm or field_norm in phrase_norm:
                    score = 0.94
                else:
                    field_terms = set(field_norm.split())
                    overlap = len(phrase_terms & field_terms) / max(len(phrase_terms), 1)
                    sequence = SequenceMatcher(None, phrase_norm, field_norm).ratio()
                    score = max(overlap, sequence * 0.82)
                best = max(best, score)

        module_hint = self._module_hint(query)
        module_norm = self._match_norm(str(result.get("module") or ""))
        if module_hint and module_norm and module_hint == module_norm:
            best = min(1.0, best + 0.06)
        return min(1.0, best)

    @staticmethod
    def _dedupe_ranked_results(results: List[Dict]) -> List[Dict]:
        deduped = []
        seen = set()
        for result in results or []:
            key = (
                result.get("parent_id")
                or result.get("document_id")
                or result.get("chunk_id")
                or result.get("source")
            )
            key = str(key)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    @staticmethod
    def _compute_overlap_score(query: str, text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        query_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        if not query_terms:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for term in query_terms if term in text_lower)
        return matches / len(query_terms)

    @staticmethod
    def _document_key(result: Dict) -> str:
        doc_id = result.get('document_id') or result.get('chunk_id')
        if doc_id is not None:
            doc_str = str(doc_id)
            if not doc_str.startswith("Document_"):
                doc_str = f"Document_{doc_str}"
        else:
            doc_str = ""
        source = str(result.get('source') or '')
        return f"{doc_str}::{source}"

    def _entities_from_query(self, query: str) -> List[str]:
        """Entités probables dérivées de la question (correspondance vectorielle graphe)."""
        vector_hits = self._entities_from_query_vector(query)
        if vector_hits:
            return [ent for ent, _ in vector_hits]
        return []

    def _compute_graph_score(self, doc_id: str, matched_entities: List[Tuple[str, float]]) -> float:
        """Calcule un score basé sur les poids des connexions entre le document et les entités."""
        if doc_id is None:
            return 0.0
        doc_str = str(doc_id)
        if not doc_str.startswith("Document_"):
            doc_str = f"Document_{doc_str}"
            
        if not self.graph or not self.graph.has_node(doc_str):
            return 0.0
        
        score = 0.0
        neighbors = set(self.graph.neighbors(doc_str))
        for entity, ent_score in matched_entities:
            if entity in neighbors:
                edge_data = self.graph.get_edge_data(doc_str, entity) or {}
                weight = float(edge_data.get("weight", 1.0))
                # log1p atténue l'influence des fréquences géantes
                score += ent_score * np.log1p(weight)
        return score

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-classe les candidats fusionnés (lexical + score vectoriel/graphe + embedding + poids graphe)."""
        if not results:
            return results
        query_norm = self._normalize_text(query)

        # Calculer les entités les plus pertinentes de la requête
        matched_entities = self._entities_from_query_vector(query, top_k=GRAPH_ENTITY_TOP_K)

        embed_scores = np.zeros(len(results), dtype="float32")
        source_embed_scores = np.zeros(len(results), dtype="float32")
        if self.embedding_model and RAG_ENABLE_EMBEDDING_RERANK:
            texts = []
            sources = []
            for result in results:
                text = result.get("text") or self._chunk_text_by_id(
                    result.get("document_id") or result.get("chunk_id")
                )
                texts.append((text or "")[:2000])
                sources.append(os.path.basename(str(result.get("source") or "")))
            if texts:
                q_vec = encode_query(self.embedding_model, query, self.embedding_model_name)
                p_vecs = encode_passages(self.embedding_model, texts, self.embedding_model_name)
                embed_scores = cosine_scores(q_vec, p_vecs)
            if sources:
                q_vec = encode_query(self.embedding_model, query, self.embedding_model_name)
                src_vecs = encode_passages(self.embedding_model, sources, self.embedding_model_name)
                source_embed_scores = cosine_scores(q_vec, src_vecs)

        for i, result in enumerate(results):
            raw_doc_id = result.get("document_id") or result.get("chunk_id")
            text = result.get("text") or self._chunk_text_by_id(raw_doc_id)
            computed_lex = self._compute_overlap_score(query_norm, self._normalize_text(text))
            lex = max(
                computed_lex,
                float(result.get("lexical_score", 0.0) or 0.0),
            )
            source = str(result.get("source") or "")
            src_boost = self._compute_overlap_score(query_norm, self._normalize_text(source))
            vec = float(result.get("score", 0) or 0)
            hybrid = float(result.get("hybrid_score", 0) or 0)
            semantic = max(vec, hybrid)
            if semantic > 1.0:
                semantic = min(1.0, semantic / 100.0)
            embed = float(embed_scores[i]) if i < len(embed_scores) else 0.0
            src_embed = float(source_embed_scores[i]) if i < len(source_embed_scores) else 0.0
            
            # Calcul du score de poids de graphe
            graph_weight_score = self._compute_graph_score(raw_doc_id, matched_entities)
            
            topic_boost = self._topic_source_boost(query, source)
            title_score = self._title_match_score_for_query(query, result)
            title_boost = 0.0
            if title_score >= 0.98:
                title_boost = RAG_TITLE_EXACT_BOOST
            elif title_score >= 0.70:
                title_boost = RAG_TITLE_PARTIAL_BOOST * title_score
            elif result.get("retrieval_source") == "title_lookup":
                title_boost = 0.35
            file_boost = 0.0
            q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
            src_tokens = self._source_tokens(source)
            if q_terms and src_tokens:
                file_boost = SOURCE_MATCH_BOOST * (
                    sum(1 for t in q_terms if t in src_tokens) / len(q_terms)
                )
            result["lexical_score"] = lex
            result["source_lexical_score"] = src_boost
            result["vector_score"] = float(result.get("vector_score", vec) or 0.0)
            result["graph_score"] = float(result.get("graph_score", 0.0) or 0.0)
            result["embedding_score"] = embed
            result["source_embedding_score"] = src_embed
            result["graph_weight_score"] = graph_weight_score
            result["source_match_boost"] = file_boost
            result["document_topic_boost"] = topic_boost
            result["title_match_boost"] = title_boost
            result["title_match_score"] = title_score
            result["rerank_score"] = (
                RERANK_LEXICAL_WEIGHT * min(1.0, lex + 0.15 * src_boost)
                + RERANK_VECTOR_WEIGHT * semantic
                + RERANK_GRAPH_WEIGHT * graph_weight_score
                + RERANK_EMBEDDING_WEIGHT * embed
                + 0.20 * src_embed
                + topic_boost
                + file_boost
                + title_boost
            )
            result["source_score"] = min(
                1.0,
                src_boost + src_embed + topic_boost + file_boost,
            )
            result["combined_score"] = result["rerank_score"]
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        for r in results:
            r["hybrid_score"] = r.get("rerank_score", r.get("hybrid_score", 0))
        return results

    def _apply_query_metadata_score(self, results: List[Dict], query_metadata: Dict) -> List[Dict]:
        """Apply SelfQuery hints as a soft score, never as a hard filter."""
        if not results or not RAG_ENABLE_SELF_QUERY:
            return results
        domain = str(query_metadata.get("domain") or "unknown").lower()
        keywords = [str(value).lower() for value in query_metadata.get("keywords") or []]
        for result in results:
            haystack = " ".join(
                str(result.get(key) or "")
                for key in ("source", "title", "section", "module", "text")
            ).lower()
            domain_score = 0.0 if domain == "unknown" else float(domain in haystack)
            keyword_score = (
                sum(1 for keyword in keywords if keyword in haystack) / max(len(keywords), 1)
                if keywords else 0.0
            )
            self_query_score = min(1.0, 0.60 * domain_score + 0.40 * keyword_score)
            result["self_query_score"] = self_query_score
            result["combined_score"] = (
                float(result.get("rerank_score", 0.0) or 0.0)
                + 0.12 * self_query_score
            )
            result["hybrid_score"] = result["combined_score"]
        results.sort(key=lambda item: item.get("combined_score", 0.0), reverse=True)
        return results

    def _apply_session_source_boost(
        self,
        results: List[Dict],
        session_context: Dict,
    ) -> List[Dict]:
        """Softly favor sources already useful in the active session."""
        if not RAG_ENABLE_SESSION_SOURCE_BOOST or not results or not session_context:
            return results
        recent_sources = {
            os.path.basename(str(source)).lower()
            for source in session_context.get("recent_sources", [])
            if source
        }
        if not recent_sources:
            return results
        for result in results:
            filename = os.path.basename(str(result.get("source") or "")).lower()
            if filename and filename in recent_sources:
                base_score = float(result.get("hybrid_score", 0.0) or 0.0)
                result["session_source_boost"] = RAG_SESSION_SOURCE_BOOST
                result["hybrid_score"] = base_score + RAG_SESSION_SOURCE_BOOST
                result["rerank_score"] = (
                    float(result.get("rerank_score", base_score) or 0.0)
                    + RAG_SESSION_SOURCE_BOOST
                )
                result["combined_score"] = (
                    float(result.get("combined_score", base_score) or 0.0)
                    + RAG_SESSION_SOURCE_BOOST
                )
        results.sort(
            key=lambda item: float(item.get("hybrid_score", 0.0) or 0.0),
            reverse=True,
        )
        return results

    # ============= RETRIEVAL PAR GRAPHE =============
    def search_graph(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche par graphe (entités) - optimisée avec alias."""
        if not self.graph:
            return []

        query_normalized = self._normalize_text(query)
        query_lower = query.lower()
        results = []

        entities_found = []

        # Correspondance vectorielle entités ↔ question
        for entity, entity_score in self._entities_from_query_vector(
            query, top_k=max(GRAPH_ENTITY_TOP_K, top_k)
        ):
            entities_found.append((entity, entity_score))

        # Complément lexical via index inversé du graphe
        candidate_nodes = set()
        for word in re.findall(r"\w+", query_lower):
            if len(word) >= 3:
                for ent in self._entity_index.get(word, []):
                    candidate_nodes.add(ent)

        for node in candidate_nodes:
            if not self.graph or not self.graph.has_node(node):
                continue
            node_data = self.graph.nodes[node] if node in self.graph else {}
            max_score = self._compute_overlap_score(query_normalized, self._normalize_text(node))
            if isinstance(node_data, dict):
                aliases = node_data.get("aliases", [])
                if not isinstance(aliases, list):
                    aliases = []
                for alias in aliases:
                    alias_score = self._compute_overlap_score(
                        query_normalized, self._normalize_text(str(alias))
                    )
                    max_score = max(max_score, alias_score)
                    if isinstance(alias, str) and alias.lower() in query_lower:
                        max_score = min(1.0, max_score + 0.15)
            if max_score > 0:
                entities_found.append((node, max_score * 0.75))

        seen_entities = set()
        deduped = []
        for ent, sc in sorted(entities_found, key=lambda x: x[1], reverse=True):
            if ent not in seen_entities:
                seen_entities.add(ent)
                deduped.append((ent, sc))
        entities_found = deduped

        for entity, entity_score in entities_found[:top_k]:
            if entity in self.graph:
                predecessors = list(self.graph.predecessors(entity))
                docs = [n for n in predecessors if isinstance(n, str) and n.startswith("Document_")]

                for doc_node in docs[:8]:
                    if doc_node in self.graph:
                        doc_data = self.graph.nodes[doc_node]
                        doc_text = doc_data.get('text', '') or self._chunk_text_by_id(doc_node)
                        doc_source = doc_data.get('source') or doc_data.get('title') or doc_node
                        results.append({
                            'entity': entity,
                            'document_id': doc_node,
                            'title': doc_data.get('title', doc_source),
                            'text': doc_text,
                            'source': doc_source,
                            'child_id': doc_data.get('child_id'),
                            'parent_id': doc_data.get('parent_id'),
                            'section': doc_data.get('section', ''),
                            'module': doc_data.get('module', ''),
                            'score': entity_score,
                            'graph_score': entity_score,
                            'method': 'graph'
                        })

        return results[:top_k]

    # ============= RETRIEVAL VECTORIEL =============
    def search_vector(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche vectorielle (similarité sémantique)."""
        if not self.faiss_index or not self.embedding_model:
            return []
        
        q = f"{E5_QUERY_PREFIX}{query}" if is_e5_model(self.embedding_model_name) else query
        query_embedding = self.embedding_model.encode([q], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                score = float(distances[0][i])
                chunk_meta = self.metadata[idx]
                chunk_text = chunk_meta.get('text', '') or ''
                source = chunk_meta.get('source') or chunk_meta.get('document_id') or f"chunk_{idx}"
                results.append({
                    'chunk_id': idx,
                    'child_id': chunk_meta.get('child_id', f'child_{idx}'),
                    'parent_id': chunk_meta.get('parent_id'),
                    'document_id': chunk_meta.get('document_id', f'Document_{idx}'),
                    'text': chunk_text,
                    'source': source,
                    'original_file_path': chunk_meta.get('original_file_path', source),
                    'title': chunk_meta.get('title', source),
                    'section': chunk_meta.get('section', ''),
                    'module': chunk_meta.get('module', ''),
                    'score': score,
                    'vector_score': score,
                    'method': 'vector'
                })
        
        return results

    @staticmethod
    def _quoted_phrases(query: str) -> List[str]:
        phrases = re.findall(r'"([^"]+)"|«([^»]+)»|“([^”]+)”', query or "")
        values = []
        for groups in phrases:
            value = next((part for part in groups if part), "")
            value = re.sub(r"\s+", " ", value).strip()
            if len(value) >= 4:
                values.append(value)
        return list(dict.fromkeys(values))

    def search_title(self, query: str, top_k: int = 10) -> List[Dict]:
        """Direct title/source lookup for questions quoting a documentation page."""
        phrases = self._quoted_phrases(query)
        if not phrases or not self.metadata:
            return []
        module_hint = ""
        module_match = re.search(
            r"\bmodule\s+([A-Za-zÀ-ÿ0-9_.-]+)",
            query or "",
            re.IGNORECASE,
        )
        if module_match:
            module_hint = self._normalize_text(module_match.group(1))

        candidates = []
        seen = set()
        for idx, chunk_meta in enumerate(self.metadata):
            title = str(chunk_meta.get("title") or "")
            section = str(chunk_meta.get("section") or "")
            source = str(
                chunk_meta.get("source")
                or chunk_meta.get("document_id")
                or f"chunk_{idx}"
            )
            module = str(chunk_meta.get("module") or "")
            haystack_values = [
                title,
                section,
                os.path.splitext(os.path.basename(source))[0],
                source,
            ]
            haystack_norm = " ".join(
                self._normalize_text(value) for value in haystack_values
            )
            best = 0.0
            for phrase in phrases:
                phrase_norm = self._normalize_text(phrase)
                if not phrase_norm:
                    continue
                if phrase_norm in haystack_norm:
                    score = 1.0
                else:
                    score = max(
                        SequenceMatcher(
                            None,
                            phrase_norm,
                            self._normalize_text(value),
                        ).ratio()
                        for value in haystack_values
                        if value
                    )
                best = max(best, score)
            if module_hint and module_hint == self._normalize_text(module):
                best = min(1.0, best + 0.08)
            if best < 0.70:
                continue
            key = str(chunk_meta.get("parent_id") or chunk_meta.get("child_id") or source)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "chunk_id": idx,
                    "child_id": chunk_meta.get("child_id", f"child_{idx}"),
                    "parent_id": chunk_meta.get("parent_id"),
                    "document_id": chunk_meta.get("document_id", f"Document_{idx}"),
                    "text": chunk_meta.get("text", "") or "",
                    "source": source,
                    "original_file_path": chunk_meta.get("original_file_path", source),
                    "title": title or source,
                    "section": section,
                    "module": module,
                    "score": best,
                    "lexical_score": best,
                    "title_match_score": best,
                    "method": "title",
                    "retrieval_source": "title_lookup",
                }
            )
        candidates.sort(key=lambda item: item["title_match_score"], reverse=True)
        return candidates[:top_k]

    # ============= FUSION DES RÉSULTATS =============
    def _merge_results(
        self,
        graph_results: List[Dict],
        vector_results: List[Dict],
        lexical_results: List[Dict] = None,
    ) -> List[Dict]:
        """Fusionne et déduplique les résultats graphe, vecteur et lexical."""
        seen = {}
        merged = []

        for result in graph_results:
            result_key = self._document_key(result)
            candidate = result.copy()
            candidate['graph_score'] = float(candidate.get('score', 0.0) or 0.0)
            candidate['hybrid_score'] = min(0.95, candidate['score'] * 1.05)
            if result_key not in seen:
                merged.append(candidate)
                seen[result_key] = candidate
            else:
                existing = seen[result_key]
                existing['hybrid_score'] = max(existing['hybrid_score'], candidate['hybrid_score'])
                existing['graph_score'] = max(
                    float(existing.get('graph_score', 0.0) or 0.0),
                    candidate['graph_score'],
                )
                existing['method'] = 'graph'

        for result in vector_results:
            result_key = self._document_key(result)
            candidate = result.copy()
            candidate['vector_score'] = float(candidate.get('score', 0.0) or 0.0)
            candidate['hybrid_score'] = min(0.95, candidate['score'] * 1.0)
            if result_key not in seen:
                merged.append(candidate)
                seen[result_key] = candidate
            else:
                existing = seen[result_key]
                combined_score = (existing.get('hybrid_score', 0.5) + candidate['hybrid_score']) / 2
                existing['hybrid_score'] = min(0.95, combined_score)
                if existing.get('method') != 'graph':
                    existing['method'] = 'hybrid'
                existing['source'] = existing.get('source') or candidate.get('source')
                existing['title'] = existing.get('title') or candidate.get('title')
                existing['vector_score'] = max(
                    float(existing.get('vector_score', 0.0) or 0.0),
                    candidate['vector_score'],
                )
                existing["hyde_used"] = bool(
                    existing.get("hyde_used") or candidate.get("hyde_used")
                )
                if candidate.get("retrieval_source") == "hyde_vector":
                    existing["retrieval_source"] = "hyde_vector"
                for key in (
                    'child_id', 'parent_id', 'original_file_path',
                    'section', 'module',
                ):
                    existing[key] = existing.get(key) or candidate.get(key)

        for result in lexical_results or []:
            result_key = self._document_key(result)
            candidate = result.copy()
            candidate["lexical_score"] = float(
                candidate.get("lexical_score", candidate.get("score", 0.0)) or 0.0
            )
            candidate["hybrid_score"] = candidate["lexical_score"]
            if result_key not in seen:
                merged.append(candidate)
                seen[result_key] = candidate
            else:
                existing = seen[result_key]
                existing["lexical_score"] = max(
                    float(existing.get("lexical_score", 0.0) or 0.0),
                    candidate["lexical_score"],
                )
                existing["hybrid_score"] = max(
                    float(existing.get("hybrid_score", 0.0) or 0.0),
                    candidate["hybrid_score"],
                )
                methods = set(str(existing.get("method") or "").split("+"))
                methods.add(str(candidate.get("method") or "lexical"))
                existing["method"] = "+".join(sorted(method for method in methods if method))
                existing["retrieval_source"] = candidate.get("retrieval_source")
                for key in (
                    "child_id", "parent_id", "original_file_path",
                    "title", "section", "module",
                ):
                    existing[key] = existing.get(key) or candidate.get(key)

        merged.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return merged

    def _map_results_to_parents(self, child_results: List[Dict]) -> List[Dict]:
        """Replace child hits with unique parent chunks for reranking and answers."""
        if not RAG_ENABLE_PARENT_CHILD_RETRIEVAL:
            print("   Child-to-parent mapping skipped (disabled)")
            return [
                {
                    **result,
                    "best_child_score": float(
                        result.get("hybrid_score", result.get("score", 0.0)) or 0.0
                    ),
                    "matched_child_text": result.get("text", ""),
                }
                for result in child_results
            ]

        parents = {}
        for result in child_results:
            matched_child_text = str(result.get("text") or "")
            child_score = max(
                float(result.get("hybrid_score", 0.0) or 0.0),
                float(result.get("vector_score", 0.0) or 0.0),
                float(result.get("graph_score", 0.0) or 0.0),
                float(result.get("lexical_score", 0.0) or 0.0),
            )
            lookup_keys = (
                result.get("parent_id"),
                result.get("child_id"),
                result.get("document_id"),
                result.get("chunk_id"),
            )
            parent_id = None
            for key in lookup_keys:
                if key is None:
                    continue
                key_str = str(key)
                if key_str in self.parents_by_id:
                    parent_id = key_str
                    break
                if key_str in self.child_to_parent:
                    parent_id = self.child_to_parent[key_str]
                    break
            if parent_id is None:
                parent_id = f"runtime_parent_{self._document_key(result)}"

            parent_meta = self.parents_by_id.get(parent_id, {})
            candidate = dict(result)
            candidate.update(
                {
                    "parent_id": parent_id,
                    "text": parent_meta.get("text") or result.get("text") or "",
                    "source": parent_meta.get("source") or result.get("source") or "unknown",
                    "original_file_path": parent_meta.get("original_file_path")
                    or result.get("original_file_path")
                    or result.get("source")
                    or "unknown",
                    "title": parent_meta.get("title") or result.get("title") or "",
                    "section": parent_meta.get("section") or result.get("section") or "",
                    "module": parent_meta.get("module") or result.get("module") or "",
                    "child_id": str(
                        result.get("child_id") or result.get("chunk_id") or ""
                    ),
                    "best_child_score": child_score,
                    "matched_child_text": matched_child_text,
                    "matched_child_ids": [
                        str(result.get("child_id") or result.get("chunk_id") or "")
                    ],
                }
            )
            existing = parents.get(parent_id)
            if existing is None:
                parents[parent_id] = candidate
                continue

            existing["hybrid_score"] = max(
                float(existing.get("hybrid_score", 0.0) or 0.0),
                float(candidate.get("hybrid_score", 0.0) or 0.0),
            )
            existing["vector_score"] = max(
                float(existing.get("vector_score", 0.0) or 0.0),
                float(candidate.get("vector_score", 0.0) or 0.0),
            )
            existing["graph_score"] = max(
                float(existing.get("graph_score", 0.0) or 0.0),
                float(candidate.get("graph_score", 0.0) or 0.0),
            )
            existing["lexical_score"] = max(
                float(existing.get("lexical_score", 0.0) or 0.0),
                float(candidate.get("lexical_score", 0.0) or 0.0),
            )
            existing["matched_child_ids"].extend(candidate["matched_child_ids"])
            existing["matched_child_ids"] = list(dict.fromkeys(existing["matched_child_ids"]))
            if child_score > float(existing.get("best_child_score", 0.0) or 0.0):
                existing["best_child_score"] = child_score
                existing["matched_child_text"] = matched_child_text
                existing["child_id"] = candidate["child_id"]
            existing["hyde_used"] = bool(
                existing.get("hyde_used") or candidate.get("hyde_used")
            )
            if existing.get("method") != candidate.get("method"):
                existing["method"] = "hybrid"
            if float(candidate.get("title_match_score", 0.0) or 0.0) > float(
                existing.get("title_match_score", 0.0) or 0.0
            ):
                existing["title_match_score"] = candidate.get("title_match_score", 0.0)
                existing["retrieval_source"] = candidate.get(
                    "retrieval_source",
                    existing.get("retrieval_source"),
                )
            queries = list(existing.get("retrieval_queries") or [])
            queries.extend(candidate.get("retrieval_queries") or [])
            if candidate.get("retrieval_query"):
                queries.append(candidate["retrieval_query"])
            existing["retrieval_queries"] = list(dict.fromkeys(queries))

        mapped = sorted(
            parents.values(),
            key=lambda item: float(item.get("hybrid_score", 0.0) or 0.0),
            reverse=True,
        )
        print(
            f"   Child-to-parent mapping success: "
            f"{len(child_results)} children -> {len(mapped)} parents"
        )
        return mapped

    def _retrieve_transformed(
        self,
        query: str,
        pool_k: int,
        force_expansion: bool = False,
        force_hyde: bool = False,
    ):
        print("   [query transformation]")
        transformation = transform_query(
            query,
            enable_expansion=RAG_ENABLE_QUERY_REWRITING or force_expansion,
            enable_hyde=RAG_ENABLE_HYDE or force_hyde,
        )
        query_metadata = (
            extract_query_metadata(query)
            if RAG_ENABLE_SELF_QUERY
            else {"domain": "unknown", "keywords": [], "filters": {}}
        )
        variants = transformation.get("query_variants") or [query] + transformation["variants"]
        variants = list(dict.fromkeys(value for value in variants if value))
        print(
            f"   Query variants ({len(variants)}): {variants} | "
            f"HyDE: {'on' if transformation['hyde_text'] else 'off'}"
        )
        if transformation["hyde_text"]:
            print(
                "   HyDE query generated: "
                + transformation["hyde_text"][:180].replace("\n", " ")
            )
        print(f"   Query metadata: {query_metadata}")

        graph_results = []
        print("   [graph retrieval]")
        graph_k = max(RAG_GRAPH_TOP_K, pool_k if force_expansion else 0)
        for retrieval_query in variants:
            hits = self.search_graph(retrieval_query, top_k=graph_k)
            for hit in hits:
                hit["retrieval_query"] = retrieval_query
                hit["retrieval_queries"] = [retrieval_query]
            graph_results.extend(hits)
        print(f"   Graphe: {len(graph_results)} resultat(s) bruts")

        vector_results = []
        print("   [FAISS retrieval]")
        vector_k = max(RAG_VECTOR_TOP_K, pool_k if force_expansion else 0)
        vector_queries = [
            (variant, "original" if index == 0 else "rewritten")
            for index, variant in enumerate(variants)
        ]
        if transformation["hyde_text"]:
            vector_queries.append((transformation["hyde_text"], "hyde"))
        for retrieval_query, query_type in vector_queries:
            hits = self.search_vector(retrieval_query, top_k=vector_k)
            for hit in hits:
                hit["retrieval_query"] = retrieval_query
                hit["retrieval_queries"] = [retrieval_query]
                hit["retrieval_query_type"] = query_type
                hit["hyde_used"] = query_type == "hyde"
                if query_type == "hyde":
                    hit["retrieval_source"] = "hyde_vector"
            vector_results.extend(hits)
        print(f"   Vectoriel: {len(vector_results)} resultat(s) bruts")

        lexical_results = []
        print("   [lexical retrieval]")
        title_hits = self.search_title(query, top_k=max(10, RAG_BM25_TOP_K // 2))
        for hit in title_hits:
            hit["retrieval_query"] = query
            hit["retrieval_queries"] = [query]
        lexical_results.extend(title_hits)
        if self.lexical_retriever and self.lexical_retriever.enabled:
            lexical_k = max(RAG_BM25_TOP_K, pool_k if force_expansion else 0)
            for retrieval_query in variants:
                hits = self.lexical_retriever.search(retrieval_query, top_k=lexical_k)
                for hit in hits:
                    hit["retrieval_query"] = retrieval_query
                    hit["retrieval_queries"] = [retrieval_query]
                lexical_results.extend(hits)
        print(
            f"   Lexical: {len(lexical_results)} resultat(s) bruts "
            f"(title={len(title_hits)})"
        )
        transformation["query_metadata"] = query_metadata
        transformation["query_variants"] = variants
        return transformation, graph_results, vector_results, lexical_results

    def _execute_pipeline(
        self,
        question: str,
        retrieval_query: str,
        top_k: int,
        retry: bool = False,
        session_context: Dict = None,
        original_question: str = None,
        reasoning_plan: List[str] = None,
    ) -> Dict:
        pool_k = max(RETRIEVAL_POOL_SIZE, top_k * 5) * (2 if retry else 1)
        transformation, graph_results, vector_results, lexical_results = self._retrieve_transformed(
            retrieval_query,
            pool_k=pool_k,
            force_expansion=retry,
            force_hyde=retry,
        )

        print("   [fusion]")
        child_results = self._merge_results(
            graph_results,
            vector_results,
            lexical_results,
        )
        print(f"   Fusion enfants: {len(child_results)} candidat(s)")

        print("   [parent retrieval]")
        parent_results = self._map_results_to_parents(child_results)
        print(f"   Parents uniques: {len(parent_results)}")

        parent_results = self._rerank_results(retrieval_query, parent_results)
        parent_results = self._apply_session_source_boost(
            parent_results,
            session_context or {},
        )
        parent_results = self._apply_query_metadata_score(
            parent_results,
            transformation["query_metadata"],
        )[:RAG_FUSION_TOP_K]

        print("   [MMR diversification]")
        before_mmr = len(parent_results)
        if RAG_ENABLE_MMR and parent_results:
            if self.embedding_model is not None:
                query_embedding = encode_query(
                    self.embedding_model,
                    retrieval_query,
                    self.embedding_model_name,
                )
                embedding_fn = lambda texts: encode_passages(
                    self.embedding_model,
                    texts,
                    self.embedding_model_name,
                )
            else:
                query_embedding = np.array([], dtype="float32")
                embedding_fn = lambda texts: (_ for _ in ()).throw(
                    RuntimeError("embedding model unavailable")
                )
            mmr_results = maximal_marginal_relevance(
                query_embedding,
                parent_results,
                embedding_fn,
                top_k=RAG_MMR_TOP_K,
                lambda_mult=RAG_MMR_LAMBDA,
            )
        else:
            mmr_results = parent_results[:RAG_MMR_TOP_K]
        protected_title_results = [
            result for result in parent_results
            if (
                result.get("retrieval_source") == "title_lookup"
                or float(result.get("title_match_score", 0.0) or 0.0) >= 0.70
            )
        ][:5]
        if protected_title_results:
            mmr_results = self._dedupe_ranked_results(
                protected_title_results + mmr_results
            )[:RAG_MMR_TOP_K]
        print(f"   MMR: {len(mmr_results)} selectionne(s)")

        print("   [cross-encoder reranking]")
        cross_k = max(RAG_FINAL_TOP_K, RAG_CROSS_ENCODER_CANDIDATE_K)
        if RAG_ENABLE_CROSS_ENCODER_RERANKER:
            reranked_pool = self.cross_encoder.rerank(
                question,
                mmr_results,
                top_k=cross_k,
            )
        else:
            reranked_pool = mmr_results[:cross_k]
        for result in reranked_pool:
            result["hybrid_score"] = float(
                result.get(
                    "cross_encoder_score",
                    result.get("combined_score", result.get("rerank_score", 0.0)),
                ) or 0.0
            )
            title_score = self._title_match_score_for_query(retrieval_query, result)
            result["title_match_score"] = max(
                float(result.get("title_match_score", 0.0) or 0.0),
                title_score,
            )
            if self._page_overview_query(retrieval_query):
                result["hybrid_score"] += 0.45 * result["title_match_score"]
        reranked_pool.sort(
            key=lambda item: float(item.get("hybrid_score", 0.0) or 0.0),
            reverse=True,
        )
        reranked_results = reranked_pool[:RAG_FINAL_TOP_K]
        print(f"   Cross-encoder: {len(reranked_results)} selectionne(s)")

        top_score = (
            float(
                reranked_results[0].get(
                    "cross_encoder_score",
                    reranked_results[0].get(
                        "combined_score",
                        reranked_results[0].get("rerank_score", 0.0),
                    ),
                ) or 0.0
            )
            if reranked_results else 0.0
        )
        retrieval_confidence = self.calibrator.calibrate(top_score)
        print(
            f"   Score top rerank: {top_score:.4f} | "
            f"Confidence retrieval: {retrieval_confidence:.2%}"
        )

        print("   [compression]")
        compression_pool = self._dedupe_ranked_results(
            reranked_results + reranked_pool + parent_results[:RAG_EXTRACTION_TOP_K]
        )[:max(RAG_FINAL_TOP_K, RAG_EXTRACTION_TOP_K)]
        compressed_results = self.compressor.compress(
            question,
            compression_pool,
            embedding_model=self.embedding_model,
            embedding_model_name=self.embedding_model_name,
        )
        print(f"   Contextes compresses: {len(compressed_results)}")

        extractive_answer = generate_answer_from_results(
            question,
            compressed_results,
            embedding_model=self.embedding_model,
            embedding_model_name=self.embedding_model_name,
        )
        generated_answer = formulate_answer_with_llm(
            original_question or question,
            extractive_answer,
            compressed_results,
            reasoning_plan=reasoning_plan or [],
            session_summary=(session_context or {}).get("summary", ""),
        )
        print("   [answer verification]")
        verification = verify_answer(
            original_question or question,
            generated_answer,
            compressed_results,
            embedding_model=self.embedding_model,
            embedding_model_name=self.embedding_model_name,
        )
        print(
            f"   Supported: {verification['answer_supported']} | "
            f"Confidence: {verification['confidence']:.2%} | "
            f"Retry: {verification['needs_retry']}"
        )
        return {
            "transformation": transformation,
            "graph_results": graph_results,
            "vector_results": vector_results,
            "lexical_results": lexical_results,
            "child_results": child_results,
            "parent_results": parent_results,
            "mmr_results": mmr_results,
            "reranked_pool": reranked_pool,
            "answer_results": compression_pool,
            "reranked_results": reranked_results,
            "compressed_results": compressed_results,
            "extractive_answer": extractive_answer,
            "generated_answer": generated_answer,
            "verification": verification,
            "retrieval_confidence": retrieval_confidence,
            "retrieval_counts": {
                "vector": len(vector_results),
                "graph": len(graph_results),
                "lexical": len(lexical_results),
                "merged": len(child_results),
                "parents": len(parent_results),
                "before_mmr": before_mmr,
                "after_mmr": len(mmr_results),
                "final": len(reranked_results),
            },
        }

    def _build_reasoning_plan(
        self,
        question: str,
        session_summary: str = "",
    ) -> List[str]:
        """Build a short internal decomposition used to guide extractive answer ranking."""
        if not RAG_ENABLE_REASONING_PLAN:
            return []
        normalized = self._normalize_text(question)
        focus_terms = [
            term for term in re.findall(r"[A-Za-zÀ-ÿ0-9_.-]{3,}", question)
            if self._normalize_text(term) not in {
                "comment", "quel", "quelle", "quels", "quelles", "dans",
                "avec", "pour", "sans", "harmony", "divalto", "est", "sont",
                "une", "des", "les", "aux", "sur", "que", "quoi",
            }
        ][:8]
        focus = " ".join(focus_terms) or question
        plan = [f"Identifier le sujet principal: {focus}."]
        if any(token in normalized for token in ("comment", "creer", "lancer", "utiliser", "configurer", "modifier")):
            plan.append("Chercher les etapes, actions, menus ou commandes associes.")
        if any(token in normalized for token in ("obligatoire", "doit", "faut", "condition", "avant", "necessaire")):
            plan.append("Chercher les conditions, prerequis, droits ou limites.")
        if any(token in normalized for token in ("erreur", "probleme", "ne fonctionne", "impossible", "corriger")):
            plan.append("Chercher les causes probables et les controles a effectuer.")
        if any(token in normalized for token in ("ou", "chemin", "fichier", "stocke", "trouver")):
            plan.append("Chercher les emplacements, fichiers, chemins ou modules cites.")
        if session_summary:
            plan.append(f"Conserver le contexte de session: {session_summary[:180]}.")
        return plan[:5]

    @staticmethod
    def _question_with_reasoning_plan(question: str, reasoning_plan: List[str]) -> str:
        if not reasoning_plan:
            return question
        return question + "\n" + "\n".join(reasoning_plan)

    # ============= INTERFACE PRINCIPALE =============
    def query(
        self,
        question: str,
        top_k: int = 1,
        retrieval_question: str = None,
        session_id: str = None,
    ) -> Dict:
        """Run transformed hybrid retrieval, parent reranking, and verification."""
        original_question = question
        history = []
        history_used = False
        session_context = {}
        session_summary = ""
        recent_sources = []
        memory = getattr(self, "session_manager", None)
        if session_id and RAG_ENABLE_CHAT_MEMORY and memory is not None:
            history = memory.get_history(
                session_id,
                limit=RAG_CHAT_HISTORY_TURNS,
            )
            if RAG_ENABLE_SESSION_SUMMARY:
                session_context = memory.get_session_context(session_id) or {}
                session_summary = str(session_context.get("global_context") or "")
                recent_sources = memory.get_recent_sources(session_id, limit=5)
            if (
                RAG_ENABLE_CONVERSATION_REFORMULATION
                and not retrieval_question
                and history
            ):
                question = reformulate_with_history(question, history)
                history_used = question != original_question
            print(
                f"   [chat memory] session={session_id} "
                f"turns={len(history)} used={history_used}"
            )
            if session_summary:
                print(f"   Session summary: {session_summary[:180]}")
            if history_used:
                print(f"   Standalone question: {question[:220]}")
        else:
            print("   [chat memory] disabled or no session_id")

        effective_query = retrieval_question or question
        active_session_context = {
            "summary": session_summary,
            "recent_sources": recent_sources,
        }
        reasoning_plan = self._build_reasoning_plan(question, session_summary)
        reasoning_question = self._question_with_reasoning_plan(
            question,
            reasoning_plan,
        )
        print(f"\nRecherche hybride pour: {effective_query}")
        if reasoning_plan:
            print(f"   [reasoning plan] {len(reasoning_plan)} etape(s) internes")

        first = self._execute_pipeline(
            question=reasoning_question,
            retrieval_query=effective_query,
            top_k=top_k,
            retry=False,
            session_context=active_session_context,
            original_question=original_question,
            reasoning_plan=reasoning_plan,
        )
        selected = first
        retry_count = 0
        for attempt in range(RAG_MAX_RETRY_COUNT):
            current_verification = selected["verification"]
            retry_reason = (
                current_verification.get("reason")
                or "weak retrieval or unsupported answer"
            )
            should_retry = (
                current_verification.get("needs_retry")
                or current_verification.get("confidence", 0.0)
                < RAG_VERIFIER_MIN_CONFIDENCE
                or not selected["reranked_results"]
                or selected["retrieval_confidence"] < ABSTENTION_THRESHOLD
            )
            if not should_retry:
                print("   [answer verification] Retry not required")
                break
            print(
                "   [answer verification] Retry triggered "
                f"({attempt + 1}/{RAG_MAX_RETRY_COUNT}): {retry_reason}"
            )
            retry_result = self._execute_pipeline(
                question=reasoning_question,
                retrieval_query=effective_query,
                top_k=top_k,
                retry=True,
                session_context=active_session_context,
                original_question=original_question,
                reasoning_plan=reasoning_plan,
            )
            retry_count += 1
            retry_verification = retry_result["verification"]
            if (
                retry_verification.get("answer_supported")
                and (
                    not current_verification.get("answer_supported")
                    or retry_verification.get("confidence", 0.0)
                    >= current_verification.get("confidence", 0.0)
                )
            ):
                selected = retry_result
            elif (
                retry_verification.get("confidence", 0.0)
                > current_verification.get("confidence", 0.0)
            ):
                selected = retry_result
        retried = retry_count > 0

        verification = selected["verification"]
        generated_answer = selected["generated_answer"]
        if not verification.get("answer_supported", False):
            fallback_answer = generate_answer_from_results(
                original_question or question,
                selected.get("compressed_results") or selected.get("reranked_results") or [],
                embedding_model=self.embedding_model,
                embedding_model_name=self.embedding_model_name,
            )
            if fallback_answer and not is_not_found_answer(fallback_answer):
                fallback_verification = verify_answer(
                    original_question or question,
                    fallback_answer,
                    selected.get("compressed_results") or selected.get("reranked_results") or [],
                    embedding_model=self.embedding_model,
                    embedding_model_name=self.embedding_model_name,
                )
                if (
                    fallback_verification.get("answer_supported", False)
                    or fallback_verification.get("confidence", 0.0)
                    > verification.get("confidence", 0.0)
                ):
                    generated_answer = fallback_answer
                    verification = fallback_verification

        abstained = (
            not verification.get("answer_supported", False)
            or verification.get("confidence", 0.0) < RAG_MIN_ANSWER_CONFIDENCE
        )
        if abstained:
            generated_answer = "Information non trouvée dans la documentation locale fournie."

        reranked_results = selected["reranked_results"]
        compressed_results = selected["compressed_results"]
        best_result = compressed_results[:top_k] if compressed_results else []
        sources = source_references(compressed_results, max_sources=RAG_FINAL_TOP_K)
        answer = answer_with_sources(generated_answer, compressed_results)
        answer_context = build_context_from_results(
            compressed_results,
            max_chunks=max(5, top_k),
        )

        context_parts = []
        for result in best_result:
            method = f"[{result.get('method', 'unknown').upper()}]"
            score = f"{float(result.get('hybrid_score', 0.0) or 0.0):.2f}"
            source = result.get("source", "inconnu")
            title = result.get("title", source)
            text_preview = str(
                result.get("compressed_text") or result.get("text") or ""
            )[:400].replace("\n", " ")
            context_parts.append(f"{method} {source} ({score})\n{title}\n{text_preview}")
        context = "\n".join(context_parts) if context_parts else "Aucune information trouvee."

        print(
            "   Final sources: "
            + (", ".join(source["filename"] for source in sources) or "aucune")
        )

        features_used = {
            "query_rewriting": RAG_ENABLE_QUERY_REWRITING,
            "self_query": RAG_ENABLE_SELF_QUERY,
            "bm25": bool(
                self.lexical_retriever
                and self.lexical_retriever.enabled
                and self.lexical_retriever.backend == "bm25"
            ),
            "tfidf": bool(
                self.lexical_retriever
                and self.lexical_retriever.enabled
                and self.lexical_retriever.backend == "tfidf"
            ),
            "lexical": bool(self.lexical_retriever and self.lexical_retriever.enabled),
            "mmr": RAG_ENABLE_MMR,
            "cross_encoder": RAG_ENABLE_CROSS_ENCODER_RERANKER,
            "compression": RAG_ENABLE_CONTEXT_COMPRESSION,
            "parent_child_retrieval": RAG_ENABLE_PARENT_CHILD_RETRIEVAL,
            "hyde": bool(selected["transformation"].get("hyde_text")),
            "answer_verifier": RAG_ENABLE_ANSWER_VERIFIER,
            "chat_memory": bool(session_id and RAG_ENABLE_CHAT_MEMORY),
            "reasoning_plan": bool(reasoning_plan),
        }

        response_object = {
            "question": original_question,
            "standalone_question": question,
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "history_used": history_used,
            "retrieved_contexts": compressed_results,
            "query_variants": selected["transformation"].get("query_variants", [question]),
            "metadata": {
                "query_metadata": selected["transformation"].get("query_metadata", {}),
                "retrieval_counts": selected["retrieval_counts"],
                "features_used": features_used,
                "history_turns": len(history),
                "session_summary": session_summary,
                "session_recent_sources": recent_sources,
                "reasoning_plan": reasoning_plan,
                "backends": {
                    "lexical": (
                        self.lexical_retriever.backend
                        if self.lexical_retriever else "disabled"
                    ),
                    "cross_encoder": self.cross_encoder.backend,
                },
            },
            "graph_results_count": len(selected["graph_results"]),
            "vector_results_count": len(selected["vector_results"]),
            "lexical_results_count": len(selected["lexical_results"]),
            "merged_results": best_result,
            "retrieval_results": reranked_results[:max(5, top_k)],
            "retrieval_candidates": selected["parent_results"][:max(10, RAG_FINAL_TOP_K)],
            "child_retrieval_results": selected["child_results"][:max(10, top_k)],
            "compressed_results": compressed_results,
            "context": context,
            "answer_context": answer_context,
            "reasoning_plan": reasoning_plan,
            "query_transformation": selected["transformation"],
            "extractive_answer": selected.get("extractive_answer", ""),
            "generated_answer": generated_answer,
            "response": answer or "Reponse hybride trouvee",
            "calibrated_confidence": selected["retrieval_confidence"],
            "answer_verification": verification,
            "verification": verification,
            "answer_supported": verification.get("answer_supported", False),
            "verification_confidence": verification.get("confidence", 0.0),
            "needs_retry": verification.get("needs_retry", False),
            "retried": retried,
            "retry_count": retry_count,
            "abstained": abstained,
        }
        if session_id and RAG_ENABLE_CHAT_MEMORY and memory is not None:
            memory.add_turn(
                session_id,
                original_question,
                answer,
                sources=sources,
                metadata={
                    "standalone_question": question,
                    "history_used": history_used,
                    "verification": verification,
                    "session_summary": session_summary,
                    "reasoning_plan": reasoning_plan,
                },
            )
        return response_object


def main():
    """Interface interactive."""
    print("🤖 SYSTÈME RAG HYBRIDE (Graph RAG + Vector RAG)")
    print("=" * 60)

    sessions = SessionManager(
        file_path=SESSIONS_FILE,
        db_path=SESSIONS_DB_FILE,
        use_db=True,
        max_history_turns=RAG_CHAT_HISTORY_TURNS,
    )
    active_session = choose_cli_session(sessions)
    rag = HybridRAG(session_manager=sessions)
    print(f"🗂️ Session active: {active_session}")

    if not rag.graph and not rag.faiss_index:
        print("❌ Erreur: Exécutez d'abord python ingest_docs.py")
        return

    while True:
        question = input("\n💬 Posez votre question (ou 'exit'): ").strip()

        if question.lower() in ['exit', 'quit', 'q']:
            break

        if not question:
            continue

        command = question.lower()
        if command == "session":
            print(f"🗂️ Session active: {sessions.current_session()}")
            continue
        if command == "sessions":
            print("\n🗂️ Sessions:")
            for sid, _info in sessions.list_sessions():
                marker = " (active)" if sid == sessions.current_session() else ""
                print(f"  • {sid}{marker}")
            continue
        if command.startswith("switch "):
            sid = question[7:].strip()
            if sid:
                sessions.switch_session(sid)
                print(f"✅ Session changée: {sessions.current_session()}")
            else:
                print("❌ Usage: switch <id_session>")
            continue
        if command in ["history", "hist"]:
            history = sessions.get_history(limit=10)
            print(f"\n📜 Historique session '{sessions.current_session()}':")
            if not history:
                print("  Aucun échange.")
            else:
                for i, turn in enumerate(history, 1):
                    print(f"  {i}. Q: {turn['question']}")
            continue
        if command in ["summary", "resume"]:
            summary = sessions.refresh_session_memory(sessions.current_session())
            print(f"\nResume session '{sessions.current_session()}':")
            print(summary.get("summary") or "  Aucun resume disponible.")
            if summary.get("topics"):
                print("Sujets: " + ", ".join(summary["topics"][:8]))
            if summary.get("sources"):
                print("Sources recentes: " + ", ".join(summary["sources"][:5]))
            continue
        if command.startswith("search-history "):
            term = question[len("search-history "):].strip()
            if not term:
                print("Usage: search-history <mot>")
                continue
            matches = sessions.search_history(term, sessions.current_session())[:10]
            print(f"\nRecherche historique '{term}': {len(matches)} resultat(s)")
            for i, turn in enumerate(matches, 1):
                print(
                    f"  {i}. {turn.get('timestamp', '')} | "
                    f"Q: {turn.get('question', '')}"
                )
            continue
        if command.startswith("export-session"):
            parts = question.split(maxsplit=1)
            fmt = parts[1].strip().lower() if len(parts) == 2 else "markdown"
            if fmt not in {"json", "markdown", "txt"}:
                print("Usage: export-session [json|markdown|txt]")
                continue
            ext = "md" if fmt == "markdown" else fmt
            out_path = f"session_export_{sessions.current_session()}.{ext}"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(sessions.export_session_context(format=fmt))
            print(f"Session exportee: {out_path}")
            continue
        if command == "/clear":
            sessions.clear_session(sessions.current_session())
            print(f"Session '{sessions.current_session()}' cleared.")
            continue

        result = rag.query(
            question,
            top_k=3,
            session_id=sessions.current_session() or "default_cli_session",
        )

        # Afficher les résultats
        print("\n" + "=" * 60)
        print("✅ MEILLEURE RÉPONSE (Hybride)")
        print("=" * 60)
        
        if result['merged_results']:
            best = result['merged_results'][0]
            answer_text = result.get("generated_answer") or best.get("text", "")
            print(f"\n📚 Source: {best.get('source', best.get('document_id', 'inconnu'))}")
            print(f"📊 Méthode: {best['method'].upper()} | Score: {best['hybrid_score']:.2%}")
            print(f"\n💬 RÉPONSE:")
            print(answer_text)
            print(f"\n📖 Extrait source ({len(best.get('text', ''))} car.):")
            print((best.get("text") or "")[:350])
        else:
            print("\n❌ Aucune réponse trouvée.")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
