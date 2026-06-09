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
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from session_manager import SessionManager
from rag_config import (
    EMBEDDING_MODEL,
    is_e5_model,
    E5_QUERY_PREFIX,
    GRAPH_PATH,
    GRAPH_SOURCES_PATH,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
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
)
from rag_canonical import parse_doc_about_topic
from rag_answer import generate_answer_from_results, build_context_from_results
from rag_vector import encode_query, encode_passages, cosine_scores, top_k_indices

SESSIONS_FILE = "hybrid_rag_sessions.json"


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
                 cot_enabled: bool = True):
        self.graph = None
        self.faiss_index = None
        self.metadata = []
        self.embedding_model = None
        self.embedding_model_name = EMBEDDING_MODEL
        self.cot_enabled = cot_enabled
        self.calibrator = ConfidenceCalibrator()
        
        self._entity_index = {}
        self._entity_nodes: List[str] = []
        self._entity_embeddings = np.zeros((0, 1), dtype="float32")
        self._load_graph(graph_path)
        self._load_vector_index(faiss_path, metadata_path)
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

    def _load_vector_index(self, faiss_path, metadata_path):
        """Charge l'index FAISS et les métadonnées."""
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            with open(faiss_path, 'rb') as f:
                self.faiss_index = pickle.load(f)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"Index FAISS chargé: {len(self.metadata)} chunks indexés (dim={self.faiss_index.d})")
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

    def _load_embedding_model(self):
        """Charge le modèle d'embeddings aligné sur l'index."""
        model_name = self.embedding_model_name
        print(f"Chargement du modèle d'embeddings: {model_name}...")
        try:
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
        if not self.graph or not self.embedding_model:
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
        if self._entity_embeddings.size == 0 or not self.embedding_model:
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
        if self.embedding_model:
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
            lex = self._compute_overlap_score(query_norm, self._normalize_text(text))
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
            file_boost = 0.0
            q_terms = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 3]
            src_tokens = self._source_tokens(source)
            if q_terms and src_tokens:
                file_boost = SOURCE_MATCH_BOOST * (
                    sum(1 for t in q_terms if t in src_tokens) / len(q_terms)
                )
            result["rerank_score"] = (
                RERANK_LEXICAL_WEIGHT * min(1.0, lex + 0.15 * src_boost)
                + RERANK_VECTOR_WEIGHT * semantic
                + RERANK_GRAPH_WEIGHT * graph_weight_score
                + RERANK_EMBEDDING_WEIGHT * embed
                + 0.20 * src_embed
                + topic_boost
                + file_boost
            )
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        for r in results:
            r["hybrid_score"] = r.get("rerank_score", r.get("hybrid_score", 0))
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
                            'score': entity_score,
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
                    'document_id': chunk_meta.get('document_id'),
                    'text': chunk_text,
                    'source': source,
                    'score': score,
                    'method': 'vector'
                })
        
        return results

    # ============= FUSION DES RÉSULTATS =============
    def _merge_results(self, graph_results: List[Dict], 
                      vector_results: List[Dict]) -> List[Dict]:
        """Fusionne et déduplique les résultats des deux méthodes avec pondération optimale."""
        seen = {}
        merged = []

        for result in graph_results:
            result_key = self._document_key(result)
            candidate = result.copy()
            candidate['hybrid_score'] = min(0.95, candidate['score'] * 1.05)
            if result_key not in seen:
                merged.append(candidate)
                seen[result_key] = candidate
            else:
                existing = seen[result_key]
                existing['hybrid_score'] = max(existing['hybrid_score'], candidate['hybrid_score'])
                existing['method'] = 'graph'

        for result in vector_results:
            result_key = self._document_key(result)
            candidate = result.copy()
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

        merged.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return merged

    def _build_cot(self, user_question: str, retrieval_question: str,
                   graph_results: List[Dict], vector_results: List[Dict],
                   merged_results: List[Dict], top_k: int) -> List[str]:
        """Construit une chain-of-thought explicable (niveau produit)."""
        steps = []
        if retrieval_question != user_question:
            steps.append("La question a ete enrichie avec le contexte recent de la session.")
        else:
            steps.append("La question utilisateur a ete utilisee telle quelle.")

        steps.append(
            f"Recherche hybride effectuee: {len(graph_results)} candidats graphe et {len(vector_results)} candidats vectoriels."
        )

        if merged_results:
            best = merged_results[0]
            steps.append(
                f"Le meilleur extrait provient de '{best.get('source', 'inconnu')}' via la methode '{best['method']}' (score {best['hybrid_score']:.2f})."
            )
            if len(graph_results) == 0:
                steps.append("Aucune correspondance par graphe n'a ete trouvee; la recherche vectorielle a fourni les candidats.")
            else:
                steps.append("La fusion a combine les resultats graphe et vectoriel pour identifier le meilleur passage.")
            steps.append(f"La reponse finale est basee sur les {min(top_k, len(merged_results))} meilleur(s) extrait(s).")
        else:
            steps.append("Aucun resultat pertinent n'a ete trouve apres fusion.")

        return steps

    # ============= INTERFACE PRINCIPALE =============
    def query(self, question: str, top_k: int = 1, retrieval_question: str = None) -> Dict:
        """Traite une question avec retrieval hybride optimisé."""
        effective_query = retrieval_question or question
        print(f"\nRecherche hybride pour: {effective_query}")
        
        pool_k = max(RETRIEVAL_POOL_SIZE, top_k * 5)
        
        # 1. Recherche par graphe
        graph_results = self.search_graph(effective_query, top_k=pool_k)
        print(f"   Graphe: {len(graph_results)} résultat(s)")
        
        # 2. Recherche vectorielle
        vector_results = self.search_vector(effective_query, top_k=pool_k)
        print(f"   Vectoriel: {len(vector_results)} résultat(s)")
        
        # 3. Fusion + re-ranking lexical/sémantique
        merged_results = self._merge_results(graph_results, vector_results)
        merged_results = self._rerank_results(effective_query, merged_results)
        
        # B. Confidence Calibration & Abstention check
        top_score = merged_results[0].get("rerank_score", 0.0) if merged_results else 0.0
        confidence = self.calibrator.calibrate(top_score)
        print(f"   Score top rerank: {top_score:.4f} | Confidence: {confidence:.2%}")
        
        abstained = False
        
        # 4. Meilleurs extraits pour réponse
        context_parts = []
        best_result = merged_results[:top_k] if merged_results else []
        answer_context = build_context_from_results(merged_results, max_chunks=max(5, top_k))
        
        if confidence < ABSTENTION_THRESHOLD:
            abstained = True
            generated_answer = "Désolé, je n'ai pas trouvé d'information pertinente dans la documentation."
        else:
            generated_answer = generate_answer_from_results(
                question,
                merged_results,
                embedding_model=self.embedding_model,
                embedding_model_name=self.embedding_model_name,
            )
        
        for result in best_result:
            method = f"[{result['method'].upper()}]"
            score = f"{result['hybrid_score']:.2f}"
            source = result.get('source', 'inconnu')
            title = result.get('title', source)
            text_preview = result['text'][:400].replace('\n', ' ')
            context_parts.append(f"{method} {source} ({score})\n{title}\n{text_preview}")
        
        context = "\n".join(context_parts) if context_parts else "Aucune information trouvée."
        
        cot_steps = self._build_cot(
            user_question=question,
            retrieval_question=effective_query,
            graph_results=graph_results,
            vector_results=vector_results,
            merged_results=merged_results,
            top_k=top_k,
        ) if self.cot_enabled else []
        
        if abstained:
            cot_steps.append(f"Abstention: La confiance ({confidence:.2%}) est inferieure au seuil ({ABSTENTION_THRESHOLD:.2%}).")

        return {
            'question': question,
            'graph_results_count': len(graph_results),
            'vector_results_count': len(vector_results),
            'merged_results': best_result,
            'context': context,
            'cot_steps': cot_steps,
            'generated_answer': generated_answer,
            'response': generated_answer or "Réponse hybride trouvée",
            'calibrated_confidence': confidence,
            'abstained': abstained,
        }


def main():
    """Interface interactive."""
    print("🤖 SYSTÈME RAG HYBRIDE (Graph RAG + Vector RAG)")
    print("=" * 60)

    enable_cot = True
    if '--no-cot' in sys.argv or '--disable-cot' in sys.argv:
        enable_cot = False

    rag = HybridRAG(cot_enabled=enable_cot)
    sessions = SessionManager(file_path=SESSIONS_FILE)
    print(f"🗂️ Session active: {sessions.current_session()}")
    if not enable_cot:
        print("⚠️ Chain-of-thought désactivée pour cette session.")

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
            for sid in sessions.list_sessions():
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

        recent_questions = sessions.get_recent_questions(limit=2)
        retrieval_question = question
        if recent_questions and len(question.split()) <= 6:
            retrieval_question = " ; ".join(recent_questions + [question])

        result = rag.query(question, top_k=3, retrieval_question=retrieval_question)

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
            if enable_cot:
                print("\n🧠 CHAIN OF THOUGHT (resume):")
                for i, step in enumerate(result.get("cot_steps", []), 1):
                    print(f"  {i}. {step}")
            else:
                print("\n🧠 Chain-of-thought désactivée.")
            sessions.add_turn(
                question,
                answer_text,
                metadata={"cot_steps": result.get("cot_steps", [])},
            )
        else:
            print("\n❌ Aucune réponse trouvée.")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
