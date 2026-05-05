#!/usr/bin/env python3
"""
Script de requête pour un système RAG hybride (Graph RAG + Vector RAG)
Combine recherche par graphe (entités) et recherche vectorielle (similarité sémantique)
"""

import os
import pickle
import json
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

# Configuration
GRAPH_PATH = "networkx_graph.pkl"
FAISS_INDEX_PATH = "faiss_index.pkl"
CHUNKS_METADATA_PATH = "chunks_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HybridRAG:
    """Système RAG hybride combinant Graph RAG et Vector RAG"""
    
    def __init__(self, graph_path: str = GRAPH_PATH, 
                 faiss_path: str = FAISS_INDEX_PATH,
                 metadata_path: str = CHUNKS_METADATA_PATH):
        self.graph = None
        self.faiss_index = None
        self.metadata = []
        self.embedding_model = None
        
        self._load_graph(graph_path)
        self._load_vector_index(faiss_path, metadata_path)
        self._load_embedding_model()

    def _load_graph(self, graph_path):
        """Charge le graphe NetworkX."""
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"✅ Graphe chargé: {self.graph.number_of_nodes()} noeuds, {self.graph.number_of_edges()} relations")
        else:
            print(f"⚠️  Graphe non trouvé: {graph_path}")

    def _load_vector_index(self, faiss_path, metadata_path):
        """Charge l'index FAISS et les métadonnées."""
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            with open(faiss_path, 'rb') as f:
                self.faiss_index = pickle.load(f)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✅ Index FAISS chargé: {len(self.metadata)} chunks indexés")
        else:
            print(f"⚠️  Index vectoriel non trouvé")

    def _load_embedding_model(self):
        """Charge le modèle d'embeddings."""
        print(f"📦 Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Modèle chargé")

    # ============= RETRIEVAL PAR GRAPHE =============
    def search_graph(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche par graphe (entités) - optimisée."""
        if not self.graph:
            return []

        results = []
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]  # Mots > 2 caractères
        
        # Chercher les entités qui matchent la requête
        entities_found = []
        for node in self.graph.nodes():
            if isinstance(node, str) and len(node) > 2 and not node.startswith("Document_"):
                node_lower = node.lower()
                # Meilleur matching pour les entités
                score = sum(1 for word in query_words if word in node_lower) / max(1, len(query_words))
                if score > 0:
                    entities_found.append((node, score * 0.9))  # Score: 0-0.9
        
        # Trier par score
        entities_found.sort(key=lambda x: x[1], reverse=True)
        
        # Pour chaque entité, récupérer les documents associés
        for entity, entity_score in entities_found[:top_k]:
            if entity in self.graph:
                neighbors = list(self.graph.neighbors(entity))
                docs = [n for n in neighbors if n.startswith("Document_")]
                
                for doc_node in docs:
                    if doc_node in self.graph:
                        doc_data = self.graph.nodes[doc_node]
                        results.append({
                            'entity': entity,
                            'document_id': doc_node,
                            'text': doc_data.get('text', ''),
                            'score': entity_score,  # Score basé sur le matching
                            'method': 'graph'
                        })
        
        return results[:top_k]

    # ============= RETRIEVAL VECTORIEL =============
    def search_vector(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recherche vectorielle (similarité sémantique)."""
        if not self.faiss_index or not self.embedding_model:
            return []
        
        # Générer l'embedding de la requête
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Rechercher dans FAISS (Inner Product = similarité cosinus avec embeddings normalisés)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                score = float(distances[0][i])  # Score de similarité cosinus [0, 1]
                chunk_meta = self.metadata[idx]
                results.append({
                    'chunk_id': idx,
                    'text': chunk_meta['text'],
                    'source': chunk_meta['source'],
                    'score': score,
                    'method': 'vector'
                })
        
        return results

    # ============= FUSION DES RÉSULTATS =============
    def _merge_results(self, graph_results: List[Dict], 
                      vector_results: List[Dict]) -> List[Dict]:
        """Fusionne et déduplique les résultats des deux méthodes avec pondération optimale."""
        seen_texts = {}
        merged = []
        
        # Ajouter les résultats du graphe (ajuster la pondération)
        for result in graph_results:
            text_key = result['text'][:100]
            if text_key not in seen_texts:
                # Augmenter le poids du graphe (0.75 au lieu de 0.7)
                result['hybrid_score'] = min(0.95, result['score'] * 0.75)
                merged.append(result)
                seen_texts[text_key] = result
        
        # Ajouter les résultats vectoriels (nouveaux)
        for result in vector_results:
            text_key = result['text'][:100]
            if text_key not in seen_texts:
                # Augmenter le poids vectoriel (0.85 au lieu de 0.8)
                result['hybrid_score'] = min(0.95, result['score'] * 0.85)
                merged.append(result)
                seen_texts[text_key] = result
            else:
                # Si le texte existe déjà, combiner les scores
                existing = seen_texts[text_key]
                combined_score = (existing.get('hybrid_score', 0.5) + result['score'] * 0.85) / 2
                existing['hybrid_score'] = min(0.95, combined_score)
                existing['method'] = 'hybrid'
        
        # Trier par score hybride décroissant
        merged.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return merged

    # ============= INTERFACE PRINCIPALE =============
    def query(self, question: str, top_k: int = 1) -> Dict:
        """Traite une question avec retrieval hybride optimisé."""
        print(f"\n🔍 Recherche hybride pour: {question}")
        
        # Augmenter les recherches internes pour avoir plus de candidats
        internal_k = max(3, top_k * 2)
        
        # 1. Recherche par graphe
        graph_results = self.search_graph(question, top_k=internal_k)
        print(f"   📊 Graphe: {len(graph_results)} résultat(s)")
        
        # 2. Recherche vectorielle
        vector_results = self.search_vector(question, top_k=internal_k)
        print(f"   🔢 Vectoriel: {len(vector_results)} résultat(s)")
        
        # 3. Fusion des résultats
        merged_results = self._merge_results(graph_results, vector_results)
        
        # 4. Construire le contexte (seulement le meilleur résultat)
        context_parts = []
        best_result = merged_results[:top_k] if merged_results else []
        
        for result in best_result:
            method = f"[{result['method'].upper()}]"
            score = f"{result['hybrid_score']:.2f}"
            text_preview = result['text'][:400].replace('\n', ' ')
            context_parts.append(f"{method} ({score})\n{text_preview}")
        
        context = "\n".join(context_parts) if context_parts else "Aucune information trouvée."
        
        return {
            'question': question,
            'graph_results_count': len(graph_results),
            'vector_results_count': len(vector_results),
            'merged_results': best_result,
            'context': context,
            'response': f"Réponse hybride trouvée"
        }


def main():
    """Interface interactive."""
    print("🤖 SYSTÈME RAG HYBRIDE (Graph RAG + Vector RAG)")
    print("=" * 60)

    rag = HybridRAG()

    if not rag.graph and not rag.faiss_index:
        print("❌ Erreur: Exécutez d'abord python ingest_docs.py")
        return

    while True:
        question = input("\n💬 Posez votre question (ou 'exit'): ").strip()

        if question.lower() in ['exit', 'quit', 'q']:
            break

        if not question:
            continue

        # Traiter la question (top_k=1 pour une seule réponse)
        result = rag.query(question, top_k=1)

        # Afficher les résultats
        print("\n" + "=" * 60)
        print("✅ MEILLEURE RÉPONSE (Hybride)")
        print("=" * 60)
        
        if result['merged_results']:
            best = result['merged_results'][0]
            print(f"\n📚 Source: {best.get('source', best.get('document_id', 'inconnu'))}")
            print(f"📊 Méthode: {best['method'].upper()} | Score: {best['hybrid_score']:.2%}")
            print(f"\n📖 CONTENU:")
            print(best['text'])
        else:
            print("\n❌ Aucune réponse trouvée.")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()