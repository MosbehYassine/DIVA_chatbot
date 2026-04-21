#!/usr/bin/env python3
"""
Script de requête pour le système GraphRAG
Utilise le graphe NetworkX créé par ingest_docs.py pour répondre aux questions.
"""

import os
import pickle
import networkx as nx
from typing import List, Dict
import json

# Configuration
GRAPH_PATH = "networkx_graph.pkl"

class GraphQuery:
    def __init__(self, graph_path: str = GRAPH_PATH):
        self.graph_path = graph_path
        self.graph = None
        self._load_graph()

    def _load_graph(self):
        """Charge le graphe NetworkX depuis le fichier pickle."""
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"✅ Graphe chargé: {self.graph.number_of_nodes()} noeuds, {self.graph.number_of_edges()} relations")
        else:
            print(f"❌ Fichier graphe non trouvé: {self.graph_path}")
            print("Exécutez d'abord: python ingest_docs.py")
            self.graph = None

    def search_entities(self, query: str, top_k: int = 5) -> List[str]:
        """Recherche d'entités pertinentes dans le graphe."""
        if not self.graph:
            return []

        query_lower = query.lower()
        entities = []

        # Recherche d'entités dont le nom contient des mots de la requête
        for node in self.graph.nodes():
            if isinstance(node, str) and len(node) > 2:
                node_lower = node.lower()
                # Vérifier si des mots de la requête sont dans le nom de l'entité
                query_words = query_lower.split()
                if any(word in node_lower for word in query_words):
                    entities.append(node)

        return entities[:top_k]

    def get_related_documents(self, entities: List[str], top_k: int = 3) -> List[Dict]:
        """Récupère les documents liés aux entités trouvées."""
        if not self.graph:
            return []

        related_docs = []

        for entity in entities:
            if entity in self.graph:
                # Trouver tous les documents connectés à cette entité
                neighbors = list(self.graph.neighbors(entity))
                docs = [n for n in neighbors if n.startswith("Document_")]

                for doc in docs:
                    if doc in self.graph:
                        doc_data = self.graph.nodes[doc]
                        text = doc_data.get('text', '')
                        related_docs.append({
                            'entity': entity,
                            'document': doc,
                            'text': text
                        })

        # Dédoublonner et limiter
        seen_docs = set()
        unique_docs = []
        for doc in related_docs:
            if doc['document'] not in seen_docs:
                unique_docs.append(doc)
                seen_docs.add(doc['document'])
                if len(unique_docs) >= top_k:
                    break

        return unique_docs

    def query(self, question: str) -> Dict:
        """Traite une question et retourne les résultats."""
        print(f"\n🔍 Recherche pour: {question}")

        # 1. Trouver les entités pertinentes
        entities = self.search_entities(question)
        print(f"📋 Entités trouvées: {entities}")

        # 2. Récupérer les documents liés
        documents = self.get_related_documents(entities)
        print(f"📄 Documents trouvés: {len(documents)}")

        # 3. Construire la réponse
        context_parts = []
        for doc in documents:
            context_parts.append(f"[{doc['entity']}] {doc['text'][:300]}...")

        context = "\n\n".join(context_parts) if context_parts else "Aucune information trouvée."

        return {
            'question': question,
            'entities_found': entities,
            'documents_found': len(documents),
            'context': context,
            'response': f"Réponse basée sur {len(documents)} documents trouvés dans le graphe."
        }

def main():
    """Interface interactive pour poser des questions."""
    print("🤖 SYSTÈME GRAPHRAG - REQUÊTES")
    print("=" * 50)

    query_system = GraphQuery()

    if not query_system.graph:
        return

    while True:
        question = input("\n💬 Posez votre question (ou 'exit'): ").strip()

        if question.lower() in ['exit', 'quit', 'q']:
            break

        if not question:
            continue

        # Traiter la question
        result = query_system.query(question)

        # Afficher les résultats
        print("\n" + "="*50)
        print("📋 ENTITÉS TROUVÉES:")
        for entity in result['entities_found']:
            print(f"  • {entity}")

        print(f"\n📄 DOCUMENTS ({result['documents_found']}):")
        if result['documents_found'] > 0:
            for i, doc in enumerate(result['documents_found'] if isinstance(result['documents_found'], list) else [], 1):
                print(f"  {i}. {doc}")
        else:
            print("  Aucun document trouvé")

        print("\n🤖 RÉPONSE:")
        print(result['response'])

        print("\n📖 CONTEXTE:")
        print(result['context'][:500] + "..." if len(result['context']) > 500 else result['context'])

if __name__ == "__main__":
    main()