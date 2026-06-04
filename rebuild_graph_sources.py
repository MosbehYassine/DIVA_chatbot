#!/usr/bin/env python3
"""
Script pour reconstruire le graphe avec les métadonnées source correctes.
Améliore l'indexation du graphe pour la recherche.
"""

import os
import json
import pickle
import networkx as nx
from pathlib import Path

GRAPH_PATH = "networkx_graph.pkl"
CHUNKS_METADATA_PATH = "chunks_metadata.json"
ENRICHED_GRAPH_PATH = "networkx_graph_enriched.pkl"
GRAPH_WITH_SOURCES_PATH = "networkx_graph_sources.pkl"

def rebuild_graph_with_sources():
    """Reconstruit le graphe avec les sources corrigées depuis les métadonnées."""
    
    # Charger les métadonnées des chunks
    print(f"Chargement des métadonnées depuis {CHUNKS_METADATA_PATH}...")
    with open(CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Métadonnées chargées: {len(metadata)} chunks")
    
    # Charger le graphe enrichi
    print(f"Chargement du graphe enrichi...")
    with open(ENRICHED_GRAPH_PATH, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Graphe chargé: {graph.number_of_nodes()} noeuds")
    
    # Mettre à jour les nœuds documents avec les sources
    doc_nodes = [n for n in graph.nodes() if isinstance(n, str) and n.startswith("Document_")]
    
    updated_count = 0
    for doc_node in doc_nodes:
        # Extraire l'index du Document_X
        try:
            doc_id = int(doc_node.replace("Document_", ""))
            if doc_id < len(metadata):
                chunk_meta = metadata[doc_id]
                source = chunk_meta.get('source', f'chunk_{doc_id}')
                
                # Mettre à jour le nœud avec la source
                node_data = graph.nodes[doc_node]
                if isinstance(node_data, dict):
                    node_data['source'] = source
                    node_data['chunk_id'] = doc_id
                    graph.nodes[doc_node].update(node_data)
                    updated_count += 1
        except Exception as e:
            pass
    
    print(f"Documents enrichis avec sources: {updated_count}")
    
    # Sauvegarder le graphe avec sources
    print(f"Sauvegarde du graphe avec sources dans {GRAPH_WITH_SOURCES_PATH}...")
    with open(GRAPH_WITH_SOURCES_PATH, 'wb') as f:
        pickle.dump(graph, f)
    
    print(f"Graphe sauvegardé: {graph.number_of_nodes()} noeuds, {graph.number_of_edges()} aretes")
    
    return graph

if __name__ == "__main__":
    rebuild_graph_with_sources()
    print("\nReconstruction complete!")
