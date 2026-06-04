#!/usr/bin/env python3
"""
Script pour enrichir le graphe existant avec des alias et synonymes.
Améliore le matching d'entités dans la recherche graphe.
"""

import pickle
import networkx as nx
import json

GRAPH_PATH = "networkx_graph.pkl"
ENRICHED_GRAPH_PATH = "networkx_graph_enriched.pkl"

# Dictionnaire de synonymes et alias pour chaque entité
ENTITY_ALIASES = {
    "Harmony": [
        "harmony",
        "Harmony ERP",
        "Harmony logiciel",
        "application harmony",
        "HRM",
    ],
    "Installation": [
        "installation",
        "installer",
        "déploiement",
        "mise en place",
        "configuration initiale",
        "setup",
    ],
    "TSE": [
        "tse",
        "terminal server",
        "environnement serveur",
        "serveur terminal",
        "architecture tse",
        "clients légers",
    ],
    "Imprimante": [
        "imprimante",
        "impression",
        "imprimantes",
        "spouleur",
        "pilote impression",
        "queue impression",
    ],
    "Chemin Harmony": [
        "chemin harmony",
        "chemins harmony",
        "chemin implicite",
        "chemin réseau",
        "mapping",
        "partage",
    ],
    "Fichier": [
        "fichier",
        "fichiers",
        "document",
        "fichier de configuration",
        "config",
        "données",
    ],
    "Client léger": [
        "client léger",
        "client leger",
        "thin client",
        "clients légers",
        "web client",
    ],
    "Serveur Xlan": [
        "serveur xlan",
        "serveur xlan",
        "xlan",
        "serveur application",
        "serveur données",
    ],
    "Configuration": [
        "configuration",
        "configurer",
        "paramètres",
        "paramétrage",
        "settings",
        "setup",
    ],
    "Diagnostic": [
        "diagnostic",
        "diagnostiquer",
        "dépannage",
        "troubleshooting",
        "problème",
        "erreur",
    ],
    "Accès": [
        "accès",
        "accéder",
        "accès fichier",
        "droit d'accès",
        "permission",
        "authentification",
    ],
    "Aides": [
        "aides",
        "aide",
        "fenêtre aide",
        "documentation",
        "help",
        "assistance",
    ],
    "ODBC": [
        "odbc",
        "ODBC",
        "base de données",
        "connexion base",
        "driver odbc",
    ],
    "Oracle": [
        "oracle",
        "Oracle",
        "base oracle",
        "oracle base",
    ],
    "MSSQL": [
        "mssql",
        "MSSQL",
        "sql server",
        "ms sql",
        "sqlserver",
    ],
}

def enrich_graph_with_aliases():
    """Charge le graphe et ajoute des alias pour chaque entité."""
    
    # Charger le graphe existant
    print(f"Chargement du graphe depuis {GRAPH_PATH}...")
    with open(GRAPH_PATH, 'rb') as f:
        graph = pickle.load(f)
    
    print(f"Graphe chargé: {graph.number_of_nodes()} noeuds, {graph.number_of_edges()} aretes")
    
    # Identifier les entités
    entity_nodes = [n for n in graph.nodes() if isinstance(n, str) and not n.startswith("Document_")]
    print(f"Entités trouvées: {len(entity_nodes)}")
    
    # Enrichir les nœuds entités avec des alias
    enriched_count = 0
    for entity in entity_nodes:
        if entity in ENTITY_ALIASES:
            aliases = ENTITY_ALIASES[entity]
            node_data = graph.nodes[entity] if entity in graph else {}
            if isinstance(node_data, dict):
                node_data['aliases'] = aliases
                node_data['label'] = entity
                graph.nodes[entity].update(node_data)
                enriched_count += 1
                print(f"  Enrichi: {entity} -> {len(aliases)} alias")
        else:
            # Ajouter au moins une version minuscule comme alias
            node_data = graph.nodes[entity] if entity in graph else {}
            if isinstance(node_data, dict):
                node_data['aliases'] = [entity.lower()]
                node_data['label'] = entity
                graph.nodes[entity].update(node_data)
    
    print(f"\nEntités enrichies: {enriched_count}")
    
    # Ajouter aussi les entités manquantes comme nœuds isolés (utiles pour le matching)
    for entity_name in ENTITY_ALIASES.keys():
        if entity_name not in graph:
            graph.add_node(entity_name, aliases=ENTITY_ALIASES[entity_name], label=entity_name)
            print(f"  Ajout entité isolée: {entity_name}")
    
    # Sauvegarder le graphe enrichi
    print(f"\nSauvegarde du graphe enrichi dans {ENRICHED_GRAPH_PATH}...")
    with open(ENRICHED_GRAPH_PATH, 'wb') as f:
        pickle.dump(graph, f)
    
    # Statistiques finales
    print(f"Graphe enrichi: {graph.number_of_nodes()} noeuds, {graph.number_of_edges()} aretes")
    print("Enrichissement complète!")
    
    return graph

if __name__ == "__main__":
    enrich_graph_with_aliases()
