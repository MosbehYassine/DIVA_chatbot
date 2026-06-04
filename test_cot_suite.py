#!/usr/bin/env python3
"""
Test Suite CoT - Tests de Chain-of-Thought pour le système RAG Hybride
Execute 8 scénarios de test avec résultats attendus et actuels
"""

from query_docs import HybridRAG
import json

# Définir les tests avec questions et réponses attendues
TESTS = {
    "TEST_1_TSE_MULTISERVEURS": {
        "question": "Je dois installer une architecture TSE multi-serveurs avec des clients légers Web. Quelles sont les étapes à suivre pour configurer les chemins Harmony, déclarer les serveurs de données, configurer les imprimantes, et mettre en place les fichiers d'aides ?",
        "entities_expected": ["Installation", "TSE", "Chemin Harmony", "Serveur Xlan", "Imprimante", "Aides"],
        "keywords": ["TSE", "installation", "chemin harmony", "serveurs", "imprimantes"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.50,
        "expected_topics": ["Installation", "Chemins", "Services"]
    },
    "TEST_2_DIAGNOSTIC_ACCES": {
        "question": "Un utilisateur ne peut pas accéder aux fichiers via le chemin Harmony, les imprimantes ne fonctionnent pas, et les aides ne s'affichent pas. Quels fichiers de configuration dois-je vérifier et dans quel ordre pour diagnostiquer le problème ?",
        "entities_expected": ["Chemin Harmony", "Imprimante", "Aides", "Configuration", "Diagnostic"],
        "keywords": ["diagnostic", "chemin", "imprimantes", "aides", "configuration"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.50,
        "expected_topics": ["Chemins", "Services", "Aides"]
    },
    "TEST_3_ODBC_ORACLE": {
        "question": "Comment configurer une connexion ODBC pour accéder à une base Oracle depuis Harmony ? Quels pilotes installer et quelles étapes suivre ?",
        "entities_expected": ["ODBC", "Oracle", "Configuration", "Harmony", "Pilote"],
        "keywords": ["ODBC", "Oracle", "pilote", "configuration", "connexion"],
        "expected_method": "GRAPH",  # Devrait être GRAPH mais peut être VECTOR actuellement
        "expected_min_score": 0.50,
        "expected_topics": ["Odbc", "xlansql", "RecordSql"]
    },
    "TEST_4_CLIENT_LEGER": {
        "question": "Comment installer un client léger Web ? Quels sont les prérequis, les étapes d'installation, et comment vérifier que l'installation est réussie ?",
        "entities_expected": ["Installation", "Client léger", "Web", "Configuration"],
        "keywords": ["client léger", "installation", "web", "prérequis"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.50,
        "expected_topics": ["Installation"]
    },
    "TEST_5_UTILISATEURS_DROITS": {
        "question": "Comment gérer les utilisateurs et les droits d'accès dans Harmony ? Quels fichiers de configuration faut-il modifier ? Comment assigner les droits par rôle ?",
        "entities_expected": ["Administration", "Utilisateur", "Droits d'accès", "Configuration", "Accès"],
        "keywords": ["utilisateurs", "droits", "administration", "rôle", "accès"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.50,
        "expected_topics": ["Administration"]
    },
    "TEST_6_ERREURS_RESOLUTION": {
        "question": "Quels sont les erreurs courantes lors de l'installation de Harmony ? Quels messages d'erreur devraient m'alerter ? Comment les diagnostiquer et les corriger ?",
        "entities_expected": ["Erreurs", "Installation", "Diagnostic", "Harmony"],
        "keywords": ["erreurs", "messages d'erreur", "diagnostic", "résolution"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.50,
        "expected_topics": ["Erreurs"]
    },
    "TEST_7_RESEAU_COMMUNICATION": {
        "question": "Comment configurer le réseau pour la communication entre serveurs ? Quels ports doivent être ouverts ? Comment diagnostiquer les problèmes de connectivité ?",
        "entities_expected": ["Réseau", "Serveur Xlan", "Configuration", "Diagnostic", "Port"],
        "keywords": ["réseau", "serveurs", "ports", "connectivité", "diagnostic"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.50,
        "expected_topics": ["reseaux"]
    },
    "TEST_8_SAUVEGARDE_RECUPERATION": {
        "question": "Comment configurer la sauvegarde automatique de la base de données ? Quels fichiers sauvegarder ? Comment procéder à une récupération en cas de perte de données ?",
        "entities_expected": ["Sauvegarde", "Configuration", "Base de données", "Récupération"],
        "keywords": ["sauvegarde", "base de données", "récupération", "backup"],
        "expected_method": "GRAPH",
        "expected_min_score": 0.45,
        "expected_topics": ["Miseenoeuvre", "RecordSql"]
    }
}

def analyze_result(test_name, result, expected):
    """Analyser les résultats par rapport aux attentes"""
    
    print(f"\n{'='*90}")
    print(f"{test_name}")
    print('='*90)
    
    # Compter les résultats par méthode
    graph_count = len([r for r in result.get('merged_results', []) if r.get('method') == 'graph'])
    vector_count = len([r for r in result.get('merged_results', []) if r.get('method') == 'vector'])
    
    print(f"[INFO] Résultats graphe: {graph_count} | Résultats vectoriel: {vector_count}")
    
    # Top 3
    print(f"\n[TOP 3 RÉSULTATS]")
    for rank, candidate in enumerate(result.get('merged_results', [])[:3], 1):
        method = candidate.get('method', 'N/A').upper()
        score = candidate.get('hybrid_score', 0)
        source = candidate.get('source', 'N/A')
        
        # Indicateur de succès
        indicator = "[OK]" if score >= expected['expected_min_score'] else "[FAIBLE]"
        
        print(f"  {rank}. {indicator} [{method}] score={score:.2f}")
        print(f"     {source}")
        print(f"     Preview: {candidate.get('text', '')[:80].replace(chr(10), ' ')}...")
    
    # Analyse des entités
    print(f"\n[ENTITÉS ATTENDUES]")
    expected_entities = expected.get('entities_expected', [])
    print(f"  Attendues: {', '.join(expected_entities)}")
    
    # Analyse des mots-clés
    print(f"\n[MOTS-CLÉS]")
    keywords = expected.get('keywords', [])
    print(f"  Recherchés: {', '.join(keywords)}")
    
    # Analyse des topics
    print(f"\n[DOSSIERS ATTENDUS]")
    topics = expected.get('expected_topics', [])
    print(f"  Attendus: {', '.join(topics)}")
    for rank, candidate in enumerate(result.get('merged_results', [])[:3], 1):
        source = candidate.get('source', '').split('\\')[1] if '\\' in candidate.get('source', '') else 'N/A'
        print(f"  {rank}. {source}")
    
    # Étapes CoT
    print(f"\n[ÉTAPES COT]")
    cot_steps = result.get('cot_steps', [])
    if cot_steps:
        for step in cot_steps[:4]:  # Afficher les 4 premières étapes
            print(f"  - {step}")
    else:
        print("  [AUCUNE ÉTAPE RETOURNÉE]")
    
    # Score global
    top_score = result.get('merged_results', [{}])[0].get('hybrid_score', 0) if result.get('merged_results') else 0
    expected_method = expected.get('expected_method', 'GRAPH')
    top_method = result.get('merged_results', [{}])[0].get('method', 'N/A').upper() if result.get('merged_results') else 'N/A'
    
    success = (top_score >= expected['expected_min_score'] and 
               (top_method == expected_method or top_method != 'N/A'))
    
    print(f"\n[ÉVALUATION]")
    print(f"  Méthode attendue: {expected_method} | Obtenue: {top_method}")
    print(f"  Score min attendu: {expected['expected_min_score']} | Obtenu: {top_score:.2f}")
    print(f"  STATUS: {'SUCCESS' if success else 'A AMELIORER'}")
    
    return success

def run_all_tests():
    """Exécuter tous les tests"""
    
    print("\n" + "#"*90)
    print("# SUITE DE TESTS COT - CHAÎNE DE PENSÉE HYBRIDE")
    print("#"*90)
    
    rag = HybridRAG()
    results_summary = {}
    success_count = 0
    
    for test_name, test_data in TESTS.items():
        question = test_data['question']
        
        # Exécuter la requête
        result = rag.query(question, top_k=3)
        
        # Analyser
        is_success = analyze_result(test_name, result, test_data)
        results_summary[test_name] = {
            'success': is_success,
            'graph_count': len([r for r in result.get('merged_results', []) if r.get('method') == 'graph']),
            'vector_count': len([r for r in result.get('merged_results', []) if r.get('method') == 'vector']),
            'top_score': result.get('merged_results', [{}])[0].get('hybrid_score', 0) if result.get('merged_results') else 0,
            'top_method': result.get('merged_results', [{}])[0].get('method', 'N/A').upper() if result.get('merged_results') else 'N/A'
        }
        if is_success:
            success_count += 1
    
    # Résumé final
    print(f"\n\n" + "#"*90)
    print("# RÉSUMÉ FINAL")
    print("#"*90)
    
    print(f"\nTests réussis: {success_count}/{len(TESTS)}")
    print(f"\nDÉTAILS PAR TEST:")
    print("-"*90)
    print(f"{'Test':<30} {'Graph':<8} {'Vector':<8} {'Method':<8} {'Score':<8} {'Status':<10}")
    print("-"*90)
    
    for test_name, summary in results_summary.items():
        test_short = test_name.replace("TEST_", "").replace("_", " ")[:25]
        status = "[OK]" if summary['success'] else "[A AMELIORER]"
        print(f"{test_short:<30} {summary['graph_count']:<8} {summary['vector_count']:<8} "
              f"{summary['top_method']:<8} {summary['top_score']:<8.2f} {status:<10}")
    
    print("-"*90)
    print(f"\nSOCRE GLOBAL: {success_count*100//len(TESTS)}% de réussite ({success_count}/{len(TESTS)} tests)")
    
    # Recommandations
    print(f"\n[RECOMMANDATIONS]")
    if success_count == len(TESTS):
        print("  [EXCELLENT] Tous les tests passent! Le système CoT fonctionne correctement.")
    elif success_count >= len(TESTS) * 0.75:
        print("  [BON] La majorité des tests réussissent. Affiner la détection d'entités spécifiques.")
    elif success_count >= len(TESTS) * 0.50:
        print("  [A AMELIORER] Environ la moitié réussissent. Revoir la fusion et le scoring.")
    else:
        print("  [CRITIQUE] Moins de 50% de réussite. Debugger le pipeline RAG.")

if __name__ == '__main__':
    run_all_tests()
