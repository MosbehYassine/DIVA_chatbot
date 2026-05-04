#!/usr/bin/env python3
"""
Script de test du système RAG hybride
Teste le modèle avec les questions du fichier test_questions.json
"""

import json
import os
from datetime import datetime
from query_docs import HybridRAG
from typing import Dict, List

class RAGTester:
    def __init__(self, test_file: str = "test_questions.json"):
        self.test_file = test_file
        self.rag = HybridRAG()
        self.test_questions = []
        self.results = []
        self._load_tests()

    def _load_tests(self):
        """Charge les questions de test."""
        if os.path.exists(self.test_file):
            with open(self.test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.test_questions = data.get('test_questions', [])
            print(f"✅ {len(self.test_questions)} questions de test chargées")
        else:
            print(f"❌ Fichier de test non trouvé: {self.test_file}")

    def run_tests(self) -> List[Dict]:
        """Exécute tous les tests."""
        print("\n" + "=" * 80)
        print("🚀 DÉMARRAGE DES TESTS DU SYSTÈME RAG HYBRIDE")
        print("=" * 80)

        if not self.rag.graph and not self.rag.faiss_index:
            print("❌ Erreur: Les index ne sont pas chargés")
            print("   Exécutez d'abord: python ingest_docs.py")
            return []

        total = len(self.test_questions)
        for idx, test in enumerate(self.test_questions, 1):
            print(f"\n[{idx}/{total}] Test: {test['id']}")
            print(f"   Catégorie: {test['category']}")
            print(f"   Difficulté: {test['difficulty']}")
            print(f"   Question: {test['question']}")

            try:
                # Exécuter la requête
                result = self.rag.query(test['question'], top_k=1)
                
                # Récupérer la meilleure réponse
                best_response = ""
                if result['merged_results']:
                    best_response = result['merged_results'][0]['text']
                
                # Enregistrer le résultat
                test_result = {
                    'test_id': test['id'],
                    'question': test['question'],
                    'expected_answer': test['expected_answer'],
                    'retrieved_text': best_response,
                    'method': result['merged_results'][0]['method'] if result['merged_results'] else 'none',
                    'score': result['merged_results'][0]['hybrid_score'] if result['merged_results'] else 0,
                    'category': test['category'],
                    'difficulty': test['difficulty'],
                    'found': len(result['merged_results']) > 0
                }
                
                self.results.append(test_result)
                
                # Afficher le résultat
                if test_result['found']:
                    print(f"   ✅ TROUVÉ (Score: {test_result['score']:.2%})")
                    print(f"   📖 Réponse: {best_response[:150]}...")
                else:
                    print(f"   ❌ AUCUNE RÉPONSE TROUVÉE")

            except Exception as e:
                print(f"   ❌ ERREUR: {e}")
                self.results.append({
                    'test_id': test['id'],
                    'question': test['question'],
                    'expected_answer': test['expected_answer'],
                    'retrieved_text': '',
                    'method': 'error',
                    'score': 0,
                    'category': test['category'],
                    'difficulty': test['difficulty'],
                    'found': False,
                    'error': str(e)
                })

        return self.results

    def generate_report(self, output_file: str = "test_results.json") -> Dict:
        """Génère un rapport de test."""
        print("\n" + "=" * 80)
        print("📊 GÉNÉRATION DU RAPPORT")
        print("=" * 80)

        total = len(self.results)
        found = sum(1 for r in self.results if r['found'])
        avg_score = sum(r['score'] for r in self.results if r['found']) / found if found > 0 else 0

        # Statistiques par catégorie
        categories_stats = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories_stats:
                categories_stats[cat] = {'total': 0, 'found': 0, 'avg_score': 0}
            categories_stats[cat]['total'] += 1
            if result['found']:
                categories_stats[cat]['found'] += 1
                categories_stats[cat]['avg_score'] += result['score']
        
        # Normaliser les scores moyens
        for cat in categories_stats:
            if categories_stats[cat]['found'] > 0:
                categories_stats[cat]['avg_score'] /= categories_stats[cat]['found']

        # Statistiques par difficulté
        difficulty_stats = {}
        for result in self.results:
            diff = result['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'found': 0, 'avg_score': 0}
            difficulty_stats[diff]['total'] += 1
            if result['found']:
                difficulty_stats[diff]['found'] += 1
                difficulty_stats[diff]['avg_score'] += result['score']
        
        # Normaliser les scores moyens
        for diff in difficulty_stats:
            if difficulty_stats[diff]['found'] > 0:
                difficulty_stats[diff]['avg_score'] /= difficulty_stats[diff]['found']

        # Créer le rapport
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total,
                'found': found,
                'not_found': total - found,
                'success_rate': f"{(found / total * 100):.1f}%" if total > 0 else "0%",
                'average_score': f"{avg_score:.2%}"
            },
            'by_category': categories_stats,
            'by_difficulty': difficulty_stats,
            'detailed_results': self.results
        }

        # Sauvegarder le rapport
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Rapport sauvegardé: {output_file}")

        # Afficher le résumé
        print("\n" + "=" * 80)
        print("📈 RÉSUMÉ DES RÉSULTATS")
        print("=" * 80)
        print(f"✅ Réponses trouvées: {found}/{total} ({(found/total*100):.1f}%)")
        print(f"❌ Aucune réponse: {total - found}/{total} ({((total-found)/total*100):.1f}%)")
        print(f"📊 Score moyen: {avg_score:.2%}")

        print("\n📋 PAR CATÉGORIE:")
        for cat, stats in categories_stats.items():
            success_rate = (stats['found'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   • {cat}: {stats['found']}/{stats['total']} ({success_rate:.0f}%) - Score: {stats['avg_score']:.2%}")

        print("\n📊 PAR DIFFICULTÉ:")
        for diff, stats in difficulty_stats.items():
            success_rate = (stats['found'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   • {diff}: {stats['found']}/{stats['total']} ({success_rate:.0f}%) - Score: {stats['avg_score']:.2%}")

        print("\n" + "=" * 80)

        return report


def main():
    tester = RAGTester()
    
    if not tester.test_questions:
        print("❌ Aucune question de test chargée")
        return

    # Exécuter les tests
    tester.run_tests()
    
    # Générer le rapport
    report = tester.generate_report("test_results_hybrid_rag.json")


if __name__ == "__main__":
    main()
