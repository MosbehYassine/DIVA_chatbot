#!/usr/bin/env python3
"""
Script de test du système RAG hybride
Teste le modèle avec les questions du fichier test_questions.json
"""

import json
import os
import io
import sys
from datetime import datetime
from difflib import SequenceMatcher
from query_docs import HybridRAG
from rag_answer import generate_answer_from_results, build_context_from_results
from rag_config import TARGET_PRECISION
from typing import Dict, List

# Forcer UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
            print(f"Chargement: {len(self.test_questions)} questions de test")
        else:
            print(f"Fichier de test non trouve: {self.test_file}")

    def run_tests(self) -> List[Dict]:
        """Exécute tous les tests."""
        print("\n" + "=" * 80)
        print("DEMARRAGE DES TESTS DU SYSTEME RAG HYBRIDE")
        print("=" * 80)

        if not self.rag.graph and not self.rag.faiss_index:
            print("Erreur: Les index ne sont pas charges")
            print("Executez d'abord: python ingest_docs.py")
            return []

        total = len(self.test_questions)
        for idx, test in enumerate(self.test_questions, 1):
            print(f"\n[{idx}/{total}] Test: {test['id']}")
            print(f"   Categorie: {test['category']}")
            print(f"   Difficulte: {test['difficulty']}")
            print(f"   Question: {test['question']}")

            try:
                # Exécuter la requête
                result = self.rag.query(test['question'], top_k=3)
                
                generated = result.get("generated_answer") or generate_answer_from_results(
                    test["question"],
                    result.get("merged_results", []),
                    embedding_model=self.rag.embedding_model,
                    embedding_model_name=self.rag.embedding_model_name,
                )
                best_response = generated
                if not best_response and result.get("merged_results"):
                    best_response = result["merged_results"][0]["text"]

                similarity = SequenceMatcher(
                    None,
                    (generated or "").lower(),
                    test["expected_answer"].lower(),
                ).ratio()
                
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
                    'found': len(result['merged_results']) > 0,
                    'similarity': similarity,
                    'quality': (
                        "EXCELLENT" if similarity >= 0.9 else
                        "TRES BON" if similarity >= 0.8 else
                        "BON" if similarity >= 0.7 else
                        "MOYEN" if similarity >= 0.6 else
                        "FAIBLE" if similarity >= 0.4 else
                        "TRES FAIBLE"
                    ),
                }
                
                self.results.append(test_result)
                
                # Afficher le résultat
                if test_result['found']:
                    print(f"   TROUVE (Score: {test_result['score']:.2%}, similarite: {similarity:.2%}, {test_result['quality']})")
                    print(f"   Reponse: {best_response[:150]}...")
                else:
                    print(f"   AUCUNE REPONSE TROUVEE")

            except Exception as e:
                print(f"   ERREUR: {e}")
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

    def generate_report(self, output_file: str = "test_results_optimized.json") -> Dict:
        """Génère un rapport de test."""
        print("\n" + "=" * 80)
        print("GENERATION DU RAPPORT")
        print("=" * 80)

        total = len(self.results)
        found = sum(1 for r in self.results if r['found'])
        avg_score = sum(r['score'] for r in self.results if r['found']) / found if found > 0 else 0
        avg_similarity = sum(r.get('similarity', 0) for r in self.results) / total if total > 0 else 0
        excellent = sum(1 for r in self.results if r.get('similarity', 0) >= 0.9)
        good = sum(1 for r in self.results if r.get('similarity', 0) >= 0.7)

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
                'average_score': f"{avg_score:.2%}",
                'average_similarity': f"{avg_similarity:.2%}",
                'excellent_similarity_count': excellent,
                'good_similarity_count': good,
            },
            'by_category': categories_stats,
            'by_difficulty': difficulty_stats,
            'detailed_results': self.results
        }

        # Sauvegarder le rapport
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Rapport sauvegarde: {output_file}")

        # Afficher le résumé
        print("\n" + "=" * 80)
        print("RESUME DES RESULTATS")
        print("=" * 80)
        print(f"Reponses trouvees: {found}/{total} ({(found/total*100):.1f}%)")
        print(f"Aucune reponse: {total - found}/{total} ({((total-found)/total*100):.1f}%)")
        print(f"Score moyen retrieval: {avg_score:.2%}")
        print(f"Similarite moyenne vs attendu: {avg_similarity:.2%}")
        print(f"Objectif precision ({TARGET_PRECISION:.0%}): {'ATTEINT' if avg_similarity >= TARGET_PRECISION else 'NON ATTEINT'}")
        print(f"Excellent (>=90%): {excellent}/{total} | Bon (>=70%): {good}/{total}")

        print("\nPAR CATEGORIE:")
        for cat, stats in categories_stats.items():
            success_rate = (stats['found'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   - {cat}: {stats['found']}/{stats['total']} ({success_rate:.0f}%) - Score: {stats['avg_score']:.2%}")

        print("\nPAR DIFFICULTE:")
        for diff, stats in difficulty_stats.items():
            success_rate = (stats['found'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   - {diff}: {stats['found']}/{stats['total']} ({success_rate:.0f}%) - Score: {stats['avg_score']:.2%}")

        print("\n" + "=" * 80)

        return report


def main():
    tester = RAGTester()
    
    if not tester.test_questions:
        print("Aucune question de test chargee")
        return

    # Exécuter les tests
    tester.run_tests()
    
    # Générer le rapport
    report = tester.generate_report("test_results_optimized.json")
    avg_sim = sum(r.get("similarity", 0) for r in tester.results) / max(len(tester.results), 1)
    if avg_sim < TARGET_PRECISION:
        print(f"\nECHEC: precision {avg_sim:.2%} < objectif {TARGET_PRECISION:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
