#!/usr/bin/env python3
"""
Script de test complet du système RAG Harmony/Divalto
Teste toutes les questions de test_questions.json et compare avec les réponses attendues
"""

import json
import sys
import os
from difflib import SequenceMatcher
from typing import List, Dict, Any

# Ajouter le répertoire courant au path
sys.path.append('.')

from answer_questions import VectorStore, generate_answer

def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    """Charge les questions de test depuis le fichier JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_questions']

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcule la similarité entre deux textes (0-1)"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def evaluate_answer(generated: str, expected: str) -> Dict[str, Any]:
    """Évalue une réponse générée par rapport à la réponse attendue"""
    similarity = calculate_similarity(generated, expected)

    # Évaluation qualitative
    if similarity >= 0.9:
        quality = "EXCELLENT"
        score = 5
    elif similarity >= 0.8:
        quality = "TRÈS BON"
        score = 4
    elif similarity >= 0.7:
        quality = "BON"
        score = 3
    elif similarity >= 0.6:
        quality = "MOYEN"
        score = 2
    elif similarity >= 0.4:
        quality = "FAIBLE"
        score = 1
    else:
        quality = "TRÈS FAIBLE"
        score = 0

    return {
        'similarity': similarity,
        'quality': quality,
        'score': score,
        'generated_length': len(generated),
        'expected_length': len(expected)
    }

def test_all_questions():
    """Teste toutes les questions et génère un rapport complet"""

    print("🚀 DÉMARRAGE DES TESTS DU SYSTÈME RAG HARMONY/DIVALTO")
    print("=" * 60)

    # Charger les questions de test
    test_file = 'test_questions.json'
    if not os.path.exists(test_file):
        print(f"❌ Fichier {test_file} non trouvé!")
        return

    questions = load_test_questions(test_file)
    print(f"📋 {len(questions)} questions à tester")

    # Initialiser le système RAG
    print("\n🔧 Initialisation du système RAG...")
    store = VectorStore()
    if not store.load():
        print("❌ Impossible de charger l'index FAISS!")
        return

    print("✅ Système RAG chargé avec succès")

    # Résultats
    results = []
    total_score = 0
    category_stats = {}
    difficulty_stats = {'facile': [], 'moyen': [], 'difficile': []}

    print("\n🧪 DÉBUT DES TESTS")
    print("-" * 60)

    for i, question_data in enumerate(questions, 1):
        question_id = question_data['id']
        question = question_data['question']
        expected = question_data['expected_answer']
        category = question_data['category']
        difficulty = question_data['difficulty']

        print(f"\n[{i:2d}/{len(questions)}] Test: {question_id}")
        print(f"Question: {question}")

        try:
            # Générer la réponse
            context_chunks = store.search(question, k=5)
            context = '\n\n'.join(context_chunks)
            generated_answer = generate_answer(question, context)

            # Évaluer la réponse
            evaluation = evaluate_answer(generated_answer, expected)

            # Stocker les résultats
            result = {
                'id': question_id,
                'question': question,
                'category': category,
                'difficulty': difficulty,
                'expected': expected,
                'generated': generated_answer,
                'evaluation': evaluation
            }
            results.append(result)

            # Statistiques
            total_score += evaluation['score']
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(evaluation['score'])
            difficulty_stats[difficulty].append(evaluation['score'])

            # Afficher le résultat
            print(f"✅ Réponse générée ({evaluation['quality']})")
            print(f"   Similarité: {evaluation['similarity']:.2%}")
            print(f"   Score: {evaluation['score']}/5")

        except Exception as e:
            print(f"❌ Erreur lors du test: {str(e)}")
            result = {
                'id': question_id,
                'question': question,
                'category': category,
                'difficulty': difficulty,
                'expected': expected,
                'generated': f"ERREUR: {str(e)}",
                'evaluation': {'similarity': 0, 'quality': 'ERREUR', 'score': 0}
            }
            results.append(result)

    # Rapport final
    print("\n" + "=" * 60)
    print("📊 RAPPORT FINAL")
    print("=" * 60)

    # Score global
    avg_score = total_score / len(questions)
    print(f"📊 SCORE GLOBAL: {avg_score:.2f}/5 ({total_score}/{len(questions)*5})")

    # Statistiques par catégorie
    print("\n📈 STATISTIQUES PAR CATÉGORIE:")
    for category, scores in category_stats.items():
        avg_cat = sum(scores) / len(scores)
        print(f"   {category}: {avg_cat:.2f}/5 ({sum(scores)}/{len(scores)*5})")

    # Statistiques par difficulté
    print("\n🎯 STATISTIQUES PAR DIFFICULTÉ:")
    for difficulty, scores in difficulty_stats.items():
        if scores:
            avg_diff = sum(scores) / len(scores)
            print(f"   {difficulty}: {avg_diff:.2f}/5 ({sum(scores)}/{len(scores)*5})")

    # Détails des réponses
    print("\n📝 DÉTAIL DES RÉPONSES:")
    print("-" * 60)

    for result in results:
        eval = result['evaluation']
        status = "✅" if eval['score'] >= 3 else "⚠️" if eval['score'] >= 2 else "❌"
        print(f"{status} {result['id']} ({result['category']}) - {eval['quality']} ({eval['score']}/5)")

        if eval['score'] < 3:  # Montrer les réponses faibles
            print(f"   Question: {result['question'][:80]}...")
            print(f"   Attendu: {result['expected'][:80]}...")
            print(f"   Généré: {result['generated'][:80]}...")
            print()

    # Sauvegarder les résultats détaillés
    output_file = 'test_results_detailed.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_questions': len(questions),
                'average_score': avg_score,
                'category_stats': {cat: sum(scores)/len(scores) for cat, scores in category_stats.items()},
                'difficulty_stats': {diff: sum(scores)/len(scores) for diff, scores in difficulty_stats.items() if scores}
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats détaillés sauvegardés dans: {output_file}")

    # Évaluation globale
    if avg_score >= 4.5:
        print("\n🎉 EXCELLENT! Le système RAG fonctionne parfaitement!")
    elif avg_score >= 3.5:
        print("\n👍 TRÈS BON! Le système RAG est opérationnel avec de bonnes performances.")
    elif avg_score >= 2.5:
        print("\n👌 BON! Le système RAG fonctionne correctement.")
    elif avg_score >= 1.5:
        print("\n⚠️ MOYEN! Le système nécessite quelques améliorations.")
    else:
        print("\n❌ FAIBLE! Le système nécessite des améliorations majeures.")

if __name__ == "__main__":
    test_all_questions()