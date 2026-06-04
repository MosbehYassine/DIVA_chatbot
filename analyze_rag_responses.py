#!/usr/bin/env python3
"""
Script pour tester et analyser les réponses du système RAG
Exécute les questions du fichier test_questions_installation_zoom.json
et analyse la qualité des réponses fournies par le modèle
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from difflib import SequenceMatcher

sys.path.append('.')

class RAGAnalyzer:
    """Analyse les réponses du système RAG"""
    
    def __init__(self, questions_file: str):
        self.questions_file = questions_file
        self.questions = []
        self.results = []
        
    def load_questions(self):
        """Charge les questions depuis le fichier JSON"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.questions = data.get('test_questions', [])
            print(f"✅ {len(self.questions)} questions chargées")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def similarity_ratio(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def analyze_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse une question"""
        question_id = question_data.get('id', 'unknown')
        question = question_data.get('question', '')
        expected_source = question_data.get('source', '')
        expected_keywords = question_data.get('expected_keywords', [])
        expected_sources_contains = question_data.get('expected_source_contains', [])
        
        analysis = {
            'id': question_id,
            'question': question[:100] + '...' if len(question) > 100 else question,
            'expected_source': expected_source,
            'expected_keywords': expected_keywords,
            'expected_sources_contain': expected_sources_contains,
            'module': question_data.get('module', ''),
            'status': '⏳ À analyser',
            'details': {}
        }
        
        return analysis
    
    def print_summary(self):
        """Affiche un résumé des questions"""
        print("\n" + "="*80)
        print("📊 RÉSUMÉ DES QUESTIONS À TESTER")
        print("="*80)
        
        # Grouper par module
        modules = {}
        for q in self.questions:
            module = q.get('module', 'Non classifié')
            if module not in modules:
                modules[module] = []
            modules[module].append(q)
        
        total_questions = 0
        for module in sorted(modules.keys()):
            questions = modules[module]
            print(f"\n📋 Module: {module}")
            print(f"   └─ Nombre de questions: {len(questions)}")
            
            # Afficher les 3 premières questions
            for i, q in enumerate(questions[:3]):
                q_text = q.get('question', '')
                if len(q_text) > 60:
                    q_text = q_text[:60] + '...'
                print(f"      [{i+1}] {q_text}")
            
            if len(questions) > 3:
                print(f"      ... et {len(questions) - 3} autres")
            
            total_questions += len(questions)
        
        print(f"\n{'─'*80}")
        print(f"Total: {total_questions} questions")
        
        # Analyser les sources
        print("\n" + "="*80)
        print("📁 SOURCES DE DOCUMENTATION")
        print("="*80)
        
        sources = {}
        for q in self.questions:
            source = q.get('source', 'Non spécifiée')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        # Afficher les 10 sources les plus fréquentes
        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 sources:")
        for i, (source, count) in enumerate(sorted_sources[:10]):
            print(f"{i+1:2d}. {source:60s} - {count:3d} question(s)")
        
        if len(sorted_sources) > 10:
            print(f"\n... et {len(sorted_sources) - 10} autres sources")
        
        print(f"\nTotal unique sources: {len(sources)}")
    
    def analyze_complexity(self):
        """Analyse la complexité des questions"""
        print("\n" + "="*80)
        print("🎯 ANALYSE DE COMPLEXITÉ")
        print("="*80)
        
        # Analyser la longueur des questions
        lengths = [len(q.get('question', '')) for q in self.questions]
        keywords_count = [len(q.get('expected_keywords', [])) for q in self.questions]
        
        print(f"\nLongueur des questions:")
        print(f"  • Minimum: {min(lengths) if lengths else 0} caractères")
        print(f"  • Maximum: {max(lengths) if lengths else 0} caractères")
        print(f"  • Moyenne: {sum(lengths) / len(lengths):.0f} caractères")
        
        print(f"\nNombre de mots-clés attendus:")
        print(f"  • Minimum: {min(keywords_count) if keywords_count else 0}")
        print(f"  • Maximum: {max(keywords_count) if keywords_count else 0}")
        print(f"  • Moyenne: {sum(keywords_count) / len(keywords_count):.1f}")
        
        # Analyser les patterns de questions
        question_patterns = {}
        for q in self.questions:
            question = q.get('question', '')
            if 'Que dit' in question:
                pattern = 'Que dit la documentation sur'
            elif 'Comment' in question:
                pattern = 'Comment'
            elif 'Quel' in question:
                pattern = 'Quel/Quelle'
            else:
                pattern = 'Autre'
            
            if pattern not in question_patterns:
                question_patterns[pattern] = 0
            question_patterns[pattern] += 1
        
        print(f"\nPatterns de questions:")
        for pattern, count in sorted(question_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {pattern:40s} - {count:3d} question(s)")
    
    def check_keywords_coverage(self):
        """Vérifie la couverture des mots-clés"""
        print("\n" + "="*80)
        print("🔍 ANALYSE DES MOTS-CLÉS")
        print("="*80)
        
        all_keywords = {}
        for q in self.questions:
            keywords = q.get('expected_keywords', [])
            for kw in keywords:
                if kw not in all_keywords:
                    all_keywords[kw] = 0
                all_keywords[kw] += 1
        
        # Mots-clés les plus fréquents
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 20 mots-clés les plus fréquents:")
        for i, (kw, count) in enumerate(sorted_keywords[:20]):
            percentage = (count / len(self.questions)) * 100
            print(f"{i+1:2d}. {kw:30s} - {count:3d} fois ({percentage:5.1f}%)")
        
        print(f"\nTotal de mots-clés uniques: {len(all_keywords)}")
        
        # Analyser les domaines
        domains = {}
        domain_keywords = {
            'Installation': ['installation', 'serveur', 'client', 'déploiement'],
            'Chemins': ['chemin', 'implicite', 'harmony', 'fichier'],
            'Imprimantes': ['imprimante', 'impression', 'spouleur', 'modle'],
            'Aides': ['aides', 'fenêtre', 'documentation'],
            'Utilisateurs': ['utilisateur', 'profil', 'droit', 'authentification'],
            'Licences': ['licence', 'dmlt', 'gestion'],
        }
        
        for kw in all_keywords:
            for domain, domain_kws in domain_keywords.items():
                if any(d in kw.lower() for d in domain_kws):
                    if domain not in domains:
                        domains[domain] = 0
                    domains[domain] += 1
                    break
        
        print(f"\nDomaines couverts:")
        for domain in sorted(domains.keys()):
            print(f"  • {domain}")
    
    def generate_report(self):
        """Génère un rapport complet"""
        self.load_questions()
        
        print("\n" + "🚀 "*20)
        print("ANALYSE COMPLÈTE DU SYSTÈME DE QUESTIONS RAG")
        print("🚀 "*20 + "\n")
        
        self.print_summary()
        self.analyze_complexity()
        self.check_keywords_coverage()
        
        print("\n" + "="*80)
        print("✅ ANALYSE TERMINÉE")
        print("="*80)
        
        # Sauvegarder l'analyse
        analysis_file = Path('analysis_questions.json')
        analysis_summary = {
            'total_questions': len(self.questions),
            'modules': list(set(q.get('module', '') for q in self.questions)),
            'unique_sources': len(set(q.get('source', '') for q in self.questions)),
            'avg_question_length': sum(len(q.get('question', '')) for q in self.questions) / len(self.questions) if self.questions else 0,
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Analyse sauvegardée: {analysis_file}")

def verify_rag_responses():
    """Vérifie les réponses du RAG sur quelques questions"""
    print("\n" + "="*80)
    print("🧪 TEST DE QUELQUES RÉPONSES DU RAG")
    print("="*80)
    
    try:
        from query_docs import HybridRAG
        
        rag = HybridRAG()
        print("✅ Système RAG initialisé")
        
        # Charger les questions
        with open('test_questions_installation_zoom.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = data.get('test_questions', [])
        
        # Tester les 5 premières questions
        print(f"\n🔹 Teste 5 questions sur {len(questions)} disponibles\n")
        
        for i, q_data in enumerate(questions[:5]):
            q_id = q_data.get('id', 'unknown')
            question = q_data.get('question', '')
            expected_source = q_data.get('source', '')
            expected_keywords = q_data.get('expected_keywords', [])
            
            print(f"\n[{i+1}/5] Question ID: {q_id}")
            print(f"       Question: {question}")
            print(f"       Source attendue: {expected_source}")
            
            try:
                # Requête RAG
                result = rag.query(question, top_k=1)
                
                if 'merged_results' in result and len(result['merged_results']) > 0:
                    top_result = result['merged_results'][0]
                    
                    print(f"       ✅ Réponse trouvée:")
                    print(f"          • Source: {top_result.get('document_id', 'N/A')}")
                    print(f"          • Score: {top_result.get('score', 0):.3f}")
                    print(f"          • Texte (50 chars): {top_result.get('text', '')[:50]}...")
                    
                    # Vérifier la correspondance avec la source attendue
                    source_match = expected_source.lower() in top_result.get('document_id', '').lower()
                    print(f"          • Correspondance source: {'✅' if source_match else '⚠️'}")
                else:
                    print(f"       ❌ Pas de résultats trouvés")
            except Exception as e:
                print(f"       ❌ Erreur lors du traitement: {e}")
        
        print(f"\n{'─'*80}")
        print("✅ Test de réponses terminé")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")

if __name__ == '__main__':
    # Analyser les questions
    analyzer = RAGAnalyzer('test_questions_installation_zoom.json')
    analyzer.generate_report()
    
    # Tester quelques réponses du RAG
    verify_rag_responses()
    
    print("\n" + "="*80)
    print("🎉 RAPPORT D'ANALYSE COMPLET GÉNÉRÉ")
    print("="*80)
