#!/usr/bin/env python3
from query_docs import HybridRAG

questions = [
    "Je dois installer une architecture TSE multi-serveurs avec des clients légers Web. Quelles sont les étapes à suivre pour configurer les chemins Harmony, déclarer les serveurs de données, configurer les imprimantes, et mettre en place les fichiers d'aides ?",
    "Un utilisateur ne peut pas accéder aux fichiers via le chemin Harmony, les imprimantes ne fonctionnent pas, et les aides ne s'affichent pas. Quels fichiers de configuration dois-je vérifier et dans quel ordre pour diagnostiquer le problème ?"
]

rag = HybridRAG()
for q in questions:
    res = rag.query(q, top_k=3)
    print('QUESTION:', q)
    print('COT STEPS:')
    for step in res['cot_steps']:
        print('-', step)
    if res['merged_results']:
        print('TOP CANDIDATS:')
        for rank, candidate in enumerate(res['merged_results'], start=1):
            print(f"  {rank}. source={candidate.get('source','N/A')} method={candidate.get('method','N/A')} score={candidate.get('hybrid_score',0):.2f}")
            print(f"     preview={candidate.get('text','')[:150].replace('\n', ' ')}")
    else:
        print('BEST SOURCE: N/A')
    print('\n' + '='*60 + '\n')
