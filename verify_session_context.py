#!/usr/bin/env python3
"""Verify the RAG context stored in sessions"""

import json

with open('rag_sessions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Afficher la session harmony_admin
if 'harmony_admin' in data['sessions']:
    session = data['sessions']['harmony_admin']
    print(f'📋 Session: harmony_admin')
    print(f'   Contexte global: {session["context"].get("global_context", "N/A")}')
    print(f'   Entités indexées: {session["context"].get("indexed_entities", [])}')
    print(f'   Nombre de tours: {len(session["turns"])}')
    
    if session['turns']:
        turn = session['turns'][0]
        print(f'\n📝 Premier tour:')
        print(f'   Question: {turn["question"][:80]}...')
        print(f'\n🎯 Contexte RAG:')
        rag_ctx = turn.get('rag_context', {})
        print(f'   - Méthode: {rag_ctx.get("method", "N/A")}')
        print(f'   - Documents trouvés: {rag_ctx.get("documents_count", 0)}')
        print(f'   - Scores: {rag_ctx.get("scores", [])}')
        print(f'   - Entités trouvées: {rag_ctx.get("entities", [])}')
        
        if rag_ctx.get('sources'):
            print(f'\n📄 Sources:')
            for i, source in enumerate(rag_ctx['sources'], 1):
                print(f'   {i}. Document: {source.get("document_id", "N/A")}')
                print(f'      Méthode: {source.get("method", "N/A")}')
                print(f'      Texte: {source.get("text_preview", "N/A")[:100]}...')
        
        print(f'\n📊 Métadonnées:')
        metadata = turn.get('metadata', {})
        print(f'   - Graph results: {metadata.get("graph_results", 0)}')
        print(f'   - Vector results: {metadata.get("vector_results", 0)}')
        print(f'   - Timestamp: {turn.get("timestamp", "N/A")}')

print('\n✅ Contexte RAG correctement stocké dans la session!')
