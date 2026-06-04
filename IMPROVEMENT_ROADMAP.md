# Feuille de Route : Amélioration RAG vers 80%

## 🎯 Objectif Global
Passer de la ligne de base actuelle à 80% de précision sur `test_questions.json`

---

## 📊 Étape 0 : Mesure Baseline
**Statut**: En cours...

```bash
python run_precision.py --no-cot
```

### Métrique à capturer
- Précision actuelle (%)
- Temps d'inférence moyen
- Top-k accuracy (1, 3, 5)

**Fichier résultat**: `precision_report.json`

---

## 🔄 Étape 1 : Ré-ingérer avec Meilleure Extraction HTML

### Problème Identifié
- Perte d'accents (é, è, à, ç, etc.)
- Caractères spéciaux mal normalisés
- Espaces superflus après extraction

### Changes Requis

#### 1.1 Améliorer `ingest_docs.py`
- **Cible**: Fonction `load_documents()`
- **Changements**:
  - Ajouter normalisation Unicode (NFC)
  - Nettoyer les entités HTML (`&nbsp;`, `&gt;`, etc.)
  - Utiliser `html.unescape()`
  
#### 1.2 Fichiers clés à modifier
```
ingest_docs.py
├── load_documents()     ← Améliorer extraction
├── chunk_documents()    ← OK (garder stratégie HTML)
└── embed_and_index()    ← OK (garder batch size)
```

### Exécution
```bash
# Lancer la réingestion avec meilleure extraction
python ingest_docs.py

# Vérifier les accents dans les chunks
python -c "
import json
with open('chunks_metadata.json') as f:
    chunks = json.load(f)
    for i, c in enumerate(chunks[:5]):
        print(f'Chunk {i}: {c[\"text\"][:100]}...')
"
```

### Validation
- ✅ Accents préservés dans `chunks_metadata.json`
- ✅ Pas de perte de caractères spéciaux
- ✅ Nombre de chunks stable (± 5%)

**Temps estimé**: 15-20 min (dépend du parsing HTML)

---

## 🔍 Étape 2 : Index FAISS au Niveau Phrase

### Motivation
- Granularité actuelle : **chunk** (1200 caractères)
- Granularité cible : **phrase** (30-80 caractères)
- **Bénéfice**: Meilleure correspondance fine-grained, moins de bruit

### Stratégie

#### 2.1 Créer `index_by_sentences.py`
```python
# Pseudocode
1. Charger chunks depuis chunks_metadata.json
2. Fractionner chaque chunk en phrases (NLTK, regex)
3. Créer un index [phrase_id → chunk_id] pour traceabilité
4. Indexer dans FAISS chaque phrase individuellement
5. Sauver : sentences_metadata.json, sentences_faiss.pkl
```

#### 2.2 Adapter `query_docs.py`
- Charger nouvel index phrase
- Adapter `search_vector()` pour retourner phrases + lien vers chunk source
- Conserver affichage du chunk complet pour contexte

### Exécution
```bash
# Créer l'index au niveau phrase
python index_by_sentences.py

# Vérifier les résultats
python -c "
import json
with open('sentences_metadata.json') as f:
    sentences = json.load(f)
    print(f'Total phrases: {len(sentences)}')
    print(f'Ratio phrase/chunk: {len(sentences) / 10175:.1f}')
"
```

### Validation
- ✅ Nombre de phrases > 50k
- ✅ Ratio phrase/chunk ≥ 5
- ✅ Métadonnées correctes (trace vers chunk source)

**Temps estimé**: 10-15 min (NLTK tokenization)

---

## 🚀 Étape 3 : Modèle Plus Fort

### Upgrade: `intfloat/multilingual-e5-large`

**Comparaison**:
| Modèle | Dimension | Qualité | Temps |
|--------|-----------|---------|-------|
| `e5-base` | 768 | ⭐⭐⭐ | ⚡⚡ |
| `e5-large` | 1024 | ⭐⭐⭐⭐ | ⚡ |

### Changements

#### 3.1 Configuration
```bash
# Passer le modèle en variable d'environnement
export RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

#### 3.2 Ré-indexer
```bash
# Lancer l'ingest complet avec le nouveau modèle
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large python ingest_docs.py

# Si on a l'index phrase aussi :
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large python index_by_sentences.py
```

### Validation
- ✅ Vérifier dimension FAISS = 1024 (vs 768)
- ✅ `chunks_metadata.json` contient `"embedding_model": "intfloat/multilingual-e5-large"`
- ✅ Temps d'inférence acceptable (< 2s par requête)

**Impact attendu**: +5-10% de précision (meilleur sémantique)

**Temps estimé**: 30-40 min (téléchargement + indexation)

---

## 📈 Étape 4 : Enrichir le Graphe

### Motivation
- Graphe actuel : basique (entités + relations simples)
- Graphe cible : enrichi (concepts, thèmes, hiérarchies)
- **Bénéfice**: Meilleure traversée sémantique, découverte de connexions

### Approche

#### 4.1 Vérifier `enrich_graph.py` existant
```bash
# Consulter ce qu'il fait
head -50 enrich_graph.py
```

#### 4.2 Améliorer l'enrichissement
- Ajouter extraction de concepts/thèmes
- Créer sous-graphes par domaine (Installation, Impression, etc.)
- Ajouter poids aux relations (force, type)

#### 4.3 Exécuter
```bash
python enrich_graph.py

# Vérifier l'enrichissement
python -c "
import pickle
with open('networkx_graph_sources.pkl', 'rb') as f:
    G = pickle.load(f)
    print(f'Nœuds: {G.number_of_nodes()}')
    print(f'Arêtes: {G.number_of_edges()}')
    print(f'Attributs nœud sample: {list(G.nodes(data=True))[0]}')
"
```

### Validation
- ✅ Nombre de nœuds augmentés (+ concepts/thèmes)
- ✅ Attributs enrichis sur les nœuds/arêtes
- ✅ Traversée du graphe possible (connexité)

**Temps estimé**: 5-10 min

---

## 🔗 Dépendances et Ordre d'Exécution

```
Baseline (mesurer avant)
    ↓
Étape 1: Meilleure extraction HTML (ré-ingest)
    ↓
Étape 2: Index FAISS phrase  [optionnel, peut être parallèle]
    ↓
Étape 3: Modèle e5-large  [redépend Étape 1]
    ↓
Étape 4: Enrichir graphe  [indépendant, peut être parallèle]
    ↓
Mesure finale (comparer avec Baseline)
```

---

## 📅 Estimation Temps Total

| Étape | Durée | Notes |
|-------|-------|-------|
| Baseline | 10 min | En attente résultats |
| 1. HTML extraction | 20 min | ré-ingest |
| 2. FAISS sentences | 15 min | optionnel |
| 3. e5-large | 40 min | ré-index complet |
| 4. Enrich graphe | 10 min | rapide |
| **Total** | **~95 min** | ~1.5-2h |

---

## 🎯 Métriques de Succès

### Avant
- Baseline precision: `?`% (à mesurer)

### Après Étape 1-2
- Precision: +2-3%
- Raison: Meilleure extraction + granularité phrase

### Après Étape 3
- Precision: +5-10% supplémentaires
- Raison: Modèle plus fort

### Après Étape 4
- Precision: +3-5% supplémentaires
- Raison: Meilleure navigation graphe

### Cible
- **Precision finale: ≥ 80%**

---

## 🛠️ Commandes Récapitulatives

### Phase 1 : Amélioration base
```bash
# Ré-ingest avec meilleure HTML extraction
python ingest_docs.py

# Test rapide
python -c "import json; data=json.load(open('chunks_metadata.json')); print(f'Chunks: {len(data)}')"
```

### Phase 2 : Indexation phrase (optionnel)
```bash
python index_by_sentences.py
```

### Phase 3 : Upgrade modèle
```bash
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large python ingest_docs.py
```

### Phase 4 : Enrichissement graphe
```bash
python enrich_graph.py
```

### Phase 5 : Mesure finale
```bash
python run_precision.py --no-cot > final_report.log
```

---

## 📝 Notes Importantes

1. **Sauvegardes**: Sauvegarder les anciens indexes avant chaque étape
2. **Logs**: Tous les scripts sauvegardent leur output
3. **Validation**: Tester après chaque étape majeure
4. **Rollback**: Si performance baisse, revenir à l'index précédent

---

**Last Updated**: 2026-06-04
**Version**: 1.0
