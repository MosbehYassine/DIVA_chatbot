# Guide d'Exécution : Atteindre 80% de Précision

**Baseline Actuelle**: 38.03% (2/21 questions)  
**Objectif**: 80% (17/21 questions)  
**Gap**: +41.97 points

---

## 🚀 Étape 1 : Meilleure Extraction HTML (Accents)

### ✅ Changements Appliqués
- `ingest_docs.py` : Ajout de `normalize_text()` 
- Décoding HTML entities (`&é;` → `é`)
- Normalisation Unicode (NFC)
- Nettoyage des espaces spéciaux

### 🔧 Exécution

```bash
# Re-ingest complet (15-20 min)
python ingest_docs.py

# Vérification (vérifier accents)
python -c "
import json
with open('chunks_metadata.json') as f:
    chunks = json.load(f)
    print(f'Total chunks: {len(chunks)}')
    # Chercher des accents
    for c in chunks[:3]:
        txt = c['text'][:100]
        print(f'  {txt}...')
"
```

### ✔️ Validation
- [ ] Fichier `chunks_metadata.json` mis à jour
- [ ] Accents préservés (é, è, à, ç, etc.)
- [ ] Nombre de chunks stable (±5%)

### 📊 Impact Estimé
- **+2-3%** de précision (meilleure extraction = meilleur matching)

---

## 🔍 Étape 2 : Index FAISS au Niveau Phrase

### ✅ Script Créé
- `index_by_sentences.py` : Nouveau script complet
- Split intelligent par phrases (gère M., Dr., etc.)
- Métadonnées de traçabilité chunk → phrase

### 🔧 Exécution

```bash
# Créer index phrase (10-15 min)
python index_by_sentences.py

# Vérification
python -c "
import json
with open('sentences_metadata.json') as f:
    sentences = json.load(f)
    print(f'Total phrases: {len(sentences)}')
    print(f'Ratio phrase/chunk: {len(sentences) / 10175:.1f}x')
    print(f'Sample phrase: {sentences[0][\"text\"][:80]}...')
"
```

### ✔️ Validation
- [ ] Fichier `sentences_metadata.json` créé
- [ ] Fichier `sentences_faiss.pkl` créé
- [ ] Ratio phrases/chunks ≥ 5

### 📊 Impact Estimé
- **+3-5%** de précision (granularité phrase)

---

## 🚀 Étape 3 : Modèle Plus Fort (e5-large)

### 🔧 Exécution

```bash
# Re-index avec modèle plus puissant (30-40 min)
export RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large
python ingest_docs.py

# Vérifier dimension = 1024 (vs 768 pour base)
python -c "
import json
with open('chunks_metadata.json') as f:
    chunks = json.load(f)
    if chunks:
        print(f'Embedding model: {chunks[0].get(\"embedding_model\", \"N/A\")}')
"
```

### ✔️ Validation
- [ ] Vérifier `embedding_model: intfloat/multilingual-e5-large`
- [ ] Dimension FAISS = 1024
- [ ] Temps d'inférence acceptable (< 2s par requête)

### 📊 Impact Estimé
- **+5-10%** de précision (modèle meilleur sémantique)

---

## 📈 Étape 4 : Enrichir le Graphe

### 🔧 Exécution

```bash
# Enrichir le graphe (5-10 min)
python enrich_graph.py

# Vérification
python -c "
import pickle
with open('networkx_graph_sources.pkl', 'rb') as f:
    G = pickle.load(f)
    print(f'Nœuds: {G.number_of_nodes()}')
    print(f'Arêtes: {G.number_of_edges()}')
    print(f'Degré moyen: {2 * G.number_of_edges() / G.number_of_nodes():.2f}')
"
```

### ✔️ Validation
- [ ] Graphe enrichi (plus de nœuds/arêtes)
- [ ] Connectivité améliorée

### 📊 Impact Estimé
- **+3-5%** de précision (meilleure navigation graphe)

---

## 📊 Mesure Finale

Après les 4 étapes :

```bash
# Mesure de précision (10-15 min)
python run_precision.py --no-cot

# Attendre résultat
# Comparer avec baseline 38.03%
```

### Projection Théorique
```
Baseline:           38.03%
Après Étape 1:     +2-3%  → ~41%
Après Étape 2:     +3-5%  → ~45%
Après Étape 3:     +5-10% → ~52%
Après Étape 4:     +3-5%  → ~58%
─────────────────────────────
Total attendu:      ~58-60%
```

**Note**: Peut atteindre 60-70%, mais pas garanti 80% en une itération.  
Peut nécessiter étapes supplémentaires (fine-tuning, prompt engineering, etc.)

---

## 🔄 Plan d'Exécution Séquentiel

```
Temps Total Estimé: ~90-100 minutes
```

### Jour 1 : Mesure + Étapes 1-2
1. **Baseline** (10 min) ✅ Done: 38.03%
2. **Étape 1** : HTML extraction (20 min)
3. **Étape 2** : Index phrases (15 min)
4. **Mesure intermédiaire** (15 min)

### Jour 2 : Étapes 3-4
5. **Étape 3** : Modèle e5-large (40 min)
6. **Étape 4** : Enrichir graphe (10 min)
7. **Mesure finale** (15 min)

---

## 🎯 Checkpoint : Avant Chaque Étape

Avant de lancer chaque étape, vérifier :

### ✅ Pre-étape 1
```bash
# Backup
cp faiss_index.pkl faiss_index.bak.pkl
cp chunks_metadata.json chunks_metadata.bak.json
```

### ✅ Post-étape 1
```
✓ chunks_metadata.json > chunks_metadata.bak.json
✓ Accents visibles dans les chunks
✓ Nombre de chunks stable
```

### ✅ Post-étape 2
```
✓ sentences_metadata.json créé
✓ sentences_faiss.pkl créé
✓ Total phrases > 50k
```

### ✅ Post-étape 3
```
✓ embedding_model = e5-large
✓ FAISS dimension = 1024
✓ Temps query < 2s
```

### ✅ Post-étape 4
```
✓ Graphe enrichi
✓ Connectivité améliorée
```

---

## 🚨 Rollback en Cas de Régression

Si performance **baisse** après une étape:

```bash
# Revenir à l'index précédent
cp faiss_index.bak.pkl faiss_index.pkl
cp chunks_metadata.bak.json chunks_metadata.json

# Re-tester
python run_precision.py --no-cot
```

---

## 📝 Notes Importantes

1. **Variable d'environnement**: `RAG_EMBEDDING_MODEL`
   ```bash
   # Vérifier valeur courante
   python -c "from rag_config import EMBEDDING_MODEL; print(EMBEDDING_MODEL)"
   ```

2. **Temps machine**: Les valeurs estimées dépendent du CPU/GPU
   - CPU: Ajouter +50%
   - GPU: Réduire de 30%

3. **Espace disque**: ~500MB par étape (indexes temporaires)

4. **Monitoring**: Tous les scripts loggent dans la sortie console
   - Consulter les logs pour diagnostiquer les problèmes

---

## 📞 Support Rapide

| Problème | Solution |
|----------|----------|
| Accents corrompus | Vérifier encoding UTF-8 dans BeautifulSoup |
| Index FAISS crash | Réduire EMBEDDING_BATCH_SIZE |
| OOM (mémoire) | Réduire top_k de 30 à 15 |
| Lenteur inférence | Vérifier si GPU utilisé (`CUDA_VISIBLE_DEVICES`) |

---

**Créé**: 2026-06-04  
**Version**: 1.0 (Baseline at 38.03%)
