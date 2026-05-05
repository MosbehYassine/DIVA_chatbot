# Système RAG Hybride - Documentation Harmony/Divalto

## 📋 Vue d'ensemble

Système de **Retrieval-Augmented Generation (RAG) hybride** combinant :
- **Graph RAG** : Recherche basée sur les entités et relations
- **Vector RAG** : Recherche par similarité sémantique (FAISS)

## 📁 Structure du projet

```
CHM_extrait/
│
├── DATA/                          ← 📚 Tous les documents (4093 fichiers)
│   ├── Administration/
│   ├── AidesFenetrees/
│   ├── Annexes/
│   ├── Chemins/
│   ├── Installation/
│   ├── ... (50 autres dossiers)
│   └── zoom/
│
├── 📜 Fichiers Python
│   ├── ingest_docs.py             ← Ingestion et indexation depuis DATA/
│   ├── query_docs.py              ← Interface de requête hybride
│   ├── test_hybrid_rag.py         ← Tests d'évaluation
│   └── requirements.txt           ← Dépendances
│
├── 📊 Index générés (ignorés par Git)
│   ├── networkx_graph.pkl         ← Graphe des entités
│   ├── faiss_index.pkl            ← Index vectoriel
│   └── chunks_metadata.json       ← Métadonnées des chunks
│
└── 📄 Documentation
    ├── README.md                  ← Doc principale
    └── README_RAG.md              ← Ce fichier
```

## 🚀 Guide d'utilisation

### 1️⃣ Installation des dépendances

```bash
pip install -r requirements.txt
```

### 2️⃣ Indexation des documents

Crée les index FAISS et le graphe NetworkX depuis le dossier `DATA/` :

```bash
python ingest_docs.py
```

**Résultat :**
```
✅ 4182 documents chargés
✅ 6040 chunks créés
✅ 6104 nœuds de graphe
✅ 8004 relations
✅ Index FAISS avec 384 dimensions
```

### 3️⃣ Requête interactive

Interface de chat avec réponse hybride unique :

```bash
python query_docs.py
```

**Exemple :**
```
💬 Posez votre question (ou 'exit'): Que couvre la documentation sur les impressions dans Harmony ?

✅ MEILLEURE RÉPONSE (Hybride)
📚 Source: data/Impressions/...
📊 Méthode: VECTOR | Score: 54.28%
📖 CONTENU: La documentation explique le fonctionnement des impressions...
```

### 4️⃣ Tests d'évaluation

Lance les tests sur les 21 questions de test_questions.json :

```bash
python test_hybrid_rag.py
```

**Résultats :**
```
✅ Taux de réussite: 100% (21/21)
📊 Score moyen: 58.25%

Catégories:
  • Gestion des utilisateurs: 3/3 (100%) - Score: 64%
  • Chemins Harmony: 3/3 (100%) - Score: 62%
  • Impressions: 3/3 (100%) - Score: 54%
  ...
```

## 🔍 Comment ça marche

### Ingestion (ingest_docs.py)

1. **Chargement** : Lit tous les `.htm`, `.html`, `.md` du dossier `DATA/`
2. **Chunking** : Découpe les documents en chunks de 1500 caractères (overlap: 150)
3. **Graph RAG** : Extrait les entités clés (Harmony, XLAN, etc.) et crée un graphe
4. **Vector RAG** : Génère des embeddings avec sentence-transformers et les indexe avec FAISS

### Requête (query_docs.py)

Chaque question est traitée par **deux méthodes simultanées** :

```
Question: "Que couvre la documentation sur les impressions ?"

1️⃣ RECHERCHE PAR GRAPHE
   └─ Cherche les entités "impressions", "documentation"
   └─ Retrouve les nœuds et les documents connectés
   └─ Score: 0.80 (confiance)

2️⃣ RECHERCHE VECTORIELLE  
   └─ Génère l'embedding de la question
   └─ Recherche les chunks les plus similaires avec FAISS
   └─ Score: 0.54 (similarité cosinus)

3️⃣ FUSION INTELLIGENTE
   ├─ Déduplique les résultats
   ├─ Pondère: Graphe (70%) + Vectoriel (80%)
   └─ Retourne: 1 meilleure réponse combinée
```

## 📊 Statistiques

| Métrique | Valeur |
|----------|--------|
| Documents | 4,182 |
| Chunks | 6,040 |
| Nœuds graphe | 6,104 |
| Relations | 8,004 |
| Dimension embeddings | 384 |
| Entités clés | 14+ (Harmony, XLAN, RecordSQL, etc.) |

## 🔧 Configuration

Modifier `ingest_docs.py` :

```python
DOCS_DIR = "data"                    # Dossier des documents
EMBEDDING_MODEL = "..."              # Modèle sentence-transformers
CHUNK_SIZE = 1500                    # Taille des chunks
CHUNK_OVERLAP = 150                  # Chevauchement
```

## 📈 Performances

- **Temps d'ingestion** : ~5 min (4000+ docs)
- **Temps de requête** : <2 sec par question
- **Taux de succès** : 100% (tous les documents trouvent une réponse)
- **Score moyen** : 58.25%

## 🐛 Troubleshooting

### Index non trouvés
```bash
# Réindexer depuis DATA/
python ingest_docs.py
```

### Trop lent
```python
# Réduire la taille des chunks dans ingest_docs.py
CHUNK_SIZE = 1000  # Au lieu de 1500
```

### Scores bas
```python
# Utiliser un modèle plus spécialisé
EMBEDDING_MODEL = "sentence-transformers/LLM2Vec-Sharpened-Mistral-7B-v2"
```

## 📚 Référence entités Graph RAG

```python
KEY_TERMS = [
    "Harmony",
    "Xwpf.exe",
    "XrtDiva.exe",
    "XLAN",
    "RecordSQL",
    "ODBC",
    "MSSQL",
    "Oracle",
    "DB2",
    "Lotus Notes",
    "Serveur Xlan",
    "Serveur d'applications",
    "Client léger",
    "Architecture 3-tiers",
]
```

## 🔗 Fichiers importants

- [ingest_docs.py](ingest_docs.py) - Ingestion et indexation
- [query_docs.py](query_docs.py) - Interface de requête
- [test_hybrid_rag.py](test_hybrid_rag.py) - Tests automatisés
- [requirements.txt](requirements.txt) - Dépendances Python

---

**Version** : 1.0 | **Date** : 2026-05-04 | **Système** : Hybrid RAG (Graph + Vector)
