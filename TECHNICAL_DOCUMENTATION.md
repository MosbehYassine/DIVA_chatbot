# Documentation Technique - Projet RAG Harmony/Divalto

## 1. Objectif du projet

Ce projet implemente un systeme RAG hybride pour interroger une documentation Harmony/Divalto extraite de fichiers CHM.

Le systeme utilise maintenant uniquement le pipeline RAG documentaire :

1. Recherche graphe dans `networkx_graph.pkl`.
2. Recherche vectorielle FAISS dans `faiss_index.pkl`.
3. Fusion des resultats graphe/vectoriels.
4. Reranking lexical, vectoriel et graphe.
5. Generation extractive depuis les meilleurs passages.

L'objectif est de fournir une reponse fiable a une question utilisateur en s'appuyant sur les sources documentaires locales.

## 2. Architecture generale

```text
Question utilisateur
        |
        v
query_docs.py / HybridRAG.query()
        |
        +--> Recherche graphe
        |       - networkx_graph.pkl
        |
        +--> Recherche vectorielle FAISS
        |       - faiss_index.pkl
        |       - chunks_metadata.json
        |
        +--> Fusion + deduplication
        |
        +--> Reranking lexical/vectoriel/graphe
        |
        +--> Generation extractive via rag_answer.py
```

## 3. Fichiers essentiels

### Pipeline principal

| Fichier | Role |
|---|---|
| `ingest_docs.py` | Lit les documents dans `data/`, cree les chunks, l'index FAISS et le graphe NetworkX. |
| `query_docs.py` | Point d'entree d'interrogation. Contient la classe `HybridRAG`. |
| `build_rag.py` | Compile, relance l'ingestion et execute les tests selon les options. |
| `run_precision.py` | Lance une mesure rapide de similarite sur `test_questions.json`. |
| `measure_precision.py` | Mesure la precision avec un mode reusable/importable. |

### Modules RAG

| Fichier | Role |
|---|---|
| `rag_config.py` | Configuration centrale : modele embeddings, chunking, chemins, poids de reranking, seuils. |
| `rag_vector.py` | Fonctions communes d'encodage embeddings et similarite cosinus. |
| `rag_answer.py` | Generation extractive de reponse depuis les meilleurs passages. |
| `rag_canonical.py` | Ressource optionnelle pour tests/reponses connues ; non utilisee par `query_docs.py`. |
| `rag_curated_vector.py` | Ressource optionnelle pour QA curatee ; non utilisee par `query_docs.py`. |
| `session_manager.py` | Gestion des sessions et historique conversationnel. |

### Donnees et index

| Fichier/Dossier | Role |
|---|---|
| `data/` | Documents source extraits des CHM. |
| `chunks_metadata.json` | Metadonnees et textes des chunks indexes. |
| `faiss_index.pkl` | Index vectoriel FAISS. |
| `networkx_graph.pkl` | Graphe de connaissances NetworkX. |
| `networkx_graph_sources.pkl` | Graphe avec informations de sources. |
| `index_config.json` | Configuration de l'index courant. |
| `curated_qa.json` | Questions/reponses validees manuellement, conservees comme ressource optionnelle. |
| `curated_qa_vectors.npz` | Embeddings pre-calcules des questions curatees, non utilises par la requete principale. |

### Tests

| Fichier | Role |
|---|---|
| `test_questions.json` | Jeu principal de 21 questions avec reponses attendues. |
| `test_questions_extended.json` | Jeu etendu pour verifier sources et mots-cles. |
| `test_hybrid_rag.py` | Tests du RAG hybride sur le jeu principal. |
| `test_hybrid_rag_extended.py` | Tests etendus par module/source. |
| `precision_report.json` | Rapport genere par `run_precision.py`. |

## 4. Pipeline d'ingestion

Commande :

```powershell
python ingest_docs.py
```

Etapes principales :

1. Lecture des fichiers `.htm`, `.html` et `.md` dans `data/`.
2. Nettoyage HTML avec BeautifulSoup.
3. Normalisation texte.
4. Decoupage en chunks selon `RAG_CHUNK_STRATEGY`.
5. Encodage des chunks avec SentenceTransformers.
6. Creation de l'index FAISS.
7. Extraction d'entites et creation du graphe NetworkX.
8. Sauvegarde des artefacts :
   - `faiss_index.pkl`
   - `chunks_metadata.json`
   - `networkx_graph.pkl`
   - `networkx_graph_sources.pkl`
   - `index_config.json`

Configuration actuelle de l'index :

```json
{
  "embedding_model": "intfloat/multilingual-e5-large",
  "embedding_dimension": 1024,
  "chunk_strategy": "semantic",
  "chunk_count": 13216,
  "chunk_size": 800,
  "chunk_overlap": 200
}
```

## 5. Pipeline de requete

Commande interactive :

```powershell
python query_docs.py
```

Flux logique :

1. La question est recue par `HybridRAG.query()`.
2. Le systeme lance la recherche graphe.
3. Le systeme lance la recherche vectorielle FAISS.
4. Les resultats sont fusionnes et dedupliques.
5. Les passages sont rerankes avec les signaux lexicaux, vectoriels, source et graphe.
6. La reponse finale est generee de maniere extractive depuis les meilleurs passages.

La requete principale n'utilise plus :

- `lookup_canonical()` ;
- `lookup_by_vector()` ;
- les reponses canoniques ;
- les reponses QA curatees.

## 6. Evaluation et precision

Commande :

```powershell
python run_precision.py
```

Note importante :

- Le score depend maintenant uniquement de la capacite du RAG documentaire a retrouver puis extraire les bons passages.
- Les resultats ne sont plus forces par les reponses attendues de `test_questions.json`.
- Le benchmark devient plus realiste pour evaluer le retrieval et la generation extractive.
- L'ancien score de 100% n'est plus representatif, car il provenait des reponses canoniques maintenant desactivees.
- Il faut relancer `python run_precision.py` pour generer un nouveau `precision_report.json` en mode RAG pur.

## 7. Variables de configuration utiles

| Variable | Role | Valeur recommandee |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | Modele d'embeddings | `intfloat/multilingual-e5-large` |
| `RAG_CHUNK_STRATEGY` | Strategie de chunking | `semantic` |
| `RAG_CHUNK_SIZE` | Taille des chunks | `800` |
| `RAG_CHUNK_OVERLAP` | Overlap entre chunks | `200` |
| `RAG_TARGET_PRECISION` | Seuil de validation | `0.80` |

## 8. Fine-tuning : est-ce possible ?

Oui, le fine-tuning est possible, mais il faut choisir le bon niveau.

### Option A - Fine-tuning du modele d'embeddings

C'est l'option la plus pertinente pour ce projet.

Objectif :

- ameliorer la recherche FAISS ;
- rapprocher les questions utilisateur des bons passages ;
- reduire les erreurs de retrieval.

Donnees necessaires :

```json
[
  {
    "question": "Qu'est-ce qu'un chemin Harmony ?",
    "positive_passage": "Les chemins Harmony permettent de raccourcir les noms de fichier...",
    "negative_passages": [
      "La gestion des utilisateurs est assuree par Xlog1.dhop...",
      "Harmony utilise le spouleur Windows..."
    ]
  }
]
```

Volume conseille :

- minimum utile : 100 a 300 paires question/passage ;
- bon niveau : 500 a 2000 paires ;
- ideal : paires positives + negatives difficiles.

Avantages :

- ameliore le coeur du RAG ;
- conserve les sources documentaires ;
- moins couteux qu'un fine-tuning LLM complet.

Limites :

- necessite de reconstruire `faiss_index.pkl` apres fine-tuning ;
- demande un jeu de donnees annote proprement.

### Option B - Fine-tuning d'un reranker

Tres utile si le retrieval retourne les bons documents dans le top 20 mais pas en top 1.

Objectif :

- reclasser les passages candidats ;
- ameliorer la precision finale sans modifier l'index principal.

Donnees necessaires :

```json
{
  "question": "...",
  "passage": "...",
  "label": 1
}
```

ou :

```json
{
  "question": "...",
  "positive": "...",
  "negative": "..."
}
```

Avantages :

- tres efficace pour la precision ;
- peut etre ajoute apres FAISS ;
- evite de trop dependre du matching lexical.

### Option C - Fine-tuning d'un LLM generateur

Possible, mais ce n'est pas la premiere recommandation ici.

Pourquoi :

- le projet utilise surtout une generation extractive ;
- le risque principal actuel est le choix du bon passage, pas la redaction ;
- fine-tuner un LLM peut produire des reponses fluides mais hallucinees si le retrieval est faible.

Fine-tuning LLM utile si :

- on veut un assistant conversationnel qui reformule mieux ;
- on dispose de nombreuses paires question/reponse validees ;
- on garde le RAG comme source de contexte.

Format typique :

```jsonl
{"messages":[{"role":"system","content":"Tu reponds uniquement depuis la documentation Harmony/Divalto."},{"role":"user","content":"Qu'est-ce qu'un chemin Harmony ?"},{"role":"assistant","content":"Les chemins Harmony permettent de raccourcir les noms de fichier..."}]}
```

Volume conseille :

- minimum : 100 exemples tres propres ;
- recommande : 500+ exemples ;
- ideal : exemples multi-domaines, avec refus quand la documentation ne contient pas la reponse.

## 9. Recommandation

Pour ce projet, l'ordre recommande est :

1. Evaluer le RAG documentaire sur des questions metier variees.
2. Annoter les bons passages retrouves pour chaque question.
3. Ajouter des negatives difficiles pour entrainer un reranker ou un modele d'embeddings.
4. Entrainer ou adapter un modele d'embeddings/reranker.
5. Reindexer les documents.
6. Fine-tuner un LLM uniquement si la qualite de formulation reste insuffisante.

Conclusion :

- Fine-tuning possible : oui.
- Fine-tuning recommande en premier : embeddings ou reranker.
- Fine-tuning LLM : possible, mais secondaire.
- Priorite actuelle : construire un dataset d'evaluation plus large et tester les questions hors benchmark canonique.

## 10. Commandes de maintenance

Compiler le projet :

```powershell
python build_rag.py --compile
```

Recreer les index :

```powershell
python ingest_docs.py
```

Tester la precision :

```powershell
python run_precision.py
```

Lancer l'interface :

```powershell
python query_docs.py
```
