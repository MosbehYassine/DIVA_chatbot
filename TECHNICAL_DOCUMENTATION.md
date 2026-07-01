# Documentation Technique - RAG Harmony/Divalto

## 1. Objectif

Ce projet est un systeme RAG local pour interroger la documentation Harmony/Divalto extraite de fichiers CHM.

Le systeme est volontairement documentaire :

- il ne retourne pas de reponse curatee avant la recherche ;
- il ne force pas les reponses attendues de `test_questions.json` ;
- il cherche dans les documents locaux ;
- il extrait une reponse depuis les passages retrouves ;
- il ajoute des sources ;
- il verifie que la reponse est supportee par le contexte.

Le mode actuel est donc un RAG pur :

```text
question
  -> transformation de requete
  -> recherche graphe
  -> recherche vectorielle FAISS
  -> recherche lexicale BM25/TF-IDF
  -> fusion des resultats
  -> mapping child -> parent
  -> MMR
  -> reranking cross-encoder ou fallback
  -> compression contextuelle
  -> generation extractive
  -> verification documentaire
```

## 2. Structure Du Projet

### Fichiers racine utiles

| Fichier | Role |
|---|---|
| `README.md` | Presentation courte du projet. |
| `TECHNICAL_DOCUMENTATION.md` | Documentation technique detaillee. |
| `requirements.txt` | Dependances Python. |
| `Dockerfile` | Execution conteneurisee optionnelle. |
| `.env` | Variables locales, notamment cles API optionnelles. |
| `build_rag.py` | Compilation, ingestion et tests. |

### Donnees et index

| Fichier ou dossier | Role |
|---|---|
| `data/` | Documentation source extraite des CHM. |
| `faiss_index.pkl` | Index FAISS des chunks enfants. |
| `chunks_metadata.json` | Metadonnees des chunks enfants. |
| `parent_chunks_metadata.json` | Metadonnees des chunks parents. |
| `networkx_graph.pkl` | Graphe de connaissances NetworkX. |
| `networkx_graph_sources.pkl` | Copie du graphe avec sources. |
| `index_config.json` | Configuration de l'index actuellement genere. |
| `hybrid_rag_sessions.db` | Base SQLite locale des sessions, questions et reponses. |
| `hybrid_rag_sessions.json` | Ancien stockage JSON, utilise comme source de migration si la base est vide. |

### Pipeline RAG

| Fichier | Role |
|---|---|
| `ingest_docs.py` | Ingestion, sectionnement HTML, parent-child chunking, FAISS, graphe. |
| `query_docs.py` | Moteur de requete `HybridRAG`. |
| `rag_config.py` | Configuration centrale. |
| `rag_vector.py` | Encodage embeddings et similarite cosinus. |
| `rag_answer.py` | Generation extractive et sources. |
| `rag_llm_answer.py` | Reformulation LLM controlee de la reponse finale. |
| `rag_canonical.py` | Helpers de normalisation et extraction de sujet documentaire. |
| `rag_lexical.py` | BM25, TF-IDF ou fallback lexical. |
| `rag_mmr.py` | Diversification MMR. |
| `rag_reranker.py` | Reranking cross-encoder avec fallback. |
| `rag_query_transform.py` | Expansion, SelfQuery, variantes de recherche. |
| `rag_hyde.py` | Generation HyDE pour le retrieval uniquement. |
| `rag_compressor.py` | Compression extractive des contextes. |
| `rag_verifier.py` | Verification du support documentaire. |
| `rag_conversation.py` | Reformulation avec historique conversationnel. |
| `session_manager.py` | Sessions et historique. |

### Evaluation

| Fichier | Role |
|---|---|
| `measure_precision.py` | Evaluation detaillee retrieval + answer. |
| `run_precision.py` | Lance une evaluation rapide et regenere `precision_report.json`. |
| `test_questions.json` | Fichier unique de test : 3 questions par module principal de `data/` (facile, moyen, difficile). |
| `session_test_questions.json` | Mini scenarios de questions liees pour tester la memoire de session. |
| `test_hybrid_rag.py` | Tests fonctionnels du RAG. |
| `test_hybrid_rag_extended.py` | Tests etendus. |
| `test_rag_pipeline_components.py` | Tests des composants internes du pipeline. |

## 3. Fichiers Supprimes

Les fichiers suivants ont ete supprimes car ils ne font plus partie du pipeline actif :

| Fichier | Raison |
|---|---|
| `__pycache__/` | Cache Python regenere automatiquement. |
| `precision_report.json` | Rapport genere, regenere par `python run_precision.py`. |
| `curated_qa.json` | Ancienne base QA curatee, non utilisee par `query_docs.py`. |
| `curated_qa_vectors.npz` | Ancien index vectoriel QA curatee. |
| `rag_curated_vector.py` | Ancien raccourci de reponse curatee, desactive. |
| `README_RAG.md` | Documentation redondante et obsolescente. |
| `EXECUTION_GUIDE.md` | Guide ancien remplace par cette documentation. |

Le pipeline actuel ne depend plus de ces fichiers.

## 4. Ingestion Semantique

Le fichier principal est `ingest_docs.py`.

### Etapes

1. Lecture des fichiers HTML/Markdown dans `data/`.
2. Detection d'encodage avec `UnicodeDammit`.
3. Nettoyage et normalisation du texte.
4. Decoupage HTML par sections `h1`, `h2`, `h3`.
5. Construction des chunks parents et enfants.
6. Enrichissement du texte envoye a FAISS.
7. Creation des embeddings.
8. Creation de l'index FAISS.
9. Creation du graphe NetworkX avec aretes ponderees.
10. Sauvegarde des artefacts.

### Section-aware chunking

Avant, le parent-child chunking etait surtout une fenetre de tokens.

Maintenant, l'ingestion cree d'abord des documents de section :

```text
document HTML
  -> section h1/h2/h3
  -> parent chunk
  -> child chunk
```

Cela evite de melanger plusieurs sections et ameliore la coherence semantique.

### Parent-child retrieval

Configuration actuelle dans `rag_config.py` :

```text
RAG_ENABLE_PARENT_CHILD_RETRIEVAL=true
RAG_PARENT_CHUNK_SIZE=1000
RAG_PARENT_CHUNK_OVERLAP=200
RAG_CHILD_CHUNK_SIZE=250
RAG_CHILD_CHUNK_OVERLAP=50
```

Principe :

- les enfants sont petits et optimises pour la recherche ;
- les parents sont plus longs et optimises pour la reponse ;
- les resultats enfants sont regroupes par `parent_id`.

### Texte indexe dans FAISS

FAISS n'encode pas seulement le contenu brut. Il encode un texte enrichi :

```text
Module: Administration
Titre: Acces aux fichiers d'Harmony
Section: Acces aux fichiers d'Harmony
Fichier: Acc_sauxfichiersd_Harmony.htm
Contenu: ...
```

Le contenu original reste conserve dans `text` pour la reponse.

Le texte reel encode est garde dans `index_text` pour debug.

### Graphe pondere

Chaque relation document -> entite contient maintenant un poids :

```text
weight = occurrences dans le chunk + bonus titre/section/module
```

Le graphe devient donc plus utile pour le reranking.

## 5. Requete

Le fichier principal est `query_docs.py`.

### Flux

```text
HybridRAG.query()
  -> memoire conversationnelle
  -> reformulation si besoin
  -> transformation de requete
  -> recherche graphe
  -> recherche vectorielle FAISS
  -> recherche lexicale
  -> fusion enfants
  -> mapping enfants vers parents
  -> MMR
  -> cross-encoder
  -> compression
  -> generation extractive
  -> verification
  -> reessai optionnel
```

### Important

`query_docs.py` ne fait plus :

- lookup canonique de reponse attendue ;
- lookup dans une base QA curatee ;
- generation d'une reponse HyDE visible par l'utilisateur.

HyDE est seulement utilise comme aide au retrieval.

## 6. Reranking

Le reranking combine plusieurs signaux :

- score vectoriel FAISS ;
- score lexical ;
- signal graphe ;
- correspondance source/fichier ;
- metadata issue de SelfQuery ;
- MMR pour diversifier ;
- cross-encoder si disponible.

Le cross-encoder cible est :

```text
BAAI/bge-reranker-v2-m3
```

Si `FlagEmbedding` ou le modele ne sont pas disponibles, le systeme revient au score hybride interne.

## 7. Generation De Reponse

La reponse suit maintenant deux etapes :

1. generation extractive depuis les meilleurs passages ;
2. reformulation LLM controlee de cette reponse extractive.

Le LLM ne doit pas inventer une reponse libre. Il recoit :

- la question utilisateur ;
- la reponse extractive ;
- les passages documentaires autorises ;
- le resume de session ;
- le plan interne de lecture documentaire.

Il reformule la reponse dans un style plus naturel, puis `rag_verifier.py` verifie que la reponse finale reste supportee par les contextes.

Avant la compression et la generation extractive, le moteur construit maintenant un plan de raisonnement interne court. Ce plan decompose la question en sous-objectifs documentaires, par exemple :

- identifier le sujet principal ;
- chercher les etapes ou actions si la question demande `comment` ;
- chercher les prerequis si la question parle de condition ou d'obligation ;
- chercher les causes et controles si la question parle d'erreur ;
- garder le contexte de session si la question est une question de suivi.

Ce plan n'est pas affiche comme un bloc de chain-of-thought. Il sert seulement a mieux orienter le classement des phrases et la verification de la reponse finale.

Avantages :

- meilleure formulation selon la question ;
- garde-fou documentaire via les passages et la verification ;
- reponses ancrees dans la documentation ;
- sources faciles a fournir.

Limites :

- necessite une cle `OPENAI_API_KEY` ou `OPENROUTER_API_KEY` ;
- si l'appel LLM echoue, le systeme revient a la reponse extractive ;
- similarite faible si la reponse attendue est une reformulation ;
- depend beaucoup de la qualite du passage retrouve.

## 8. Verification

`rag_verifier.py` verifie que la reponse est supportee par le contexte :

- support lexical ;
- support par embeddings ;
- detection de reponses generiques ;
- option LLM verifier si active.

Si le support est insuffisant, le pipeline peut retenter une recherche elargie.

## 9. Configuration Principale

Variables utiles :

| Variable | Role | Defaut |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | Modele embeddings | `intfloat/multilingual-e5-large` |
| `RAG_ENABLE_PARENT_CHILD_RETRIEVAL` | Active parent-child | `true` |
| `RAG_PARENT_CHUNK_SIZE` | Taille parent | `1000` |
| `RAG_CHILD_CHUNK_SIZE` | Taille enfant | `250` |
| `RAG_ENABLE_BM25` | Active BM25 | `true` |
| `RAG_ENABLE_MMR` | Active MMR | `true` |
| `RAG_ENABLE_CROSS_ENCODER_RERANKER` | Active reranker | `true` |
| `RAG_ENABLE_CONTEXT_COMPRESSION` | Active compression | `true` |
| `RAG_ENABLE_HYDE` | Active HyDE retrieval | `true` |
| `RAG_ENABLE_ANSWER_VERIFIER` | Active verification | `true` |
| `RAG_ENABLE_LLM_ANSWER_GENERATION` | Active la reformulation LLM de la reponse finale | `true` |
| `RAG_ANSWER_LLM_MODEL` | Modele LLM pour formuler la reponse | `gpt-4o-mini` |
| `RAG_ENABLE_REASONING_PLAN` | Active la decomposition interne de la question avant generation | `true` |
| `RAG_MAX_RETRY_COUNT` | Nombre de reessais | `1` |

## 10. Commandes

### Compilation

```powershell
python build_rag.py --compile
```

### Ingestion complete

```powershell
python ingest_docs.py
```

Cette commande regenere :

- `faiss_index.pkl`
- `chunks_metadata.json`
- `parent_chunks_metadata.json`
- `networkx_graph.pkl`
- `networkx_graph_sources.pkl`
- `index_config.json`

### Evaluation

```powershell
python run_precision.py
```

ou :

```powershell
python measure_precision.py
```

Le fichier utilise par defaut est `test_questions.json`. Il couvre les dossiers de premier niveau dans `data/` avec trois questions par module :

- `facile` : question generale sur une page du module ;
- `moyen` : question sur le contenu principal d'une autre page ;
- `difficile` : question plus detaillee sur une troisieme page, ou sur la meilleure page disponible si le module contient peu de contenu exploitable.

Le rapport `precision_report.json` donne maintenant une indication plus complete de performance :

- `average_performance_score` : score composite retrieval + qualite reponse + source ;
- `retrieval.recall_at_5`, `recall_at_10`, `mrr`, `ndcg_at_5` : qualite de recherche documentaire ;
- `answer.keyword_overlap`, `supported_answer_rate`, `hallucination_rate` : qualite de reponse ;
- `by_module`, `by_difficulty`, `by_category` : performance detaillee par groupe ;
- `failure_reasons` : cause principale des echecs ;
- `weakest_cases` : questions a inspecter en priorite.

Commandes utiles :

```powershell
python measure_precision.py --output precision_report.json --no-fail
python measure_precision.py --test-file test_questions.json --no-fail
```

### Interface interactive

```powershell
python query_docs.py
```

Les sessions sont stockees dans la base SQLite locale :

```text
hybrid_rag_sessions.db
```

Au premier lancement, si la base est vide et que `hybrid_rag_sessions.json` existe, `SessionManager` migre automatiquement les anciennes sessions JSON vers SQLite.

Schema principal :

```text
sessions(session_id, description, created, modified, global_context, indexed_entities)
turns(id, session_id, timestamp, question, answer, metadata, rag_context)
settings(key, value)
```

Au lancement, l'interface demande quelle session utiliser :

- choisir une session existante avec son numero ;
- appuyer sur `Entree` pour garder la session active ;
- saisir `N` pour creer une nouvelle session ;
- saisir directement un identifiant pour activer ou creer une session.

Pendant la conversation, les commandes utiles sont :

```text
session              affiche la session active
sessions             liste les sessions disponibles
switch <session_id>  change de session
history              affiche les derniers tours de la session
summary              recalcule et affiche le resume de la session
search-history <mot> cherche dans les questions/reponses de la session
export-session       exporte la session active en Markdown
export-session json  exporte la session active en JSON
/clear               vide l'historique de la session active
exit                 quitte l'interface
```

L'historique est utilise par `HybridRAG.query()` avant la recherche. Si une question depend du contexte precedent, `rag_conversation.py` reformule la question avec les derniers tours de la meme session.

La session garde aussi un resume deterministe dans `sessions.global_context` :

- sujets detectes depuis les dernieres questions ;
- sources documentaires recentes ;
- derniere question importante.

Ce resume est mis a jour apres chaque reponse. Les sources recentes peuvent donner un petit bonus au reranking, ce qui aide les questions de suivi a rester dans le bon contexte sans forcer la reponse.

Variables de configuration :

```text
RAG_ENABLE_SESSION_SUMMARY=true
RAG_ENABLE_SESSION_SOURCE_BOOST=true
RAG_SESSION_SOURCE_BOOST=0.04
```

Flux conversationnel :

```text
question de suivi
  -> historique de la session active
  -> resume de session
  -> reformulation contextuelle
  -> recherche graphe / FAISS / lexical
  -> bonus leger sur les sources recentes de la session
  -> fusion, reranking, generation extractive
```

Pour tester cette partie, utiliser `session_test_questions.json`. Chaque scenario contient trois questions a poser dans l'ordre dans la meme session. Les tours 2 et 3 doivent utiliser l'historique, par exemple les pronoms ou expressions comme `dedans`, `apres ca`, `ces parametres`.

Verifier rapidement la base SQLite :

```powershell
python -c "import sqlite3; con=sqlite3.connect('hybrid_rag_sessions.db'); print(con.execute('select count(*) from sessions').fetchone()[0], 'sessions'); print(con.execute('select count(*) from turns').fetchone()[0], 'tours')"
```

## 11. Etat Actuel Des Index

L'index actuel a ete regenere apres le passage en ingestion semantique.

Les artefacts importants sont :

```text
faiss_index.pkl
chunks_metadata.json
parent_chunks_metadata.json
networkx_graph.pkl
networkx_graph_sources.pkl
index_config.json
```

Verifier la configuration :

```powershell
Get-Content index_config.json
```

## 12. Points A Surveiller

### Encodage

Les fichiers CHM peuvent avoir des encodages anciens. L'ingestion utilise `UnicodeDammit`, mais il faut verifier les sorties si des textes comme `reprsente` ou `ncessite` apparaissent encore.

### Cross-encoder

Le reranker peut fonctionner en fallback si le backend n'est pas disponible.

Verifier les logs :

```text
Cross-encoder FlagEmbedding charge
Cross-encoder SentenceTransformers charge
Cross-encoder indisponible; fallback combined_score
```

### Evaluation

Un score bas peut venir de trois endroits differents :

1. retrieval : la bonne source n'est pas dans le top-k ;
2. reranking : la bonne source est trouvee mais mal classee ;
3. generation extractive : le bon passage est present mais la phrase extraite est mauvaise.

Il faut donc regarder :

- `recall_at_5`
- `recall_at_10`
- `mrr`
- `source_match`
- `keyword_overlap`
- `supported_answer_rate`

## 13. Fine-tuning

Le fine-tuning est possible, mais il faut choisir le bon niveau.

### Recommande en premier : embeddings ou reranker

Pour ce projet, le plus utile est de fine-tuner :

- le modele d'embeddings ;
- ou un reranker.

Format de donnees utile :

```json
{
  "question": "Qu'est-ce qu'un chemin Harmony ?",
  "positive_passage": "Les chemins Harmony permettent...",
  "negative_passages": [
    "La gestion des utilisateurs...",
    "Les impressions Windows..."
  ]
}
```

### Fine-tuning LLM

Possible, mais secondaire.

Le projet utilise surtout une generation extractive. Avant de fine-tuner un LLM, il faut d'abord stabiliser :

- l'ingestion ;
- le retrieval ;
- le reranking ;
- la verification.

Un LLM generateur est utile si l'objectif devient :

- reponses plus fluides ;
- synthese multi-sources ;
- reformulation professionnelle.

Il doit rester contraint par les sources.

## 14. Checklist De Maintenance

Avant livraison :

```powershell
python build_rag.py --compile
python run_precision.py
```

Apres modification de l'ingestion :

```powershell
python ingest_docs.py
python run_precision.py
```

Apres modification de `query_docs.py` ou des modules RAG :

```powershell
python test_rag_pipeline_components.py
python run_precision.py
```
