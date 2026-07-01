# Documentation explicative - ingest_docs.py

> Note : dans ce projet, le fichier d'ingestion principal s'appelle
> `ingest_docs.py`. Si on parle de `ingest.py`, on parle donc ici de
> `ingest_docs.py`.

## 1. Role du fichier

`ingest_docs.py` prepare toute la base documentaire utilisee ensuite par
`query_docs.py`.

Son objectif est de transformer les fichiers de documentation places dans
`data/` en index exploitables par le pipeline RAG :

```text
data/
  -> lecture HTML / Markdown
  -> nettoyage texte
  -> extraction title / section / module
  -> filtrage des pages inutiles
  -> creation parent chunks / child chunks
  -> embeddings
  -> index FAISS
  -> metadonnees JSON
  -> graphe NetworkX
```

Apres ingestion, le systeme peut faire :

- recherche vectorielle avec FAISS ;
- recherche graphe avec NetworkX ;
- parent-child retrieval ;
- reranking ;
- generation extractive avec sources.

## 2. Commande d'execution

Commande principale :

```powershell
python ingest_docs.py
```

Ou via le script de build :

```powershell
python build_rag.py --ingest
```

Pour compiler avant ingestion :

```powershell
python build_rag.py --compile
```

## 3. Entrees utilisees

Le dossier source est defini dans `rag_config.py` :

```python
DOCS_DIR = "data"
```

Le script lit :

- les fichiers `.htm` ;
- les fichiers `.html` ;
- les fichiers `.md`.

Ces fichiers peuvent etre organises par modules :

```text
data/
  Administration/
  Chemins/
  Xlog/
  zoom/
  ...
```

Le nom du premier dossier sous `data/` devient le champ `module`.

## 4. Sorties generees

L'ingestion regenere plusieurs fichiers importants :

| Fichier | Role |
|---|---|
| `faiss_index.pkl` | Index vectoriel FAISS des chunks enfants. |
| `chunks_metadata.json` | Metadonnees des chunks enfants. |
| `parent_chunks_metadata.json` | Metadonnees des chunks parents. |
| `networkx_graph.pkl` | Graphe NetworkX pour Graph RAG. |
| `networkx_graph_sources.pkl` | Copie du graphe avec sources. |
| `index_config.json` | Resume de la configuration utilisee pendant l'ingestion. |

Ces fichiers sont ensuite charges par `query_docs.py`.

## 5. Pipeline detaille

### Etape 1 - Lecture des documents

Fonction principale :

```python
load_documents(directory)
```

Elle cherche tous les fichiers HTML et Markdown dans `data/`.

Pour chaque HTML :

```python
read_file_as_unicode(file_path)
_documents_from_html_sections(file_path, html_content)
```

Pour chaque Markdown :

```python
normalize_text(...)
_source_metadata(...)
```

Objectif :

- lire les fichiers meme si l'encodage varie ;
- extraire le texte ;
- recuperer les metadonnees utiles : source, titre, section, module.

### Etape 2 - Nettoyage texte

Fonction :

```python
normalize_text(text)
```

Elle fait :

- decodage des entites HTML ;
- normalisation Unicode ;
- remplacement des espaces speciaux ;
- suppression des lignes vides ;
- nettoyage des espaces inutiles.

Cette etape evite que les embeddings et la recherche lexicale soient pollues
par des caracteres parasites.

### Etape 3 - Detection d'encodage

Fonction :

```python
read_file_as_unicode(file_path)
```

Elle lit le fichier en bytes, puis utilise `UnicodeDammit` pour detecter
l'encodage.

Fallbacks :

1. detection automatique ;
2. UTF-8 ;
3. Windows-1252 avec `errors="ignore"`.

C'est important pour les anciens fichiers CHM/HTML qui ne sont pas toujours
en UTF-8.

### Etape 4 - Extraction des metadonnees

Fonction :

```python
_source_metadata(file_path, title="", section="")
```

Elle produit un dictionnaire comme :

```json
{
  "source": "data/Administration/page.htm",
  "original_file_path": "C:/.../data/Administration/page.htm",
  "title": "Titre de la page",
  "section": "Section courante",
  "module": "Administration"
}
```

Ces metadonnees sont tres importantes pour :

- retrouver la bonne source ;
- booster les recherches par titre ;
- calculer `source_match` dans `measure_precision.py` ;
- afficher les sources dans la reponse finale.

### Etape 5 - Decoupage HTML par sections

Fonction :

```python
_documents_from_html_sections(file_path, html_content)
```

Elle utilise BeautifulSoup pour lire :

- `title` ;
- `h1` ;
- `h2` ;
- `h3` ;
- `p` ;
- `li` ;
- `pre` ;
- `td` ;
- `th` ;
- `div`.

Le but est de creer des documents par section logique avant le chunking.

Exemple :

```text
Page HTML
  h1: Installation
  h2: Prerequis
  p: ...
  h2: Configuration
  p: ...
```

Devient :

```text
Document 1 -> section Prerequis
Document 2 -> section Configuration
```

Cette logique rend le chunking plus semantique qu'un simple decoupage par
taille.

## 6. Chunking

La strategie est configuree dans `rag_config.py` :

```python
CHUNK_STRATEGY = "semantic"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
```

Fonction :

```python
chunk_documents(documents)
```

Strategies possibles :

| Strategie | Description |
|---|---|
| `semantic` | Utilise `SemanticChunker` si disponible. |
| `html` | Utilise les titres HTML `h1/h2/h3`, puis split recursive. |
| `recursive` | Decoupe simple par taille avec overlap. |

Si `semantic` echoue, le script bascule automatiquement vers :

```text
html + recursive
```

Si le split HTML echoue aussi, il bascule vers :

```text
recursive uniquement
```

## 7. Parent-child retrieval

Le projet utilise un decoupage parent/enfant.

Configuration :

```python
RAG_ENABLE_PARENT_CHILD_RETRIEVAL = True
RAG_PARENT_CHUNK_SIZE = 1000
RAG_PARENT_CHUNK_OVERLAP = 200
RAG_CHILD_CHUNK_SIZE = 250
RAG_CHILD_CHUNK_OVERLAP = 50
```

Fonction :

```python
create_parent_child_chunks(documents)
```

Principe :

```text
Document
  -> parent chunk large pour la reponse
      -> child chunk court pour la recherche
```

Pourquoi ?

- Les child chunks sont plus precis pour FAISS/retrieval.
- Les parent chunks donnent plus de contexte pour la reponse finale.

Exemple :

```text
parent_000123
  child_000901
  child_000902
  child_000903
```

Pendant la requete, le systeme retrouve d'abord un child chunk, puis remonte
vers son parent.

## 8. Texte indexe dans FAISS

Fonction :

```python
_build_index_text(document)
```

Le texte envoye a l'embedding n'est pas seulement le contenu brut. Il est
enrichi avec :

```text
Module: ...
Titre: ...
Section: ...
Fichier: ...
Contenu: ...
```

Cela aide beaucoup les questions de type :

```text
Que couvre la page "..." dans le module ... ?
```

Le modele d'embedding peut ainsi prendre en compte le titre, le module et le
nom du fichier.

## 9. Filtrage des documents

Fonction :

```python
is_valid_document(document)
```

Elle garde :

- les documents avec plus de 300 caracteres ;
- ou les petites pages CHM si elles ont quand meme un titre, une section, une
  source, un module et un minimum de texte.

But :

- supprimer les pages vides ou purement decoratives ;
- garder les pages courtes mais utiles, par exemple une page avec un titre
  important.

## 10. Creation de l'index FAISS

Fonction :

```python
create_vector_embeddings(chunks, parent_chunks=None)
```

Elle fait :

1. charge le modele `SentenceTransformer` ;
2. construit le texte enrichi avec `_build_index_text` ;
3. ajoute le prefixe `passage:` si le modele est de type E5 ;
4. genere les embeddings ;
5. normalise les vecteurs ;
6. cree un index FAISS `IndexFlatIP` ;
7. sauvegarde l'index dans `faiss_index.pkl`.

Modele par defaut :

```python
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
```

FAISS utilise ici `IndexFlatIP`.

Comme les embeddings sont normalises :

```python
normalize_embeddings=True
```

le produit scalaire correspond a une similarite cosinus.

## 11. Metadonnees des chunks

Toujours dans :

```python
create_vector_embeddings(...)
```

Le script sauvegarde `chunks_metadata.json`.

Chaque chunk enfant contient notamment :

```json
{
  "chunk_id": "child_000001",
  "child_id": "child_000001",
  "parent_id": "parent_000001",
  "document_id": "Document_1",
  "text": "...",
  "index_text": "...",
  "parent_text": "...",
  "source": "...",
  "title": "...",
  "section": "...",
  "module": "...",
  "embedding_model": "...",
  "chunk_strategy": "semantic_parent_child"
}
```

Ces metadonnees permettent a `query_docs.py` de :

- afficher les sources ;
- retrouver le parent d'un child chunk ;
- appliquer les boosts de titre/module/source ;
- calculer les scores de precision.

## 12. Metadonnees parents

Fichier :

```text
parent_chunks_metadata.json
```

Il contient les chunks parents :

```json
{
  "parent_id": "parent_000001",
  "text": "...",
  "source": "...",
  "title": "...",
  "section": "...",
  "module": "..."
}
```

Ces chunks sont plus longs et servent surtout pour :

- generation extractive ;
- verification documentaire ;
- affichage du contexte final.

## 13. Configuration d'index

Fichier :

```text
index_config.json
```

Il sauvegarde :

- modele d'embedding ;
- dimension des vecteurs ;
- strategie de chunking ;
- activation parent-child ;
- nombre de chunks enfants ;
- nombre de chunks parents ;
- tailles et overlaps.

Ce fichier est utile pour verifier que l'index correspond bien a la
configuration actuelle.

## 14. Creation du graphe NetworkX

Fonction :

```python
create_graph(chunks)
```

Elle cree un graphe dirige :

```text
Document_x -> Entite
```

Chaque chunk devient un noeud document :

```text
Document_0
Document_1
Document_2
...
```

Chaque entite detectee devient un noeud :

```text
Harmony
XLAN
ODBC
LDAP
Zoom
...
```

Puis le graphe ajoute une relation :

```text
Document_0 --CONTIENT--> ODBC
```

Avec un poids :

```python
weight = _entity_weight(entity, chunk)
```

## 15. Extraction d'entites

Fonction :

```python
extract_entities_rule_based(text)
```

Elle combine deux approches :

1. liste de termes connus :

```python
KEY_TERMS = ["Harmony", "XLAN", "RecordSQL", "ODBC", ...]
```

2. detection de mots techniques commencant par une majuscule :

```python
r"\b[A-Z][A-Za-z0-9_.-]{2,}\b"
```

Cela permet de construire le graphe sans LLM.

## 16. Poids des entites

Fonction :

```python
_entity_weight(entity, chunk)
```

Le poids augmente si :

- l'entite apparait plusieurs fois dans le texte ;
- l'entite apparait dans le titre, la section ou le module.

Donc une entite presente dans un titre est consideree plus importante.

## 17. Fonctionnement du main

Bloc :

```python
if __name__ == "__main__":
```

Execution complete :

```text
1. Charger les documents depuis data/
2. Filtrer les documents vides
3. Creer les chunks parents/enfants
4. Creer l'index FAISS
5. Creer le graphe NetworkX
6. Sauvegarder tous les fichiers
```

## 18. Variables importantes dans rag_config.py

| Variable | Role |
|---|---|
| `DOCS_DIR` | Dossier source des documents. |
| `EMBEDDING_MODEL` | Modele d'embeddings. |
| `EMBEDDING_BATCH_SIZE` | Taille de batch pour encoder les textes. |
| `CHUNK_STRATEGY` | Strategie de chunking. |
| `CHUNK_SIZE` | Taille des chunks legacy/html. |
| `CHUNK_OVERLAP` | Overlap des chunks legacy/html. |
| `RAG_ENABLE_PARENT_CHILD_RETRIEVAL` | Active parent-child retrieval. |
| `RAG_PARENT_CHUNK_SIZE` | Taille des parents. |
| `RAG_PARENT_CHUNK_OVERLAP` | Overlap parents. |
| `RAG_CHILD_CHUNK_SIZE` | Taille des enfants. |
| `RAG_CHILD_CHUNK_OVERLAP` | Overlap enfants. |
| `SEMANTIC_BREAKPOINT_THRESHOLD` | Seuil du chunking semantique. |

## 19. Quand relancer l'ingestion ?

Il faut relancer :

```powershell
python ingest_docs.py
```

quand :

- tu ajoutes des fichiers dans `data/` ;
- tu supprimes des fichiers dans `data/` ;
- tu modifies le contenu des fichiers source ;
- tu changes le modele `EMBEDDING_MODEL` ;
- tu changes la strategie ou la taille des chunks ;
- tu modifies le parent-child retrieval ;
- tu modifies fortement la logique d'extraction des metadonnees.

Pas besoin de relancer l'ingestion quand tu modifies seulement :

- le reranking ;
- le MMR ;
- le verifier ;
- la generation extractive ;
- `measure_precision.py`.

## 20. Problemes frequents

### FAISS ou sentence-transformers manquant

Erreur typique :

```text
L'indexation vectorielle requiert sentence-transformers et faiss-cpu.
```

Solution :

```powershell
pip install -r requirements.txt
```

### Ingestion tres lente

Ca peut venir de :

- `intfloat/multilingual-e5-large` ;
- `SemanticChunker` ;
- nombre important de fichiers HTML ;
- CPU uniquement.

Solutions possibles :

- reduire `EMBEDDING_BATCH_SIZE` si memoire insuffisante ;
- passer temporairement a `CHUNK_STRATEGY=html` ;
- utiliser un modele plus leger comme `intfloat/multilingual-e5-base`.

### Mauvais resultats sur les questions de titre

Verifier dans `chunks_metadata.json` :

- `title` ;
- `section` ;
- `module` ;
- `source`.

Si ces champs sont vides ou incorrects, le retrieval par titre sera moins bon.

### Les sources ne correspondent pas dans precision

Verifier :

- le champ `source` dans `chunks_metadata.json` ;
- les `source_files` dans `test_questions.json` ;
- les accents ou caracteres speciaux dans les noms de fichiers.

## 21. Verification rapide apres ingestion

Verifier la config :

```powershell
python -c "import json; print(json.load(open('index_config.json', encoding='utf-8')))"
```

Verifier le nombre de chunks :

```powershell
python -c "import json; print(len(json.load(open('chunks_metadata.json', encoding='utf-8'))))"
```

Verifier les parents :

```powershell
python -c "import json; print(len(json.load(open('parent_chunks_metadata.json', encoding='utf-8'))))"
```

Lancer une evaluation courte :

```powershell
python measure_precision.py --facile --limit 3 --output precision_ingest_check.json --no-fail
```

## 22. Resume court

`ingest_docs.py` est responsable de construire la memoire documentaire du
projet.

Il ne repond pas aux questions directement. Il prepare :

- l'index vectoriel FAISS ;
- les metadonnees des chunks ;
- les chunks parents ;
- le graphe NetworkX ;
- la configuration d'index.

La qualite du RAG depend fortement de cette etape, car une mauvaise ingestion
donne ensuite un mauvais retrieval, meme si `query_docs.py` est bien regle.
