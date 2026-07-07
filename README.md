# RAG Harmony/Divalto

Systeme RAG local pour interroger la documentation Harmony/Divalto extraite de fichiers CHM.

Le pipeline actif combine :

- recherche graphe NetworkX ;
- recherche vectorielle FAISS ;
- recherche lexicale BM25/TF-IDF ;
- fusion et parent-child retrieval ;
- MMR ;
- reranking cross-encoder si disponible ;
- compression contextuelle ;
- generation extractive ;
- reformulation LLM optionnelle et controlee ;
- verification documentaire ;
- sessions persistantes SQLite.

## Documentation

| Fichier | Contenu |
|---|---|
| `TECHNICAL_DOCUMENTATION.md` | Documentation technique globale du projet. |
| `SESSION_MANAGEMENT_DOCUMENTATION.md` | Documentation detaillee des sessions, historique, reformulation et SQLite. |
| `INGEST_DOCS_DOCUMENTATION.md` | Documentation de l'ingestion et du chunking. |
| `MEASURE_PRECISION_FUNCTIONS.md` | Documentation des fonctions d'evaluation. |

## Installation

```powershell
pip install -r requirements.txt
```

Pour utiliser BM25 :

```powershell
pip install rank_bm25
```

## Ingestion

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

## Requete Interactive

```powershell
python query_docs.py
```

Au lancement, l'application demande quelle session utiliser.

Commandes utiles dans l'interface :

```text
sessions
session
switch <session_id>
history
summary
search-history <mot>
export-session
/clear
exit
```

Les sessions sont stockees dans :

```text
hybrid_rag_sessions.db
```

## LLM

Le systeme peut fonctionner sans LLM generateur : il garde une reponse extractive locale.

Pour utiliser Ollama en local :

```powershell
$env:RAG_LLM_PROVIDER="ollama"
$env:RAG_OLLAMA_MODEL="llama3.2:1b"
python query_docs.py
```

Ou dans `.env` :

```env
RAG_LLM_PROVIDER=ollama
RAG_OLLAMA_MODEL=llama3.2:1b
```

Pour OpenAI/OpenRouter, utiliser les variables habituelles :

```env
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

Si le quota ou le reseau est indisponible, le systeme revient a la reponse extractive.

## Evaluation

Lancer l'evaluation interactive :

```powershell
python run_precision.py
```

Ou directement :

```powershell
python measure_precision.py --output precision_report.json --no-fail
```

Le fichier principal de test est :

```text
test_questions.json
```

Il contient des questions faciles, moyennes et difficiles reparties par modules.

## Tests

Compilation :

```powershell
python build_rag.py --compile
```

Tests composants :

```powershell
python -m unittest test_rag_pipeline_components.py
```

Tests session cibles :

```powershell
python -m unittest test_rag_pipeline_components.PipelineComponentTests.test_chat_memory_reformulation_and_clear test_rag_pipeline_components.PipelineComponentTests.test_hybrid_query_uses_memory_and_retries_once
```

## Fichiers Importants

| Fichier | Role |
|---|---|
| `ingest_docs.py` | Ingestion, chunking, FAISS, graphe. |
| `query_docs.py` | Pipeline de requete et interface interactive. |
| `session_manager.py` | Sessions SQLite et resume conversationnel. |
| `rag_conversation.py` | Reformulation conversationnelle deterministe. |
| `rag_answer.py` | Generation extractive et extraction ciblee. |
| `rag_llm_answer.py` | Reformulation LLM controlee, OpenAI/OpenRouter/Ollama. |
| `measure_precision.py` | Evaluation retrieval + reponse. |
| `run_precision.py` | Lanceur d'evaluation. |
