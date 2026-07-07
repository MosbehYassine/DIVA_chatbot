# Documentation Technique - Session Management

## 1. Objectif

La couche session permet au RAG de conserver l'historique des questions/reponses et d'utiliser ce contexte pour les questions de suivi.

Elle sert a :

- choisir une session au lancement de `query_docs.py` ;
- stocker les questions, reponses, sources et metadonnees dans SQLite ;
- reformuler les questions dependantes de l'historique ;
- maintenir un resume de session exploitable par le retrieval ;
- booster legerement les sources et sujets recents ;
- auditer les reponses via la base `hybrid_rag_sessions.db`.

Le systeme reste local. La reformulation conversationnelle est deterministe et ne depend pas d'un LLM.

## 2. Fichiers Concernes

| Fichier | Role |
|---|---|
| `session_manager.py` | Gestion SQLite/JSON des sessions, historique, resume structure et sources recentes. |
| `rag_conversation.py` | Reformulation conversationnelle deterministe et score de confiance. |
| `query_docs.py` | Integration de la session dans `HybridRAG.query()`, boost session, stockage des metadonnees. |
| `session_test_questions.json` | Scenarios manuels pour tester les questions liees. |
| `hybrid_rag_sessions.db` | Base SQLite locale active. |
| `hybrid_rag_sessions.json` | Ancien stockage JSON, utilise comme source de migration si necessaire. |

## 3. Stockage SQLite

La base active est :

```text
hybrid_rag_sessions.db
```

Schema principal :

```text
sessions(
  session_id TEXT PRIMARY KEY,
  description TEXT,
  created TEXT,
  modified TEXT,
  global_context TEXT,
  indexed_entities TEXT
)

turns(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  timestamp TEXT,
  question TEXT,
  answer TEXT,
  metadata TEXT,
  rag_context TEXT
)

settings(
  key TEXT PRIMARY KEY,
  value TEXT
)
```

`metadata` et `rag_context` sont des champs JSON. Cela evite de modifier le schema quand on ajoute de nouvelles informations.

## 4. Cycle De Vie D'une Session

Au lancement de `query_docs.py` :

1. `SessionManager` charge `hybrid_rag_sessions.db`.
2. Si la base est vide, les anciennes sessions JSON peuvent etre migrees depuis `hybrid_rag_sessions.json`.
3. L'interface demande quelle session utiliser.
4. `HybridRAG` recoit le `SessionManager`.
5. Chaque appel a `rag.query(..., session_id=...)` lit l'historique de la session active.
6. Apres la reponse, le tour est stocke et le resume de session est rafraichi.

Flux simplifie :

```text
question utilisateur
  -> historique de session
  -> reformulation conversationnelle si necessaire
  -> retrieval hybride
  -> generation/verfication de reponse
  -> stockage SQLite
  -> refresh du resume de session
```

## 5. Reformulation Conversationnelle

Le module `rag_conversation.py` expose deux fonctions :

```python
reformulate_with_history(question, history) -> str
reformulate_with_history_info(question, history) -> ReformulationResult
```

`reformulate_with_history()` garde la compatibilite avec l'ancien code.

`reformulate_with_history_info()` retourne :

```python
ReformulationResult(
    question="Comment gerer les utilisateurs dans le module Administration dans Harmony/Divalto ?",
    history_used=True,
    confidence=0.90,
    module="Administration",
    topic="utilisateurs",
)
```

### Exemples

| Question de suivi | Historique | Reformulation |
|---|---|---|
| `Et comment gerer les utilisateurs dedans ?` | module Administration | `Comment gerer les utilisateurs dans le module Administration dans Harmony/Divalto ?` |
| `Quels droits faut-il verifier avant de modifier ces parametres ?` | module Administration | `Quels droits faut-il verifier avant de modifier les parametres du module Administration dans Harmony/Divalto ?` |
| `et comment imprimer ?` | facture | `Comment imprimer la facture dans Harmony/Divalto ?` |
| `Et comment la modifier ?` | facture | `Comment modifier la facture dans Harmony/Divalto ?` |
| `Comment les configurer ?` | chemins Harmony | `Comment configurer les chemins Harmony ?` |

### Score De Confiance

Le score `reformulation_confidence` est deterministe :

- base si une question de suivi est detectee ;
- bonus si un module est retrouve ;
- bonus si un sujet est retrouve ;
- bonus leger pour certains cas directs comme `imprimer`.

Il sert a l'audit et peut etre utilise plus tard pour choisir entre question originale et question reformulee.

## 6. Metadonnees Stockees Par Tour

Apres chaque reponse, `query_docs.py` stocke dans `turns.metadata` :

```json
{
  "standalone_question": "Comment gerer les utilisateurs dans le module Administration dans Harmony/Divalto ?",
  "history_used": true,
  "reformulation_confidence": 0.9,
  "reformulation_focus": {
    "module": "Administration",
    "topic": "utilisateurs"
  },
  "session_recent_sources": [
    "Administrationd_Harmony__Introduction.htm"
  ],
  "verification": {
    "answer_supported": true,
    "confidence": 0.95
  },
  "session_summary": "...",
  "reasoning_plan": [...]
}
```

Les sources sont stockees dans `turns.rag_context` :

```json
{
  "sources": [
    {
      "filename": "Administrationd_Harmony__Introduction.htm",
      "module": "Administration",
      "title": "Administration d'Harmony: Introduction"
    }
  ]
}
```

## 7. Resume Structure De Session

`SessionManager.refresh_session_memory()` appelle `build_session_summary()`.

Le resume stocke dans `sessions.global_context` contient maintenant :

```text
Module actif: Administration
Sujet actif: utilisateurs
Derniere intention: gestion
Sujets: gerer, utilisateurs, module, administration
Sources recentes: Administrationd_Harmony__Introduction.htm
Question autonome: Comment gerer les utilisateurs dans le module Administration dans Harmony/Divalto ?
Derniere question: Et comment gerer les utilisateurs dedans ?
```

Les entites indexees dans `sessions.indexed_entities` incluent :

- sujets frequents ;
- module actif ;
- sujet actif ;
- intention recente.

Ces valeurs sont reutilisees par `query_docs.py` pendant le retrieval.

## 8. Boost Session Dans Le Retrieval

`HybridRAG._apply_session_source_boost()` applique un bonus doux.

Il regarde :

- les fichiers sources recents ;
- les entites indexees de session ;
- le focus de reformulation : module/sujet.

Le boost reste faible pour eviter de forcer une mauvaise source :

```text
RAG_SESSION_SOURCE_BOOST=0.04
```

Si une source ou un module recent correspond au candidat, le score hybride est augmente legerement.

## 9. Commandes CLI

Dans `query_docs.py`, les commandes utiles sont :

```text
session              affiche la session active
sessions             liste les sessions disponibles
switch <session_id>  change de session
history              affiche les derniers tours de la session
summary              recalcule et affiche le resume de la session
search-history <mot> cherche dans la session
export-session       exporte la session active en Markdown
export-session json  exporte la session active en JSON
/clear               vide l'historique de la session active
exit                 quitte l'interface
```

Au lancement, l'interface demande quelle session choisir.

## 10. Commandes SQL Utiles

Voir les sessions :

```powershell
python -c "import sqlite3; con=sqlite3.connect('hybrid_rag_sessions.db'); print(con.execute('select session_id, description, modified from sessions').fetchall())"
```

Voir les derniers tours :

```powershell
python -c "import sqlite3; con=sqlite3.connect('hybrid_rag_sessions.db'); rows=con.execute('select session_id, timestamp, question, substr(answer,1,120) from turns order by id desc limit 10').fetchall(); print(rows)"
```

Voir les metadonnees de reformulation :

```powershell
python -c "import sqlite3, json; con=sqlite3.connect('hybrid_rag_sessions.db'); rows=con.execute('select question, metadata from turns order by id desc limit 3').fetchall(); [print(q, json.loads(m or '{}')) for q,m in rows]"
```

## 11. Configuration

| Variable | Role | Defaut |
|---|---|---|
| `RAG_ENABLE_CHAT_MEMORY` | Active la memoire conversationnelle | `true` |
| `RAG_CHAT_HISTORY_TURNS` | Nombre de tours lus pour reformuler | `5` |
| `RAG_ENABLE_CONVERSATION_REFORMULATION` | Active la reformulation deterministe | `true` |
| `RAG_ENABLE_SESSION_SUMMARY` | Active le resume de session | `true` |
| `RAG_ENABLE_SESSION_SOURCE_BOOST` | Active le boost sources/sujets recents | `true` |
| `RAG_SESSION_SOURCE_BOOST` | Intensite du boost session | `0.04` |
| `RAG_SESSIONS_DB` | Chemin optionnel de la base SQLite | `hybrid_rag_sessions.db` |
| `RAG_SESSIONS_FILE` | Chemin optionnel du JSON legacy | `hybrid_rag_sessions.json` |

## 12. Tests

Tests unitaires rapides :

```powershell
python -m unittest test_rag_pipeline_components.PipelineComponentTests.test_chat_memory_reformulation_and_clear test_rag_pipeline_components.PipelineComponentTests.test_hybrid_query_uses_memory_and_retries_once
```

Compilation :

```powershell
python build_rag.py --compile
```

Tests manuels :

```text
1. python query_docs.py
2. choisir ou creer une session
3. poser une question initiale
4. poser une question liee avec dedans / ces parametres / la modifier / les configurer
5. verifier history_used, les sources et la reponse
```

Le fichier `session_test_questions.json` contient des scenarios de test par theme.

## 13. Limites Connues

- La reformulation est volontairement deterministe, donc elle peut produire des phrases un peu mecaniques.
- Un changement brutal de sujet peut encore garder un petit boost de l'ancien contexte, mais le boost est faible.
- Les questions tres elliptiques comme `Et local ?` restent difficiles sans LLM.
- La qualite finale depend toujours du retrieval et des sources disponibles.

## 14. Evolution Possible

Ameliorations futures possibles :

- utiliser `reformulation_confidence` pour ignorer une reformulation faible ;
- ajouter un detecteur de changement de sujet ;
- ajouter un fallback LLM local Ollama uniquement si la confiance est basse ;
- afficher les metadonnees session dans une commande CLI dediee ;
- ajouter une vue SQLite/tableau pour inspecter les sessions.
