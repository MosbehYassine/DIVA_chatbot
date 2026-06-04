# 📋 Session Management Guide

Système de gestion de sessions persistantes pour le RAG Harmony. Permet de conserver l'historique complet des conversations Q/R.

## 🚀 Démarrage rapide

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Utilisation en CLI

#### Lister les sessions
```bash
python manage_sessions.py list
```

#### Créer une nouvelle session
```bash
python manage_sessions.py create -n "ma_session" -d "Description optionnelle"
```

#### Basculer vers une session
```bash
python manage_sessions.py switch -n "ma_session"
```

#### Voir l'historique
```bash
python manage_sessions.py history --limit 10
```

#### Chercher dans l'historique
```bash
python manage_sessions.py search "harmony utilisateur"
```

#### Exporter une session
```bash
# En JSON
python manage_sessions.py export -n "ma_session" -f json -o session.json

# En Markdown
python manage_sessions.py export -n "ma_session" -f markdown -o session.md

# En texte simple
python manage_sessions.py export -n "ma_session" -f txt -o session.txt
```

#### Supprimer une session
```bash
python manage_sessions.py delete -n "ma_session" -y
```

## 🔧 Utilisation Programmatique

### Exemple basique

```python
from session_manager import SessionManager

# Initialiser le gestionnaire de sessions
sm = SessionManager()

# Créer une nouvelle session
sm.create_session("projet_harmony", "Documentation du projet Harmony")

# Basculer vers la session
sm.switch_session("projet_harmony")

# Ajouter un tour Q/R
sm.add_turn(
    question="Comment gérer les utilisateurs dans Harmony?",
    answer="La gestion des utilisateurs est assurée par le programme Xlog1.dhop"
)

# Récupérer l'historique
history = sm.get_history(limit=5)
for turn in history:
    print(f"Q: {turn['question']}")
    print(f"R: {turn['answer']}")
    print(f"Temps: {turn['timestamp']}\n")

# Chercher dans l'historique
results = sm.search_history("Xlog", "projet_harmony")
print(f"Trouvé {len(results)} résultats")
```

### Intégration avec le RAG

```python
from session_manager import SessionManager
from answer_questions import VectorStore

# Initialiser
sm = SessionManager()
vector_store = VectorStore()
vector_store.load()

# Créer une session pour la conversation
sm.create_session("conversation_001", "Conversation avec l'utilisateur")
sm.switch_session("conversation_001")

# Traiter les questions
question = "Qu'est-ce que Xlog?"
# ... traiter la question avec le RAG ...
answer = "Xlog est un système de gestion des utilisateurs..."

# Sauvegarder dans la session
sm.add_turn(question, answer)

# Récupérer l'historique si nécessaire
previous_context = sm.get_history(limit=10)
```

### 🎯 Gestion du Contexte RAG

#### Créer une session avec contexte global

```python
from session_manager import SessionManager

sm = SessionManager()

# Créer une session avec contexte initial
sm.create_session(
    session_id="support_harmony",
    description="Support utilisateurs Harmony",
    context="Contexte: Documentation d'administration, guides de dépannage"
)
```

#### Stocker le contexte RAG dans chaque tour

```python
# Ajouter un tour avec le contexte RAG (documents retrouvés, scores, etc)
sm.add_turn(
    question="Où stocker le fichier Xlogf?",
    answer="Le fichier Xlogf doit être stocké dans le répertoire data...",
    rag_context={
        "sources": [
            {
                "document_id": "doc_123",
                "text_preview": "Fichier de configuration Xlogf..."
            }
        ],
        "scores": [0.85],
        "method": "hybrid",  # graph, vector, ou hybrid
        "entities": ["Xlogf", "stockage"],
        "documents_count": 1
    }
)
```

#### Gérer le contexte de session

```python
# Définir le contexte global d'une session
sm.set_session_context(
    session_id="support_harmony",
    context="Sujet: Administration système, Focus: Gestion des fichiers utilisateurs"
)

# Ajouter des entités indexées
sm.add_indexed_entities(
    entities=["Xlogf", "Xlog1", "utilisateurs", "stockage"],
    session_id="support_harmony"
)

# Récupérer le contexte de la session
context = sm.get_session_context("support_harmony")
print(f"Contexte global: {context['global_context']}")
print(f"Entités connues: {context['indexed_entities']}")
```

#### Récupérer et exporter le contexte

```python
# Récupérer le contexte d'un tour spécifique
turn_context = sm.get_turn_context(0, "support_harmony")
print(f"Méthode RAG: {turn_context['method']}")
print(f"Documents trouvés: {turn_context['documents_count']}")

# Exporter toute la session avec contexte
json_export = sm.export_session_context("support_harmony", format="json")
markdown_export = sm.export_session_context("support_harmony", format="markdown")

with open("session_context.json", "w") as f:
    f.write(json_export)
```

### Intégration complète RAG + Sessions

```python
from rag_with_sessions import RAGWithSessions

# Initialiser le système complet
rag_sessions = RAGWithSessions()

# Créer une session avec contexte
rag_sessions.create_session_with_context(
    session_id="admin_support",
    description="Support administration",
    context="Documentation: Administration système Harmony"
)

# Poser une question (le contexte RAG est automatiquement stocké!)
result = rag_sessions.query_with_context(
    question="Où peut être stocké le fichier des utilisateurs Xlogf?",
    session_id="admin_support",
    store_context=True
)

print(f"Question: {result['question']}")
print(f"Réponse: {result['answer']}")
print(f"Documents trouvés: {result['rag_context']['documents_count']}")
print(f"Méthode: {result['rag_context']['method']}")

# Exporter la session complète
export = rag_sessions.export_session_with_context("admin_support", format="markdown")
print(export)
```

## 📁 Structure des données avec contexte

Les sessions sont stockées dans `rag_sessions.json` avec la structure suivante:

```json
{
  "active_session": "default",
  "sessions": {
    "default": {
      "metadata": {
        "created": "2026-04-27T10:30:00",
        "modified": "2026-04-27T11:45:00",
        "description": "Session par défaut"
      },
      "context": {
        "global_context": "Contexte général de la session",
        "indexed_entities": ["Xlog", "Xlogf", "utilisateurs"]
      },
      "turns": [
        {
          "timestamp": "2026-04-27T10:30:15",
          "question": "Où peut être stocké le fichier Xlogf?",
          "answer": "Le fichier Xlogf doit être stocké...",
          "rag_context": {
            "sources": [
              {
                "document_id": "doc_456",
                "text_preview": "Fichier de configuration Xlogf pour...",
                "method": "hybrid"
              }
            ],
            "scores": [0.87],
            "method": "hybrid",
            "entities": ["Xlogf", "stockage", "configuration"],
            "documents_count": 1
          },
          "metadata": {
            "graph_results": 2,
            "vector_results": 5
          }
        }
      ]
    }
  }
}
```

## 📊 Métadonnées du contexte RAG

Chaque `rag_context` contient:

| Champ | Type | Description |
|-------|------|-------------|
| `sources` | Array | Liste des documents retrouvés avec `document_id`, `text_preview`, et `method` |
| `scores` | Array | Scores de pertinence (0-1) pour chaque document |
| `method` | String | Méthode de recherche utilisée (`graph`, `vector`, `hybrid`) |
| `entities` | Array | Entités extraites et matchées du texte |
| `documents_count` | Number | Nombre total de documents retrouvés |

## 🔑 Méthodes principales

### SessionManager

| Méthode | Description |
|---------|-------------|
| `create_session(id, description, context)` | Crée une nouvelle session avec contexte optionnel |
| `delete_session(id)` | Supprime une session |
| `switch_session(id)` | Bascule vers une session |
| `list_sessions()` | Liste toutes les sessions |
| `current_session()` | Retourne la session active |
| `add_turn(question, answer, metadata, rag_context)` | Ajoute un tour Q/R avec contexte RAG |
| `get_history(session_id, limit)` | Récupère l'historique |
| `get_session_context(session_id)` | Récupère le contexte global de la session |
| `set_session_context(session_id, context)` | Définit le contexte global |
| `add_indexed_entities(entities, session_id)` | Ajoute des entités indexées |
| `get_turn_context(turn_index, session_id)` | Récupère le contexte d'un tour |
| `export_session_context(session_id, format)` | Exporte session + contexte complet |
| `get_session_info(session_id)` | Infos détaillées avec contexte |
| `search_history(query, session_id)` | Cherche dans l'historique |
| `export_session(session_id, format)` | Exporte une session |
| `clear_session(session_id)` | Efface l'historique |
| `update_session_description(id, desc)` | Met à jour la description |

### RAGWithSessions

| Méthode | Description |
|---------|-------------|
| `query_with_context(question, session_id, store_context)` | Exécute une requête RAG et stocke le contexte |
| `create_session_with_context(id, description, context)` | Crée une session avec contexte pour le RAG |
| `set_session_context(context, session_id)` | Définit le contexte |
| `get_session_with_context(session_id)` | Récupère session + contexte |
| `list_sessions_overview()` | Aperçu de toutes les sessions |
| `export_session_with_context(session_id, format)` | Exporte session + contexte RAG |

## 📊 Formats d'export

### JSON
Structure complète avec métadonnées et tous les tours.

### Markdown
Format lisible avec sections pour chaque tour, idéal pour la documentation.

### Texte
Format simple, facile à lire et traiter par d'autres outils.

## 🔐 Variable d'environnement

```bash
# Définir le chemin du fichier de sessions
export RAG_SESSIONS_FILE="/chemin/custom/sessions.json"
```

## 💡 Bonnes pratiques

1. **Nommage des sessions**: Utilisez des noms parlants (ex: "projet_harmony_v1", "test_mai_2026")
2. **Descriptions**: Ajoutez des descriptions pour identifier le contexte
3. **Export régulier**: Exportez les sessions importantes en Markdown pour archivage
4. **Recherche**: Utilisez la recherche pour retrouver des questions/réponses antérieures
5. **Nettoyage**: Supprimez les sessions obsolètes pour optimiser les performances

## ⚠️ Limitations

- La session "default" ne peut pas être supprimée
- Le fichier `rag_sessions.json` peut devenir volumineux avec beaucoup de sessions
- Pas de chiffrement des données (sensibilité des données à considérer)
- Les métadonnées des tours (metadata) sont optionnelles

## 🔄 Migration depuis l'ancien format

Le système migre automatiquement les anciennes sessions du format liste vers le nouveau format avec métadonnées à la première utilisation.
