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

## 📁 Structure des données

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
      "turns": [
        {
          "timestamp": "2026-04-27T10:30:15",
          "question": "Comment utiliser Harmony?",
          "answer": "Harmony est une plateforme de gestion...",
          "metadata": {}
        }
      ]
    }
  }
}
```

## 🔑 Méthodes principales

### SessionManager

| Méthode | Description |
|---------|-------------|
| `create_session(id, description)` | Crée une nouvelle session |
| `delete_session(id)` | Supprime une session |
| `switch_session(id)` | Bascule vers une session |
| `list_sessions()` | Liste toutes les sessions |
| `current_session()` | Retourne la session active |
| `add_turn(question, answer)` | Ajoute un tour Q/R |
| `get_history(session_id, limit)` | Récupère l'historique |
| `search_history(query, session_id)` | Cherche dans l'historique |
| `export_session(session_id, format)` | Exporte une session |
| `clear_session(session_id)` | Efface l'historique |
| `get_session_info(session_id)` | Infos détaillées |
| `update_session_description(id, desc)` | Met à jour la description |

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
