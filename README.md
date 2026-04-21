# Chatbot RAG Harmony - OpenRouter Version

Ce projet est un chatbot RAG (Retrieval-Augmented Generation) basé sur la documentation Harmony/Divalto extraite de fichiers CHM.

## Configuration

### Variables d'environnement requises

#### Méthode 1: Fichier .env (recommandé)
Créez un fichier `.env` dans le répertoire du projet :
```
OPENROUTER_API_KEY=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
```

#### Méthode 2: Variable d'environnement système (Windows)
1. Rechercher "variables d'environnement" dans le menu Démarrer
2. Cliquer sur "Variables d'environnement"
3. Dans "Variables utilisateur", cliquer sur "Nouvelle..."
4. Nom: `OPENROUTER_API_KEY`
5. Valeur: `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`

#### Méthode 3: PowerShell (session temporaire)
```powershell
$env:OPENROUTER_API_KEY = "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"
```

### Modèle utilisé

Le chatbot utilise le modèle `qwen/qwen-2.5-72b-instruct` via l'API OpenRouter.

## Installation et utilisation

### Avec Docker (recommandé)

1. Construire l'image :
```bash
docker build -t diva-chatbot .
```

2. Lancer le conteneur :
```bash
# Avec fichier .env
docker run -it --env-file .env diva-chatbot

# Ou avec variable d'environnement
docker run -it -e OPENROUTER_API_KEY="votre_clé" diva-chatbot
```

### Sans Docker

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancer l'ingestion des documents :
```bash
python ingest_docs.py
```

3. Lancer le chatbot :
```bash
python query_docs.py
```

## Fonctionnement

Le système propose plusieurs approches pour traiter et interroger la documentation :

### Architecture RAG Modulaire (Recommandé)
- `process_documents.py` : Charge et traite les documents, crée l'index FAISS
- `answer_questions.py` : Interface interactive pour poser des questions et obtenir des réponses générées

### GraphRAG (Alternative)
- `ingest_docs.py` : Charge les documents et crée un graphe de connaissances NetworkX
- `query_docs.py` : Interface de requête basée sur le graphe pour recherche d'entités

### Pipeline Complet (Legacy)
- `rag_pipeline.py` : Pipeline complet en un seul script

## Workflows d'utilisation

### Workflow RAG Vectoriel (Recommandé)
```bash
# Étape 1: Indexer les documents (une seule fois)
python process_documents.py

# Étape 2: Poser des questions (interface interactive)
python answer_questions.py
```

**Fonctionnalités de `process_documents.py` :**
- 📂 Analyse récursive de tous les dossiers de documentation
- 📊 Statistiques détaillées du traitement (documents, chunks, index)
- 🧠 Construction d'index vectoriel FAISS optimisé
- 💾 Sauvegarde automatique dans `faiss_index/`

**Fonctionnalités de `answer_questions.py` :**
- 💬 Interface de chat avec historique des questions
- 📊 Statistiques de l'index (`stats`)
- 🔄 Rechargement à chaud (`reload`)
- 📖 Aide intégrée (`help`)
- 🧹 Effacement d'écran (`clear`)
- 📜 Historique des questions (`history`)

### Workflow GraphRAG
```bash
# Étape 1: Créer le graphe de connaissances
python ingest_docs.py

# Étape 2: Interroger le graphe
python query_docs.py
```

### Workflow Legacy
```bash
# Tout en un (recharge les documents à chaque fois)
python rag_pipeline.py
```

2. **Évaluation des réponses** : Après génération, l'utilisateur peut :
   - Noter la réponse (1-5 étoiles)
   - Fournir des commentaires
   - Corriger la réponse si nécessaire

3. **Analyse du feedback** : Le système enregistre toutes les interactions pour analyse future.

### Utilisation

#### Mode interactif normal
```bash
python query_docs.py
```

### Utilisation avancée
```bash
# Indexer les documents
python process_documents.py

# Poser des questions interactivement
python answer_questions.py
```

## Tests et validation

Un jeu complet de tests a été créé pour valider le système :

### Fichiers de test
- `test_questions.json` : 20 questions de test avec réponses attendues
- `test_runner.py` : Script d'exécution des tests automatisés
- `TEST_README.md` : Documentation détaillée des tests

### Exécution des tests
```bash
# Résumé des tests disponibles
python test_runner.py

# Exécution des tests automatisés
python test_runner.py run
```

### Couverture des tests
- **20 questions** réparties en 9 catégories
- Niveaux de difficulté : Facile (30%), Moyen (50%), Difficile (20%)
- Sujets : Chemins Harmony, Utilisateurs, Impressions, LDAP, etc.

## Changements récents

- Refactorisation en architecture modulaire (process_documents.py + answer_questions.py)
- Utilisation de FAISS pour la recherche vectorielle
- Génération de réponses avec GPT-2
- Suppression du Human-in-the-Loop pour automatisation complète