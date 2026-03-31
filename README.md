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

- `ingest_docs.py` : Charge et traite les documents HTML/MD, crée un graphe de connaissances
- `query_docs.py` : Interface interactive du chatbot utilisant OpenRouter pour générer des réponses

## Changements récents

- Passage de modèles locaux Qwen (Hugging Face) à l'API OpenRouter
- Support automatique du fichier `.env`
- Configuration permanente de la clé API
- Suppression des dépendances locales (transformers, torch)