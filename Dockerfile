# Utiliser une image Python officielle (slim pour réduire la taille)
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires pour certaines librairies Python (ex: compilation de bitsandbytes ou faiss)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances (sans cache pour alléger l'image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optionnel : Forcer l'installation de pytorch version CPU si vous n'avez pas de GPU Nvidia lié à Docker
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copier le reste des fichiers du projet dans le conteneur
COPY . .

# Définir l'environnement par défaut (évite les avertissements HuggingFace et gère l'encodage)
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Par défaut, on lance le script de requêtage (mode interactif). 
# Pour l'ingestion, il faudra surcharger la commande (ex: docker run my-rag-app python ingest_docs.py)
CMD ["python", "query_docs.py"]
