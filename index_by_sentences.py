#!/usr/bin/env python3
"""
Créer un index FAISS au niveau phrase (pas chunk).
Améliore la granularité de recherche pour de meilleures correspondances fine-grained.

Entrées : chunks_metadata.json + faiss_index.pkl
Sorties : sentences_metadata.json + sentences_faiss.pkl
"""

import json
import pickle
import numpy as np
import re
import sys
import io
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Forcer UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from rag_config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CHUNKS_METADATA_PATH,
    FAISS_INDEX_PATH,
)

SENTENCES_METADATA_PATH = "sentences_metadata.json"
SENTENCES_FAISS_PATH = "sentences_faiss.pkl"


def split_into_sentences(text: str) -> List[str]:
    """
    Divise un texte en phrases.
    Gère les cas français (M., Dr., etc.)
    """
    # Remplacer les abréviations communes pour éviter les faux splits
    text = re.sub(r'\bM\.\s', 'M_ABBR_ ', text)
    text = re.sub(r'\bDr\.\s', 'Dr_ABBR_ ', text)
    text = re.sub(r'\bProf\.\s', 'Prof_ABBR_ ', text)
    text = re.sub(r'\bMr\.\s', 'Mr_ABBR_ ', text)
    text = re.sub(r'\bMme\.\s', 'Mme_ABBR_ ', text)
    text = re.sub(r'\bEtc\.\s', 'Etc_ABBR_ ', text)
    
    # Split par . ! ? suivis d'un espace et une majuscule
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restaurer les abréviations
    sentences = [s.replace('M_ABBR_', 'M. ') for s in sentences]
    sentences = [s.replace('Dr_ABBR_', 'Dr. ') for s in sentences]
    sentences = [s.replace('Prof_ABBR_', 'Prof. ') for s in sentences]
    sentences = [s.replace('Mr_ABBR_', 'Mr. ') for s in sentences]
    sentences = [s.replace('Mme_ABBR_', 'Mme. ') for s in sentences]
    sentences = [s.replace('Etc_ABBR_', 'Etc. ') for s in sentences]
    
    # Nettoyer les phrases vides et courtes (< 5 caractères)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    return sentences


def load_chunks():
    """Charger les chunks depuis chunks_metadata.json."""
    print(f"Chargement des chunks depuis {CHUNKS_METADATA_PATH}...")
    with open(CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks chargés")
    return chunks


def chunks_to_sentences(chunks: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Convertir les chunks en phrases avec métadonnées de traçabilité.
    
    Retourne:
    - sentences_metadata: liste de dicts avec { text, chunk_id, chunk_index, source, document_id }
    - total_sentences: nombre total de phrases
    """
    sentences_metadata = []
    total_sentences = 0
    
    print("\nFractionnement des chunks en phrases...")
    for chunk_idx, chunk in enumerate(chunks):
        chunk_text = chunk.get('text', '')
        if not chunk_text.strip():
            continue
        
        # Diviser en phrases
        phrases = split_into_sentences(chunk_text)
        
        for phrase_idx, phrase in enumerate(phrases):
            sentence_meta = {
                'text': phrase,
                'chunk_id': chunk.get('id', chunk_idx),
                'chunk_index': chunk_idx,
                'chunk_text_preview': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text,
                'source': chunk.get('source', 'unknown'),
                'document_id': chunk.get('document_id', chunk.get('source', 'unknown')),
                'sentence_index_in_chunk': phrase_idx,
            }
            sentences_metadata.append(sentence_meta)
            total_sentences += 1
        
        if (chunk_idx + 1) % 1000 == 0:
            print(f"  Traité: {chunk_idx + 1}/{len(chunks)} chunks → {total_sentences} phrases")
    
    print(f"Total: {len(chunks)} chunks → {total_sentences} phrases")
    return sentences_metadata, total_sentences


def embed_sentences(sentences_metadata: List[Dict], model: SentenceTransformer) -> np.ndarray:
    """
    Créer les embeddings pour chaque phrase.
    """
    print(f"\nCréation des embeddings ({len(sentences_metadata)} phrases)...")
    
    texts = [s['text'] for s in sentences_metadata]
    
    # Ajouter le préfixe "passage:" pour les modèles E5
    from rag_config import is_e5_model, E5_PASSAGE_PREFIX
    if is_e5_model(EMBEDDING_MODEL):
        texts = [E5_PASSAGE_PREFIX + t for t in texts]
    
    # Encoder par batch
    embeddings = model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True)
    
    print(f"  Forme embeddings: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Construire l'index FAISS."""
    print("\nConstruction de l'index FAISS...")
    
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    
    # Index simple (L2 distance)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    print(f"  Index créé: {index.ntotal} vecteurs, dimension {dim}")
    return index


def save_index_and_metadata(index: faiss.IndexFlatL2, sentences_metadata: List[Dict]):
    """Sauvegarder l'index et les métadonnées."""
    print("\nSauvegarde de l'index et des métadonnées...")
    
    # Sauvegarder FAISS
    with open(SENTENCES_FAISS_PATH, 'wb') as f:
        pickle.dump(index, f)
    print(f"  ✅ {SENTENCES_FAISS_PATH}")
    
    # Sauvegarder métadonnées
    with open(SENTENCES_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(sentences_metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {SENTENCES_METADATA_PATH}")
    
    print(f"\nRésumé:")
    print(f"  Phrases indexées: {len(sentences_metadata)}")
    print(f"  Dimension: {index.d}")
    print(f"  Ratio phrases/chunks: {len(sentences_metadata) / 10175:.1f}x")


def main():
    print("=" * 60)
    print("INDEX PAR PHRASES - FAISS")
    print("=" * 60)
    
    # 1. Charger modèle
    print(f"\nChargement du modèle: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Dimension: {model.get_sentence_embedding_dimension()}")
    
    # 2. Charger chunks
    chunks = load_chunks()
    
    # 3. Convertir chunks → phrases
    sentences_metadata, total = chunks_to_sentences(chunks)
    
    # 4. Créer embeddings
    embeddings = embed_sentences(sentences_metadata, model)
    
    # 5. Construire index FAISS
    index = build_faiss_index(embeddings)
    
    # 6. Sauvegarder
    save_index_and_metadata(index, sentences_metadata)
    
    print("\n" + "=" * 60)
    print("✅ Index par phrases créé avec succès!")
    print("=" * 60)


if __name__ == "__main__":
    main()
