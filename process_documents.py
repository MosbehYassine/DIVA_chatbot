import os
import glob
import logging
import json
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)

DOCS_DIR = os.getenv("RAG_DOCS_DIR", ".")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))

# ==============================
# 📥 LOAD DOCUMENTS
# ==============================

def load_documents(directory: str) -> List[Document]:
    documents = []

    if not os.path.isdir(directory):
        logging.warning(f"Directory not found: {directory}. No documents loaded.")
        return documents

    source_dir = os.path.abspath(directory)
    html_files = glob.glob(os.path.join(source_dir, "**/*.html"), recursive=True)
    htm_files = glob.glob(os.path.join(source_dir, "**/*.htm"), recursive=True)
    md_files = glob.glob(os.path.join(source_dir, "**/*.md"), recursive=True)

    all_html = html_files + htm_files

    logging.info(f"📂 Loading from: {source_dir}")
    logging.info(f"Found {len(all_html)} HTML/HTM and {len(md_files)} Markdown files")

    # HTML
    for file_path in all_html:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator="\n")

                if text.strip():
                    documents.append(Document(page_content=text))
        except Exception as e:
            logging.error(f"HTML error {file_path}: {e}")

    # Markdown
    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(Document(page_content=f.read()))
        except Exception as e:
            logging.error(f"MD error {file_path}: {e}")

    return documents


# ==============================
# ✂️ CHUNKING
# ==============================

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# ==============================
# 🧠 VECTOR STORE (FAISS)
# ==============================

class VectorStore:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.texts = []

    def build(self, chunks):
        self.texts = [c.page_content for c in chunks]

        embeddings = self.model.encode(
            self.texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")

        # IndexFlatIP + vecteurs normalises = similarite cosinus
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        logging.info(f"✅ FAISS built with {len(self.texts)} chunks")

    def save(self, index_path="faiss_index", pkl_path="faiss_index/index.pkl"):
        """Save the FAISS index and texts to disk"""
        os.makedirs(index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_path, "index.faiss"))

        with open(pkl_path, 'wb') as f:
            pickle.dump(self.texts, f)

        metadata = {
            "embedding_model": self.model_name,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "index_type": "IndexFlatIP",
            "normalized_embeddings": True,
            "vector_dimension": 384,
        }
        with open(os.path.join(index_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"💾 Index saved to {index_path}")


# ==============================
# 🚀 MAIN PIPELINE - DOCUMENTS & CHUNKING
# ==============================

def show_processing_stats(docs, chunks, store):
    """Affiche les statistiques détaillées du traitement"""
    print("\n" + "="*60)
    print("📊 STATISTIQUES DU TRAITEMENT")
    print("="*60)

    # Statistiques des documents
    print(f"📄 Documents chargés: {len(docs)}")

    if docs:
        doc_lengths = [len(doc.page_content.split()) for doc in docs]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths)
        total_words = sum(doc_lengths)

        print(f"📏 Longueur moyenne des documents: {avg_doc_length:.1f} mots")
        print(f"📝 Nombre total de mots: {total_words:,}")
        print(f"📄 Document le plus long: {max(doc_lengths)} mots")
        print(f"📄 Document le plus court: {min(doc_lengths)} mots")

    # Statistiques des chunks
    print(f"\n✂️ Chunks créés: {len(chunks)}")

    if chunks:
        chunk_lengths = [len(chunk.page_content.split()) for chunk in chunks]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths)

        print(f"📏 Longueur moyenne des chunks: {avg_chunk_length:.1f} mots")
        print(f"📏 Chunk le plus long: {max(chunk_lengths)} mots")
        print(f"📏 Chunk le plus court: {min(chunk_lengths)} mots")

    # Statistiques de l'index
    print(f"\n🧠 Index vectoriel: {len(store.texts)} vecteurs")
    print(f"📏 Dimension des embeddings: 384")
    print(f"🧠 Modèle d'embedding utilisé: {store.model_name}")
    print(f"💾 Index sauvegardé dans: faiss_index/")

    print("="*60)

def main():
    print("\n" + "="*60)
    print("🏗️  TRAITEMENT DES DOCUMENTS - SYSTÈME RAG VECTORIEL")
    print("="*60)

    print("\n========== 1️⃣ CHARGEMENT DES DOCUMENTS ==========")
    docs = load_documents(DOCS_DIR)

    # Filtrage des documents vides ou trop courts
    valid_docs = [d for d in docs if len(d.page_content.strip()) > 100]
    filtered_count = len(docs) - len(valid_docs)

    print(f"📂 Dossier analysé: {os.path.abspath(DOCS_DIR)}")
    print(f"📄 Documents trouvés: {len(docs)}")
    if filtered_count > 0:
        print(f"🗑️ Documents filtrés (trop courts): {filtered_count}")
    print(f"✅ Documents valides: {len(valid_docs)}")

    if not valid_docs:
        print("❌ Aucun document valide trouvé!")
        print("💡 Vérifiez que les fichiers HTML/MD sont présents dans les sous-dossiers.")
        return

    print("\n========== 2️⃣ DÉCOUPAGE EN CHUNKS ==========")
    chunks = chunk_documents(valid_docs)
    print(f"✂️ Chunks créés: {len(chunks)}")
    print("🔧 Paramètres de chunking:")
    print("  • Taille des chunks: 500 caractères")
    print("  • Chevauchement: 50 caractères")

    print("\n========== 3️⃣ CONSTRUCTION DE L'INDEX VECTORIEL ==========")
    print(f"🧠 Modèle d'embedding: {EMBEDDING_MODEL}")
    print(f"⚡ Taille de batch embeddings: {EMBEDDING_BATCH_SIZE}")
    print("⏳ Cette opération peut prendre quelques minutes...")

    store = VectorStore()
    store.build(chunks)

    print("✅ Index vectoriel construit avec succès!")

    print("\n========== 4️⃣ SAUVEGARDE DE L'INDEX ==========")
    store.save()
    print("💾 Index sauvegardé dans le dossier 'faiss_index/'")

    # Afficher les statistiques complètes
    show_processing_stats(valid_docs, chunks, store)

    print("\n🎉 TRAITEMENT TERMINÉ!")
    print("💡 Vous pouvez maintenant utiliser: python answer_questions.py")
    print("="*60)


# ==============================
# ▶️ RUN
# ==============================

if __name__ == "__main__":
    main()