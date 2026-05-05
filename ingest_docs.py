import os
import glob
import pickle
import json
import numpy as np
import io
import sys
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss

# Forcer UTF-8 pour éviter les problèmes d'encodage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
DOCS_DIR = "data"  # Lis les documents depuis le dossier data
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GRAPH_PATH = "networkx_graph.pkl"
FAISS_INDEX_PATH = "faiss_index.pkl"
CHUNKS_METADATA_PATH = "chunks_metadata.json"

def load_documents(directory):
    documents = []
    # Find all HTML and Markdown files
    html_files = glob.glob(os.path.join(directory, "**/*.htm"), recursive=True) + \
                 glob.glob(os.path.join(directory, "**/*.html"), recursive=True)
    md_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)

    print(f"Found {len(html_files)} HTML files and {len(md_files)} Markdown files.")

    # Process HTML files
    for file_path in html_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text(separator="\n")
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"source": file_path}))
        except Exception as e:
            print(f"Error loading HTML {file_path}: {e}")

    # Process Markdown files
    for file_path in md_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading Markdown {file_path}: {e}")

    return documents

def chunk_documents(documents):
    if not documents:
        return []

    print("Initialisation du TextSplitter classique pour le Graph (besoin de contexte)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, # Augmenté pour plus de contexte et meilleure pertinence
        chunk_overlap=250  # Augmenté pour meilleure continuité
    )

    print("Découpage des documents...")
    chunks = text_splitter.split_documents(documents)
    return chunks

from langchain_community.graphs.networkx_graph import NetworkxEntityGraph

# Termes clés typiques de la doc Harmony / Divalto (optimisé)
KEY_TERMS = [
    "Harmony",
    "Xwpf.exe",
    "XrtDiva.exe",
    "XLAN",
    "RecordSQL",
    "ODBC",
    "MSSQL",
    "Oracle",
    "DB2",
    "Lotus Notes",
    "Serveur Xlan",
    "Serveur d'applications",
    "Client léger",
    "Architecture 3-tiers",
    "Chemin Harmony",
    "Utilisateur",
    "Imprimante",
    "Impression",
    "Graphique",
    "LDAP",
    "Fichier",
    "Divalto",
    "Installation",
    "Configuration",
    "Droit d'accès",
    "Base de données",
    "Paramètre",
]

def extract_entities_rule_based(text: str):
    """Extraction simple d'entités par mots-clés connus (rule-based)."""
    found = set()
    lower_text = text.lower()
    for term in KEY_TERMS:
        if term.lower() in lower_text:
            found.add(term)
    return list(found)

def create_vector_embeddings(chunks):
    """Crée les embeddings vectoriels avec sentence-transformers et FAISS."""
    print("\n--- CRÉATION DE L'INDEX VECTORIEL (FAISS) ---")
    
    # Charger le modèle d'embeddings
    print(f"Chargement du modèle: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Extraire les textes
    texts = [chunk.page_content for chunk in chunks]
    
    # Générer les embeddings
    print(f"Génération des embeddings pour {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Créer l'index FAISS
    print("Création de l'index FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product pour similarité cosinus
    index.add(embeddings)
    
    # Sauvegarder l'index
    print(f"Sauvegarde de l'index FAISS dans {FAISS_INDEX_PATH}...")
    with open(FAISS_INDEX_PATH, 'wb') as f:
        pickle.dump(index, f)
    
    # Sauvegarder les métadonnées (textes originaux)
    metadata = [
        {
            'chunk_id': i,
            'text': chunk.page_content[:500],  # Preview
            'source': chunk.metadata.get('source', 'unknown')
        }
        for i, chunk in enumerate(chunks)
    ]
    print(f"Sauvegarde des métadonnées dans {CHUNKS_METADATA_PATH}...")
    with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Index FAISS créé avec {len(texts)} vecteurs de dimension {dimension}.")
    return index, embeddings, model

def create_graph(chunks):
    """Crée le graphe NetworkX pour Graph RAG."""
    print("\n--- CONSTRUCTION DU GRAPHE (GraphRAG) ---")
    graph = NetworkxEntityGraph()
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        if (i + 1) % max(1, total_chunks // 10) == 0:
            print(f"Traitement du chunk {i+1}/{total_chunks}...")
        try:
            doc_id = f"Document_{i}"

            # Nœud document avec un extrait de texte utile comme attribut
            if not graph._graph.has_node(doc_id):
                snippet = chunk.page_content[:400]
                graph._graph.add_node(doc_id, text=snippet)

            # Entités par règles
            entities = set(extract_entities_rule_based(chunk.page_content))

            # Entité "module" basée sur le chemin du fichier source, si dispo
            source = chunk.metadata.get("source")
            if source:
                rel = os.path.relpath(source, DOCS_DIR)
                parts = rel.split(os.sep)
                if parts:
                    module_name = parts[0]
                    if len(module_name) > 2:
                        entities.add(module_name)

            # Ajout des nœuds entités et relations Document -> Entité
            for node_name in entities:
                if len(node_name) > 2 and len(node_name) < 80:
                    if not graph._graph.has_node(node_name):
                        graph._graph.add_node(node_name)
                    graph._graph.add_edge(doc_id, node_name, relation="CONTIENT")

        except Exception as e:
            print(f"Erreur lors du traitement du chunk {i}: {e}")

    # Sauvegarder le graphe
    print(f"Sauvegarde du graphe dans {GRAPH_PATH}...")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph._graph, f)

    print(f"Graphe créé avec {graph._graph.number_of_nodes()} noeuds et {graph._graph.number_of_edges()} relations.")
    return graph._graph

if __name__ == "__main__":
    print("🚀 Starting hybrid RAG ingestion process...")
    docs = load_documents(DOCS_DIR)
    print(f"📚 Loaded {len(docs)} documents.")

    if docs:
        print("\n--- TRAITEMENT DE TOUS LES DOCUMENTS ---")
        # On filtre les pages vides ou de pure mise en page
        valid_docs = [d for d in docs if len(d.page_content.strip()) > 300]
        docs = valid_docs
        print(f"✅ Documents valides après filtrage: {len(docs)}")

        chunks = chunk_documents(docs)
        print(f"📝 Création de {len(chunks)} chunks à partir de tous les documents valides.")

        # Créer les deux index
        create_vector_embeddings(chunks)
        create_graph(chunks)
        
        print("\n✅ Ingestion hybride (Graph RAG + Vector RAG) complète!")
        print("   - Index FAISS créé")
        print("   - Graphe NetworkX créé")
    else:
        print("❌ No documents found to ingest.")
