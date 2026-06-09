import os
import glob
import pickle
import json
import numpy as np
import io
import sys
import re
import unicodedata
import html as html_module
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss

from rag_config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CHUNK_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    is_e5_model,
    E5_PASSAGE_PREFIX,
    DOCS_DIR,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    GRAPH_PATH,
    GRAPH_SOURCES_PATH,
    INDEX_CONFIG_PATH,
)

# Forcer UTF-8 pour éviter les problèmes d'encodage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration (voir rag_config.py)

def normalize_text(text: str) -> str:
    """
    Nettoie et normalise le texte extrait :
    - Décode les entités HTML (&nbsp;, &gt;, &lt;, &é;, etc.)
    - Normalise Unicode (NFC)
    - Supprime les espaces superflus
    - Nettoie les retours à la ligne multiples
    """
    # Décoder les entités HTML
    text = html_module.unescape(text)
    
    # Normaliser Unicode (NFC = composition canonique)
    # Cela combine les caractères combinants (ex: é = e + accent) en forme composée
    text = unicodedata.normalize('NFC', text)
    
    # Remplacer les espaces non-breaking et autres espaces spéciaux par un espace normal
    text = text.replace('\xa0', ' ').replace('\u2009', ' ').replace('\u200b', '')
    
    # Nettoyer les retours à la ligne multiples
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    # Supprimer les espaces superflus avant/après
    text = text.strip()
    
    return text

def read_file_as_unicode(file_path: str) -> str:
    """Lit un fichier de manière robuste en gérant l'encodage (UTF-8 / CP1252)."""
    with open(file_path, "rb") as f:
        content_bytes = f.read()
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode("windows-1252", errors="ignore")

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
            with open(file_path, "rb") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                text = soup.get_text(separator="\n")
                # Normaliser le texte extrait
                text = normalize_text(text)
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

def _fallback_recursive_split(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def chunk_documents(documents):
    """
    Découpe les documents selon RAG_CHUNK_STRATEGY:
    - recursive: par taille (NON sémantique)
    - html: par titres HTML (structure sémantique, recommandé pour la doc Harmony)
    - semantic: par ruptures de similarité entre phrases (sémantique, plus lent)
    """
    if not documents:
        return []

    strategy = CHUNK_STRATEGY
    print(f"Stratégie de chunking: {strategy} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if strategy == "semantic":
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_huggingface import HuggingFaceEmbeddings

            print("Chunking sémantique (SemanticChunker) — peut prendre plusieurs minutes...")
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            splitter = SemanticChunker(
                embedder,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_THRESHOLD,
            )
            chunks = splitter.split_documents(documents)
            print(f"  -> {len(chunks)} chunks sémantiques")
            return chunks
        except Exception as exc:
            print(f"  SemanticChunker indisponible ({exc}), repli sur html + recursive")

    if strategy in ("html", "semantic"):
        # Découpe d'abord par structure HTML (sections logiques de la doc CHM)
        try:
            html_splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )
            html_chunks = []
            for doc in documents:
                source = doc.metadata.get("source", "")
                if source.lower().endswith((".htm", ".html")):
                    try:
                        html_content = read_file_as_unicode(source)
                        for part in html_splitter.split_text(html_content):
                            part.metadata = {**doc.metadata, **part.metadata}
                            html_chunks.append(part)
                    except Exception:
                        html_chunks.append(doc)
                else:
                    html_chunks.append(doc)

            if html_chunks:
                chunks = _fallback_recursive_split(html_chunks, CHUNK_SIZE, CHUNK_OVERLAP)
                print(f"  -> {len(chunks)} chunks (html + recursive)")
                return chunks
        except Exception as exc:
            print(f"  HTMLHeaderTextSplitter en échec ({exc}), repli recursive")

    chunks = _fallback_recursive_split(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  -> {len(chunks)} chunks (recursive uniquement — non sémantique)")
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
    "Zoom",
    "Xpath",
    "Xlog",
    "Xlogf",
    "Xtools",
    "Xwin",
]

def extract_entities_rule_based(text: str):
    """
    Extraction hybride :
    - mots-clés Harmony/Divalto
    - termes techniques détectés automatiquement
    """

    entities = set()

    lower_text = text.lower()

    # 1. Entités connues
    for term in KEY_TERMS:
        if term.lower() in lower_text:
            entities.add(term)

    # 2. Mots techniques commençant par une majuscule
    candidates = re.findall(r'\b[A-Z][A-Za-z0-9_.-]{2,}\b', text)

    for candidate in candidates:
        if len(candidate) > 2:
            entities.add(candidate)

    return list(entities)
def create_vector_embeddings(chunks):
    """Crée les embeddings vectoriels avec sentence-transformers et FAISS."""
    print("\n--- CRÉATION DE L'INDEX VECTORIEL (FAISS) ---")
    
    # Charger le modèle d'embeddings
    print(f"Chargement du modèle: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Extraire les textes
    texts = [chunk.page_content for chunk in chunks]
    
    # Générer les embeddings (préfixe passage: pour modèles E5)
    if is_e5_model(EMBEDDING_MODEL):
        texts_for_encode = [f"{E5_PASSAGE_PREFIX}{t}" for t in texts]
    else:
        texts_for_encode = texts

    print(f"Génération des embeddings pour {len(texts)} chunks...")
    embeddings = model.encode(
        texts_for_encode,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
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
            'text': chunk.page_content,
            'source': chunk.metadata.get('source', 'unknown'),
            'embedding_model': EMBEDDING_MODEL,
            'chunk_strategy': CHUNK_STRATEGY,
        }
        for i, chunk in enumerate(chunks)
    ]
    print(f"Sauvegarde des métadonnées dans {CHUNKS_METADATA_PATH}...")
    with open(CHUNKS_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    index_config = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": int(dimension),
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_count": len(texts),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    with open(INDEX_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(index_config, f, ensure_ascii=False, indent=2)
    print(f"Config index sauvegardée: {INDEX_CONFIG_PATH}")

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
                source = chunk.metadata.get("source", "unknown")
                graph._graph.add_node(
                    doc_id,
                    text=chunk.page_content[:2000],
                    source=source,
                    chunk_id=i,
                )

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
    print(f"Sauvegarde du graphe dans {GRAPH_PATH} et {GRAPH_SOURCES_PATH}...")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph._graph, f)
    with open(GRAPH_SOURCES_PATH, "wb") as f:
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
