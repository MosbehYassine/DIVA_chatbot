import os
import glob
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Configuration
DOCS_DIR = "."

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
        chunk_size=1500, # Assez grand pour qu'il y ait des relations à extraire
        chunk_overlap=150
    )
    
    print("Découpage des documents...")
    chunks = text_splitter.split_documents(documents)
    return chunks

from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import pickle

# Quelques termes clés typiques de la doc Harmony / Divalto
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
]

def extract_entities_rule_based(text: str):
    """Extraction simple d'entités par mots-clés connus (rule-based)."""
    found = set()
    lower_text = text.lower()
    for term in KEY_TERMS:
        if term.lower() in lower_text:
            found.add(term)
    return list(found)

def create_vector_store(chunks):
    print("\n--- CONSTRUCTION DU GRAPHE (GraphRAG) SANS LLM ---")
    graph = NetworkxEntityGraph()
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        print(f"Construction du graphe pour le document {i+1}/{total_chunks}...")
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

            if i == 0:
                print(f"DEBUG - Entités trouvées pour le 1er chunk : {entities}")

            # Ajout des nœuds entités et relations Document -> Entité
            for node_name in entities:
                if len(node_name) > 2 and len(node_name) < 80:
                    if not graph._graph.has_node(node_name):
                        graph._graph.add_node(node_name)
                    graph._graph.add_edge(doc_id, node_name, relation="CONTIENT")

        except Exception as e:
            print(f"Erreur lors de la construction du graphe : {e}")

    graph_path = "networkx_graph.pkl"
    print(f"Sauvegarde du graphe dans {graph_path}...")
    with open(graph_path, "wb") as f:
        pickle.dump(graph._graph, f)
        
    print(f"Graphe créé avec {graph._graph.number_of_nodes()} noeuds et {graph._graph.number_of_edges()} relations.")

if __name__ == "__main__":
    print("Starting ingestion process...")
    docs = load_documents(DOCS_DIR)
    print(f"Loaded {len(docs)} documents.")
    
    if docs:
        print("\n--- MODE TEST : ÉCHANTILLONNAGE ACTIVÉ (Documents significatifs) ---")
        # On filtre les pages vides ou de pure mise en page
        valid_docs = [d for d in docs if len(d.page_content.strip()) > 300]
        docs = valid_docs[:30] # On prend 30 documents valides
        print(f"Limitation à {len(docs)} documents pertinents pour le test.")
        
        chunks = chunk_documents(docs)
        print(f"Création de {len(chunks)} chunks à partir de l'échantillon.")
        
        create_vector_store(chunks)
        print("Ingestion complete!")
    else:
        print("No documents found to ingest.")
