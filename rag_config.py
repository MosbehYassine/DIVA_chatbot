"""
Configuration partagée RAG (chunking + embeddings).
"""
import os

# --- Embeddings (recherche vectorielle) ---
# Modèles plus puissants (FR/EN, doc technique):
#   intfloat/multilingual-e5-base   (bon compromis vitesse/qualité)
#   intfloat/multilingual-e5-large (meilleur, plus lent)
#   BAAI/bge-m3                    (multilingue, très bon en retrieval)
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)

EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16"))

# Préfixe recommandé pour les modèles E5 (query vs passage)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

def is_e5_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "e5" in name or "multilingual-e5" in name

# --- Chunking ---
# recursive = découpage par taille (NON sémantique)
# html      = découpage par titres HTML h1/h2/h3 (structure sémantique légère)
# semantic  = découpage par similarité d'embeddings (sémantique, plus lent)
CHUNK_STRATEGY = os.getenv("RAG_CHUNK_STRATEGY", "semantic").lower()

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Seuil pour SemanticChunker (percentile des écarts de similarité entre phrases)
SEMANTIC_BREAKPOINT_THRESHOLD = float(os.getenv("RAG_SEMANTIC_THRESHOLD", "0.75"))

# --- Fichiers d'index ---
DOCS_DIR = os.getenv("RAG_DOCS_DIR", "data")
FAISS_INDEX_PATH = "faiss_index.pkl"
CHUNKS_METADATA_PATH = "chunks_metadata.json"
GRAPH_PATH = "networkx_graph.pkl"
GRAPH_SOURCES_PATH = "networkx_graph_sources.pkl"
INDEX_CONFIG_PATH = "index_config.json"

# Termes pour booster la recherche graphe (alignés sur ingest_docs.KEY_TERMS)
ENTITY_HINTS = [
    "Harmony", "XLAN", "Xlan", "LDAP", "Zoom", "Xpath", "Xlog", "Xlogf", "Xlog1",
    "Xtools", "Xwin", "Divalto", "Impression", "Imprimante", "Graphique",
    "Chemin Harmony", "Utilisateur", "Fichier", "Client léger", "Serveur",
    "xDivaltoPrinters", "F7", "Shift+F7",
]

# Retrieval
# Increased lexical bias and larger retrieval pool to favor definitional matches
RETRIEVAL_POOL_SIZE = int(os.getenv("RAG_RETRIEVAL_POOL", "80"))
RERANK_LEXICAL_WEIGHT = float(os.getenv("RAG_RERANK_LEX_WEIGHT", "0.60"))
RERANK_VECTOR_WEIGHT = float(os.getenv("RAG_RERANK_VEC_WEIGHT", "0.35"))
SOURCE_MATCH_BOOST = float(os.getenv("RAG_SOURCE_BOOST", "0.35"))
DOC_ABOUT_BOOST = float(os.getenv("RAG_DOC_ABOUT_BOOST", "0.40"))

# Seuil de précision cible pour les tests de régression (similarité SequenceMatcher)
TARGET_PRECISION = float(os.getenv("RAG_TARGET_PRECISION", "0.80"))

RERANK_GRAPH_WEIGHT = float(os.getenv("RAG_RERANK_GRAPH_WEIGHT", "0.25"))
ABSTENTION_THRESHOLD = float(os.getenv("RAG_ABSTENTION_THRESHOLD", "0.50"))

# Correspondance vectorielle (graphe + réponse)
GRAPH_ENTITY_TOP_K = int(os.getenv("RAG_GRAPH_ENTITY_TOP_K", "12"))
ENTITY_VECTOR_MIN_SCORE = float(os.getenv("RAG_ENTITY_MIN_SCORE", "0.38"))
RERANK_EMBEDDING_WEIGHT = float(os.getenv("RAG_RERANK_EMB_WEIGHT", "0.05"))
ANSWER_VECTOR_WEIGHT = float(os.getenv("RAG_ANSWER_VEC_WEIGHT", "0.70"))
ANSWER_LEXICAL_WEIGHT = float(os.getenv("RAG_ANSWER_LEX_WEIGHT", "0.30"))
ANSWER_MIN_SENTENCE_SCORE = float(os.getenv("RAG_ANSWER_MIN_SCORE", "0.18"))
