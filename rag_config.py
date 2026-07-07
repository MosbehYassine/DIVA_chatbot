"""
Configuration partagée RAG (chunking + embeddings).
"""
import os


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

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

# Parent-child retrieval uses approximate whitespace-token counts.
RAG_ENABLE_PARENT_CHILD_RETRIEVAL = env_bool(
    "RAG_ENABLE_PARENT_CHILD_RETRIEVAL",
    True,
)
RAG_PARENT_CHUNK_SIZE = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "1000"))
RAG_PARENT_CHUNK_OVERLAP = int(os.getenv("RAG_PARENT_CHUNK_OVERLAP", "200"))
RAG_CHILD_CHUNK_SIZE = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "250"))
RAG_CHILD_CHUNK_OVERLAP = int(os.getenv("RAG_CHILD_CHUNK_OVERLAP", "50"))

# Backward-compatible names used by the ingestion code.
PARENT_CHUNK_SIZE = RAG_PARENT_CHUNK_SIZE
PARENT_CHUNK_OVERLAP = RAG_PARENT_CHUNK_OVERLAP
CHILD_CHUNK_SIZE = RAG_CHILD_CHUNK_SIZE
CHILD_CHUNK_OVERLAP = RAG_CHILD_CHUNK_OVERLAP

# Seuil pour SemanticChunker (percentile des écarts de similarité entre phrases)
SEMANTIC_BREAKPOINT_THRESHOLD = float(os.getenv("RAG_SEMANTIC_THRESHOLD", "0.75"))

# --- Fichiers d'index ---
DOCS_DIR = os.getenv("RAG_DOCS_DIR", "data")
FAISS_INDEX_PATH = "faiss_index.pkl"
CHUNKS_METADATA_PATH = "chunks_metadata.json"
PARENTS_METADATA_PATH = "parent_chunks_metadata.json"
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
RAG_ENABLE_ENTITY_VECTOR_MATCH = env_bool("RAG_ENABLE_ENTITY_VECTOR_MATCH", True)
RAG_ENABLE_EMBEDDING_RERANK = env_bool("RAG_ENABLE_EMBEDDING_RERANK", True)
GRAPH_ENTITY_TOP_K = int(os.getenv("RAG_GRAPH_ENTITY_TOP_K", "12"))
ENTITY_VECTOR_MIN_SCORE = float(os.getenv("RAG_ENTITY_MIN_SCORE", "0.38"))
RERANK_EMBEDDING_WEIGHT = float(os.getenv("RAG_RERANK_EMB_WEIGHT", "0.05"))
ANSWER_VECTOR_WEIGHT = float(os.getenv("RAG_ANSWER_VEC_WEIGHT", "0.70"))
ANSWER_LEXICAL_WEIGHT = float(os.getenv("RAG_ANSWER_LEX_WEIGHT", "0.30"))
ANSWER_MIN_SENTENCE_SCORE = float(os.getenv("RAG_ANSWER_MIN_SCORE", "0.18"))

# Cross-encoder reranking. FlagEmbedding is optional; the custom reranker remains
# the deterministic fallback when the package or model is unavailable.
RAG_ENABLE_QUERY_REWRITING = env_bool("RAG_ENABLE_QUERY_REWRITING", True)
RAG_ENABLE_SELF_QUERY = env_bool("RAG_ENABLE_SELF_QUERY", True)
RAG_ENABLE_BM25 = env_bool("RAG_ENABLE_BM25", True)
RAG_ENABLE_TFIDF = env_bool("RAG_ENABLE_TFIDF", False)
RAG_ENABLE_MMR = env_bool("RAG_ENABLE_MMR", True)
RAG_ENABLE_CROSS_ENCODER_RERANKER = env_bool(
    "RAG_ENABLE_CROSS_ENCODER_RERANKER",
    env_bool("RAG_ENABLE_CROSS_ENCODER", True),
)
RAG_ENABLE_CROSS_ENCODER = RAG_ENABLE_CROSS_ENCODER_RERANKER
RAG_CROSS_ENCODER_MODEL = os.getenv(
    "RAG_CROSS_ENCODER_MODEL",
    os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
)
RAG_RERANKER_MODEL = RAG_CROSS_ENCODER_MODEL
RAG_RERANKER_USE_FP16 = env_bool("RAG_RERANKER_USE_FP16", True)
RAG_RERANKER_BATCH_SIZE = int(os.getenv("RAG_RERANKER_BATCH_SIZE", "8"))

RAG_BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "20"))
RAG_VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "20"))
RAG_GRAPH_TOP_K = int(os.getenv("RAG_GRAPH_TOP_K", "20"))
RAG_FUSION_TOP_K = int(os.getenv("RAG_FUSION_TOP_K", "40"))
RAG_MMR_TOP_K = int(os.getenv("RAG_MMR_TOP_K", "20"))
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "5"))
RAG_CROSS_ENCODER_CANDIDATE_K = int(os.getenv("RAG_CROSS_ENCODER_CANDIDATE_K", "12"))
RAG_EXTRACTION_TOP_K = int(os.getenv("RAG_EXTRACTION_TOP_K", "10"))
RAG_TITLE_EXACT_BOOST = float(os.getenv("RAG_TITLE_EXACT_BOOST", "1.20"))
RAG_TITLE_PARTIAL_BOOST = float(os.getenv("RAG_TITLE_PARTIAL_BOOST", "0.70"))
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.70"))

# Query transformation. HyDE content is retrieval-only and must never be
# passed to answer generation as an answer.
RAG_ENABLE_QUERY_EXPANSION = env_bool(
    "RAG_ENABLE_QUERY_EXPANSION",
    RAG_ENABLE_QUERY_REWRITING,
)
RAG_ENABLE_HYDE = env_bool("RAG_ENABLE_HYDE", True)
RAG_HYDE_MAX_SENTENCES = int(os.getenv("RAG_HYDE_MAX_SENTENCES", "6"))
RAG_HYDE_USE_LLM = env_bool("RAG_HYDE_USE_LLM", False)
RAG_QUERY_VARIANTS_COUNT = int(os.getenv("RAG_QUERY_VARIANTS_COUNT", "3"))
RAG_QUERY_LLM_MODEL = os.getenv("RAG_QUERY_LLM_MODEL", "gpt-4o-mini")

# Context compression.
RAG_ENABLE_CONTEXT_COMPRESSION = env_bool("RAG_ENABLE_CONTEXT_COMPRESSION", True)
RAG_MAX_COMPRESSED_SENTENCES = int(os.getenv("RAG_MAX_COMPRESSED_SENTENCES", "6"))

# Answer verification and bounded retry.
RAG_ENABLE_ANSWER_VERIFIER = env_bool(
    "RAG_ENABLE_ANSWER_VERIFIER",
    env_bool("RAG_ENABLE_ANSWER_VERIFICATION", True),
)
RAG_ENABLE_ANSWER_VERIFICATION = RAG_ENABLE_ANSWER_VERIFIER
RAG_ENABLE_LLM_ANSWER_GENERATION = env_bool(
    "RAG_ENABLE_LLM_ANSWER_GENERATION",
    True,
)
RAG_ANSWER_LLM_MODEL = os.getenv(
    "RAG_ANSWER_LLM_MODEL",
    os.getenv("RAG_QUERY_LLM_MODEL", "gpt-4o-mini"),
)
RAG_LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "auto").strip().lower()
RAG_OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3.2:1b")
RAG_OLLAMA_BASE_URL = os.getenv(
    "RAG_OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
).rstrip("/")
RAG_LLM_ANSWER_MAX_CONTEXT_CHARS = int(
    os.getenv("RAG_LLM_ANSWER_MAX_CONTEXT_CHARS", "9000")
)
RAG_ENABLE_LLM_ANSWER_VERIFIER = env_bool(
    "RAG_ENABLE_LLM_ANSWER_VERIFIER",
    False,
)
RAG_VERIFIER_MIN_CONFIDENCE = float(
    os.getenv("RAG_VERIFIER_MIN_CONFIDENCE", "0.65")
)
RAG_MAX_RETRY_COUNT = max(0, int(os.getenv("RAG_MAX_RETRY_COUNT", "1")))
RAG_VERIFIER_SUPPORT_THRESHOLD = float(os.getenv("RAG_VERIFIER_SUPPORT_THRESHOLD", "0.52"))
RAG_VERIFIER_RETRY_THRESHOLD = float(os.getenv("RAG_VERIFIER_RETRY_THRESHOLD", "0.62"))
RAG_MIN_ANSWER_CONFIDENCE = float(
    os.getenv("RAG_MIN_ANSWER_CONFIDENCE", str(RAG_VERIFIER_MIN_CONFIDENCE))
)

# Conversation memory is in-memory unless a caller supplies a persistence file.
RAG_ENABLE_CHAT_MEMORY = env_bool("RAG_ENABLE_CHAT_MEMORY", True)
RAG_CHAT_HISTORY_TURNS = max(1, int(os.getenv("RAG_CHAT_HISTORY_TURNS", "5")))
RAG_ENABLE_CONVERSATION_REFORMULATION = env_bool(
    "RAG_ENABLE_CONVERSATION_REFORMULATION",
    True,
)
RAG_ENABLE_SESSION_SUMMARY = env_bool("RAG_ENABLE_SESSION_SUMMARY", True)
RAG_ENABLE_SESSION_SOURCE_BOOST = env_bool("RAG_ENABLE_SESSION_SOURCE_BOOST", True)
RAG_SESSION_SOURCE_BOOST = float(os.getenv("RAG_SESSION_SOURCE_BOOST", "0.04"))
RAG_ENABLE_REASONING_PLAN = env_bool("RAG_ENABLE_REASONING_PLAN", True)
