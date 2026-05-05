#!/usr/bin/env python3
"""
Exemple d'intégration du Session Management avec le RAG
Montre comment utiliser les sessions dans un pipeline Q/R
"""

import logging
from session_manager import SessionManager
from answer_questions import VectorStore, generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGWithSessions:
    """Pipeline RAG avec gestion de sessions."""

    def __init__(self):
        """Initialise le pipeline RAG avec sessions."""
        self.sm = SessionManager()
        self.vs = VectorStore()

        # Charger l'index FAISS
        if not self.vs.load():
            logger.error("Impossible de charger l'index FAISS")
            return

        logger.info("✅ Pipeline RAG initialisé")

    def create_conversation(self, session_name: str, description: str = "") -> bool:
        """Crée une nouvelle conversation."""
        return self.sm.create_session(session_name, description or session_name)

    def ask_question(self, question: str) -> str:
        """Pose une question et sauvegarde dans la session active."""
        logger.info(f"❓ Question: {question}")

        # Rechercher dans le vector store
        try:
            context_chunks = self.vs.search(question, k=5)
            context = "\n".join(context_chunks)

            # Générer la réponse
            answer = generate_answer(question, context)

            # Sauvegarder dans la session
            self.sm.add_turn(question, answer)

            logger.info(f"✅ Réponse sauvegardée dans la session")
            return answer

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement: {e}")
            return "Erreur lors du traitement de la question"

    def get_conversation_history(self, limit: int = 10) -> list:
        """Récupère l'historique de la conversation actuelle."""
        return self.sm.get_history(limit=limit)

    def switch_conversation(self, session_name: str) -> bool:
        """Bascule vers une autre conversation."""
        return self.sm.switch_session(session_name)

    def list_conversations(self) -> list:
        """Liste toutes les conversations."""
        return self.sm.list_sessions()

    def export_conversation(self, format: str = "markdown") -> str:
        """Exporte la conversation active."""
        return self.sm.export_session(format=format)

    def search_in_history(self, query: str) -> list:
        """Cherche dans l'historique de la session active."""
        return self.sm.search_history(query)


# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

def main():
    """Exemple d'utilisation du RAG avec sessions."""

    # Initialiser le pipeline
    rag = RAGWithSessions()

    # Créer une nouvelle conversation
    print("\n📝 Création d'une nouvelle conversation...")
    rag.create_conversation(
        "harmonie_session_01",
        "Questions sur la gestion des utilisateurs dans Harmony"
    )

    # Basculer vers cette conversation
    rag.switch_conversation("harmonie_session_01")

    # Poser des questions
    questions = [
        "Comment Harmony gère les utilisateurs?",
        "Qu'est-ce que Xlog?",
        "Comment gérer les chemins d'accès dans Harmony?",
    ]

    print("\n🤖 Traitement des questions...\n")
    for q in questions:
        print(f"Q: {q}")
        answer = rag.ask_question(q)
        print(f"R: {answer}\n")

    # Afficher l'historique
    print("\n📋 Historique de la conversation:\n")
    history = rag.get_conversation_history(limit=5)
    for i, turn in enumerate(history, 1):
        print(f"[{i}] {turn['timestamp']}")
        print(f"    Q: {turn['question'][:60]}...")
        print(f"    R: {turn['answer'][:60]}...\n")

    # Chercher dans l'historique
    print("\n🔍 Recherche: 'utilisateur'\n")
    results = rag.search_in_history("utilisateur")
    print(f"Résultats trouvés: {len(results)}")
    for result in results:
        print(f"  - Q: {result['question']}")

    # Exporter la conversation
    print("\n💾 Export de la conversation en Markdown:\n")
    markdown = rag.export_conversation(format="markdown")
    print(markdown[:500] + "...")

    # Lister les conversations
    print("\n📚 Conversations disponibles:\n")
    conversations = rag.list_conversations()
    for session_id, info in conversations:
        print(f"  - {session_id}: {info['description']} ({info['turns']} tours)")


if __name__ == "__main__":
    # Note: Pour tester, il faut d'abord créer l'index FAISS avec ingest_docs.py
    try:
        main()
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
