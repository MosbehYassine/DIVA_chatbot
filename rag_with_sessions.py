#!/usr/bin/env python3
"""
Integration RAG + Session Management avec contexte
Wrapper autour de HybridRAG pour stocker automatiquement le contexte dans les sessions
"""

import sys
import io
import logging
from typing import Dict, Optional
from query_docs import HybridRAG
from session_manager import SessionManager

# Fix encoding pour Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGWithSessions:
    """Intègre HybridRAG avec SessionManager pour persister le contexte"""

    def __init__(self, sessions_file: str = "rag_sessions.json"):
        """Initialise RAG et Session Manager"""
        self.rag = HybridRAG()
        self.sessions = SessionManager(sessions_file)
        logger.info("✅ RAG avec gestion de sessions initialisé")

    def query_with_context(
        self,
        question: str,
        session_id: Optional[str] = None,
        store_context: bool = True,
        top_k: int = 1
    ) -> Dict:
        """
        Exécute une requête RAG et stocke le contexte dans la session
        
        Args:
            question: La question à poser
            session_id: ID de la session (utilise la session active si None)
            store_context: Si True, stocke le contexte dans la session
            top_k: Nombre de résultats à retourner
            
        Returns:
            Dict avec question, réponse, contexte et métadonnées
        """
        try:
            # Changer de session si nécessaire
            if session_id and self.sessions.current_session() != session_id:
                self.sessions.switch_session(session_id)

            # Exécuter la requête RAG
            rag_results = self.rag.query(question, top_k=top_k)

            # Préparer le contexte pour la session
            rag_context = {
                "sources": [],
                "scores": [],
                "method": "hybrid",
                "entities": [],
                "documents_count": 0,
            }

            # Extraire les informations des résultats
            if "merged_results" in rag_results:
                for result in rag_results["merged_results"][:top_k]:
                    rag_context["sources"].append({
                        "document_id": result.get("document_id", ""),
                        "text_preview": result.get("text", "")[:200],  # Premier 200 caractères
                        "method": result.get("method", "unknown"),
                    })
                    rag_context["scores"].append(result.get("score", 0))
                    rag_context["entities"].append(result.get("entity", ""))
                    rag_context["documents_count"] += 1

            # Compiler la réponse
            answer = ""
            if "merged_results" and len(rag_results["merged_results"]) > 0:
                answer = rag_results["merged_results"][0].get("text", "")

            result = {
                "question": question,
                "answer": answer,
                "rag_context": rag_context,
                "graph_results_count": rag_results.get("graph_results_count", 0),
                "vector_results_count": rag_results.get("vector_results_count", 0),
            }

            # Stocker le contexte dans la session
            if store_context:
                self.sessions.add_turn(
                    question=question,
                    answer=answer,
                    rag_context=rag_context,
                    metadata={
                        "graph_results": rag_results.get("graph_results_count", 0),
                        "vector_results": rag_results.get("vector_results_count", 0),
                    }
                )
                logger.info(f"✅ Contexte stocké dans la session '{self.sessions.current_session()}'")

            return result

        except Exception as e:
            logger.error(f"❌ Erreur lors de la requête: {e}")
            return {
                "question": question,
                "answer": "",
                "error": str(e),
            }

    def get_session_with_context(self, session_id: Optional[str] = None) -> Dict:
        """Récupère les informations d'une session avec le contexte"""
        session_info = self.sessions.get_session_info(session_id)
        session_context = self.sessions.get_session_context(session_id)
        
        return {
            **session_info,
            "context": session_context,
        }

    def set_session_context(
        self,
        context: str,
        session_id: Optional[str] = None
    ) -> bool:
        """Définit le contexte global d'une session"""
        return self.sessions.set_session_context(session_id, context)

    def create_session_with_context(
        self,
        session_id: str,
        description: str = "",
        context: str = ""
    ) -> bool:
        """Crée une nouvelle session avec contexte"""
        success = self.sessions.create_session(session_id, description, context)
        if success:
            logger.info(f"✅ Session '{session_id}' créée avec contexte")
        return success

    def list_sessions_overview(self) -> str:
        """Retourne un aperçu de toutes les sessions avec contexte"""
        overview = "📊 Aperçu des Sessions\n" + "=" * 50 + "\n"
        
        for session_id, info in self.sessions.list_sessions():
            context = self.sessions.get_session_context(session_id)
            overview += f"\n🔹 {session_id}\n"
            overview += f"   Description: {info['description']}\n"
            overview += f"   Turns: {info['turns']}\n"
            overview += f"   Contexte: {context.get('global_context', 'N/A')[:50]}...\n"
            overview += f"   Entités: {len(context.get('indexed_entities', []))} indexées\n"
        
        return overview

    def export_session_with_context(
        self,
        session_id: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """Exporte une session avec tout son contexte"""
        return self.sessions.export_session_context(session_id, format)


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le système RAG avec sessions
    rag_sessions = RAGWithSessions()

    # Créer une session avec contexte
    rag_sessions.create_session_with_context(
        session_id="harmony_admin",
        description="Questions sur l'administration Harmony",
        context="Documentation d'administration du système Harmony"
    )

    # Basculer vers la nouvelle session
    rag_sessions.sessions.switch_session("harmony_admin")

    # Poser une question (le contexte est automatiquement stocké)
    result = rag_sessions.query_with_context(
        "Où peut être stocké le fichier des utilisateurs Xlogf ?",
        store_context=True
    )

    print("\n📝 Résultat de la requête:")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer'][:100]}...")
    print(f"Sources: {result['rag_context']['documents_count']}")

    # Afficher l'aperçu des sessions
    print(rag_sessions.list_sessions_overview())

    # Exporter la session avec contexte
    print("\n📄 Export de la session (Markdown):")
    print(rag_sessions.export_session_with_context("harmony_admin", format="markdown")[:500])
