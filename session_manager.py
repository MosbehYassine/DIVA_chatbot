#!/usr/bin/env python3
"""
Session Management Module
Gère les sessions persistantes pour l'historique Q/R du système RAG.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SESSIONS_FILE = os.getenv("RAG_SESSIONS_FILE", "rag_sessions.json")


class SessionManager:
    """Gère des sessions persistantes pour l'historique Q/R."""

    def __init__(self, file_path: str = SESSIONS_FILE):
        self.file_path = file_path
        self.data = {
            "active_session": "default",
            "sessions": {
                "default": {
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "modified": datetime.now().isoformat(),
                        "description": "Session par défaut",
                    },
                    "turns": [],
                }
            },
        }
        self._load()

    def _load(self):
        """Charge les sessions depuis le fichier."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)

                if "sessions" in loaded and isinstance(loaded["sessions"], dict):
                    # Migrer vers le nouveau format si nécessaire
                    for session_id, session_data in loaded["sessions"].items():
                        if isinstance(session_data, list):
                            # Format ancien (liste directe)
                            loaded["sessions"][session_id] = {
                                "metadata": {
                                    "created": datetime.now().isoformat(),
                                    "modified": datetime.now().isoformat(),
                                    "description": session_id,
                                },
                                "turns": session_data,
                            }
                        elif isinstance(session_data, dict) and "turns" not in session_data:
                            # Format intermédiaire
                            loaded["sessions"][session_id] = {
                                "metadata": {
                                    "created": datetime.now().isoformat(),
                                    "modified": datetime.now().isoformat(),
                                    "description": session_id,
                                },
                                "turns": session_data.get("data", []),
                            }

                    self.data = loaded

                if "active_session" not in self.data:
                    self.data["active_session"] = "default"

                if self.data["active_session"] not in self.data["sessions"]:
                    self.data["sessions"][self.data["active_session"]] = {
                        "metadata": {
                            "created": datetime.now().isoformat(),
                            "modified": datetime.now().isoformat(),
                            "description": self.data["active_session"],
                        },
                        "turns": [],
                    }

                logger.info(f"✅ Sessions chargées: {len(self.data['sessions'])} session(s)")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger {self.file_path}: {e}")
                self._ensure_default_session()
        else:
            self._ensure_default_session()

    def _ensure_default_session(self):
        """Assure qu'une session par défaut existe."""
        if "default" not in self.data["sessions"]:
            self.data["sessions"]["default"] = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "description": "Session par défaut",
                },
                "turns": [],
            }

    def _save(self):
        """Sauvegarde les sessions dans le fichier."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")

    def current_session(self) -> str:
        """Retourne l'ID de la session active."""
        return self.data["active_session"]

    def list_sessions(self) -> List[Tuple[str, Dict]]:
        """Retourne la liste des sessions avec leurs métadonnées."""
        sessions = []
        for session_id in sorted(self.data["sessions"].keys()):
            session = self.data["sessions"][session_id]
            metadata = session.get("metadata", {})
            turn_count = len(session.get("turns", []))
            sessions.append(
                (
                    session_id,
                    {
                        "description": metadata.get("description", session_id),
                        "created": metadata.get("created", "N/A"),
                        "modified": metadata.get("modified", "N/A"),
                        "turns": turn_count,
                    },
                )
            )
        return sessions

    def create_session(
        self, session_id: str, description: str = ""
    ) -> bool:
        """Crée une nouvelle session."""
        if session_id in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' existe déjà")
            return False

        self.data["sessions"][session_id] = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "description": description or session_id,
            },
            "turns": [],
        }
        logger.info(f"✅ Session '{session_id}' créée")
        self._save()
        return True

    def delete_session(self, session_id: str) -> bool:
        """Supprime une session."""
        if session_id == "default":
            logger.warning("⚠️ Impossible de supprimer la session par défaut")
            return False

        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return False

        del self.data["sessions"][session_id]

        if self.data["active_session"] == session_id:
            self.data["active_session"] = "default"

        logger.info(f"✅ Session '{session_id}' supprimée")
        self._save()
        return True

    def switch_session(self, session_id: str) -> bool:
        """Bascule vers une autre session."""
        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return False

        self.data["active_session"] = session_id
        logger.info(f"✅ Basculé vers la session '{session_id}'")
        self._save()
        return True

    def add_turn(
        self, question: str, answer: str, metadata: Optional[Dict] = None
    ) -> bool:
        """Ajoute un tour (Q/R) à la session actuelle."""
        try:
            session_id = self.current_session()
            session = self.data["sessions"][session_id]

            turn = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "answer": answer,
            }

            if metadata:
                turn["metadata"] = metadata

            session["turns"].append(turn)
            session["metadata"]["modified"] = datetime.now().isoformat()

            self._save()
            return True
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout du tour: {e}")
            return False

    def get_history(self, session_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Récupère l'historique d'une session."""
        if session_id is None:
            session_id = self.current_session()

        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return []

        turns = self.data["sessions"][session_id].get("turns", [])

        if limit <= 0:
            return turns
        return turns[-limit:]

    def get_session_info(self, session_id: Optional[str] = None) -> Dict:
        """Retourne les informations détaillées d'une session."""
        if session_id is None:
            session_id = self.current_session()

        if session_id not in self.data["sessions"]:
            return {}

        session = self.data["sessions"][session_id]
        metadata = session.get("metadata", {})

        return {
            "id": session_id,
            "description": metadata.get("description", "N/A"),
            "created": metadata.get("created", "N/A"),
            "modified": metadata.get("modified", "N/A"),
            "turns_count": len(session.get("turns", [])),
            "is_active": session_id == self.current_session(),
        }

    def update_session_description(self, session_id: str, description: str) -> bool:
        """Met à jour la description d'une session."""
        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return False

        self.data["sessions"][session_id]["metadata"]["description"] = description
        self.data["sessions"][session_id]["metadata"]["modified"] = datetime.now().isoformat()
        logger.info(f"✅ Description de '{session_id}' mise à jour")
        self._save()
        return True

    def clear_session(self, session_id: Optional[str] = None) -> bool:
        """Efface tout l'historique d'une session."""
        if session_id is None:
            session_id = self.current_session()

        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return False

        self.data["sessions"][session_id]["turns"] = []
        self.data["sessions"][session_id]["metadata"]["modified"] = datetime.now().isoformat()
        logger.info(f"✅ Session '{session_id}' effacée")
        self._save()
        return True

    def search_history(self, query: str, session_id: Optional[str] = None) -> List[Dict]:
        """Recherche dans l'historique d'une session."""
        if session_id is None:
            session_id = self.current_session()

        if session_id not in self.data["sessions"]:
            return []

        turns = self.data["sessions"][session_id].get("turns", [])
        query_lower = query.lower()

        results = [
            turn
            for turn in turns
            if query_lower in turn.get("question", "").lower()
            or query_lower in turn.get("answer", "").lower()
        ]

        return results

    def export_session(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> str:
        """Exporte une session dans un format donné."""
        if session_id is None:
            session_id = self.current_session()

        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return ""

        session = self.data["sessions"][session_id]

        if format == "json":
            return json.dumps(session, ensure_ascii=False, indent=2)

        elif format == "markdown":
            lines = [
                f"# Session: {session_id}\n",
                f"**Description**: {session['metadata'].get('description', 'N/A')}\n",
                f"**Créée**: {session['metadata'].get('created', 'N/A')}\n",
                f"**Modifiée**: {session['metadata'].get('modified', 'N/A')}\n",
                "\n---\n\n",
            ]

            for i, turn in enumerate(session.get("turns", []), 1):
                lines.append(f"## Tour {i}\n")
                lines.append(f"**Horodatage**: {turn.get('timestamp', 'N/A')}\n\n")
                lines.append(f"### Question\n{turn.get('question', 'N/A')}\n\n")
                lines.append(f"### Réponse\n{turn.get('answer', 'N/A')}\n\n")
                lines.append("---\n\n")

            return "".join(lines)

        elif format == "txt":
            lines = [
                f"Session: {session_id}\n",
                f"Description: {session['metadata'].get('description', 'N/A')}\n",
                f"Créée: {session['metadata'].get('created', 'N/A')}\n",
                f"Modifiée: {session['metadata'].get('modified', 'N/A')}\n",
                "\n" + "=" * 80 + "\n\n",
            ]

            for i, turn in enumerate(session.get("turns", []), 1):
                lines.append(f"Tour {i} - {turn.get('timestamp', 'N/A')}\n")
                lines.append(f"Q: {turn.get('question', 'N/A')}\n")
                lines.append(f"R: {turn.get('answer', 'N/A')}\n")
                lines.append("\n" + "-" * 80 + "\n\n")

            return "".join(lines)

        return ""

    def import_session(self, session_id: str, data: Dict) -> bool:
        """Importe une session depuis un dictionnaire."""
        try:
            if session_id in self.data["sessions"]:
                logger.warning(f"⚠️ Session '{session_id}' existe déjà")
                return False

            self.data["sessions"][session_id] = {
                "metadata": data.get("metadata", {
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "description": session_id,
                }),
                "turns": data.get("turns", []),
            }

            logger.info(f"✅ Session '{session_id}' importée")
            self._save()
            return True
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'import: {e}")
            return False
