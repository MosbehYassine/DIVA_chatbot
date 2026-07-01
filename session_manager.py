#!/usr/bin/env python3
"""
Session Management Module avec gestion du contexte
Gère les sessions persistantes pour l'historique Q/R du système RAG avec contexte.
"""

import os
import json
import logging
import sqlite3
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SESSIONS_FILE = os.getenv("RAG_SESSIONS_FILE", "hybrid_rag_sessions.json")
DB_SESSIONS_PATH = os.getenv("RAG_SESSIONS_DB", "hybrid_rag_sessions.db")

SESSION_STOPWORDS = {
    "comment", "faire", "quel", "quelle", "quels", "quelles", "pourquoi",
    "dans", "avec", "sans", "depuis", "harmony", "divalto", "peut", "pour",
    "une", "un", "des", "les", "la", "le", "de", "du", "au", "aux", "et",
    "est", "sont", "etre", "ce", "cette", "ces", "quoi", "plus", "moins",
}


class SessionManager:
    """Gère des sessions persistantes pour l'historique Q/R avec contexte RAG."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        db_path: Optional[str] = None,
        use_db: bool = False,
        max_history_turns: int = 5,
    ):
        self.file_path = file_path
        self.max_history_turns = max(1, int(max_history_turns))
        self.db_path = None
        self.use_db = False
        self.conn: Optional[sqlite3.Connection] = None
        self.data = {
            "active_session": "default",
            "sessions": {
                "default": {
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "modified": datetime.now().isoformat(),
                        "description": "Session par défaut",
                    },
                    "context": {
                        "global_context": "",
                        "indexed_entities": [],
                    },
                    "turns": [],
                }
            },
        }

        if use_db or os.getenv("RAG_SESSIONS_DB"):
            self.db_path = db_path or os.getenv("RAG_SESSIONS_DB", DB_SESSIONS_PATH)
            self.use_db = bool(self.db_path)
            if self.use_db:
                self._connect_db()
                self._ensure_db_schema()

        self._load()

    def _connect_db(self):
        if self.conn:
            return
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        except Exception as e:
            logger.error(f"❌ Impossible de se connecter à la base SQLite: {e}")
            self.conn = None
            self.use_db = False

    def _ensure_db_schema(self):
        if not self.conn:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                description TEXT,
                created TEXT,
                modified TEXT,
                global_context TEXT,
                indexed_entities TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                question TEXT,
                answer TEXT,
                metadata TEXT,
                rag_context TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self.conn.commit()

    def _get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        if not self.conn:
            return default
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else default

    def _set_setting(self, key: str, value: str):
        if not self.conn:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def _load(self):
        if self.use_db and self.conn:
            self._load_db()
        else:
            self._load_json()

    def _load_json(self):
        """Charge les sessions depuis le fichier JSON."""
        if self.file_path and os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)

                if "sessions" in loaded and isinstance(loaded["sessions"], dict):
                    for session_id, session_data in loaded["sessions"].items():
                        if isinstance(session_data, list):
                            loaded["sessions"][session_id] = {
                                "metadata": {
                                    "created": datetime.now().isoformat(),
                                    "modified": datetime.now().isoformat(),
                                    "description": session_id,
                                },
                                "context": {
                                    "global_context": "",
                                    "indexed_entities": [],
                                },
                                "turns": session_data,
                            }
                        elif isinstance(session_data, dict) and "turns" not in session_data:
                            loaded["sessions"][session_id] = {
                                "metadata": {
                                    "created": datetime.now().isoformat(),
                                    "modified": datetime.now().isoformat(),
                                    "description": session_id,
                                },
                                "context": session_data.get("context", {
                                    "global_context": "",
                                    "indexed_entities": [],
                                }),
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
                        "context": {
                            "global_context": "",
                            "indexed_entities": [],
                        },
                        "turns": [],
                    }

                logger.info(f"✅ Sessions chargées: {len(self.data['sessions'])} session(s)")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de charger {self.file_path}: {e}")
                self._ensure_default_session()
        else:
            self._ensure_default_session()

    def _load_db(self):
        """Charge les sessions depuis la base SQLite."""
        if not self.conn:
            self._ensure_default_session()
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT session_id FROM sessions")
        sessions = cursor.fetchall()

        if not sessions and self.file_path and os.path.exists(self.file_path):
            self._migrate_json_to_db()
        elif self.file_path and os.path.exists(self.file_path):
            cursor.execute("SELECT COUNT(1) AS count FROM turns")
            if cursor.fetchone()["count"] == 0:
                self._migrate_json_turns_to_db()

        active_session = self._get_setting("active_session", "default")
        self.data["active_session"] = active_session

        cursor.execute("SELECT session_id, description, created, modified, global_context, indexed_entities FROM sessions")
        rows = cursor.fetchall()
        for row in rows:
            indexed_entities = []
            try:
                indexed_entities = json.loads(row["indexed_entities"] or "[]")
            except Exception:
                indexed_entities = []

            self.data["sessions"][row["session_id"]] = {
                "metadata": {
                    "description": row["description"],
                    "created": row["created"],
                    "modified": row["modified"],
                },
                "context": {
                    "global_context": row["global_context"] or "",
                    "indexed_entities": indexed_entities,
                },
                "turns": [],
            }

        cursor.execute(
            "SELECT session_id, timestamp, question, answer, metadata, rag_context FROM turns ORDER BY id"
        )
        rows = cursor.fetchall()
        for turn in rows:
            session_id = turn["session_id"]
            if session_id not in self.data["sessions"]:
                continue
            metadata = {}
            rag_context = {}
            try:
                metadata = json.loads(turn["metadata"] or "{}")
            except Exception:
                metadata = {}
            try:
                rag_context = json.loads(turn["rag_context"] or "{}")
            except Exception:
                rag_context = {}

            self.data["sessions"][session_id]["turns"].append(
                {
                    "timestamp": turn["timestamp"],
                    "question": turn["question"],
                    "answer": turn["answer"],
                    "metadata": metadata,
                    "rag_context": rag_context,
                }
            )

        if self.data["active_session"] not in self.data["sessions"]:
            self.data["active_session"] = "default"

        self._ensure_default_session()
        logger.info(f"✅ Sessions SQLite chargées: {len(self.data['sessions'])} session(s)")

    def _migrate_json_to_db(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if "sessions" not in loaded or not isinstance(loaded["sessions"], dict):
                return

            for session_id, session_data in loaded["sessions"].items():
                if isinstance(session_data, list):
                    session_obj = {
                        "metadata": {
                            "created": datetime.now().isoformat(),
                            "modified": datetime.now().isoformat(),
                            "description": session_id,
                        },
                        "context": {
                            "global_context": "",
                            "indexed_entities": [],
                        },
                        "turns": session_data,
                    }
                elif isinstance(session_data, dict):
                    session_obj = {
                        "metadata": session_data.get("metadata", {
                            "created": datetime.now().isoformat(),
                            "modified": datetime.now().isoformat(),
                            "description": session_id,
                        }),
                        "context": session_data.get("context", {
                            "global_context": "",
                            "indexed_entities": [],
                        }),
                        "turns": session_data.get("turns", []),
                    }
                else:
                    continue

                self._insert_session_db(session_id, session_obj)
                for turn in session_obj.get("turns", []):
                    self._insert_turn_db(session_id, turn)

            active_session = loaded.get("active_session", "default")
            self._set_setting("active_session", active_session)
        except Exception as e:
            logger.warning(f"⚠️ Migration JSON vers SQLite échouée: {e}")

    def _migrate_json_turns_to_db(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            sessions = loaded.get("sessions", {})
            if not isinstance(sessions, dict):
                return
            for session_id, session_data in sessions.items():
                if isinstance(session_data, list):
                    turns = session_data
                elif isinstance(session_data, dict):
                    turns = session_data.get("turns", [])
                else:
                    turns = []
                for turn in turns:
                    self._insert_turn_db(session_id, turn)
        except Exception as e:
            logger.warning(f"Migration des tours JSON vers SQLite echouee: {e}")

    def _ensure_default_session(self):
        """Assure qu'une session par défaut existe."""
        if self.use_db and self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(1) as count FROM sessions WHERE session_id = ?", ("default",))
            if cursor.fetchone()["count"] == 0:
                self.create_session("default", "Session par défaut", "")
            if not self._get_setting("active_session"):
                self._set_setting("active_session", "default")
            return

        if "default" not in self.data["sessions"]:
            self.data["sessions"]["default"] = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "description": "Session par défaut",
                },
                "context": {
                    "global_context": "",
                    "indexed_entities": [],
                },
                "turns": [],
            }

    def _save(self):
        """Sauvegarde les sessions dans le fichier ou la base."""
        if (self.use_db and self.conn) or not self.file_path:
            return

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")

    def _insert_session_db(self, session_id: str, session_obj: Dict) -> bool:
        if not self.conn:
            return False
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO sessions(session_id, description, created, modified, global_context, indexed_entities) VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                session_obj["metadata"].get("description", session_id),
                session_obj["metadata"].get("created", datetime.now().isoformat()),
                session_obj["metadata"].get("modified", datetime.now().isoformat()),
                session_obj["context"].get("global_context", ""),
                json.dumps(session_obj["context"].get("indexed_entities", []), ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return True

    def _insert_turn_db(self, session_id: str, turn: Dict) -> bool:
        if not self.conn:
            return False
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO turns(session_id, timestamp, question, answer, metadata, rag_context) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
                turn.get("question", ""),
                turn.get("answer", ""),
                json.dumps(turn.get("metadata", {}), ensure_ascii=False),
                json.dumps(turn.get("rag_context", {}), ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return True

    def current_session(self) -> str:
        if self.use_db and self.conn:
            active = self._get_setting("active_session", "default")
            return active or "default"
        return self.data["active_session"]

    def list_sessions(self) -> List[Tuple[str, Dict]]:
        if self.use_db and self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT s.session_id, s.description, s.created, s.modified, COUNT(t.id) AS turns "
                "FROM sessions s LEFT JOIN turns t ON s.session_id = t.session_id "
                "GROUP BY s.session_id ORDER BY s.session_id"
            )
            sessions = []
            active = self.current_session()
            for row in cursor.fetchall():
                sessions.append(
                    (
                        row["session_id"],
                        {
                            "description": row["description"],
                            "created": row["created"],
                            "modified": row["modified"],
                            "turns": row["turns"],
                            "is_active": row["session_id"] == active,
                        },
                    )
                )
            return sessions

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
                        "is_active": session_id == self.current_session(),
                    },
                )
            )
        return sessions

    def create_session(
        self, session_id: str, description: str = "", context: str = ""
    ) -> bool:
        if self.use_db and self.conn:
            if self.get_session_info(session_id):
                logger.warning(f"Session '{session_id}' existe déjà")
                return False
            session_obj = {
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "description": description or session_id,
                },
                "context": {
                    "global_context": context,
                    "indexed_entities": [],
                },
                "turns": [],
            }
            self._insert_session_db(session_id, session_obj)
            logger.info(f"Session '{session_id}' créée")
            return True

        if session_id in self.data["sessions"]:
            logger.warning(f"Session '{session_id}' existe déjà")
            return False
        self.data["sessions"][session_id] = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "description": description or session_id,
            },
            "context": {
                "global_context": context,
                "indexed_entities": [],
            },
            "turns": [],
        }
        logger.info(f"Session '{session_id}' créée")
        self._save()
        return True

    def delete_session(self, session_id: str) -> bool:
        if session_id == "default":
            logger.warning("⚠️ Impossible de supprimer la session par défaut")
            return False
        if self.use_db and self.conn:
            if not self.get_session_info(session_id):
                logger.warning(f"⚠️ Session '{session_id}' non trouvée")
                return False
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self.conn.commit()
            if self.current_session() == session_id:
                self.switch_session("default")
            logger.info(f"✅ Session '{session_id}' supprimée")
            return True

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
        if self.use_db and self.conn:
            if not self.get_session_info(session_id):
                logger.warning(f"⚠️ Session '{session_id}' non trouvée")
                return False
            self._set_setting("active_session", session_id)
            logger.info(f"✅ Basculé vers la session '{session_id}'")
            return True

        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return False
        self.data["active_session"] = session_id
        logger.info(f"✅ Basculé vers la session '{session_id}'")
        self._save()
        return True

    def _add_turn_legacy(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict] = None,
        rag_context: Optional[Dict] = None,
    ) -> bool:
        if self.use_db and self.conn:
            session_id = self.current_session()
            if not self.get_session_info(session_id):
                logger.warning(f"⚠️ Session '{session_id}' non trouvée")
                return False
            turn = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "answer": answer,
                "metadata": metadata or {},
                "rag_context": rag_context or {},
            }
            self._insert_turn_db(session_id, turn)
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sessions SET modified = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            self.conn.commit()
            return True

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
            if rag_context:
                turn["rag_context"] = {
                    "sources": rag_context.get("sources", []),
                    "scores": rag_context.get("scores", []),
                    "method": rag_context.get("method", ""),
                    "entities": rag_context.get("entities", []),
                    "documents_count": rag_context.get("documents_count", 0),
                }
            session["turns"].append(turn)
            session["metadata"]["modified"] = datetime.now().isoformat()
            self._save()
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du tour: {e}")
            return False

    def add_turn(
        self,
        *args,
        session_id: Optional[str] = None,
        user_message: Optional[str] = None,
        assistant_answer: Optional[str] = None,
        sources: Optional[List] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        metadata: Optional[Dict] = None,
        rag_context: Optional[Dict] = None,
    ) -> bool:
        """Store a turn using either the session-aware or legacy API."""
        if len(args) >= 3:
            session_id, user_message, assistant_answer = args[:3]
        elif len(args) == 2:
            question, answer = args
        elif len(args) == 1 and question is None and user_message is None:
            question = args[0]

        question = str(user_message if user_message is not None else question or "")
        answer = str(
            assistant_answer if assistant_answer is not None else answer or ""
        )
        target_session = str(session_id or self.current_session())
        if not self.get_session_info(target_session):
            self.create_session(target_session, target_session, "")

        previous_session = self.current_session()
        if target_session != previous_session:
            self.switch_session(target_session)
        context = dict(rag_context or {})
        if sources is not None:
            context["sources"] = sources
        try:
            stored = self._add_turn_legacy(
                question,
                answer,
                metadata=metadata,
                rag_context=context,
            )
            if stored:
                if self.use_db and self.conn:
                    # SQLite is the durable audit/history store. Limit context at
                    # read time, not by deleting old question/answer turns.
                    pass
                else:
                    self._save()
                self.refresh_session_memory(target_session)
            return stored
        finally:
            if target_session != previous_session:
                self.switch_session(previous_session)

    @staticmethod
    def _source_label(source_item) -> str:
        if isinstance(source_item, dict):
            return str(
                source_item.get("filename")
                or source_item.get("source")
                or source_item.get("title")
                or ""
            ).strip()
        return str(source_item or "").strip()

    @staticmethod
    def _important_terms(text: str) -> List[str]:
        words = re.findall(r"[A-Za-zÀ-ÿ0-9_.-]{3,}", str(text or "").lower())
        terms = []
        for word in words:
            normalized = word.strip("._-")
            ascii_norm = unicodedata.normalize("NFD", normalized)
            ascii_norm = "".join(
                char for char in ascii_norm
                if unicodedata.category(char) != "Mn"
            )
            if normalized and ascii_norm not in SESSION_STOPWORDS:
                terms.append(normalized)
        return terms

    def get_recent_sources(
        self,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[str]:
        sources = []
        seen = set()
        for turn in reversed(self.get_history(session_id=session_id, limit=0)):
            context = turn.get("rag_context") or {}
            for item in context.get("sources", []) or []:
                label = self._source_label(item)
                if label and label not in seen:
                    seen.add(label)
                    sources.append(label)
                    if len(sources) >= limit:
                        return sources
        return sources

    def build_session_summary(
        self,
        session_id: Optional[str] = None,
        max_turns: int = 8,
    ) -> Dict:
        turns = self.get_history(session_id=session_id, limit=max_turns)
        if not turns:
            return {"summary": "", "topics": [], "sources": []}

        term_counts = Counter()
        for turn in turns:
            term_counts.update(self._important_terms(turn.get("question", "")))
        topics = [term for term, _count in term_counts.most_common(8)]
        sources = self.get_recent_sources(session_id=session_id, limit=5)
        last_question = turns[-1].get("question", "")

        pieces = []
        if topics:
            pieces.append("Sujets: " + ", ".join(topics[:6]))
        if sources:
            pieces.append("Sources recentes: " + ", ".join(sources[:4]))
        if last_question:
            pieces.append("Derniere question: " + last_question[:180])
        return {
            "summary": " | ".join(pieces),
            "topics": topics,
            "sources": sources,
        }

    def refresh_session_memory(self, session_id: Optional[str] = None) -> Dict:
        if session_id is None:
            session_id = self.current_session()
        summary = self.build_session_summary(session_id=session_id)
        self.set_session_context(session_id, summary.get("summary", ""))
        if summary.get("topics"):
            self.add_indexed_entities(summary["topics"], session_id=session_id)
        return summary

    def set_session_context(self, session_id: Optional[str] = None, context: str = "") -> bool:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            if not self.get_session_info(session_id):
                logger.warning(f"Session '{session_id}' non trouvée")
                return False
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sessions SET global_context = ?, modified = ? WHERE session_id = ?",
                (context, datetime.now().isoformat(), session_id),
            )
            self.conn.commit()
            return True

        try:
            if session_id is None:
                session_id = self.current_session()
            if session_id not in self.data["sessions"]:
                logger.warning(f"Session '{session_id}' non trouvée")
                return False
            if "context" not in self.data["sessions"][session_id]:
                self.data["sessions"][session_id]["context"] = {
                    "global_context": "",
                    "indexed_entities": [],
                }
            self.data["sessions"][session_id]["context"]["global_context"] = context
            self.data["sessions"][session_id]["metadata"]["modified"] = datetime.now().isoformat()
            self._save()
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la définition du contexte: {e}")
            return False

    def add_indexed_entities(
        self,
        entities: List[str],
        session_id: Optional[str] = None,
    ) -> bool:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            session_info = self.get_session_info(session_id)
            if not session_info:
                return False
            current_entities = session_info["context"].get("indexed_entities", [])
            merged = list(set(current_entities) | set(entities))
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sessions SET indexed_entities = ?, modified = ? WHERE session_id = ?",
                (json.dumps(merged, ensure_ascii=False), datetime.now().isoformat(), session_id),
            )
            self.conn.commit()
            return True

        try:
            if session_id is None:
                session_id = self.current_session()
            if session_id not in self.data["sessions"]:
                return False
            if "context" not in self.data["sessions"][session_id]:
                self.data["sessions"][session_id]["context"] = {
                    "global_context": "",
                    "indexed_entities": [],
                }
            existing = set(self.data["sessions"][session_id]["context"]["indexed_entities"])
            existing.update(entities)
            self.data["sessions"][session_id]["context"]["indexed_entities"] = list(existing)
            self.data["sessions"][session_id]["metadata"]["modified"] = datetime.now().isoformat()
            self._save()
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des entités: {e}")
            return False

    def get_session_context(self, session_id: Optional[str] = None) -> Dict:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            info = self.get_session_info(session_id)
            return info.get("context", {"global_context": "", "indexed_entities": []})

        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            return {}
        return self.data["sessions"][session_id].get("context", {"global_context": "", "indexed_entities": []})

    def get_recent_questions(self, limit: int = 2) -> List[str]:
        """Dernières questions de la session active (pour enrichir le retrieval)."""
        turns = self.get_history(limit=limit)
        return [t.get("question", "") for t in turns if t.get("question")]

    def get_history(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        if limit is None:
            limit = self.max_history_turns
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            cursor = self.conn.cursor()
            if limit is not None and limit > 0:
                cursor.execute(
                    "SELECT timestamp, question, answer, metadata, rag_context "
                    "FROM ("
                    "  SELECT id, timestamp, question, answer, metadata, rag_context "
                    "  FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (session_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT timestamp, question, answer, metadata, rag_context "
                    "FROM turns WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
            rows = cursor.fetchall()
            turns = []
            for row in rows:
                metadata = {}
                rag_context = {}
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except Exception:
                    metadata = {}
                try:
                    rag_context = json.loads(row["rag_context"] or "{}")
                except Exception:
                    rag_context = {}
                turns.append(
                    {
                        "timestamp": row["timestamp"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "metadata": metadata,
                        "rag_context": rag_context,
                    }
                )
            return turns

        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return []
        turns = self.data["sessions"][session_id].get("turns", [])
        return turns if limit <= 0 else turns[-limit:]

    def get_session_info(self, session_id: Optional[str] = None) -> Dict:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT session_id, description, created, modified, global_context, indexed_entities FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            indexed_entities = []
            try:
                indexed_entities = json.loads(row["indexed_entities"] or "[]")
            except Exception:
                indexed_entities = []
            return {
                "id": row["session_id"],
                "description": row["description"],
                "created": row["created"],
                "modified": row["modified"],
                "turns_count": len(self.get_history(session_id, limit=0)),
                "is_active": row["session_id"] == self.current_session(),
                "context": {
                    "global_context": row["global_context"] or "",
                    "indexed_entities": indexed_entities,
                },
            }

        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            return {}
        session = self.data["sessions"][session_id]
        metadata = session.get("metadata", {})
        context = session.get("context", {})
        return {
            "id": session_id,
            "description": metadata.get("description", "N/A"),
            "created": metadata.get("created", "N/A"),
            "modified": metadata.get("modified", "N/A"),
            "turns_count": len(session.get("turns", [])),
            "is_active": session_id == self.current_session(),
            "context": context,
        }

    def get_turn_context(
        self,
        turn_index: int,
        session_id: Optional[str] = None,
    ) -> Dict:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            turns = self.get_history(session_id, limit=0)
            if turn_index < 0 or turn_index >= len(turns):
                return {}
            return turns[turn_index].get("rag_context", {})

        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            return {}
        turns = self.data["sessions"][session_id].get("turns", [])
        if turn_index < 0 or turn_index >= len(turns):
            return {}
        return turns[turn_index].get("rag_context", {})

    def export_session_context(
        self,
        session_id: Optional[str] = None,
        format: str = "json",
    ) -> str:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            info = self.get_session_info(session_id)
            turns = self.get_history(session_id, limit=0)
            context_data = {
                "session_id": session_id,
                "metadata": {
                    "description": info.get("description", "N/A"),
                    "created": info.get("created", "N/A"),
                    "modified": info.get("modified", "N/A"),
                },
                "context": info.get("context", {}),
                "turns_with_context": turns,
            }
            if format == "json":
                return json.dumps(context_data, ensure_ascii=False, indent=2)
            if format == "markdown":
                md = f"# Session: {session_id}\n\n"
                md += f"**Contexte Global**: {context_data['context'].get('global_context', '')}\n\n"
                md += f"**Entités Indexées**: {', '.join(context_data['context'].get('indexed_entities', []))}\n\n"
                md += "## Historique\n\n"
                for turn in context_data["turns_with_context"]:
                    md += f"### Q: {turn['question']}\n"
                    md += f"**A**: {turn['answer']}\n"
                    if turn.get("rag_context"):
                        md += f"- **Sources**: {turn['rag_context'].get('documents_count', 0)} docs\n"
                        md += f"- **Méthode**: {turn['rag_context'].get('method', 'N/A')}\n"
                    md += "\n"
                return md
            return json.dumps(context_data, ensure_ascii=False, indent=2)

        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            return ""
        session = self.data["sessions"][session_id]
        context_data = {
            "session_id": session_id,
            "metadata": session.get("metadata", {}),
            "context": session.get("context", {}),
            "turns_with_context": session.get("turns", []),
        }
        if format == "json":
            return json.dumps(context_data, ensure_ascii=False, indent=2)
        if format == "markdown":
            md = f"# Session: {session_id}\n\n"
            md += f"**Contexte Global**: {context_data['context'].get('global_context', '')}\n\n"
            md += f"**Entités Indexées**: {', '.join(context_data['context'].get('indexed_entities', []))}\n\n"
            md += "## Historique\n\n"
            for turn in context_data["turns_with_context"]:
                md += f"### Q: {turn['question']}\n"
                md += f"**A**: {turn['answer']}\n"
                if turn.get("rag_context"):
                    md += f"- **Sources**: {turn['rag_context'].get('documents_count', 0)} docs\n"
                    md += f"- **Méthode**: {turn['rag_context'].get('method', 'N/A')}\n"
                md += "\n"
            return md
        return json.dumps(context_data, ensure_ascii=False, indent=2)

    def update_session_description(self, session_id: str, description: str) -> bool:
        if self.use_db and self.conn:
            if not self.get_session_info(session_id):
                logger.warning(f"⚠️ Session '{session_id}' non trouvée")
                return False
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE sessions SET description = ?, modified = ? WHERE session_id = ?",
                (description, datetime.now().isoformat(), session_id),
            )
            self.conn.commit()
            return True
        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return False
        self.data["sessions"][session_id]["metadata"]["description"] = description
        self.data["sessions"][session_id]["metadata"]["modified"] = datetime.now().isoformat()
        logger.info(f"✅ Description de '{session_id}' mise à jour")
        self._save()
        return True

    def clear_session(self, session_id: Optional[str] = None) -> bool:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            if not self.get_session_info(session_id):
                logger.warning(f"⚠️ Session '{session_id}' non trouvée")
                return False
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor.execute(
                "UPDATE sessions SET modified = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            self.conn.commit()
            return True
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
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            query_lower = f"%{query.lower()}%"
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT timestamp, question, answer, metadata, rag_context FROM turns "
                "WHERE session_id = ? AND (LOWER(question) LIKE ? OR LOWER(answer) LIKE ?) "
                "ORDER BY id DESC",
                (session_id, query_lower, query_lower),
            )
            results = []
            for row in cursor.fetchall():
                metadata = {}
                rag_context = {}
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except Exception:
                    metadata = {}
                try:
                    rag_context = json.loads(row["rag_context"] or "{}")
                except Exception:
                    rag_context = {}
                results.append(
                    {
                        "timestamp": row["timestamp"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "metadata": metadata,
                        "rag_context": rag_context,
                    }
                )
            return results
        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            return []
        query_lower = query.lower()
        return [
            turn
            for turn in self.data["sessions"][session_id].get("turns", [])
            if query_lower in turn.get("question", "").lower()
            or query_lower in turn.get("answer", "").lower()
        ]

    def export_session(self, session_id: Optional[str] = None, format: str = "json") -> str:
        if self.use_db and self.conn:
            if session_id is None:
                session_id = self.current_session()
            info = self.get_session_info(session_id)
            if not info:
                logger.warning(f"⚠️ Session '{session_id}' non trouvée")
                return ""
            session = {
                "metadata": {
                    "description": info["description"],
                    "created": info["created"],
                    "modified": info["modified"],
                },
                "context": info.get("context", {}),
                "turns": self.get_history(session_id, limit=0),
            }
            if format == "json":
                return json.dumps(session, ensure_ascii=False, indent=2)
            if format == "markdown":
                lines = [
                    f"# Session: {session_id}\n",
                    f"**Description**: {session['metadata'].get('description', 'N/A')}\n",
                    f"**Créée**: {session['metadata'].get('created', 'N/A')}\n",
                    f"**Modifiée**: {session['metadata'].get('modified', 'N/A')}\n",
                    "\n---\n\n",
                ]
                for i, turn in enumerate(session["turns"], 1):
                    lines.append(f"## Tour {i}\n")
                    lines.append(f"**Horodatage**: {turn.get('timestamp', 'N/A')}\n\n")
                    lines.append(f"### Question\n{turn.get('question', 'N/A')}\n\n")
                    lines.append(f"### Réponse\n{turn.get('answer', 'N/A')}\n\n")
                    lines.append("---\n\n")
                return "".join(lines)
            if format == "txt":
                lines = [
                    f"Session: {session_id}\n",
                    f"Description: {session['metadata'].get('description', 'N/A')}\n",
                    f"Créée: {session['metadata'].get('created', 'N/A')}\n",
                    f"Modifiée: {session['metadata'].get('modified', 'N/A')}\n",
                    "\n" + "=" * 80 + "\n\n",
                ]
                for i, turn in enumerate(session["turns"], 1):
                    lines.append(f"Tour {i} - {turn.get('timestamp', 'N/A')}\n")
                    lines.append(f"Q: {turn.get('question', 'N/A')}\n")
                    lines.append(f"R: {turn.get('answer', 'N/A')}\n")
                    lines.append("\n" + "-" * 80 + "\n\n")
                return "".join(lines)
            return ""

        if session_id is None:
            session_id = self.current_session()
        if session_id not in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' non trouvée")
            return ""
        session = self.data["sessions"][session_id]
        if format == "json":
            return json.dumps(session, ensure_ascii=False, indent=2)
        if format == "markdown":
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
        if format == "txt":
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
        if self.use_db and self.conn:
            if self.get_session_info(session_id):
                logger.warning(f"⚠️ Session '{session_id}' existe déjà")
                return False
            session_obj = {
                "metadata": data.get("metadata", {
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "description": session_id,
                }),
                "context": data.get("context", {
                    "global_context": "",
                    "indexed_entities": [],
                }),
                "turns": data.get("turns", []),
            }
            self._insert_session_db(session_id, session_obj)
            for turn in session_obj["turns"]:
                self._insert_turn_db(session_id, turn)
            return True

        if session_id in self.data["sessions"]:
            logger.warning(f"⚠️ Session '{session_id}' existe déjà")
            return False
        self.data["sessions"][session_id] = {
            "metadata": data.get("metadata", {
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "description": session_id,
            }),
            "context": data.get("context", {
                "global_context": "",
                "indexed_entities": [],
            }),
            "turns": data.get("turns", []),
        }
        logger.info(f"✅ Session '{session_id}' importée")
        self._save()
        return True
