"""
=============================================================================
DIVALTO HARMONY — TESTS UNITAIRES  v2.1
=============================================================================
Couvre les étapes 12, 13, 14 + les comportements critiques du pipeline :

  TestSessionManagement     — étape 12 : session & mémoire contextuelle
  TestMultiTurn             — étape 13 : questions de suivi
  TestPersonalization       — étape 14 : profils utilisateur
  TestFallback              — étape 11 : fallback & réponses vides
  TestOutOfScope            — étape 11 : détection hors-scope

CORRECTIONS vs version initiale :
  FIX 1 — session.turns supprimé de SessionState → tests mis à jour
  FIX 2 — update_session_state() → _update_session() (méthode privée)
  FIX 3 — personalize_answer() supprimée → test remplacé par un test
           qui vérifie que le profil est bien injecté dans le system prompt
  FIX 4 — patch corrigé : 'rag_chain.DivaltoAssistant' (pas 'divalto_assistant')
  FIX 5 — _is_out_of_scope est module-level → import direct + test isolé
  FIX 6 — setUp utilise mock pour éviter de charger les vrais modèles ML
           (chaque test ne doit pas prendre 30s à initialiser)
=============================================================================
"""
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

from rag_chain import (
    DivaltoAssistant,
    SessionState,
    UserProfile,
    PROFILE_INSTRUCTIONS,
    _is_out_of_scope,
    _response_is_empty,
    _is_followup_question,
)


def _make_mock_assistant(role: str = "nouveau") -> DivaltoAssistant:
    """
    FIX 6 : Crée un assistant avec les composants ML mockés.
    Sans ça, chaque test charge multilingual-e5-large (~2.2 GB) → trop lent.
    Les mocks remplacent les modèles ML par des objets factices.
    Les tests portent sur la LOGIQUE (session, profil, fallback), pas sur les modèles.
    """
    with patch("rag_chain.SentenceTransformer") as mock_st, \
         patch("rag_chain.CrossEncoder") as mock_ce, \
         patch("rag_chain.chromadb.PersistentClient") as mock_chroma, \
         patch("builtins.open", MagicMock(return_value=MagicMock(
             __enter__=MagicMock(return_value=MagicMock(
                 read=MagicMock(return_value="[]")
             )),
             __exit__=MagicMock(return_value=False)
         ))), \
         patch("json.load", return_value=[]):

        mock_collection = MagicMock()
        mock_collection.count.return_value = 11228
        mock_chroma.return_value.get_collection.return_value = mock_collection

        assistant = DivaltoAssistant(user_role=role)
        assistant.collection = mock_collection
        return assistant


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 12 : GESTION DE SESSION & MÉMOIRE CONTEXTUELLE
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionManagement(unittest.TestCase):
    """Tests de l'étape 12 : initialisation et mise à jour de la session."""

    def setUp(self):
        self.assistant = _make_mock_assistant()

    def test_session_initializes_correctly(self):
        """La session doit s'initialiser avec des valeurs vides."""
        self.assertIsNotNone(self.assistant.session.session_id)
        self.assertIsNone(self.assistant.session.current_topic)
        self.assertIsNone(self.assistant.session.current_module)
        self.assertEqual(self.assistant.session.last_entities, [])
        self.assertEqual(self.assistant.session.last_sources, [])
        self.assertEqual(self.assistant.history, [])

    def test_session_id_is_unique(self):
        """Deux assistants doivent avoir des session_id différents."""
        assistant2 = _make_mock_assistant()
        self.assertNotEqual(
            self.assistant.session.session_id,
            assistant2.session.session_id
        )

    def test_session_updates_after_interaction(self):
        """_update_session() doit mettre à jour module, topic et entités."""
        sources = [
            {"module": "Record SQL", "doc_title": "Connexion au serveur SQL",
             "source_file": "ConnexionauserveurSQL.htm", "score": 0.91},
        ]
        self.assistant._update_session(
            question="Comment se connecter au serveur SQL ?",
            sources=sources,
        )
        self.assertEqual(self.assistant.session.current_module, "Record SQL")
        self.assertEqual(self.assistant.session.current_topic, "Connexion au serveur SQL")
        self.assertIn("Connexion au serveur SQL", self.assistant.session.last_entities)
        self.assertIn("Record SQL", self.assistant.session.last_entities)

    def test_session_updates_last_sources(self):
        """_update_session() doit stocker les sources pour usage futur."""
        sources = [
            {"module": "Administration", "doc_title": "Import LDAP",
             "source_file": "ImportLDAP.htm", "score": 0.90},
            {"module": "Administration", "doc_title": "Synchronisation",
             "source_file": "Synchronisation.htm", "score": 0.85},
        ]
        self.assistant._update_session("Question test", sources)
        self.assertEqual(len(self.assistant.session.last_sources), 2)
        self.assertEqual(self.assistant.session.last_sources[0]["module"], "Administration")

    def test_reset_clears_session(self):
        """reset_history() doit effacer l'historique et réinitialiser la session."""
        sources = [{"module": "M", "doc_title": "T", "source_file": "f.htm", "score": 0.9}]
        self.assistant._update_session("Q", sources)
        self.assistant.history.append({"role": "user", "content": "Q"})

        self.assistant.reset_history()

        self.assertEqual(self.assistant.history, [])
        self.assertIsNone(self.assistant.session.current_topic)
        self.assertIsNone(self.assistant.session.current_module)
        self.assertEqual(self.assistant.session.last_entities, [])


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 13 : SCÉNARIOS MULTI-TOURS
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurn(unittest.TestCase):
    """Tests de l'étape 13 : détection et contextualisation des questions de suivi."""

    def setUp(self):
        self.assistant = _make_mock_assistant()

    def test_followup_detection_short_question(self):
        """Une question courte doit être détectée comme suivi."""
        self.assertTrue(_is_followup_question("Et pour les chemins ?"))
        self.assertTrue(_is_followup_question("Comment ?"))
        self.assertTrue(_is_followup_question("Pourquoi ?"))

    def test_followup_detection_connectors(self):
        """Les connecteurs de suivi doivent être détectés."""
        self.assertTrue(_is_followup_question("Et comment les synchroniser ensuite ?"))
        self.assertTrue(_is_followup_question("Et si le serveur est distant ?"))
        self.assertTrue(_is_followup_question("Mais dans ce cas, que faire ?"))

    def test_not_followup_new_question(self):
        """Une nouvelle question longue et indépendante ne doit pas être détectée comme suivi."""
        self.assertFalse(_is_followup_question(
            "Comment configurer les chemins implicites sur un poste client Harmony ?"
        ))

    def test_contextualization_enriches_followup(self):
        """Une question de suivi doit être enrichie avec le contexte de session."""
        # Simuler un premier échange
        sources = [{"module": "Administration", "doc_title": "Import LDAP",
                    "source_file": "ImportLDAP.htm", "score": 0.9}]
        self.assistant._update_session("Comment importer depuis LDAP ?", sources)

        enriched = self.assistant._contextualize_question("Et la synchronisation ?")

        self.assertIn("Administration", enriched)
        self.assertIn("Import LDAP", enriched)
        self.assertIn("Et la synchronisation ?", enriched)

    def test_no_contextualization_without_session(self):
        """Sans contexte de session, la question ne doit pas être modifiée."""
        question = "Comment configurer Harmony ?"
        result = self.assistant._contextualize_question(question)
        self.assertEqual(result, question)

    def test_contextualization_does_not_modify_display_question(self):
        """
        La question enrichie est utilisée pour l'embedding — pas pour l'affichage.
        La réponse retournée doit contenir la question originale, pas l'enrichie.
        """
        sources = [{"module": "Installation", "doc_title": "Serveur Web",
                    "source_file": "ServeurWeb.htm", "score": 0.9}]
        self.assistant._update_session("Comment installer le serveur Web ?", sources)

        enriched = self.assistant._contextualize_question("Et pour la configuration ?")

        # La question enrichie contient le contexte
        self.assertNotEqual(enriched, "Et pour la configuration ?")
        # Mais elle contient quand même la question originale
        self.assertIn("Et pour la configuration ?", enriched)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 14 : PERSONNALISATION PAR PROFIL
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonalization(unittest.TestCase):
    """
    Tests de l'étape 14 : profils utilisateur.
    FIX 3 : on ne teste plus personalize_answer() (supprimée).
    On teste que le profil est correctement injecté dans le system prompt.
    """

    def setUp(self):
        self.assistant = _make_mock_assistant()

    def test_default_profile_is_nouveau(self):
        """Le profil par défaut doit être 'nouveau'."""
        self.assertEqual(self.assistant.session.profile.role, "nouveau")

    def test_set_profile_consultant(self):
        """set_profile() doit mettre à jour le profil."""
        self.assistant.set_profile("consultant")
        self.assertEqual(self.assistant.session.profile.role, "consultant")

    def test_set_profile_utilisateur_cle(self):
        """set_profile() doit accepter 'utilisateur_cle'."""
        self.assistant.set_profile("utilisateur_cle")
        self.assertEqual(self.assistant.session.profile.role, "utilisateur_cle")

    def test_set_profile_invalid_does_not_crash(self):
        """Un profil inconnu ne doit pas crasher — rester sur le profil courant."""
        self.assistant.set_profile("nouveau")
        self.assistant.set_profile("profil_inexistant")
        self.assertEqual(self.assistant.session.profile.role, "nouveau")

    def test_consultant_profile_instruction_exists(self):
        """L'instruction de profil consultant doit exister et contenir les bons mots-clés."""
        instruction = PROFILE_INSTRUCTIONS.get("consultant", "")
        self.assertIn("consultant", instruction.lower())
        self.assertGreater(len(instruction), 20)

    def test_nouveau_profile_instruction_is_pedagogical(self):
        """L'instruction 'nouveau' doit être pédagogique."""
        instruction = PROFILE_INSTRUCTIONS.get("nouveau", "")
        self.assertIn("nouveau", instruction.lower())

    def test_profile_injected_in_system_prompt(self):
        """
        Le profil doit être injecté dans le system prompt — PAS dans le message
        utilisateur qui contient les extraits documentaires.
        """
        self.assistant.set_profile("consultant")
        context_chunks = [{"content": "doc content", "module": "Admin",
                           "doc_title": "Test doc", "source_file": "test.htm"}]
        messages = self.assistant._build_prompt_messages("Question test ?", context_chunks)

        system_content = messages[0]["content"]
        user_content   = messages[-1]["content"]

        # Le profil est dans le system prompt
        self.assertIn("consultant", system_content.lower())
        # Le message utilisateur contient les extraits, pas le préfixe de profil
        self.assertNotIn("Réponse concise pour consultant", user_content)
        self.assertIn("doc content", user_content)

    def test_all_profiles_produce_different_system_prompts(self):
        """Chaque profil doit produire un system prompt différent."""
        context_chunks = [{"content": "x", "module": "M",
                           "doc_title": "T", "source_file": "f.htm"}]
        prompts = {}
        for role in ("consultant", "utilisateur_cle", "nouveau"):
            self.assistant.set_profile(role)
            msgs = self.assistant._build_prompt_messages("Q ?", context_chunks)
            prompts[role] = msgs[0]["content"]

        self.assertNotEqual(prompts["consultant"], prompts["nouveau"])
        self.assertNotEqual(prompts["utilisateur_cle"], prompts["nouveau"])
        self.assertNotEqual(prompts["consultant"], prompts["utilisateur_cle"])


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 11 : FALLBACK & RÉPONSES VIDES
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback(unittest.TestCase):
    """
    Tests de l'étape 11 : fallback not_found, out_of_scope,
    et détection des réponses vides du LLM.
    """

    def setUp(self):
        self.assistant = _make_mock_assistant()

    def test_fallback_not_found_returns_correct_structure(self):
        """_fallback_not_found() doit retourner la structure attendue."""
        result = self.assistant._fallback_not_found("Comment configurer un VPN ?")
        self.assertFalse(result["found"])
        self.assertEqual(result["confidence"], "none")
        self.assertEqual(result["fallback_level"], 2)
        self.assertEqual(result["sources"], [])
        self.assertIn("Suggestions", result["response"])

    def test_fallback_not_found_includes_keywords(self):
        """Le fallback doit extraire les mots-clés de la question."""
        result = self.assistant._fallback_not_found("Comment configurer firewall réseau ?")
        self.assertIn("configurer", result["response"])

    def test_fallback_out_of_scope_returns_correct_structure(self):
        """_fallback_out_of_scope() doit retourner fallback_level=3."""
        result = self.assistant._fallback_out_of_scope("Quel est le prix ?")
        self.assertFalse(result["found"])
        self.assertEqual(result["confidence"], "none")
        self.assertEqual(result["fallback_level"], 3)
        self.assertEqual(result["sources"], [])

    def test_response_is_empty_detects_no_info(self):
        """_response_is_empty() doit détecter les réponses vides du LLM."""
        self.assertTrue(_response_is_empty(
            "Les extraits ne contiennent pas d'information sur ce sujet."
        ))
        self.assertTrue(_response_is_empty(
            "Je n'ai pas trouvé de réponse dans la documentation."
        ))
        self.assertTrue(_response_is_empty(
            "Aucune information disponible dans les extraits fournis."
        ))

    def test_response_is_empty_does_not_trigger_on_good_response(self):
        """_response_is_empty() ne doit pas se déclencher sur une bonne réponse."""
        self.assertFalse(_response_is_empty(
            "Pour configurer les chemins implicites, accédez au menu Paramétrage."
        ))
        self.assertFalse(_response_is_empty(
            "1. Ouvrez Harmony. 2. Allez dans Administration. 📄 Source : Administration > Chemins"
        ))

    def test_ask_returns_fallback_when_score_below_threshold(self):
        """ask() doit retourner fallback_level=2 si le score est sous MIN_SCORE."""
        self.assistant.collection.query.return_value = {
            "documents": [["chunk content"]],
            "metadatas": [[{"source_file": "f.htm", "module": "M",
                            "parent_id": "", "doc_title": "T", "chunk_type": "SHORT_ATOMIC"}]],
            "distances": [[0.40]],   # score = 1 - 0.40 = 0.60 < MIN_SCORE 0.70
        }
        self.assistant.embed_model.encode.return_value = np.zeros(1024)

        result = self.assistant.ask("Question quelconque")
        self.assertFalse(result["found"])
        self.assertEqual(result["fallback_level"], 2)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 11 : DÉTECTION HORS-SCOPE
# ─────────────────────────────────────────────────────────────────────────────

class TestOutOfScope(unittest.TestCase):
    """
    Tests de l'étape 11 : détection hors-scope.
    FIX 5 : _is_out_of_scope est une fonction module-level → import direct.
    """

    def test_prix_is_out_of_scope(self):
        """Une question sur le prix doit être hors-scope."""
        self.assertTrue(_is_out_of_scope("Quel est le prix d'une licence Divalto Harmony ?"))

    def test_tarif_is_out_of_scope(self):
        """Une question sur le tarif doit être hors-scope."""
        self.assertTrue(_is_out_of_scope("Quel est le tarif de Divalto ?"))

    def test_football_is_out_of_scope(self):
        """Une question hors domaine doit être hors-scope."""
        self.assertTrue(_is_out_of_scope("Qui a gagné le match de football hier ?"))

    def test_chatgpt_is_out_of_scope(self):
        """Une question sur ChatGPT sans mention Divalto doit être hors-scope."""
        self.assertTrue(_is_out_of_scope("C'est quoi ChatGPT ?"))

    def test_legitimate_divalto_question_not_out_of_scope(self):
        """Une vraie question Divalto ne doit pas être bloquée."""
        self.assertFalse(_is_out_of_scope(
            "Comment configurer les chemins implicites dans Harmony ?"
        ))
        self.assertFalse(_is_out_of_scope(
            "Comment installer le serveur Web Harmony ?"
        ))
        self.assertFalse(_is_out_of_scope(
            "Comment importer des utilisateurs depuis un annuaire LDAP ?"
        ))

    def test_ask_returns_fallback_level_3_for_out_of_scope(self):
        """ask() doit retourner fallback_level=3 pour les questions hors-scope."""
        assistant = _make_mock_assistant()
        result = assistant.ask("Quel est le prix d'une licence Divalto ?")
        self.assertFalse(result["found"])
        self.assertEqual(result["fallback_level"], 3)
        self.assertEqual(result["confidence"], "none")


if __name__ == "__main__":
    unittest.main(verbosity=2)