#!/usr/bin/env python3
"""Fast dependency-tolerant regression tests for the modular RAG stages."""
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np

from rag_compressor import compress_context
from rag_lexical import LexicalRetriever
from rag_mmr import maximal_marginal_relevance
from rag_query_transform import extract_query_metadata, rewrite_query
from rag_reranker import CrossEncoderReranker
from rag_hyde import build_hyde_queries, generate_hypothetical_document
from rag_verifier import verify_answer
from rag_conversation import reformulate_with_history, reformulate_with_history_info
from session_manager import SessionManager
from ingest_docs import Document, create_parent_child_chunks


class PipelineComponentTests(unittest.TestCase):
    def test_query_rewriting_preserves_original(self):
        query = "Comment configurer LDAP dans Harmony ?"
        variants = rewrite_query(query)
        self.assertEqual(variants[0], query)
        self.assertEqual(len(variants), len(set(value.lower() for value in variants)))
        self.assertGreaterEqual(len(variants), 2)
        self.assertEqual(extract_query_metadata(query)["domain"], "utilisateur")

    def test_lexical_retriever_returns_candidates(self):
        retriever = LexicalRetriever(
            [
                {"chunk_id": 0, "text": "Harmony configure les chemins.", "source": "a.htm"},
                {"chunk_id": 1, "text": "Gestion des imprimantes Windows.", "source": "b.htm"},
            ]
        )
        results = retriever.search("chemins Harmony", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], 0)
        self.assertIn(results[0]["retrieval_source"], {"bm25", "tfidf", "token_overlap"})

    def test_mmr_reduces_duplicates(self):
        candidates = [
            {"text": "alpha beta", "source": "same"},
            {"text": "alpha beta", "source": "same"},
            {"text": "gamma delta", "source": "other"},
        ]
        vectors = {
            "alpha beta": np.array([1.0, 0.0], dtype="float32"),
            "gamma delta": np.array([0.0, 1.0], dtype="float32"),
        }
        selected = maximal_marginal_relevance(
            np.array([1.0, 0.0], dtype="float32"),
            candidates,
            lambda texts: np.vstack([vectors[text] for text in texts]),
            top_k=2,
            lambda_mult=0.5,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({item["text"] for item in selected}), 2)

    def test_cross_encoder_disabled_fallback(self):
        reranker = CrossEncoderReranker(enabled=False)
        results = reranker.rerank(
            "question",
            [
                {"text": "low", "combined_score": 0.1},
                {"text": "high", "combined_score": 0.9},
            ],
            top_k=1,
        )
        self.assertEqual(results[0]["text"], "high")
        self.assertTrue(results[0]["cross_encoder_fallback"])

    def test_compression_only_uses_original_text(self):
        original = (
            "Harmony gere les chemins de fichiers. "
            "Les imprimantes utilisent le spouleur Windows."
        )
        result = compress_context(
            "Comment Harmony gere les chemins ?",
            [{"text": original, "source": "doc.htm"}],
            max_sentences=1,
        )[0]
        self.assertEqual(result["text"], original)
        self.assertIn(result["compressed_text"], original)

    def test_parent_child_chunks_and_mapping(self):
        parents, children = create_parent_child_chunks(
            [
                Document(
                    page_content=" ".join(f"token{i}" for i in range(1300)),
                    metadata={"source": "manual.htm", "title": "Manual"},
                )
            ]
        )
        self.assertTrue(parents)
        self.assertGreater(len(children), len(parents))
        parent_ids = {parent.metadata["parent_id"] for parent in parents}
        self.assertTrue(
            all(child.metadata["parent_id"] in parent_ids for child in children)
        )

        from query_docs import HybridRAG

        rag = HybridRAG.__new__(HybridRAG)
        rag.parents_by_id = {
            parents[0].metadata["parent_id"]: {
                "parent_id": parents[0].metadata["parent_id"],
                "text": parents[0].page_content,
                "source": "manual.htm",
            }
        }
        rag.child_to_parent = {
            children[0].metadata["child_id"]: children[0].metadata["parent_id"]
        }
        mapped = rag._map_results_to_parents(
            [
                {
                    "child_id": children[0].metadata["child_id"],
                    "parent_id": children[0].metadata["parent_id"],
                    "text": children[0].page_content,
                    "source": "manual.htm",
                    "hybrid_score": 0.8,
                    "retrieval_source": "hyde_vector",
                    "hyde_used": True,
                }
            ]
        )
        self.assertEqual(mapped[0]["text"], parents[0].page_content)
        self.assertEqual(mapped[0]["matched_child_text"], children[0].page_content)
        self.assertTrue(mapped[0]["hyde_used"])

    def test_hyde_is_retrieval_only(self):
        question = "Comment configurer LDAP dans Harmony ?"
        document = generate_hypothetical_document(
            question,
            extract_query_metadata(question),
        )
        queries = build_hyde_queries(question, extract_query_metadata(question))
        self.assertEqual(queries[0], question)
        self.assertIn(document, queries)
        self.assertGreaterEqual(len(document.split(".")), 3)
        self.assertNotIn("source", document.lower())

    def test_answer_verifier_supported_and_unsupported(self):
        contexts = [
            {
                "text": (
                    "LDAP configure les utilisateurs dans Harmony. "
                    "La configuration utilise les parametres de l'annuaire."
                )
            }
        ]
        supported = verify_answer(
            "Comment configurer LDAP dans Harmony ?",
            "LDAP configure les utilisateurs dans Harmony.",
            contexts,
        )
        unsupported = verify_answer(
            "Comment configurer LDAP dans Harmony ?",
            "La lune controle automatiquement toutes les imprimantes.",
            contexts,
        )
        self.assertGreater(supported["confidence"], unsupported["confidence"])
        self.assertTrue(supported["answer_supported"])
        self.assertTrue(unsupported["needs_retry"])
        self.assertTrue(unsupported["unsupported_claims"])

    def test_chat_memory_reformulation_and_clear(self):
        sessions = SessionManager(max_history_turns=2)
        sessions.add_turn(
            "test_session",
            "Comment creer une facture ?",
            "Utilisez le module de facturation.",
            sources=[{"filename": "facture.htm"}],
        )
        history = sessions.get_history("test_session")
        standalone = reformulate_with_history(
            "et comment l'imprimer ?",
            history,
        )
        self.assertIn("facture", standalone.lower())
        self.assertIn("imprimer", standalone.lower())
        sessions.add_turn(
            "admin_session",
            "A quoi sert le module Administration ?",
            "Le module Administration sert a gerer les utilisateurs et les profils.",
            sources=[{"filename": "Administrationd_Harmony__Introduction.htm"}],
        )
        admin_history = sessions.get_history("admin_session")
        admin_standalone = reformulate_with_history(
            "Et comment gerer les utilisateurs dedans ?",
            admin_history,
        )
        admin_info = reformulate_with_history_info(
            "Et comment gerer les utilisateurs dedans ?",
            admin_history,
        )
        self.assertIn("module administration", admin_standalone.lower())
        self.assertIn("gerer les utilisateurs", admin_standalone.lower())
        self.assertNotIn("quoi sert", admin_standalone.lower())
        self.assertTrue(admin_info.history_used)
        self.assertGreaterEqual(admin_info.confidence, 0.75)
        self.assertEqual(admin_info.module.lower(), "administration")
        summary = sessions.build_session_summary("admin_session")
        self.assertIn("Module actif: Administration", summary["summary"])
        sessions.clear_session("test_session")
        self.assertEqual(sessions.get_history("test_session"), [])

    def test_hybrid_query_uses_memory_and_retries_once(self):
        from query_docs import HybridRAG

        sessions = SessionManager(max_history_turns=3)
        sessions.add_turn(
            "query_session",
            "Comment creer une facture ?",
            "Utilisez le module de facturation.",
        )
        rag = HybridRAG.__new__(HybridRAG)
        rag.session_manager = sessions
        rag.cot_enabled = False
        rag.lexical_retriever = None
        rag.cross_encoder = SimpleNamespace(backend="disabled")
        calls = []
        context = {
            "text": "La facture est imprimee depuis le module Harmony.",
            "compressed_text": "La facture est imprimee depuis le module Harmony.",
            "source": "facture.htm",
            "title": "Facture",
            "method": "hybrid",
            "hybrid_score": 0.9,
            "parent_id": "parent_000001",
            "child_id": "child_000001",
        }

        def execute(question, retrieval_query, top_k, retry=False, **_kwargs):
            calls.append((question, retrieval_query, retry))
            verification = {
                "answer_supported": retry,
                "confidence": 0.9 if retry else 0.2,
                "needs_retry": not retry,
                "reason": "supported" if retry else "low support",
                "unsupported_claims": [] if retry else ["claim"],
            }
            return {
                "transformation": {
                    "query_variants": [retrieval_query],
                    "query_metadata": {},
                    "hyde_text": "retrieval only",
                },
                "graph_results": [],
                "vector_results": [],
                "lexical_results": [],
                "child_results": [context],
                "parent_results": [context],
                "mmr_results": [context],
                "reranked_results": [context],
                "compressed_results": [context],
                "generated_answer": "La facture est imprimee depuis le module Harmony.",
                "verification": verification,
                "retrieval_confidence": 1.0,
                "retrieval_counts": {
                    "vector": 0,
                    "graph": 0,
                    "lexical": 0,
                    "merged": 1,
                    "parents": 1,
                    "before_mmr": 1,
                    "after_mmr": 1,
                    "final": 1,
                },
            }

        rag._execute_pipeline = execute
        result = rag.query(
            "et comment l'imprimer ?",
            top_k=1,
            session_id="query_session",
        )
        self.assertTrue(result["history_used"])
        self.assertIn("facture", result["standalone_question"].lower())
        self.assertEqual(result["retry_count"], 1)
        self.assertTrue(result["verification"]["answer_supported"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(sessions.get_history("query_session")), 2)

    def test_all_new_features_can_be_disabled(self):
        env = os.environ.copy()
        env.update(
            {
                "RAG_ENABLE_QUERY_REWRITING": "0",
                "RAG_ENABLE_SELF_QUERY": "0",
                "RAG_ENABLE_BM25": "0",
                "RAG_ENABLE_TFIDF": "0",
                "RAG_ENABLE_MMR": "0",
                "RAG_ENABLE_CROSS_ENCODER_RERANKER": "0",
                "RAG_ENABLE_CONTEXT_COMPRESSION": "0",
                "RAG_ENABLE_PARENT_CHILD_RETRIEVAL": "0",
                "RAG_ENABLE_HYDE": "0",
                "RAG_ENABLE_ANSWER_VERIFIER": "0",
                "RAG_ENABLE_CHAT_MEMORY": "0",
                "RAG_MAX_RETRY_COUNT": "0",
            }
        )
        code = (
            "from rag_query_transform import rewrite_query; "
            "from rag_lexical import LexicalRetriever; "
            "from rag_reranker import CrossEncoderReranker; "
            "from rag_compressor import ContextualCompressor; "
            "from query_docs import HybridRAG, ConfidenceCalibrator; "
            "assert rewrite_query('Question') == ['Question']; "
            "assert not LexicalRetriever([{'text':'doc'}]).enabled; "
            "assert not CrossEncoderReranker().enabled; "
            "assert not ContextualCompressor().enabled; "
            "rag=HybridRAG.__new__(HybridRAG); "
            "rag.graph=None; rag.faiss_index=None; rag.metadata=[]; "
            "rag.parents_by_id={}; rag.child_to_parent={}; "
            "rag.embedding_model=None; rag.embedding_model_name=''; "
            "rag.cot_enabled=False; rag.calibrator=ConfidenceCalibrator(); "
            "rag.cross_encoder=CrossEncoderReranker(); "
            "rag.compressor=ContextualCompressor(); "
            "rag.lexical_retriever=LexicalRetriever([]); "
            "rag._entity_embeddings=__import__('numpy').zeros((0,1),dtype='float32'); "
            "rag._entity_nodes=[]; rag._entity_index={}; "
            "rag.search_graph=lambda query,top_k=5:["
            "{'document_id':'Document_0','chunk_id':0,'text':"
            "'Harmony permet de configurer les chemins de fichiers pour les applications locales.',"
            "'source':'CheminsHarmony.htm','score':0.9,'graph_score':0.9,'method':'graph'}]; "
            "rag.search_vector=lambda query,top_k=5:[]; "
            "out=rag.query('Comment configurer les chemins Harmony ?',top_k=1); "
            "assert out['answer']; assert out['metadata']['features_used']['mmr'] is False"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
