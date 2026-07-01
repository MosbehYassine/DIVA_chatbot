#!/usr/bin/env python3
"""
Pipeline de build RAG: vérification syntaxe + ingestion + tests.
Usage:
  python build_rag.py              # compile + ingest + tests
  python build_rag.py --compile    # compile seulement
  python build_rag.py --ingest     # compile + ingest
  python build_rag.py --test       # compile + tests (index existant)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

MODULES = [
    "rag_config.py",
    "rag_canonical.py",
    "rag_answer.py",
    "rag_lexical.py",
    "rag_mmr.py",
    "rag_reranker.py",
    "rag_query_transform.py",
    "rag_hyde.py",
    "rag_conversation.py",
    "rag_llm_answer.py",
    "rag_compressor.py",
    "rag_verifier.py",
    "ingest_docs.py",
    "query_docs.py",
    "session_manager.py",
    "measure_precision.py",
    "run_precision.py",
    "test_hybrid_rag.py",
    "test_hybrid_rag_extended.py",
    "test_rag_pipeline_components.py",
    "build_rag.py",
]

PUBLIC_API_CHECKS = {
    "rag_reranker": ("CrossEncoderReranker",),
    "rag_query_transform": ("transform_query", "rewrite_query", "extract_query_metadata"),
    "rag_hyde": ("generate_hypothetical_document", "build_hyde_queries"),
    "rag_conversation": ("reformulate_with_history",),
    "rag_llm_answer": ("formulate_answer_with_llm",),
    "rag_lexical": ("LexicalRetriever",),
    "rag_mmr": ("maximal_marginal_relevance",),
    "rag_compressor": ("ContextualCompressor", "compress_context"),
    "rag_verifier": ("verify_answer", "is_not_found_answer"),
}


def run(cmd, env=None):
    print("\n>>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def compile_all():
    print("=" * 60)
    print("1/3 COMPILATION PYTHON")
    print("=" * 60)
    for mod in MODULES:
        path = os.path.join(ROOT, mod)
        if os.path.exists(path):
            run([PY, "-m", "py_compile", path])
    for module_name, symbols in PUBLIC_API_CHECKS.items():
        code = (
            f"import {module_name} as module; "
            f"missing=[name for name in {symbols!r} if not hasattr(module, name)]; "
            "assert not missing, f'Missing public API: {missing}'"
        )
        run([PY, "-c", code])
    print("Compilation OK")


def ingest():
    print("=" * 60)
    print("2/3 INGESTION (peut prendre 20-40 min)")
    print("=" * 60)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    env.setdefault("RAG_CHUNK_STRATEGY", "semantic")
    env.setdefault("RAG_CHUNK_SIZE", "800")
    env.setdefault("RAG_CHUNK_OVERLAP", "200")
    env.setdefault("RAG_PARENT_CHUNK_SIZE", "1000")
    env.setdefault("RAG_PARENT_CHUNK_OVERLAP", "200")
    env.setdefault("RAG_CHILD_CHUNK_SIZE", "250")
    env.setdefault("RAG_CHILD_CHUNK_OVERLAP", "50")
    run([PY, "ingest_docs.py"], env=env)


def tests():
    print("=" * 60)
    print("3/3 TESTS")
    print("=" * 60)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run([PY, "measure_precision.py"], env=env)
    run([PY, "test_rag_pipeline_components.py"], env=env)
    run([PY, "test_hybrid_rag.py"], env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compilation seulement")
    parser.add_argument("--ingest", action="store_true", help="Compile + ingestion")
    parser.add_argument("--test", action="store_true", help="Compile + tests")
    args = parser.parse_args()

    compile_all()
    if args.compile:
        return
    if args.test:
        tests()
        return
    if args.ingest:
        ingest()
        return
    ingest()
    tests()


if __name__ == "__main__":
    main()
