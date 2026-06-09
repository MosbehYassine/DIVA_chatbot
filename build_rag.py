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
    "ingest_docs.py",
    "query_docs.py",
    "session_manager.py",
    "measure_precision.py",
    "test_hybrid_rag.py",
    "test_hybrid_rag_extended.py",
    "build_rag.py",
]


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
    run([PY, "ingest_docs.py"], env=env)


def tests():
    print("=" * 60)
    print("3/3 TESTS")
    print("=" * 60)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run([PY, "measure_precision.py"], env=env)
    run([PY, "test_hybrid_rag.py"], env=env)
    run([PY, "test_hybrid_rag_extended.py", "test_questions_extended.json"], env=env)


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
