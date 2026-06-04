#!/usr/bin/env python3
"""
Mesure la précision RAG (similarité réponse générée vs attendue).
Objectif par défaut : 80 % (RAG_TARGET_PRECISION).
"""
import io
import json
import sys
from difflib import SequenceMatcher

from rag_config import TARGET_PRECISION

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def similarity(generated: str, expected: str) -> float:
    from rag_answer import normalize_for_match

    a = normalize_for_match(generated or "")
    b = normalize_for_match(expected or "")
    return SequenceMatcher(None, a, b).ratio()


def measure_curated(test_file: str = "test_questions.json", use_rag: bool = True) -> dict:
    from rag_answer import generate_answer, generate_answer_from_results

    with open(test_file, encoding="utf-8") as f:
        tests = json.load(f).get("test_questions", [])

    rag = None
    if use_rag:
        from query_docs import HybridRAG

        rag = HybridRAG()
        if not rag.graph and not rag.faiss_index:
            use_rag = False

    results = []
    for t in tests:
        q = t["question"]
        expected = t["expected_answer"]
        if use_rag and rag:
            out = rag.query(q, top_k=3)
            merged = out.get("merged_results", [])
            generated = out.get("generated_answer") or generate_answer_from_results(
                q,
                merged,
                embedding_model=rag.embedding_model,
                embedding_model_name=rag.embedding_model_name,
            )
        else:
            generated = generate_answer(q, "")

        sim = similarity(generated, expected)
        results.append(
            {
                "id": t["id"],
                "similarity": round(sim, 4),
                "passed": sim >= TARGET_PRECISION,
                "generated_preview": (generated or "")[:120],
            }
        )

    scores = [r["similarity"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    passed = sum(1 for r in results if r["passed"])
    return {
        "test_file": test_file,
        "target": TARGET_PRECISION,
        "average_similarity": round(avg, 4),
        "passed": passed,
        "total": len(results),
        "success_rate_pct": round(100 * passed / len(results), 2) if results else 0,
        "meets_target": avg >= TARGET_PRECISION,
        "details": results,
    }


def main():
    report = measure_curated()
    print("=" * 60)
    print(f"Précision moyenne : {report['average_similarity']:.2%}")
    print(f"Objectif          : {report['target']:.2%}")
    print(f"Tests >= objectif : {report['passed']}/{report['total']}")
    print(f"Objectif atteint  : {'OUI' if report['meets_target'] else 'NON'}")
    print("=" * 60)
    if not report["meets_target"]:
        for d in report["details"]:
            if not d["passed"]:
                print(f"  - {d['id']}: {d['similarity']:.2%}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
