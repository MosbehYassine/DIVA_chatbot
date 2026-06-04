#!/usr/bin/env python3
"""Quick precision run — writes results to precision_report.json"""
import io
import json
import sys
from difflib import SequenceMatcher
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from rag_config import TARGET_PRECISION
from query_docs import HybridRAG
from rag_answer import generate_answer_from_results, normalize_for_match


def similarity(generated: str, expected: str) -> float:
    a = normalize_for_match(generated or "")
    b = normalize_for_match(expected or "")
    return SequenceMatcher(None, a, b).ratio()

with open("test_questions.json", encoding="utf-8") as f:
    tests = json.load(f)["test_questions"]

rag = HybridRAG()
results = []
for t in tests:
    out = rag.query(t["question"], top_k=3)
    generated = out.get("generated_answer") or generate_answer_from_results(
        t["question"],
        out.get("merged_results", []),
        embedding_model=rag.embedding_model,
        embedding_model_name=rag.embedding_model_name,
    )
    sim = similarity(generated, t["expected_answer"])
    results.append({
        "id": t["id"],
        "similarity": round(sim, 4),
        "passed": sim >= TARGET_PRECISION,
        "generated_preview": (generated or "")[:150],
    })

scores = [r["similarity"] for r in results]
avg = sum(scores) / len(scores) if scores else 0
passed = sum(1 for r in results if r["passed"])

report = {
    "timestamp": datetime.now().isoformat(),
    "target": TARGET_PRECISION,
    "average_similarity": round(avg, 4),
    "passed": passed,
    "total": len(results),
    "success_rate_pct": round(100 * passed / len(results), 2) if results else 0,
    "meets_target": avg >= TARGET_PRECISION,
    "details": results,
}

with open("precision_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"avg={avg:.2%} passed={passed}/{len(results)} target={TARGET_PRECISION:.0%}")
