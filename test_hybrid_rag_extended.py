#!/usr/bin/env python3
"""
Jeu de tests etendu pour couvrir plus de fichiers/modules de la documentation.
Evaluation: correspondance de source + couverture de mots-cles attendus.
"""

import io
import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from typing import Dict, List

from query_docs import HybridRAG

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


class ExtendedTester:
    def __init__(self, test_file: str = "test_questions_extended.json"):
        self.test_file = test_file
        self.rag = HybridRAG()
        self.tests = self._load_tests()
        self.results: List[Dict] = []

    def _load_tests(self) -> List[Dict]:
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Fichier introuvable: {self.test_file}")
        with open(self.test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("test_questions", [])

    def _evaluate(self, test: Dict, best: Dict) -> Dict:
        retrieved_text = best.get("text", "") if best else ""
        retrieved_source = best.get("source", best.get("document_id", "")) if best else ""
        score = float(best.get("hybrid_score", 0.0)) if best else 0.0
        method = best.get("method", "none") if best else "none"

        src_norm = normalize_text(retrieved_source)
        txt_norm = normalize_text(retrieved_text)

        source_match = False
        matched_source_token = ""
        for token in test.get("expected_source_contains", []):
            token_norm = normalize_text(token)
            if token_norm and token_norm in src_norm:
                source_match = True
                matched_source_token = token
                break

        kw_expected = test.get("expected_keywords", [])
        kw_hits = []
        for kw in kw_expected:
            if normalize_text(kw) in txt_norm:
                kw_hits.append(kw)
        kw_ratio = (len(kw_hits) / len(kw_expected)) if kw_expected else 0.0

        passed = source_match or kw_ratio >= 0.50
        quality = "OK" if passed else "ECHEC"
        if passed and source_match and kw_ratio >= 0.50:
            quality = "TRES BON"

        return {
            "id": test["id"],
            "module": test["module"],
            "question": test["question"],
            "method": method,
            "hybrid_score": score,
            "retrieved_source": retrieved_source,
            "retrieved_text_preview": retrieved_text[:220],
            "source_match": source_match,
            "matched_source_token": matched_source_token,
            "keywords_expected": kw_expected,
            "keywords_hits": kw_hits,
            "keyword_ratio": kw_ratio,
            "result": quality,
            "passed": passed,
        }

    def run(self) -> Dict:
        print("=" * 90)
        print("TESTS ETENDUS HYBRID RAG")
        print("=" * 90)
        print(f"Nombre de tests: {len(self.tests)}")

        if not self.rag.graph and not self.rag.faiss_index:
            raise RuntimeError("Index non charges. Executez d'abord python ingest_docs.py")

        for idx, test in enumerate(self.tests, 1):
            print(f"\n[{idx}/{len(self.tests)}] {test['id']} - {test['module']}")
            print(f"Q: {test['question']}")
            result = self.rag.query(test["question"], top_k=3)
            best = result["merged_results"][0] if result["merged_results"] else {}
            answer_text = result.get("generated_answer") or best.get("text", "")
            evaluated = self._evaluate(test, {**best, "text": answer_text})
            self.results.append(evaluated)
            print(
                f"-> {evaluated['result']} | method={evaluated['method']} | score={evaluated['hybrid_score']:.2%} | "
                f"source_match={evaluated['source_match']} | kw={len(evaluated['keywords_hits'])}/{len(evaluated['keywords_expected'])}"
            )

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        avg_score = sum(r["hybrid_score"] for r in self.results) / total if total else 0.0

        by_module: Dict[str, Dict] = {}
        for r in self.results:
            mod = r["module"]
            by_module.setdefault(mod, {"total": 0, "passed": 0})
            by_module[mod]["total"] += 1
            by_module[mod]["passed"] += 1 if r["passed"] else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": round((passed / total * 100), 2) if total else 0.0,
                "average_hybrid_score": round(avg_score, 4),
            },
            "by_module": by_module,
            "results": self.results,
        }

        base = os.path.splitext(os.path.basename(self.test_file))[0]
        output = f"test_results_{base}.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 90)
        print(
            f"RESUME: {passed}/{total} passes ({report['summary']['success_rate']}%), "
            f"score moyen={avg_score:.2%}"
        )
        for module, stats in by_module.items():
            rate = (stats["passed"] / stats["total"] * 100) if stats["total"] else 0.0
            print(f"- {module}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        print(f"Rapport detaille: {output}")
        print("=" * 90)
        return report


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "test_questions_extended.json"
    tester = ExtendedTester(test_file=test_file)
    tester.run()
