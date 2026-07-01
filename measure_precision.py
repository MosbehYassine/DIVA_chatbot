#!/usr/bin/env python3
"""Evaluate retrieval quality separately from answer quality."""
import json
import math
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional

if "--allow-llm" not in sys.argv:
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["RAG_ENABLE_LLM_ANSWER_GENERATION"] = "false"
    os.environ["RAG_HYDE_USE_LLM"] = "false"
    os.environ["RAG_ENABLE_LLM_ANSWER_VERIFIER"] = "false"

from rag_config import (
    RAG_ENABLE_BM25,
    RAG_ENABLE_CONTEXT_COMPRESSION,
    RAG_ENABLE_CROSS_ENCODER_RERANKER,
    RAG_ENABLE_MMR,
    RAG_ENABLE_QUERY_REWRITING,
    RAG_ENABLE_PARENT_CHILD_RETRIEVAL,
    RAG_ENABLE_HYDE,
    RAG_ENABLE_ANSWER_VERIFIER,
    RAG_ENABLE_CHAT_MEMORY,
    RAG_ENABLE_LLM_ANSWER_GENERATION,
    RAG_ENABLE_REASONING_PLAN,
    TARGET_PRECISION,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def similarity(generated: str, expected: str) -> float:
    from rag_answer import normalize_for_match

    a = normalize_for_match(generated or "")
    b = normalize_for_match(expected or "")
    return SequenceMatcher(None, a, b).ratio()


def _normalize_source(value: str) -> str:
    from rag_answer import normalize_for_match

    return normalize_for_match(os.path.basename(value or ""))


def _expected_sources(test: Dict) -> List[str]:
    values = test.get("source_files") or test.get("expected_source_contains") or []
    return [_normalize_source(str(value)) for value in values if str(value).strip()]


def _source_is_relevant(source: str, expected_sources: List[str]) -> bool:
    source_norm = _normalize_source(source)
    return any(
        expected in source_norm or source_norm in expected
        for expected in expected_sources
        if expected and source_norm
    )


def _retrieval_metrics(results: List[Dict], expected_sources: List[str], k: int = 5) -> Dict:
    if not expected_sources:
        return {
            "recall": None,
            "recall_at_5": None,
            "reciprocal_rank": None,
            "ndcg_at_5": None,
            "first_relevant_rank": None,
        }

    top = results[:k]
    unmatched = set(expected_sources)
    matched_expected = set()
    relevance = []
    for result in top:
        source_norm = _normalize_source(str(result.get("source") or ""))
        match = next(
            (
                expected for expected in unmatched
                if expected in source_norm or source_norm in expected
            ),
            None,
        )
        relevance.append(1 if match else 0)
        if match:
            matched_expected.add(match)
            unmatched.remove(match)
    recall = len(matched_expected) / max(len(set(expected_sources)), 1)
    first_rank: Optional[int] = next(
        (index + 1 for index, relevant in enumerate(relevance) if relevant),
        None,
    )
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance))
    ideal_relevant = min(len(set(expected_sources)), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_relevant))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "recall": recall,
        "recall_at_5": recall,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_5": ndcg,
        "first_relevant_rank": first_rank,
    }


def _expects_not_found(test: Dict) -> bool:
    return bool(
        test.get("expected_not_found")
        or test.get("should_find") is False
        or test.get("answerable") is False
    )


def _keyword_overlap(generated: str, test: Dict) -> float:
    from rag_answer import extract_question_terms, normalize_for_match

    expected_keywords = test.get("expected_keywords") or []
    if expected_keywords:
        keywords = [normalize_for_match(str(value)) for value in expected_keywords]
    else:
        keywords = extract_question_terms(test.get("expected_answer", ""))
    keywords = [keyword for keyword in keywords if keyword]
    generated_norm = normalize_for_match(generated)
    return (
        sum(1 for keyword in keywords if keyword in generated_norm) / len(keywords)
        if keywords else 0.0
    )


def _quality_label(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.70:
        return "bon"
    if score >= 0.55:
        return "moyen"
    return "faible"


def _safe_avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_summary(rows: List[Dict], key: str) -> Dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)

    output = {}
    for group_name, items in sorted(grouped.items()):
        total = len(items)
        retrieval_values = [
            float(item["recall_at_5"])
            for item in items
            if item.get("recall_at_5") is not None
        ]
        output[group_name] = {
            "total": total,
            "passed": sum(1 for item in items if item.get("passed")),
            "pass_rate": round(
                sum(1 for item in items if item.get("passed")) / total,
                4,
            ) if total else 0.0,
            "average_similarity": round(
                _safe_avg([float(item.get("similarity", 0.0)) for item in items]),
                4,
            ),
            "average_performance_score": round(
                _safe_avg([float(item.get("performance_score", 0.0)) for item in items]),
                4,
            ),
            "keyword_overlap": round(
                _safe_avg([float(item.get("keyword_overlap", 0.0)) for item in items]),
                4,
            ),
            "source_match_rate": round(
                sum(1 for item in items if item.get("source_match")) / total,
                4,
            ) if total else 0.0,
            "supported_rate": round(
                sum(1 for item in items if item.get("answer_supported")) / total,
                4,
            ) if total else 0.0,
            "recall_at_5": round(_safe_avg(retrieval_values), 4),
            "quality": _quality_label(
                _safe_avg([
                    float(item.get("performance_score", 0.0))
                    for item in items
                ])
            ),
        }
    return output


def _failure_reason(row: Dict) -> str:
    if not row.get("source_match"):
        return "retrieval_source_miss"
    if row.get("recall_at_5") == 0:
        return "retrieval_recall_zero"
    if not row.get("answer_supported"):
        return "unsupported_answer"
    if row.get("keyword_overlap", 0.0) < 0.35:
        return "low_keyword_overlap"
    if row.get("similarity", 0.0) < TARGET_PRECISION:
        return "low_similarity"
    return "ok"


def _load_resume_details(output_path: str, resume: bool) -> List[Dict]:
    if not resume or not output_path or not os.path.exists(output_path):
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            report = json.load(f)
        details = report.get("details", [])
        if isinstance(details, list):
            return [row for row in details if isinstance(row, dict) and row.get("id")]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Resume ignore: impossible de lire {output_path} ({exc})")
    return []


def _save_resume_details(output_path: str, details: List[Dict], total: int) -> None:
    if not output_path:
        return
    payload = {
        "partial": True,
        "timestamp": datetime.now().isoformat(),
        "completed": len(details),
        "total": total,
        "details": details,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def evaluate_tests(
    test_file: str = "test_questions.json",
    use_rag: bool = True,
    show_progress: bool = True,
    limit: int = 0,
    module_filter: str = "",
    difficulty_filter: str = "",
    output_path: str = "",
    resume: bool = False,
) -> Dict:
    from rag_answer import (
        generate_answer,
        generate_answer_from_results,
        is_not_found_answer as fallback_is_not_found_answer,
    )
    try:
        from rag_verifier import is_not_found_answer
    except (ImportError, AttributeError):
        is_not_found_answer = fallback_is_not_found_answer

    with open(test_file, encoding="utf-8-sig") as f:
        tests = json.load(f).get("test_questions", [])
    if module_filter:
        wanted = module_filter.casefold()
        tests = [
            test for test in tests
            if str(test.get("module") or test.get("category") or "").casefold() == wanted
        ]
    if difficulty_filter:
        wanted = difficulty_filter.casefold()
        if wanted not in {"all", "tout", "*"}:
            tests = [
                test for test in tests
                if str(test.get("difficulty") or "").casefold() == wanted
            ]
    if limit and limit > 0:
        tests = tests[:limit]

    tests_by_id = {str(test.get("id")): test for test in tests if test.get("id")}
    details = _load_resume_details(output_path, resume)
    completed_ids = {str(row.get("id")) for row in details}
    if completed_ids:
        before = len(tests)
        tests = [test for test in tests if str(test.get("id")) not in completed_ids]
        if show_progress:
            skipped = before - len(tests)
            print(f"Resume: {skipped} question(s) deja calculee(s), {len(tests)} restante(s).")

    rag = None
    if use_rag:
        from query_docs import HybridRAG

        rag = HybridRAG()
        if not rag.graph and not rag.faiss_index:
            use_rag = False

    retrieval_rows = []
    supported_count = 0
    answerable_count = 0
    hallucination_count = 0
    negative_count = 0
    correct_not_found_count = 0
    verification_confidences = []
    unsupported_answer_count = 0
    retry_count = 0
    parent_source_hits = []
    child_source_hits = []

    for row in details:
        test = tests_by_id.get(str(row.get("id")), {})
        expected_sources = _expected_sources(test)
        if row.get("recall_at_5") is not None:
            retrieval_rows.append(
                {
                    "recall_at_5": row.get("recall_at_5"),
                    "recall_at_10": row.get("recall_at_10"),
                    "reciprocal_rank": row.get("reciprocal_rank"),
                    "ndcg_at_5": row.get("ndcg_at_5"),
                }
            )
        if expected_sources:
            parent_source_hits.append(bool(row.get("parent_source_match")))
            child_source_hits.append(bool(row.get("child_source_match")))
        expected_not_found = bool(row.get("expected_not_found"))
        predicted_not_found = bool(row.get("predicted_not_found"))
        answer_supported = bool(row.get("answer_supported"))
        if not answer_supported:
            unsupported_answer_count += 1
        verification_confidences.append(float(row.get("verification_confidence", 0.0) or 0.0))
        retry_count += int(bool(row.get("retried")))
        if expected_not_found:
            negative_count += 1
            if predicted_not_found:
                correct_not_found_count += 1
        else:
            answerable_count += 1
            if answer_supported:
                supported_count += 1
        if not predicted_not_found and not answer_supported:
            hallucination_count += 1

    total_tests = len(details) + len(tests)
    completed_before = len(details)
    for offset, test in enumerate(tests, 1):
        index = completed_before + offset
        question = test["question"]
        expected = test.get("expected_answer", "")
        if show_progress:
            test_id = str(test.get("id") or f"question_{index}")
            difficulty = str(test.get("difficulty") or "unknown")
            print(f"[{index}/{total_tests}] {test_id} - {difficulty}", flush=True)
        if use_rag and rag:
            output = rag.query(question, top_k=5)
            retrieval_results = output.get("retrieval_results") or output.get("merged_results", [])
            retrieval_candidates = (
                output.get("retrieval_candidates")
                or retrieval_results
            )
            generated = output.get("generated_answer") or generate_answer_from_results(
                question,
                output.get("merged_results", []),
                embedding_model=rag.embedding_model,
                embedding_model_name=rag.embedding_model_name,
            )
            verification = output.get("answer_verification") or {}
            child_results = output.get("child_retrieval_results") or []
        else:
            output = {}
            retrieval_results = []
            generated = generate_answer(question, "")
            verification = {}
            retrieval_candidates = retrieval_results
            child_results = []

        sim = similarity(generated, expected)
        expected_sources = _expected_sources(test)
        retrieval = _retrieval_metrics(retrieval_results, expected_sources, k=5)
        retrieval_at_10 = _retrieval_metrics(retrieval_candidates, expected_sources, k=10)
        if retrieval["recall_at_5"] is not None:
            retrieval_rows.append(
                {
                    **retrieval,
                    "recall_at_10": retrieval_at_10["recall"],
                }
            )

        predicted_not_found = is_not_found_answer(generated)
        expected_not_found = _expects_not_found(test)
        answer_supported = bool(verification.get("answer_supported", False))
        verification_confidence = float(verification.get("confidence", 0.0) or 0.0)
        keyword_overlap = _keyword_overlap(generated, test)
        source_match = bool(
            expected_sources
            and any(
                _source_is_relevant(str(result.get("source") or ""), expected_sources)
                for result in retrieval_results[:5]
            )
        )
        child_source_match = bool(
            expected_sources
            and any(
                _source_is_relevant(str(result.get("source") or ""), expected_sources)
                for result in child_results[:10]
            )
        )
        if expected_sources:
            parent_source_hits.append(source_match)
            child_source_hits.append(child_source_match)
        verification_confidences.append(verification_confidence)
        if not answer_supported:
            unsupported_answer_count += 1
        retry_count += int(output.get("retry_count", bool(output.get("retried", False))) or 0)

        if expected_not_found:
            negative_count += 1
            if predicted_not_found:
                correct_not_found_count += 1
        else:
            answerable_count += 1
            if answer_supported:
                supported_count += 1

        if not predicted_not_found and not answer_supported:
            hallucination_count += 1

        retrieval_score = (
            float(retrieval["recall_at_5"])
            if retrieval["recall_at_5"] is not None
            else 0.0
        )
        answer_quality_score = (
            0.45 * sim
            + 0.35 * keyword_overlap
            + 0.20 * (1.0 if answer_supported else 0.0)
        )
        performance_score = (
            0.40 * retrieval_score
            + 0.45 * answer_quality_score
            + 0.15 * (1.0 if source_match else 0.0)
        )

        details.append(
            {
                "id": test["id"],
                "module": test.get("module") or test.get("category") or "unknown",
                "category": test.get("category") or test.get("module") or "unknown",
                "difficulty": test.get("difficulty", "unknown"),
                "question": question,
                "similarity": round(sim, 4),
                "passed": sim >= TARGET_PRECISION,
                "performance_score": round(performance_score, 4),
                "performance_label": _quality_label(performance_score),
                "answer_quality_score": round(answer_quality_score, 4),
                "generated_preview": (generated or "")[:120],
                "expected_preview": (expected or "")[:120],
                "recall_at_5": (
                    round(retrieval["recall_at_5"], 4)
                    if retrieval["recall_at_5"] is not None else None
                ),
                "reciprocal_rank": (
                    round(retrieval["reciprocal_rank"], 4)
                    if retrieval["reciprocal_rank"] is not None else None
                ),
                "ndcg_at_5": (
                    round(retrieval["ndcg_at_5"], 4)
                    if retrieval["ndcg_at_5"] is not None else None
                ),
                "recall_at_10": (
                    round(retrieval_at_10["recall"], 4)
                    if retrieval_at_10["recall"] is not None else None
                ),
                "first_relevant_rank": retrieval["first_relevant_rank"],
                "keyword_overlap": round(keyword_overlap, 4),
                "source_match": source_match,
                "parent_source_match": source_match,
                "child_source_match": child_source_match,
                "answer_supported": answer_supported,
                "verification_confidence": round(verification_confidence, 4),
                "predicted_not_found": predicted_not_found,
                "expected_not_found": expected_not_found,
                "retried": bool(output.get("retried", False)),
                "failure_reason": "ok",
            }
        )
        details[-1]["failure_reason"] = _failure_reason(details[-1])
        _save_resume_details(output_path, details, total_tests)

    scores = [row["similarity"] for row in details]
    average_similarity = sum(scores) / len(scores) if scores else 0.0
    passed = sum(1 for row in details if row["passed"])

    def average_metric(name: str) -> float:
        values = [float(row[name]) for row in retrieval_rows if row[name] is not None]
        return sum(values) / len(values) if values else 0.0

    retrieval_metrics = {
        "recall_at_5": round(average_metric("recall_at_5"), 4),
        "recall_at_10": round(average_metric("recall_at_10"), 4),
        "mrr": round(average_metric("reciprocal_rank"), 4),
        "ndcg_at_5": round(average_metric("ndcg_at_5"), 4),
        "evaluated_queries": len(retrieval_rows),
        "parent_hit_rate": round(
            sum(parent_source_hits) / len(parent_source_hits)
            if parent_source_hits else 0.0,
            4,
        ),
        "child_hit_rate": round(
            sum(child_source_hits) / len(child_source_hits)
            if child_source_hits else 0.0,
            4,
        ),
    }
    answer_metrics = {
        "average_similarity": round(average_similarity, 4),
        "keyword_overlap": round(
            sum(row["keyword_overlap"] for row in details) / len(details)
            if details else 0.0,
            4,
        ),
        "source_match": round(
            sum(1 for row in details if row["source_match"]) / len(details)
            if details else 0.0,
            4,
        ),
        "supported_answer_rate": round(
            supported_count / answerable_count if answerable_count else 0.0,
            4,
        ),
        "hallucination_rate": round(
            hallucination_count / len(details) if details else 0.0,
            4,
        ),
        "not_found_accuracy": round(
            correct_not_found_count / negative_count if negative_count else 0.0,
            4,
        ),
        "answerable_queries": answerable_count,
        "not_found_queries": negative_count,
    }
    performance_scores = [
        float(row.get("performance_score", 0.0) or 0.0)
        for row in details
    ]
    failure_reasons = defaultdict(int)
    for row in details:
        failure_reasons[row.get("failure_reason", "unknown")] += 1
    weakest_cases = sorted(
        details,
        key=lambda row: (
            float(row.get("performance_score", 0.0) or 0.0),
            float(row.get("similarity", 0.0) or 0.0),
        ),
    )[:15]
    global_score = round(
        (
            retrieval_metrics["recall_at_5"]
            + retrieval_metrics["ndcg_at_5"]
            + answer_metrics["keyword_overlap"]
            + answer_metrics["supported_answer_rate"]
        ) / 4.0,
        4,
    )
    config = {
        "query_rewriting": RAG_ENABLE_QUERY_REWRITING,
        "bm25": RAG_ENABLE_BM25,
        "mmr": RAG_ENABLE_MMR,
        "cross_encoder": RAG_ENABLE_CROSS_ENCODER_RERANKER,
        "compression": RAG_ENABLE_CONTEXT_COMPRESSION,
    }
    features = {
        "parent_child_retrieval": RAG_ENABLE_PARENT_CHILD_RETRIEVAL,
        "hyde": RAG_ENABLE_HYDE,
        "answer_verifier": RAG_ENABLE_ANSWER_VERIFIER,
        "chat_memory": RAG_ENABLE_CHAT_MEMORY,
        "llm_answer_generation": RAG_ENABLE_LLM_ANSWER_GENERATION,
        "reasoning_plan": RAG_ENABLE_REASONING_PLAN,
    }
    verification_metrics = {
        "average_confidence": round(
            sum(verification_confidences) / len(verification_confidences)
            if verification_confidences else 0.0,
            4,
        ),
        "unsupported_answer_count": unsupported_answer_count,
        "retry_count": retry_count,
    }
    return {
        "timestamp": datetime.now().isoformat(),
        "test_file": test_file,
        "target": TARGET_PRECISION,
        "average_similarity": round(average_similarity, 4),
        "average_performance_score": round(_safe_avg(performance_scores), 4),
        "performance_label": _quality_label(_safe_avg(performance_scores)),
        "passed": passed,
        "total": len(details),
        "success_rate_pct": round(100 * passed / len(details), 2) if details else 0,
        "meets_target": _safe_avg(performance_scores) >= TARGET_PRECISION,
        "meets_performance_target": _safe_avg(performance_scores) >= TARGET_PRECISION,
        "meets_similarity_target": average_similarity >= TARGET_PRECISION,
        "global_score": global_score,
        "retrieval": retrieval_metrics,
        "answer": answer_metrics,
        "config": config,
        "features": features,
        "verification": verification_metrics,
        "by_module": _group_summary(details, "module"),
        "by_category": _group_summary(details, "category"),
        "by_difficulty": _group_summary(details, "difficulty"),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "weakest_cases": [
            {
                "id": row["id"],
                "module": row["module"],
                "difficulty": row["difficulty"],
                "performance_score": row["performance_score"],
                "similarity": row["similarity"],
                "keyword_overlap": row["keyword_overlap"],
                "source_match": row["source_match"],
                "answer_supported": row["answer_supported"],
                "failure_reason": row["failure_reason"],
                "question": row["question"],
                "generated_preview": row["generated_preview"],
                "expected_preview": row["expected_preview"],
            }
            for row in weakest_cases
        ],
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "recall_at_5": retrieval_metrics["recall_at_5"],
        "recall_at_10": retrieval_metrics["recall_at_10"],
        "mrr": retrieval_metrics["mrr"],
        "ndcg_at_5": retrieval_metrics["ndcg_at_5"],
        "supported_answer_rate": answer_metrics["supported_answer_rate"],
        "keyword_overlap": answer_metrics["keyword_overlap"],
        "source_match": answer_metrics["source_match"],
        "hallucination_rate": answer_metrics["hallucination_rate"],
        "not_found_accuracy": answer_metrics["not_found_accuracy"],
        "details": details,
    }


def measure_curated(
    test_file: str = "test_questions.json",
    use_rag: bool = True,
    show_progress: bool = True,
    limit: int = 0,
    module_filter: str = "",
    difficulty_filter: str = "",
    output_path: str = "",
    resume: bool = False,
) -> Dict:
    """Backward-compatible entry point."""
    return evaluate_tests(
        test_file=test_file,
        use_rag=use_rag,
        show_progress=show_progress,
        limit=limit,
        module_filter=module_filter,
        difficulty_filter=difficulty_filter,
        output_path=output_path,
        resume=resume,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance on test_questions.json."
    )
    parser.add_argument(
        "--test-file",
        default="test_questions.json",
        help="Question set JSON file.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Evaluate fallback answer generation without HybridRAG.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit with code 1 when the target is not reached.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-question progress lines.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N questions after filters.",
    )
    parser.add_argument(
        "--module",
        default="",
        help="Evaluate only one module name.",
    )
    parser.add_argument(
        "--difficulty",
        default="",
        choices=["facile", "moyen", "difficile", "all", "tout"],
        help="Evaluate one difficulty or all: facile, moyen, difficile, all/tout.",
    )
    difficulty_shortcuts = parser.add_mutually_exclusive_group()
    difficulty_shortcuts.add_argument(
        "--facile",
        action="store_true",
        help="Shortcut for --difficulty facile.",
    )
    difficulty_shortcuts.add_argument(
        "--moyen",
        action="store_true",
        help="Shortcut for --difficulty moyen.",
    )
    difficulty_shortcuts.add_argument(
        "--difficile",
        action="store_true",
        help="Shortcut for --difficulty difficile.",
    )
    difficulty_shortcuts.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all difficulties.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing --output JSON file and skip completed question ids.",
    )
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help="Allow OpenAI LLM calls during evaluation. By default precision is local/offline.",
    )
    args = parser.parse_args()
    difficulty = args.difficulty
    if args.facile:
        difficulty = "facile"
    elif args.moyen:
        difficulty = "moyen"
    elif args.difficile:
        difficulty = "difficile"
    elif args.all:
        difficulty = "all"

    report = measure_curated(
        args.test_file,
        use_rag=not args.no_rag,
        show_progress=not args.quiet,
        limit=args.limit,
        module_filter=args.module,
        difficulty_filter=difficulty,
        output_path=args.output,
        resume=args.resume,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("RAPPORT PERFORMANCE RAG")
    print("=" * 60)
    print(f"Score performance : {report['average_performance_score']:.2%} ({report['performance_label']})")
    print(f"Precision moyenne : {report['average_similarity']:.2%}")
    print(f"Pass rate         : {report['success_rate_pct']:.2f}% ({report['passed']}/{report['total']})")
    print(f"Recall@5          : {report['recall_at_5']:.2%}")
    print(f"Recall@10         : {report['recall_at_10']:.2%}")
    print(f"MRR               : {report['mrr']:.4f}")
    print(f"nDCG@5            : {report['ndcg_at_5']:.4f}")
    print(f"Supported answers : {report['supported_answer_rate']:.2%}")
    print(f"Keyword overlap   : {report['keyword_overlap']:.2%}")
    print(f"Source match      : {report['source_match']:.2%}")
    print(f"Hallucination rate: {report['hallucination_rate']:.2%}")
    print(f"Objectif perf RAG : {'OUI' if report.get('meets_performance_target') else 'NON'}")
    print(f"Objectif sim stricte: {'OUI' if report.get('meets_similarity_target') else 'NON'}")
    print("=" * 60)

    print("\nPerformance par difficulte")
    for difficulty, row in report["by_difficulty"].items():
        print(
            f"  {difficulty:10s} | score={row['average_similarity']:.2%} "
            f"perf={row['quality']} pass={row['pass_rate']:.2%} "
            f"retrieval@5={row['recall_at_5']:.2%}"
        )

    weak_modules = sorted(
        report["by_module"].items(),
        key=lambda item: (item[1]["pass_rate"], item[1]["average_similarity"]),
    )[:10]
    print("\nModules a ameliorer en priorite")
    for module, row in weak_modules:
        print(
            f"  {module:25s} | pass={row['pass_rate']:.2%} "
            f"sim={row['average_similarity']:.2%} "
            f"kw={row['keyword_overlap']:.2%} "
            f"src={row['source_match_rate']:.2%}"
        )

    print("\nCauses principales")
    for reason, count in report["failure_reasons"].items():
        print(f"  {reason:24s}: {count}")

    print("\nCas les plus faibles")
    for row in report["weakest_cases"][:10]:
        print(
            f"  {row['id']} | {row['module']} | {row['difficulty']} | "
            f"perf={row['performance_score']:.2%} | {row['failure_reason']}"
        )
    if not report["meets_target"] and not args.no_fail:
        for detail in report["details"]:
            if not detail["passed"]:
                print(f"  - {detail['id']}: {detail['similarity']:.2%}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
