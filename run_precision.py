#!/usr/bin/env python3
"""Interactive precision runner with difficulty selection and resume support."""
import json
import os
import sys
from typing import Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


CHOICES: Dict[str, Tuple[str, str, str]] = {
    "1": ("facile", "Questions faciles", "precision_facile.json"),
    "2": ("moyen", "Questions moyennes", "precision_moyen.json"),
    "3": ("difficile", "Questions difficiles", "precision_difficile.json"),
    "4": ("all", "Toutes les questions", "precision_report.json"),
}


def _ask_choice() -> Tuple[str, str]:
    print("Quel type de questions veux-tu tester ?")
    for key, (_, label, output) in CHOICES.items():
        print(f"  {key}. {label} -> {output}")

    while True:
        answer = input("Choix [1-4] : ").strip()
        if answer in CHOICES:
            difficulty, _, output = CHOICES[answer]
            return difficulty, output
        print("Choix invalide. Tape 1, 2, 3 ou 4.")


def _completed_count(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    try:
        with open(output_path, encoding="utf-8") as f:
            report = json.load(f)
        details = report.get("details", [])
        return len(details) if isinstance(details, list) else 0
    except (OSError, json.JSONDecodeError):
        return 0


def _ask_resume(output_path: str) -> bool:
    completed = _completed_count(output_path)
    if completed <= 0:
        return False

    print(f"\nUn rapport existe deja: {output_path}")
    print(f"Questions deja calculees dans ce fichier: {completed}")
    while True:
        answer = input("Continuer depuis ce fichier ? [O/n] : ").strip().lower()
        if answer in {"", "o", "oui", "y", "yes"}:
            return True
        if answer in {"n", "non", "no"}:
            return False
        print("Reponds par O pour continuer ou N pour refaire a zero.")


def _print_summary(report: Dict, output_path: str) -> None:
    print("\n" + "=" * 60)
    print("RAPPORT PERFORMANCE RAG")
    print("=" * 60)
    print(f"Fichier rapport    : {output_path}")
    print(f"Score performance : {report['average_performance_score']:.2%} ({report['performance_label']})")
    print(f"Precision moyenne : {report['average_similarity']:.2%}")
    print(f"Pass rate         : {report['success_rate_pct']:.2f}% ({report['passed']}/{report['total']})")
    print(f"Recall@5          : {report['recall_at_5']:.2%}")
    print(f"Recall@10         : {report['recall_at_10']:.2%}")
    print(f"MRR               : {report['mrr']:.3f}")
    print(f"Supported answers : {report['supported_answer_rate']:.2%}")
    print(f"Source match      : {report['source_match']:.2%}")
    print(f"Hallucination rate: {report['hallucination_rate']:.2%}")
    print(f"Objectif perf RAG : {'OUI' if report.get('meets_performance_target') else 'NON'}")
    print(f"Objectif sim stricte: {'OUI' if report.get('meets_similarity_target') else 'NON'}")
    print("=" * 60)

    weakest = sorted(
        report["by_module"].items(),
        key=lambda item: (item[1]["pass_rate"], item[1]["average_similarity"]),
    )[:5]
    print("Modules faibles   : " + ", ".join(
        f"{module}:{row['pass_rate']:.0%}" for module, row in weakest
    ))
    print("Causes            : " + ", ".join(
        f"{reason}:{count}" for reason, count in report["failure_reasons"].items()
    ))


def _configure_local_evaluation() -> None:
    """Disable paid/remote LLM calls during precision measurements by default."""
    local_llm_overrides = {
        "OPENAI_API_KEY": "",
        "RAG_ENABLE_LLM_ANSWER_GENERATION": "false",
        "RAG_HYDE_USE_LLM": "false",
        "RAG_ENABLE_LLM_ANSWER_VERIFIER": "false",
    }
    for name, value in local_llm_overrides.items():
        os.environ[name] = value


def main():
    _configure_local_evaluation()
    from measure_precision import measure_curated

    difficulty, output_path = _ask_choice()
    resume = _ask_resume(output_path)
    if resume:
        print(f"\nReprise activee depuis {output_path}.")
    else:
        print(f"\nNouveau calcul. Le rapport sera ecrit dans {output_path}.")
    print("Mode evaluation locale: LLM OpenAI desactive pendant la precision.")

    report = measure_curated(
        "test_questions.json",
        use_rag=True,
        output_path=output_path,
        resume=resume,
        difficulty_filter=difficulty,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_summary(report, output_path)


if __name__ == "__main__":
    main()
