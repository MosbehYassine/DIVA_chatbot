#!/usr/bin/env python3
"""
Genere automatiquement un jeu de tests de couverture a partir de chunks_metadata.json.
Objectif: couvrir un maximum de fichiers sources avec des questions simples basees sur les titres.
"""

import json
import os
import re
import unicodedata
from collections import defaultdict

INPUT_FILE = "chunks_metadata.json"
OUTPUT_FILE = "test_questions_auto_coverage.json"
MAX_TESTS = 999999


def normalize(text: str) -> str:
    text = (text or "").strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def file_stem(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return normalize(stem)


def guess_title(text: str, fallback: str) -> str:
    # Prend la premiere ligne non vide significative comme titre
    for line in (text or "").splitlines():
        line = normalize(line).strip()
        if len(line) >= 8 and len(line) <= 120:
            # eviter lignes trop "bruitees"
            if re.search(r"[A-Za-z]", line):
                return line
    return fallback.replace("_", " ")


def build_question_from_title(title: str) -> str:
    # Question simple pour forcer le retrieval sur le bon document
    return f"Que dit la documentation a propos de: {title} ?"


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Fichier introuvable: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Regroupe par source, puis prend un chunk representatif
    by_source = defaultdict(list)
    for ch in chunks:
        source = ch.get("source", "")
        if source:
            by_source[source].append(ch)

    tests = []
    # Prioriser une couverture par repertoire de 1er niveau sous data/
    per_group_count = defaultdict(int)
    max_per_group = 999999

    for source, items in by_source.items():
        rel = source.replace("\\", "/")
        parts = rel.split("/")
        group = parts[1] if len(parts) > 1 else "other"
        if per_group_count[group] >= max_per_group:
            continue

        sample = items[0]
        stem = file_stem(source)
        title = guess_title(sample.get("text", ""), stem)
        q = build_question_from_title(title)

        tests.append(
            {
                "id": f"auto_{len(tests)+1:03d}",
                "module": group,
                "question": q,
                "expected_source_contains": [stem],
                "expected_keywords": [w for w in re.split(r"\W+", title.lower()) if len(w) >= 5][:4],
                "source": source,
            }
        )
        per_group_count[group] += 1
        if len(tests) >= MAX_TESTS:
            break

    payload = {
        "metadata": {
            "generated_from": INPUT_FILE,
            "total_tests": len(tests),
            "max_tests": MAX_TESTS,
            "strategy": "one representative test per source with per-group cap",
        },
        "test_questions": tests,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(tests)} tests -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
