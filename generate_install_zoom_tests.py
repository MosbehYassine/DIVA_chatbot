#!/usr/bin/env python3
"""
Genere un jeu de tests cible sur les themes Installation et Zoom.
"""

import json
import os
import re
import unicodedata

INPUT_FILE = "chunks_metadata.json"
OUTPUT_FILE = "test_questions_installation_zoom.json"
MAX_INSTALL = 99999
MAX_ZOOM = 99999


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def source_stem(source: str) -> str:
    base = os.path.basename(source)
    return os.path.splitext(base)[0]


def guess_title(text: str, fallback: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if 10 <= len(line) <= 120 and re.search(r"[A-Za-z]", line):
            return line
    return fallback.replace("_", " ")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tests = []
    install_count = 0
    zoom_count = 0
    seen_sources = set()

    for ch in chunks:
        source = ch.get("source", "")
        text = ch.get("text", "")
        if not source or source in seen_sources:
            continue

        src_norm = normalize(source)
        txt_norm = normalize(text)

        is_install = "installation" in src_norm or "installation" in txt_norm
        is_zoom = "zoom" in src_norm or "zoom " in txt_norm or "zooms " in txt_norm

        module = None
        if is_install and install_count < MAX_INSTALL:
            module = "Installation"
            install_count += 1
        elif is_zoom and zoom_count < MAX_ZOOM:
            module = "Zoom"
            zoom_count += 1
        else:
            continue

        stem = source_stem(source)
        title = guess_title(text, stem)
        words = [w for w in normalize(title).split() if len(w) >= 5][:4]
        question = f"Que dit la documentation sur: {title} ?"

        tests.append(
            {
                "id": f"{module.lower()}_{len(tests)+1:03d}",
                "module": module,
                "question": question,
                "expected_source_contains": [stem],
                "expected_keywords": words,
                "source": source,
            }
        )
        seen_sources.add(source)

        if install_count >= MAX_INSTALL and zoom_count >= MAX_ZOOM:
            break

    payload = {
        "metadata": {
            "generated_from": INPUT_FILE,
            "installation_tests": install_count,
            "zoom_tests": zoom_count,
            "total_tests": len(tests),
        },
        "test_questions": tests,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(tests)} tests ({install_count} Installation, {zoom_count} Zoom) -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
