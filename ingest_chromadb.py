"""
=============================================================================
DIVALTO HARMONY — CHROMADB INGESTION SCRIPT
=============================================================================
Loads chunks_children.json + embeddings.npy into a persistent ChromaDB
vector database ready for RAG retrieval.

USAGE:
    pip install chromadb numpy

    python ingest_chromadb.py
    # expects chunks_children.json and embeddings.npy in the same folder
    # creates ./chroma_db/ folder (persistent, reusable across restarts)

WHAT THIS FIXES vs the naive version:
    1. Batching        — ChromaDB crashes above ~41k items in one .add() call
                         we batch in groups of 2000 (safe for any machine)
    2. ID sanitization — ChromaDB rejects accented chars in IDs
                         (é, è, à etc.) — we replace them with ASCII
    3. None metadata   — ChromaDB rejects None values in metadata fields
                         we replace with empty string ""
    4. Embedding shape check — catches mismatch between chunks and embeddings
                         before touching the DB (saves you from a corrupt index)
    5. Idempotent      — safe to re-run: deletes and recreates the collection
                         so you never get duplicates from multiple runs
=============================================================================
"""

import os
import re
import json
import unicodedata
import numpy as np
import chromadb

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CHUNKS_FILE    = "chunks_children.json"
EMBEDDINGS_FILE= "embeddings.npy"
CHROMA_PATH    = "./chroma_db"
COLLECTION_NAME= "divalto_harmony"
BATCH_SIZE     = 2000   # safe for any machine; increase to 5000 if you have RAM

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def sanitize_id(chunk_id: str) -> str:
    """
    ChromaDB requires IDs to match [a-zA-Z0-9_\\-\\.]+
    We normalize accented chars to ASCII then strip anything else.
    Example: 'Chemin_d_accès__001' → 'Chemin_d_acces__001'
    """
    # Decompose accented chars → base + combining mark, then drop marks
    nfkd = unicodedata.normalize('NFKD', chunk_id)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    # Replace any remaining non-allowed chars with underscore
    safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_str)
    return safe


def clean_metadata(chunk: dict) -> dict:
    """
    ChromaDB metadata values must be str, int, or float — never None.
    Replace None with empty string.
    """
    return {
        "source_file": chunk.get("source_file") or "",
        "module"     : chunk.get("module")      or "",
        "parent_id"  : chunk.get("parent_id")   or "",
        "doc_title"  : chunk.get("doc_title")   or "",
        "chunk_type" : chunk.get("chunk_type")  or "",
        "section_hint": chunk.get("section_hint") or "",
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("DIVALTO HARMONY — ChromaDB Ingestion")
    print("=" * 65)

    # ── 1. Load chunks ────────────────────────────────────────────────
    print(f"\nLoading chunks from: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Keep only child chunks (not PARENT aggregates)
    chunks = [c for c in all_chunks if c.get("chunk_type") != "PARENT"]
    print(f"  {len(all_chunks)} total → {len(chunks)} child chunks (parents excluded)")

    # ── 2. Load embeddings ────────────────────────────────────────────
    print(f"\nLoading embeddings from: {EMBEDDINGS_FILE}")
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"  Shape: {embeddings.shape}")

    # Shape check — must match chunk count exactly
    if embeddings.shape[0] != len(chunks):
        print(f"\n  ERROR: embedding count ({embeddings.shape[0]}) "
              f"!= chunk count ({len(chunks)})")
        print("  Your embeddings.npy was computed on a different chunk set.")
        print("  Re-run eval_semantic_v3.py without --embeddings flag to recompute.")
        return

    print(f"  Shape check passed: {embeddings.shape[0]} embeddings for {len(chunks)} chunks")

    # ── 3. Connect to ChromaDB ────────────────────────────────────────
    print(f"\nConnecting to ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection to avoid duplicates on re-run
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Collection '{COLLECTION_NAME}' exists — deleting for clean re-ingest")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity (matches your eval)
    )
    print(f"  Collection '{COLLECTION_NAME}' created")

    # ── 4. Prepare data ───────────────────────────────────────────────
    print("\nPreparing data...")
    ids        = [sanitize_id(c["chunk_id"]) for c in chunks]
    documents  = [c["content"] for c in chunks]
    metadatas  = [clean_metadata(c) for c in chunks]
    emb_list   = embeddings.tolist()

    # Verify no duplicate IDs after sanitization
    unique_ids = set(ids)
    if len(unique_ids) < len(ids):
        dupes = len(ids) - len(unique_ids)
        print(f"  WARNING: {dupes} duplicate IDs after sanitization — making unique")
        seen, deduped = set(), []
        for i, id_ in enumerate(ids):
            if id_ in seen:
                id_ = f"{id_}_{i}"
            seen.add(id_)
            deduped.append(id_)
        ids = deduped

    print(f"  {len(ids)} items ready for ingestion")

    # ── 5. Batch ingest ───────────────────────────────────────────────
    print(f"\nIngesting in batches of {BATCH_SIZE}...")
    total    = len(chunks)
    ingested = 0

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        collection.add(
            ids        = ids[start:end],
            embeddings = emb_list[start:end],
            documents  = documents[start:end],
            metadatas  = metadatas[start:end],
        )

        ingested += (end - start)
        pct = ingested / total * 100
        print(f"  [{pct:5.1f}%] {ingested}/{total} chunks ingested")

    # ── 6. Verify ─────────────────────────────────────────────────────
    count = collection.count()
    print(f"\nVerification: ChromaDB reports {count} items in collection")

    if count == total:
        print("  All chunks ingested successfully")
    else:
        print(f"  WARNING: expected {total}, got {count}")

    # ── 7. Quick smoke test ───────────────────────────────────────────
    print("\nSmoke test — querying with a sample embedding...")
    result = collection.query(
        query_embeddings=[emb_list[0]],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    print("  Top 3 results for first chunk:")
    for i, (doc, meta, dist) in enumerate(zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    )):
        print(f"    {i+1}. [{meta['module']}] {meta['source_file']}  dist={dist:.4f}")
        print(f"       {doc[:80]}...")

    print(f"\n{'='*65}")
    print(f"  ChromaDB ready at: {os.path.abspath(CHROMA_PATH)}")
    print(f"  Collection: '{COLLECTION_NAME}'  |  {count} chunks indexed")
    print(f"  Next step: build the RAG chain (retrieval + LLM generation)")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()