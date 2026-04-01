"""
=============================================================================
DIVALTO HARMONY — SEMANTIC EMBEDDING EVALUATION SCRIPT
=============================================================================
This script replaces the keyword-overlap proxy in pipeline_v5 with REAL
semantic similarity using multilingual-e5-large embeddings.

HOW TO RUN:
    pip install sentence-transformers numpy tqdm

    # First run  (~5 min CPU / ~45s GPU) — computes & saves embeddings
    python eval_semantic.py --chunks chunks_children.json

    # Subsequent runs — loads saved embeddings instantly (skips recompute)
    python eval_semantic.py --chunks chunks_children.json --embeddings embeddings.npy

OUTPUT FILES:
    embeddings.npy          — chunk embeddings (save once, reuse forever)
    eval_semantic.json      — full per-question results + module breakdown
    eval_semantic_report.txt — human-readable report you can paste into your report

WHAT THIS MEASURES:
    recall@1  — correct source file is the TOP result
    recall@3  — correct source file is in the TOP 3 results
    recall@5  — correct source file is in the TOP 5 results
    MRR       — Mean Reciprocal Rank (standard IR metric)
    per-module recall — which modules are weakest (tells you where to fix chunking)
=============================================================================
"""

import os
import json
import argparse
import time
import re
from collections import defaultdict

import numpy as np

# ─── GOLD STANDARD — 40 QUESTIONS (comprehensive production eval set) ─────────
# Covers: Administration, Installation, RecordSql, Erreurs, and cross-module topics
# Structured as: question, expected_file, module, category
# Category: direct (answer is the title) / indirect (answer is inside the doc)

GOLD_QA = [
    # ── ADMINISTRATION (20 questions) ──────────────────────────────────────────
    {
        "question"     : "Comment accéder à un fichier réseau Windows avec son nom Harmony ?",
        "expected_file": "Acc_s___un_fichier_r_seau_Windows_sous_son_nom_Harmony.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment créer ou modifier un modèle d'imprimante dans Harmony ?",
        "expected_file": "Gestiondesmod_lesd_imprimante.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Que fait la commande /Ga dans un fichier pilote ?",
        "expected_file": "Fichierdecommandes.htm",
        "module"       : "Administration",
        "category"     : "indirect",
    },
    {
        "question"     : "Comment déclarer un chemin Harmony sur un serveur Xlan ?",
        "expected_file": "Nomsdefichierenr_seauXlan.htm",
        "module"       : "Administration",
        "category"     : "indirect",
    },
    {
        "question"     : "Comment configurer l'imprimante par défaut de Windows dans Harmony ?",
        "expected_file": "Imprimantepard_fautdeWindows.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment importer des utilisateurs depuis un annuaire LDAP dans Harmony ?",
        "expected_file": "Import_et_synchronisation_des_utilisateurs_d\u2019un_annuaire_LDAP.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Quelle est la différence entre un modèle préconisé et un modèle impératif ?",
        "expected_file": "Mod_lepr_conis_etmod_leimp_ratif.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment lancer automatiquement une tâche de fond au démarrage d'Harmony ?",
        "expected_file": "Lancementautomatiqued_unet_chedefond.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment configurer les chemins implicites sur un poste client Harmony ?",
        "expected_file": "Cheminsd_acc_simplicites.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment passer des paramètres à un programme Harmony version 6 ?",
        "expected_file": "Passagedeparam_tresauxprogrammes.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment configurer une liaison entre un modèle et une imprimante Windows ?",
        "expected_file": "Liaisonentremod_leetimprimante_param_trageWindows.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment fonctionne le format spool dans un modèle d'imprimante Harmony ?",
        "expected_file": "Gestiondesmod_lesd_imprimante.htm",
        "module"       : "Administration",
        "category"     : "indirect",
    },
    {
        "question"     : "Comment iconiser une application Harmony version 6 ?",
        "expected_file": "Iconiseruneapplication.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment configurer le gestionnaire d'impression Windows (spouleur) dans Harmony ?",
        "expected_file": "Gestionnaired_impressiondeWindows_spouleur_.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment synchroniser les utilisateurs Harmony avec un annuaire LDAP ?",
        "expected_file": "Synchronisation.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Quelles sont les règles de correspondance des propriétés LDAP dans Harmony ?",
        # FIX: original GOLD had ASCII-mangled name; actual file in corpus uses URL-decoded Unicode name
        "expected_file": "Règles_de_correspondance_des_propriétés.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment paramétrer une icône pour une tâche de fond dans Harmony ?",
        # FIX: original GOLD had full name; disk file is physically truncated to 'unet_che.htm'
        "expected_file": "Param_tragedel_iconed_unet_che.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment déclarer un serveur dans la table des serveurs Harmony ?",
        # FIX: original GOLD was missing 'logiques' suffix; actual file includes it
        "expected_file": "D_clarationdesserveursetdesunit_slogiques.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment déclarer un chemin dans la table des chemins Harmony ?",
        "expected_file": "D_clarationdescheminsHarmony.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    {
        "question"     : "Comment accéder au fichier des utilisateurs Harmony ?",
        "expected_file": "Fichierdesutilisateurs.htm",
        "module"       : "Administration",
        "category"     : "direct",
    },
    # ── INSTALLATION (5 questions) ──────────────────────────────────────────────
    {
        "question"     : "Comment installer le serveur Web Harmony ?",
        "expected_file": "Installation_du_serveur_Web.htm",
        "module"       : "Installation",
        "category"     : "direct",
    },
    {
        "question"     : "Comment installer et paramétrer le service dhsTelnet (Xtelnet) ?",
        "expected_file": "Installation_et_param_trage_du_service_Xtelnet.htm",
        "module"       : "Installation",
        "category"     : "direct",
    },
    {
        "question"     : "Comment configurer la mémoire commune dans Harmony ?",
        "expected_file": "Configurationdelam_moirecommune.htm",
        "module"       : "Installation",
        "category"     : "direct",
    },
    {
        "question"     : "Comment configurer le client léger HTML dans Harmony ?",
        "expected_file": "Configuration_du_client_l_ger_Html.htm",
        "module"       : "Installation",
        "category"     : "direct",
    },
    {
        "question"     : "Comment activer la compression des trames pour le client léger Web ?",
        "expected_file": "Compression_des_trames_pour_un_client_l_ger_Web.htm",
        "module"       : "Installation",
        "category"     : "direct",
    },
    # ── RECORD SQL (5 questions) ────────────────────────────────────────────────
    {
        "question"     : "Comment se connecter au serveur SQL depuis Harmony ?",
        "expected_file": "ConnexionauserveurSQL.htm",
        "module"       : "RecordSql",
        "category"     : "direct",
    },
    {
        "question"     : "Quelles sont les particularités de DB2 sur un serveur IBM i dans Harmony ?",
        "expected_file": "ConnexionauserveurSQL.htm",
        "module"       : "RecordSql",
        "category"     : "indirect",
    },
    {
        "question"     : "Comment désactiver le contexte d'utilisation par défaut dans Harmony ?",
        "expected_file": "D_sactivation_du_contexte_d_utilisation_par_d_faut.htm",
        "module"       : "RecordSql",
        "category"     : "direct",
    },
    {
        "question"     : "Comment insérer et supprimer des objets dans Harmony ?",
        "expected_file": "Ins_rer_et_supprimer_des_objets.htm",
        "module"       : "RecordSql",
        "category"     : "direct",
    },
    {
        "question"     : "Comment paramétrer un agent d'impression dans Harmony ?",
        "expected_file": "Param_trage.htm",
        "module"       : "RecordSql",
        "category"     : "direct",
    },
    # ── ERREURS / CROSS-MODULE (10 questions) ───────────────────────────────────
    {
        "question"     : "Quels sont les codes d'erreur Windows renvoyés par Harmony (erreurs 10xxx) ?",
        "expected_file": "10xxx_Erreurssignal_esparWindows.htm",
        "module"       : "Erreurs",
        "category"     : "direct",
    },
    {
        "question"     : "Comment fonctionne la sélection multiple dans le zoom Harmony ?",
        "expected_file": "S_lection_multiple_du_zoom.htm",
        "module"       : "zoom",
        "category"     : "direct",
    },
    {
        "question"     : "Comment effectuer une consultation en mode fiche dans un zoom Harmony ?",
        "expected_file": "Consultationenmodefiche.htm",
        "module"       : "zoom",
        "category"     : "direct",
    },
    {
        "question"     : "Comment lancer un zoom depuis un menu dans Harmony ?",
        "expected_file": "Appeld_unzoomdepuisunmenu.htm",
        "module"       : "yzoom",
        "category"     : "direct",
    },
    {
        "question"     : "Comment afficher un agenda dans Harmony ymeg2 ?",
        "expected_file": "Affichage_d_un_agenda.htm",
        "module"       : "ymeg2",
        "category"     : "direct",
    },
    {
        "question"     : "Comment programmer un tableau dans Harmony ymeg ?",
        "expected_file": "Programmationd_untableau.htm",
        "module"       : "ymeg",
        "category"     : "direct",
    },
    {
        "question"     : "Quelles sont les restrictions imposées par les navigateurs Web dans Harmony ?",
        "expected_file": "Restrictions_impos_es_par_les_navigateurs_Web.htm",
        "module"       : "ymeg2",
        "category"     : "direct",
    },
    {
        "question"     : "Comment utiliser le drag and drop dans un tableau Harmony ymeg ?",
        "expected_file": "Drag_and_drop_d__l_ments_de_tableau.htm",
        "module"       : "ymeg",
        "category"     : "direct",
    },
    {
        "question"     : "Comment gérer la liste associée à un tableau dans Harmony ymeg ?",
        "expected_file": "Gestiondelalisteassoci_e_untableau.htm",
        "module"       : "ymeg",
        "category"     : "direct",
    },
    {
        "question"     : "Comment utiliser XmeListSetAttribut pour modifier les attributs d'une liste ?",
        "expected_file": "XmeListSetAttribut.htm",
        "module"       : "ymeg",
        "category"     : "direct",
    },
]


# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Semantic RAG eval for Divalto Harmony")
    p.add_argument('--chunks',     default='chunks_children.json',
                   help='Path to chunks_children.json (from pipeline_v5)')
    p.add_argument('--embeddings', default=None,
                   help='Path to pre-saved embeddings .npy (skip recompute if exists)')
    p.add_argument('--save_emb',   default='embeddings.npy',
                   help='Where to save computed embeddings')
    p.add_argument('--model',      default='intfloat/multilingual-e5-large',
                   help='HuggingFace model name')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--top_k',      type=int, default=10,
                   help='How many results to retrieve per query for analysis')
    p.add_argument('--reranker',   default='cross-encoder/ms-marco-MiniLM-L-6-v2',
                   help='Cross-encoder reranker model. Pass --reranker none to disable.')
    p.add_argument('--output',     default='eval_semantic.json')
    p.add_argument('--report',     default='eval_semantic_report.txt')
    return p.parse_args()


# ─── EMBEDDING UTILS ──────────────────────────────────────────────────────────

def load_model(model_name: str):
    """Load multilingual-e5-large. Downloads ~2.2GB on first use."""
    print(f"\nLoading model: {model_name}")
    print("(First run downloads ~2.2 GB — subsequent runs use local cache)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print("Model loaded.")
    return model


def encode_chunks(model, chunks: list, batch_size: int = 32) -> np.ndarray:
    """
    Encode all chunk contents.
    multilingual-e5-large requires 'passage: ' prefix for documents.
    """
    print(f"\nEncoding {len(chunks)} chunks (batch_size={batch_size})...")
    print("Estimated time: ~5 min CPU / ~45s GPU")
    t0 = time.time()

    texts = ["passage: " + c['content'] for c in chunks]

    try:
        from tqdm import tqdm
        embeddings = model.encode(
            texts,
            batch_size    = batch_size,
            show_progress_bar = True,
            normalize_embeddings = True,   # cosine similarity = dot product
            convert_to_numpy = True,
        )
    except ImportError:
        embeddings = model.encode(
            texts,
            batch_size    = batch_size,
            normalize_embeddings = True,
            convert_to_numpy = True,
        )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  "
          f"({len(chunks)/elapsed:.0f} chunks/s)")
    return embeddings


def encode_queries(model, questions: list) -> np.ndarray:
    """
    Encode gold questions.
    multilingual-e5-large requires 'query: ' prefix for queries.
    """
    texts = ["query: " + q for q in questions]
    return model.encode(
        texts,
        normalize_embeddings = True,
        convert_to_numpy = True,
        show_progress_bar = False,
    )


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def _normalize_filename(name: str) -> str:
    """
    Normalize a filename for comparison:
    - ASCII-escape accented chars  (é -> _)
    - Lowercase
    This bridges the gap between GOLD_QA names (cp1252-mangled, ASCII-only)
    and chunk source_file names (real Unicode filenames from the filesystem).
    """
    import re
    return re.sub(r'[^\x00-\x7F]', '_', name).lower()


# Pre-build a lookup: normalized_name -> original stored name
# Built once at eval time for O(1) lookup per question
def build_filename_lookup(chunks: list) -> dict:
    lookup = {}
    for c in chunks:
        sf = c['source_file']
        key = _normalize_filename(sf)
        if key not in lookup:
            lookup[key] = sf
        # also keep verbatim match
        lookup[sf.lower()] = sf
    return lookup


def retrieve_top_k(
    query_emb   : np.ndarray,   # (dim,)
    chunk_embs  : np.ndarray,   # (N, dim)
    chunks      : list,
    k           : int = 10,
) -> list:
    """
    Pure numpy cosine similarity (embeddings already normalized → dot product).
    Returns list of (chunk, score) sorted by score desc.
    """
    scores = chunk_embs @ query_emb          # (N,)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ─── CROSS-ENCODER RERANKER ───────────────────────────────────────────────────

def load_reranker(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Load the cross-encoder reranker.
    Downloads ~85 MB on first use (much faster than the embedding model).

    Why this works:
      The bi-encoder (multilingual-e5-large) compresses query and chunk into
      independent vectors — fast but lossy. At recall@3=60%, roughly 10 of our
      failures are cases where the RIGHT chunk is at rank 4-6, just below the
      cutoff.
      The cross-encoder sees the FULL (query, chunk) pair and can catch subtle
      keyword matches the bi-encoder missed. It is ~100× slower per pair, so we
      only run it on the top-20 candidates already narrowed by the bi-encoder.
    """
    print(f"\nLoading cross-encoder reranker: {model_name}")
    print("(First run downloads ~85 MB)")
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(model_name)
        print("Reranker loaded.")
        return reranker
    except ImportError:
        print("  ⚠ sentence-transformers not installed.")
        print("    pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"  ⚠ Reranker load failed: {e}")
        return None


def rerank(
    question    : str,
    candidates  : list,          # list of (chunk, bi_encoder_score)
    reranker,                    # CrossEncoder or None
    top_k       : int = 3,
) -> list:
    """
    Rerank top-N candidates with cross-encoder, return top_k.

    HYBRID SCORING: final_score = reranker_score + 0.15 * bi_encoder_score
    This prevents the cross-encoder from completely overriding the bi-encoder
    when the reranker is uncertain (it's English-trained on French content).
    The bi-encoder weight (0.15) acts as a tiebreaker for near-equal reranker scores,
    fixing the negative-gap cases where the right file had higher bi-encoder score
    but was demoted by the reranker.

    If reranker is None, falls back to original bi-encoder ordering.
    """
    if reranker is None or not candidates:
        return candidates[:top_k]

    pairs  = [(question, c['content']) for c, _ in candidates]
    ce_scores = reranker.predict(pairs)

    # Normalise bi-encoder scores to [0,1] range for stable blending
    be_scores = [float(s) for _, s in candidates]
    be_min, be_max = min(be_scores), max(be_scores)
    be_range = be_max - be_min if be_max > be_min else 1.0
    be_norm  = [(s - be_min) / be_range for s in be_scores]

    hybrid = [
        (cand, float(ce) + 0.15 * float(be))
        for cand, ce, be in zip(candidates, ce_scores, be_norm)
    ]
    reranked = sorted(hybrid, key=lambda x: -x[1])
    return [cand for cand, _ in reranked[:top_k]]

    pairs  = [(question, c['content']) for c, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: -x[1]
    )
    return [cand for cand, _ in reranked[:top_k]]




# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(
    gold_qa    : list,
    chunks     : list,
    chunk_embs : np.ndarray,
    query_embs : np.ndarray,
    top_k      : int = 10,
    reranker   = None,           # CrossEncoder or None
) -> dict:

    results       = []
    recall_hits   = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum       = 0.0
    module_stats  = defaultdict(lambda: {'total': 0, 'hit@1': 0, 'hit@3': 0, 'hit@5': 0})
    category_stats= defaultdict(lambda: {'total': 0, 'hit@1': 0, 'hit@3': 0})

    # Build filename lookup ONCE for the entire eval run
    # This resolves unicode/encoding mismatches between GOLD_QA and stored chunk names
    fname_lookup = build_filename_lookup(chunks)

    for i, qa in enumerate(gold_qa):
        q_emb    = query_embs[i]
        # Step 1: bi-encoder retrieves top-20 candidates fast
        top20    = retrieve_top_k(q_emb, chunk_embs, chunks, k=max(top_k, 20))
        # Step 2: cross-encoder reranks top-20, returns top-3 precisely
        top3_reranked = rerank(qa['question'], top20, reranker, top_k=3)
        # Step 3: keep full top-k for recall@5/10 stats (bi-encoder only — reranker too slow for 10)
        top      = top3_reranked + [c for c in top20 if c not in top3_reranked][:(top_k-3)]
        top_files= [c['source_file'] for c, _ in top]
        module   = qa.get('module', '?')
        category = qa.get('category', 'direct')

        # Resolve expected_file: try exact match first, then normalized lookup
        # This fixes the "expected=0.0000" bug where GOLD uses ASCII-mangled names
        # but chunks store real Unicode filenames (e.g. 'léger' vs 'l_ger')
        raw_expected = qa['expected_file']
        expected = (
            fname_lookup.get(raw_expected)                        # exact match
            or fname_lookup.get(raw_expected.lower())             # case-insensitive
            or fname_lookup.get(_normalize_filename(raw_expected))# unicode-normalized
            or raw_expected                                        # fallback (will likely miss)
        )

        # Find rank of expected file (1-indexed, 0 = not found in top_k)
        # Compare normalized to handle any remaining encoding edge cases
        expected_norm = _normalize_filename(expected)
        rank = 0
        for r, (c, _) in enumerate(top, 1):
            if _normalize_filename(c['source_file']) == expected_norm:
                rank = r
                break

        hit1  = rank == 1
        hit3  = 0 < rank <= 3
        hit5  = 0 < rank <= 5
        hit10 = 0 < rank <= 10

        if hit1:  recall_hits[1]  += 1
        if hit3:  recall_hits[3]  += 1
        if hit5:  recall_hits[5]  += 1
        if hit10: recall_hits[10] += 1

        mrr_sum += (1.0 / rank) if rank > 0 else 0.0

        # Per-module
        module_stats[module]['total'] += 1
        if hit1: module_stats[module]['hit@1'] += 1
        if hit3: module_stats[module]['hit@3'] += 1
        if hit5: module_stats[module]['hit@5'] += 1

        # Per-category
        category_stats[category]['total'] += 1
        if hit1: category_stats[category]['hit@1'] += 1
        if hit3: category_stats[category]['hit@3'] += 1

        # Score for the resolved expected file (normalized comparison)
        expected_score = 0.0
        for c, s in top:
            if _normalize_filename(c['source_file']) == expected_norm:
                expected_score = float(s)
                break

        results.append({
            'question'         : qa['question'],
            'expected_file_raw': raw_expected,    # what GOLD_QA says
            'expected_file'    : expected,         # resolved to actual stored name
            'module'           : module,
            'category'         : category,
            'rank'             : rank,
            'hit@1'            : hit1,
            'hit@3'            : hit3,
            'hit@5'            : hit5,
            'top_score'        : round(float(top[0][1]), 4) if top else 0,
            'expected_score'   : round(expected_score, 4),
            'top3_files'       : top_files[:3],
            'top1_chunk_type'  : top[0][0]['chunk_type'] if top else '',
        })

    n = len(gold_qa)
    return {
        'n_questions': n,
        'recall@1'   : round(recall_hits[1]  / n * 100, 1),
        'recall@3'   : round(recall_hits[3]  / n * 100, 1),
        'recall@5'   : round(recall_hits[5]  / n * 100, 1),
        'recall@10'  : round(recall_hits[10] / n * 100, 1),
        'mrr'        : round(mrr_sum / n, 4),
        'grade'      : grade(recall_hits[3] / n),
        'per_module' : {
            m: {
                'total'   : v['total'],
                'recall@1': round(v['hit@1'] / v['total'] * 100, 1),
                'recall@3': round(v['hit@3'] / v['total'] * 100, 1),
                'recall@5': round(v['hit@5'] / v['total'] * 100, 1),
            }
            for m, v in module_stats.items()
        },
        'per_category': {
            c: {
                'total'   : v['total'],
                'recall@1': round(v['hit@1'] / v['total'] * 100, 1),
                'recall@3': round(v['hit@3'] / v['total'] * 100, 1),
            }
            for c, v in category_stats.items()
        },
        'details'  : results,
        'model'    : 'intfloat/multilingual-e5-large',
        'reranker' : 'cross-encoder/ms-marco-MiniLM-L-6-v2' if reranker else 'disabled',
        'n_chunks' : len(chunks),
    }


def grade(ratio: float) -> str:
    if ratio > 0.92: return 'A+'
    if ratio > 0.85: return 'A'
    if ratio > 0.75: return 'B+'
    if ratio > 0.65: return 'B'
    if ratio > 0.55: return 'C+'
    return 'C'


# ─── REPORT PRINTER ───────────────────────────────────────────────────────────

def print_report(R: dict) -> str:
    W    = 72
    lines= []

    def p(s=""): lines.append(s); print(s)

    p("=" * W)
    p("DIVALTO HARMONY — SEMANTIC RAG EVALUATION")
    p(f"Model   : {R['model']}")
    p(f"Reranker: {R.get('reranker', 'disabled')}")
    p(f"Corpus  : {R['n_chunks']} chunks  |  {R['n_questions']} gold questions")
    p("=" * W)

    p()
    p("RETRIEVAL METRICS (semantic embeddings — real numbers)")
    p("-" * W)
    p(f"  recall@1   : {R['recall@1']:5.1f}%   (correct file is #1 result)")
    p(f"  recall@3   : {R['recall@3']:5.1f}%   (correct file in top 3)   ← main KPI")
    p(f"  recall@5   : {R['recall@5']:5.1f}%   (correct file in top 5)")
    p(f"  recall@10  : {R['recall@10']:5.1f}%   (correct file in top 10)")
    p(f"  MRR        : {R['mrr']:.4f}   (mean reciprocal rank)")
    p(f"  Grade      : {R['grade']}")

    # Interpretation
    r3 = R['recall@3']
    p()
    p("INTERPRETATION")
    p("-" * W)
    if r3 >= 92:
        p("  EXCELLENT. Chunking is not your bottleneck.")
        p("  Next: add a cross-encoder reranker (ms-marco-MiniLM-L-6-v2)")
        p("  and hybrid BM25 + vector retrieval for short queries.")
    elif r3 >= 85:
        p("  GOOD. Chunking is solid. Small targeted fixes will push you over 90%.")
        p("  Check per-module table below for which modules need attention.")
    elif r3 >= 75:
        p("  ACCEPTABLE but not production-ready. Systematic issues in chunking.")
        p("  Likely: SHORT_ATOMIC chunks too thin for embedding signal.")
        p("  Fix: merge SHORT_ATOMIC chunks up to TARGET_CHUNK_WORDS=150.")
    elif r3 >= 65:
        p("  NEEDS WORK. Structural chunking problem.")
        p("  Check if xdiva1 or ymeg modules are dragging the score.")
        p("  Consider per-module chunking strategies.")
    else:
        p("  CRITICAL. Major chunking issue — possibly encoding corruption.")
        p("  Check that multilingual-e5-large can read your chunk content.")

    p()
    p("PER-MODULE RECALL@3")
    p("-" * W)
    mod_sorted = sorted(R['per_module'].items(),
                        key=lambda x: x[1]['recall@3'])
    for m, v in mod_sorted:
        bar = "█" * int(v['recall@3'] / 5)
        flag = " ← NEEDS FIX" if v['recall@3'] < 70 and v['total'] >= 3 else ""
        p(f"  {m:20s}: {v['recall@3']:5.1f}%  {bar}{flag}")

    p()
    p("PER-CATEGORY RECALL")
    p("-" * W)
    for cat, v in R['per_category'].items():
        p(f"  {cat:10s}: @1={v['recall@1']}%  @3={v['recall@3']}%  (n={v['total']})")
    p("  Note: 'indirect' questions (answer buried in doc, not in title) are harder.")
    p("        If indirect recall@3 < direct recall@3 - 20%, add more context overlap.")

    p()
    p("QUESTION BREAKDOWN")
    p("-" * W)
    failures = []
    for d in R['details']:
        icon = "✅" if d['hit@3'] else "❌"
        r1   = "🎯" if d['hit@1'] else "  "
        p(f"  {icon}{r1}  [{d['module']:15s}]  {d['question'][:52]}")
        if not d['hit@3']:
            failures.append(d)
            # Show resolved filename; flag if it differs from GOLD (encoding fix was applied)
            raw = d.get('expected_file_raw', d['expected_file'])
            resolved = d['expected_file']
            if raw != resolved:
                p(f"         Expected  : {resolved[:60]}  [resolved from: {raw[:40]}]")
            else:
                p(f"         Expected  : {resolved[:60]}")
            p(f"         Got top-3 : {[f[:35] for f in d['top3_files']]}")
            p(f"         Scores    : expected={d['expected_score']:.4f}  top1={d['top_score']:.4f}")

    if failures:
        p()
        p("FAILURE ANALYSIS")
        p("-" * W)
        p(f"  {len(failures)} questions failed recall@3. Patterns:")
        p()

        # Score gap analysis
        large_gap = [d for d in failures if d['top_score'] - d['expected_score'] > 0.05]
        small_gap = [d for d in failures if d['top_score'] - d['expected_score'] <= 0.05]

        if large_gap:
            p(f"  SCORE GAP > 0.05 ({len(large_gap)} questions) — wrong file wins convincingly")
            p("  → The expected chunk needs more distinctive content.")
            p("    Options: (1) inject more title/section signal into chunk content")
            p("             (2) the expected file may need a dedicated SHORT_ATOMIC chunk")
            p("             with just its title + key terms as a 'dense retrieval anchor'")
            for d in large_gap:
                gap = d['top_score'] - d['expected_score']
                p(f"    - {d['question'][:55]}  gap={gap:.4f}")

        if small_gap:
            p()
            p(f"  SCORE GAP ≤ 0.05 ({len(small_gap)} questions) — close miss, reranker would fix")
            p("  → These are retrieval boundary cases.")
            p("    A cross-encoder reranker (ms-marco-MiniLM-L-6-v2) will fix most of these.")
            for d in small_gap:
                gap = d['top_score'] - d['expected_score']
                p(f"    - {d['question'][:55]}  gap={gap:.4f}")

    p()
    p("NEXT STEPS (priority order)")
    p("-" * W)

    # Compute weak modules for targeted advice
    weak_modules = [m for m, v in R['per_module'].items()
                    if v['recall@3'] < 70 and v['total'] >= 3]

    step = 1
    if r3 < 90:
        p(f"  {step}. ADD CROSS-ENCODER RERANKER")
        p("     from sentence_transformers import CrossEncoder")
        p("     reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')")
        p("     # retrieve top-20, rerank, return top-3")
        p("     Expected lift: +5 to +12% recall@3")
        step += 1

    if weak_modules:
        p()
        p(f"  {step}. FIX WEAK MODULES: {weak_modules}")
        p("     Check if these modules have many SHORT_ATOMIC chunks.")
        p("     If so, lower MIN_CHUNK_WORDS from 35 to 20 and")
        p("     force SHORT_ATOMIC files to merge with their nearest neighbour.")
        step += 1

    p()
    p(f"  {step}. ADD HYBRID BM25 + VECTOR RETRIEVAL")
    p("     pip install rank-bm25")
    p("     BM25 catches exact technical terms (function names, error codes)")
    p("     that embeddings sometimes miss.")
    p("     Expected lift: +3 to +8% on technical/API questions.")
    step += 1

    if r3 < 85:
        p()
        p(f"  {step}. EXPAND GOLD SET to 80+ questions covering all 30 modules")
        p("     Current 40 questions are too Administration/ymeg-heavy")
        p("     to give reliable signal on xdiva1 (your largest module).")

    p()
    p("=" * W)

    return "\n".join(lines)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load chunks ────────────────────────────────────────────────────────
    print(f"Loading chunks from: {args.chunks}")
    with open(args.chunks, encoding='utf-8') as f:
        all_chunks = json.load(f)

    # Filter out PARENT chunks if mixed file
    chunks = [c for c in all_chunks if c.get('chunk_type') != 'PARENT']
    print(f"  {len(all_chunks)} total → {len(chunks)} child chunks (parents excluded)")

    # ── Diagnostic: check GOLD_QA expected files against corpus ──────────────
    import re as _re
    def _norm(s): return _re.sub(r'[^\x00-\x7F]', '_', s).lower()
    fname_lookup_diag = {}
    for c in chunks:
        sf = c['source_file']
        fname_lookup_diag[_norm(sf)] = sf
        fname_lookup_diag[sf.lower()] = sf

    resolved_count = 0
    missing_files  = []
    for qa in GOLD_QA:
        ef  = qa['expected_file']
        hit = (fname_lookup_diag.get(ef)
               or fname_lookup_diag.get(ef.lower())
               or fname_lookup_diag.get(_norm(ef)))
        if hit:
            resolved_count += 1
        else:
            missing_files.append(ef)

    print(f"\n── GOLD_QA FILE RESOLUTION ──────────────────────────────────────────")
    print(f"  {resolved_count}/{len(GOLD_QA)} expected files resolved in corpus")
    if missing_files:
        print(f"  ⚠️  {len(missing_files)} file(s) NOT FOUND in corpus (will always score 0):")
        for mf in missing_files:
            print(f"     - {mf}")
    else:
        print(f"  ✅ All expected files found — encoding fix is working correctly")
    print(f"─────────────────────────────────────────────────────────────────────\n")

    # ── 2. Load or compute embeddings ─────────────────────────────────────────
    emb_path = args.embeddings or args.save_emb
    if args.embeddings and os.path.exists(args.embeddings):
        print(f"\nLoading pre-computed embeddings from: {args.embeddings}")
        chunk_embs = np.load(args.embeddings)
        print(f"  Shape: {chunk_embs.shape}")
        if chunk_embs.shape[0] != len(chunks):
            print(f"  WARNING: embedding count ({chunk_embs.shape[0]}) != chunk count ({len(chunks)})")
            print("  Recomputing embeddings...")
            chunk_embs = None
    else:
        chunk_embs = None

    if chunk_embs is None:
        model      = load_model(args.model)
        chunk_embs = encode_chunks(model, chunks, args.batch_size)
        np.save(args.save_emb, chunk_embs)
        print(f"Embeddings saved to: {args.save_emb}")
        print("(Next run: pass --embeddings embeddings.npy to skip recompute)")
    else:
        model = load_model(args.model)

    # ── 3. Encode queries ─────────────────────────────────────────────────────
    print(f"\nEncoding {len(GOLD_QA)} gold questions...")
    questions  = [qa['question'] for qa in GOLD_QA]
    query_embs = encode_queries(model, questions)

    # ── 4. Load reranker ──────────────────────────────────────────────────────
    reranker = None
    if args.reranker.lower() != 'none':
        reranker = load_reranker(args.reranker)
    else:
        print("\nReranker disabled (--reranker none)")

    # ── 5. Run evaluation ─────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    results = evaluate(GOLD_QA, chunks, chunk_embs, query_embs,
                       top_k=args.top_k, reranker=reranker)

    # ── 5. Save JSON results ──────────────────────────────────────────────────
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON results saved to: {args.output}")

    # ── 6. Print and save report ──────────────────────────────────────────────
    print()
    report_text = print_report(results)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nText report saved to: {args.report}")

    # ── 7. Quick summary ──────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════╗
  recall@1  : {results['recall@1']:5.1f}%
  recall@3  : {results['recall@3']:5.1f}%   ← main KPI
  recall@5  : {results['recall@5']:5.1f}%
  MRR       : {results['mrr']:.4f}
  Grade     : {results['grade']}
╚══════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()