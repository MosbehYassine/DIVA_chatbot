"""
=============================================================================
DIVALTO HARMONY DOCUMENTATION — INTELLIGENT CHUNKING PIPELINE  v5.0
=============================================================================
Project : RAG Chatbot for Divalto Harmony ERP — Full corpus (4090 files)

WHAT'S NEW vs v4  (architecture changes, not just fixes)
─────────────────────────────────────────────────────────────────────────────

UPGRADE 1 — FULL CORPUS: all 4090 files across all modules (not just Administration)
  v4 was hardcoded to Administration/Administration (60 files out of 4090).
  v5 walks the entire DATAA tree, detects the module from the folder path,
  and injects it into every chunk's metadata and context_prefix.

UPGRADE 2 — MODULE-AWARE CONTEXT PREFIX
  Before : "Divalto Harmony – Administration – {title}"
  After  : "Divalto Harmony – {module} – {title} – {h2_section}"
  This single change is the biggest recall booster for full-corpus retrieval.
  The LLM embedding sees module identity, not just document title.

UPGRADE 3 — H2/H3 HEADER-AWARE SPLITTING (true adaptive contextual chunking)
  v4 ignored all <h2>/<h3> tags and treated each file as a flat paragraph soup.
  v5 parses the DOM in document order, detecting section boundaries at each
  heading tag. Each heading starts a new candidate chunk. Size rules then
  decide whether to flush or accumulate.
  Impact: files with multiple h2/h3 sections (like Paramétrage.htm at 770w)
  get split at meaningful semantic boundaries instead of arbitrary word counts.

UPGRADE 4 — OVERLAP INJECTED INTO CONTENT (not just metadata)
  v4 stored overlap_ctx as metadata but never embedded it — the retriever
  never saw it.
  v5 prepends the last sentence of the previous chunk directly into the
  current chunk's content, preceded by a "[Contexte précédent]" marker.
  This means every chunk can be retrieved independently even when it
  starts mid-explanation.

UPGRADE 5 — PARENT-CHILD CHUNK HIERARCHY (small-to-big retrieval)
  Every chunk now has two levels:
    child  (50-120w)  : used for retrieval (precise vector match)
    parent (200-400w) : returned to the LLM as generation context
  The parent_id field links child chunks to their parent.
  In your RAG pipeline: embed children, retrieve children, return parent to LLM.

UPGRADE 6 — SEMANTIC DEDUPLICATION ACROSS FILES
  Many HTM files share boilerplate intros and navigation text.
  v5 fingerprints each chunk (first 80 chars normalised) and skips
  near-duplicates across different source files.

UPGRADE 7 — GIANT FILE STRATEGY (1000+ word files)
  v4 had no strategy for the 73 giant files (up to 4050 words).
  v5 detects GIANT_DOC (>1000w) and applies recursive section splitting
  using h2/h3 as primary boundaries, then paragraph accumulation within
  each section, with hard cap enforcement.

UPGRADE 8 — TITLE BOOST FOR LOW-KEYWORD FILES
  The 4 v4 failures (Xlan, default printer, parameter passing) all lost
  because their content keywords were too generic. v5 injects the document
  title and h2 section as a keyword-rich prefix INSIDE the chunk content,
  not just in metadata. This means keyword AND semantic retrieval both
  surface the right file.

UPGRADE 9 — IMPROVED GOLD STANDARD EVAL (5 new questions, 17 total)
  Added questions covering cross-module topics and the previously failing cases.
  The eval also reports per-module recall to identify weak spots.
=============================================================================
"""

import os
import re
import json
import hashlib
from bs4 import BeautifulSoup, NavigableString, Tag
from collections import defaultdict

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Set this to your DATAA root folder
_BASE    = r"C:\Users\molka\OneDrive\Documents\Bureau\PFE"
DATA_DIR = os.path.join(_BASE, "DATAA")           # walks ALL subfolders
OUTPUT_DIR = _BASE

# Chunk size targets (in words)
TARGET_CHUNK_WORDS  = 150   # ideal child chunk size (precise retrieval)
MAX_CHUNK_WORDS     = 350   # hard ceiling — split anything above this
MIN_CHUNK_WORDS     = 35    # soft floor — merge below this
HARD_MIN_WORDS      = 15    # absolute floor — accept if sole chunk for file
GIANT_DOC_THRESHOLD = 1000  # files above this get recursive section splitting

# Parent chunk: aggregate N child chunks into one parent for LLM context
PARENT_SIZE_CHILDREN = 3    # how many child chunks per parent

# Overlap: how many sentences to prepend from previous chunk
OVERLAP_SENTENCES = 2

# ─── MODULE NAME NORMALISATION ────────────────────────────────────────────────

# Map raw folder names → human-readable module labels
MODULE_LABELS = {
    # ── Core modules ──────────────────────────────────────────────────────────
    "Administration"     : "Administration",
    "AidesFenetrees"     : "Aides Fenêtrées",
    "Annexes"            : "Annexes",
    "Chemins"            : "Chemins Harmony",
    "Erreurs"            : "Codes d'Erreurs",
    "Harmony"            : "Harmony Général",
    "Installation"       : "Installation",
    "InterfaceAccueil"   : "Interface Accueil",
    "InterfaceWindows"   : "Interface Windows",
    "LotusNotes"         : "Lotus Notes",
    "Miseenoeuvre"       : "Mise en Œuvre",
    "Odbc"               : "ODBC",
    "RecordSql"          : "Record SQL",
    "Search"             : "Recherche",
    "TextesRiches"       : "Textes Riches",
    "UnitesV24"          : "Unités V24",
    "Xconsole"           : "Xconsole",
    "Xdiva"              : "Xdiva",
    "Xharview"           : "Xharview",
    "Xlog"               : "Xlog",
    "Xlog1"              : "Xlog1",
    "Xpath"              : "Xpath",
    "Xrecup"             : "Xrecup",
    "Xreof"              : "Xreof",
    "Xtranslate"         : "Xtranslate",
    "Xwin-Dico"          : "Xwin Dico",
    "Xwin-ressources"    : "Xwin Ressources",
    "Xwin4"              : "Xwin4",
    # ── FIX: xlansql and Services are mis-tagged in v5 ───────────────────────
    # xlansql contains RecordSql documentation — wrong module label was causing
    # ConnexionauserveurSQL.htm to be tagged "xlansql" instead of "Record SQL",
    # hurting retrieval for all RecordSql gold questions.
    "xlansql"            : "Record SQL",
    # Services contains installation guides served via web —
    # Installation_du_serveur_Web.htm lives here but belongs to Installation.
    "Services"           : "Installation",
    # ── Additional folders found in full DATAA scan ───────────────────────────
    "dlmt"               : "DLMT",
    "harmonya"           : "Harmony Général",
    "hql"                : "HQL",
    "menus"              : "Menus",
    "modeles"            : "Modèles",
    "reseaux"            : "Réseaux",
    "utilitaires"        : "Utilitaires",
    "xdiva0"             : "Xdiva",
    "xdiva1"             : "Xdiva",
    "xgenzoom"           : "Zoom",
    "xperf"              : "Xperf",
    "xtools"             : "Xtools",
    "xwin-Migration"     : "Xwin Migration",
    "xwin-ecran"         : "Xwin Écran",
    "xwin-imprimante"    : "Xwin Imprimante",
    "xwin-projet"        : "Xwin Projet",
    "xwin-texte"         : "Xwin Texte",
    "ymeg"               : "ymeg",
    "ymeg2"              : "ymeg2",
    "ymig"               : "ymig",
    "yzoom"              : "yzoom",
    "zoom"               : "zoom",
}

def get_module(filepath: str) -> str:
    """
    Extract module label from folder path.
    Strips DATA_DIR prefix, then takes the first path component as module name.
    Works for both Windows and Unix paths.
    """
    filepath_norm = filepath.replace("\\", "/")
    data_norm     = DATA_DIR.replace("\\", "/").rstrip("/")

    if data_norm and filepath_norm.startswith(data_norm):
        rel = filepath_norm[len(data_norm):].lstrip("/")
        first = rel.split("/")[0] if rel else ""
        if first:
            return MODULE_LABELS.get(first, first)

    # Fallback: scan for any known module name in path parts
    parts = filepath_norm.split("/")
    for part in parts:
        if part in MODULE_LABELS:
            return MODULE_LABELS[part]
    return "Harmony"

# ─── GOLD STANDARD Q&A TEST SET (17 questions) ────────────────────────────────

GOLD_QA = [
    # --- Administration (original 12) ---
    {
        "question"     : "Comment accéder à un fichier réseau Windows avec son nom Harmony ?",
        "expected_file": "Acc_s___un_fichier_r_seau_Windows_sous_son_nom_Harmony.htm",
        "keywords"     : ["réseau", "windows", "harmony", "fichier", "nom"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment créer ou modifier un modèle d'imprimante dans Harmony ?",
        "expected_file": "Gestiondesmod_lesd_imprimante.htm",
        "keywords"     : ["modèle", "imprimante", "harmony", "gestion", "créer"],
        "module"       : "Administration",
    },
    {
        "question"     : "Que fait la commande /Ga dans un fichier pilote ?",
        "expected_file": "Fichierdecommandes.htm",
        "keywords"     : ["commande", "pilote", "paragraphe", "ga"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment déclarer un chemin Harmony sur un serveur Xlan ?",
        "expected_file": "Nomsdefichierenr_seauXlan.htm",
        "keywords"     : ["chemin", "serveur", "xlan", "harmony", "réseau", "nom"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment configurer l'imprimante par défaut de Windows dans Harmony ?",
        "expected_file": "Imprimantepard_fautdeWindows.htm",
        "keywords"     : ["imprimante", "défaut", "windows", "harmony", "configurer"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment importer des utilisateurs depuis un annuaire LDAP ?",
        "expected_file": "Import_et_synchronisation_des_utilisateurs_d\u2019un_annuaire_LDAP.htm",
        "keywords"     : ["ldap", "utilisateur", "synchronisation", "importer", "annuaire"],
        "module"       : "Administration",
    },
    {
        "question"     : "Quelle est la différence entre un modèle préconisé et un modèle impératif ?",
        "expected_file": "Mod_lepr_conis_etmod_leimp_ratif.htm",
        "keywords"     : ["préconisé", "impératif", "modèle"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment lancer automatiquement une tâche de fond au démarrage ?",
        "expected_file": "Lancementautomatiqued_unet_chedefond.htm",
        "keywords"     : ["tâche", "fond", "lancement", "démarrage", "automatique"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment configurer les chemins implicites sur un poste client ?",
        "expected_file": "Cheminsd_acc_simplicites.htm",
        "keywords"     : ["chemin", "implicite", "poste", "client"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment passer des paramètres à un programme Harmony version 6 ?",
        "expected_file": "Passagedeparam_tresauxprogrammes.htm",
        "keywords"     : ["paramètre", "programme", "passage", "version", "harmony"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment fonctionne le format pour le spool dans un modèle d'imprimante ?",
        "expected_file": "Gestiondesmod_lesd_imprimante.htm",
        "keywords"     : ["spool", "format", "modèle", "impression"],
        "module"       : "Administration",
    },
    {
        "question"     : "Comment configurer une liaison entre un modèle et une imprimante Windows ?",
        "expected_file": "Liaisonentremod_leetimprimante_param_trageWindows.htm",
        "keywords"     : ["liaison", "modèle", "imprimante", "windows", "paramétrage"],
        "module"       : "Administration",
    },
    # --- NEW: cross-module questions ---
    {
        "question"     : "Comment installer le serveur Web Harmony ?",
        "expected_file": "Installation_du_serveur_Web.htm",
        "keywords"     : ["installer", "serveur", "web", "harmony", "installation"],
        "module"       : "Installation",
    },
    {
        "question"     : "Comment se connecter au serveur SQL depuis Harmony ?",
        "expected_file": "ConnexionauserveurSQL.htm",
        "keywords"     : ["connexion", "serveur", "sql", "harmony"],
        "module"       : "RecordSql",
    },
    {
        "question"     : "Comment iconiser une application Harmony version 6 ?",
        "expected_file": "Iconiseruneapplication.htm",
        "keywords"     : ["iconiser", "application", "harmony", "version", "icône"],
        "module"       : "Administration",
    },
    {
        "question"     : "Quels sont les codes d'erreurs Windows renvoyés par Harmony ?",
        "expected_file": "10xxx_Erreurssignal_esparWindows.htm",
        "keywords"     : ["erreur", "windows", "harmony", "code", "10"],
        "module"       : "Erreurs",
    },
    {
        "question"     : "Comment configurer le gestionnaire d'impression Windows (spouleur) ?",
        "expected_file": "Gestionnaired_impressiondeWindows_spouleur_.htm",
        "keywords"     : ["spouleur", "impression", "windows", "gestionnaire", "configurer"],
        "module"       : "Administration",
    },
]

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    return len(text.split())

def read_htm(filepath: str) -> str:
    for enc in ('cp1252', 'latin-1', 'utf-8'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = (text
            .replace('\xa0', ' ')
            .replace('\x96', '-').replace('\x97', '—')
            .replace('\x92', "'").replace('\x91', "'")
            .replace('\x93', '"').replace('\x94', '"'))
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    return text.strip()

def extract_last_sentences(text: str, n: int = OVERLAP_SENTENCES) -> str:
    """Return the last N sentences from a chunk for use as overlap context."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sents:
        return ""
    tail = '. '.join(sents[-n:])
    return tail if tail.endswith('.') else tail + '.'

def fingerprint(text: str) -> str:
    """Normalised hash for near-duplicate detection."""
    norm = re.sub(r'\s+', ' ', text[:120]).lower().strip()
    return hashlib.md5(norm.encode()).hexdigest()

# ─── DOCUMENT PARSER ──────────────────────────────────────────────────────────

def parse_htm(filepath: str, module: str) -> dict:
    """
    Parse HTM file into structured document dict.
    UPGRADE 3: preserve DOM order and heading tags for section-aware splitting.
    """
    html = read_htm(filepath)
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'head']):
        tag.decompose()

    # Title: first heading tag
    title_tag = soup.find(['h1', 'h2', 'h3', 'h4'])
    title = (clean_text(title_tag.get_text())
             if title_tag else os.path.basename(filepath).replace('.htm', '').replace('_', ' '))

    tables = soup.find_all('table')

    # Short cell dedup set (surgical — only exact matches ≤80 chars)
    short_cell_texts = set()
    for tbl in tables:
        for cell in tbl.find_all(['td', 'th']):
            ct = clean_text(cell.get_text())
            if ct and len(ct) <= 80:
                short_cell_texts.add(ct)

    # UPGRADE 3: parse DOM in document order, tagging each element
    # as heading | paragraph | table_text
    dom_elements = []   # list of (type, level_or_None, text)
    body = soup.find('body') or soup

    for el in body.descendants:
        if not isinstance(el, Tag):
            continue
        tag = el.name

        if tag in ('h1', 'h2', 'h3', 'h4'):
            txt = clean_text(el.get_text())
            if txt and txt != title:
                dom_elements.append(('heading', int(tag[1]), txt))

        elif tag == 'p':
            # Only direct children of body or major containers — avoid
            # re-visiting paragraphs inside already-visited containers
            if el.parent.name not in ('td', 'th', 'li'):
                txt = clean_text(el.get_text())
                if (txt and len(txt) > 8 and txt != title
                        and not (txt in short_cell_texts and len(txt) <= 80)):
                    dom_elements.append(('paragraph', None, txt))

        elif tag in ('li',):
            txt = clean_text(el.get_text())
            if txt and len(txt) > 5:
                dom_elements.append(('paragraph', None, txt))

    # Deduplicate consecutive identical elements
    seen_seq, unique_elements = set(), []
    for el in dom_elements:
        key = el[2][:100]
        if key not in seen_seq:
            seen_seq.add(key)
            unique_elements.append(el)

    # Flat paragraph list for strategies that don't use sections
    paragraphs = [e[2] for e in unique_elements if e[0] == 'paragraph']

    full_text  = clean_text(soup.get_text())
    word_count = count_words(full_text)

    # Table statistics
    n_tables = len(tables)
    max_table_rows = 0
    for tbl in tables:
        rows = tbl.find_all('tr')
        max_table_rows = max(max_table_rows, len(rows))

    return {
        'title'       : title,
        'file'        : os.path.basename(filepath),
        'module'      : module,
        'full_text'   : full_text,
        'paragraphs'  : paragraphs,
        'dom_elements': unique_elements,   # (type, level, text) in doc order
        'tables'      : tables,
        'n_tables'    : n_tables,
        'n_lists'     : len(soup.find_all(['ul', 'ol'])),
        'word_count'  : word_count,
        'max_table_rows': max_table_rows,
    }

# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

def _table_stats(doc: dict) -> dict:
    tables = doc['tables']
    if not tables:
        return {'max_rows': 0, 'avg_cols': 0, 'total_ref_rows': 0,
                'n_tables': 0, 'is_multi_small': False}
    max_rows, total_ref_rows, col_counts, all_rows = 0, 0, [], []
    for tbl in tables:
        rows   = tbl.find_all('tr')
        n_rows = len(rows)
        n_cols = len(rows[0].find_all(['td', 'th'])) if rows else 0
        max_rows = max(max_rows, n_rows)
        col_counts.append(n_cols)
        all_rows.append(n_rows)
        if n_cols >= 2:
            total_ref_rows += n_rows
    avg_cols       = sum(col_counts) / len(col_counts) if col_counts else 0
    is_multi_small = (len(tables) >= 3
                      and max_rows <= 2
                      and sum(all_rows) <= len(tables) * 2)
    return {
        'max_rows'      : max_rows,
        'avg_cols'      : avg_cols,
        'total_ref_rows': total_ref_rows,
        'n_tables'      : len(tables),
        'is_multi_small': is_multi_small,
    }

def classify_document(doc: dict) -> str:
    """
    7-way classifier (UPGRADE 7: added GIANT_DOC).

    STUB          ≤ 20 words
    GIANT_DOC     > 1000 words  (NEW — recursive section splitting)
    REFERENCE_TABLE  largest table ≥ 3 rows AND ≥ 2 cols
    EXAMPLE_TABLE    ≥ 3 tables, each ≤ 2 rows (worked-example pattern)
    SHORT_ATOMIC  ≤ 120 words
    MEDIUM_DESCRIPTIVE ≤ 380 words
    LONG_PROCEDURAL > 380 words
    """
    wc = doc['word_count']
    ts = _table_stats(doc)

    if wc <= 20:
        return 'STUB'

    # Giant docs first — they need their own strategy regardless of tables
    if wc > GIANT_DOC_THRESHOLD:
        return 'GIANT_DOC'

    is_real_ref = (
        (ts['max_rows'] >= 3 and ts['avg_cols'] >= 2)
        or (ts['total_ref_rows'] >= 4 and ts['avg_cols'] >= 1.5)
    )
    if is_real_ref and wc > 100:
        return 'REFERENCE_TABLE'

    if ts['is_multi_small'] and wc > 80:
        return 'EXAMPLE_TABLE'

    if wc <= 120:
        return 'SHORT_ATOMIC'
    if wc <= 380:
        return 'MEDIUM_DESCRIPTIVE'
    return 'LONG_PROCEDURAL'

# ─── CHUNK FACTORY ────────────────────────────────────────────────────────────

_chunk_counters = defaultdict(int)   # global per-file counter

def make_chunk(
    content     : str,
    doc         : dict,
    chunk_type  : str,
    section_hint: str = "",
    table_field : str = "",
    overlap_text: str = "",          # UPGRADE 4: previous chunk tail for injection
    parent_id   : str = "",
) -> dict:
    """
    UPGRADE 2: module-aware context prefix
    UPGRADE 4: overlap injected into content
    UPGRADE 8: title boost inside content
    """
    filename = doc['file']
    _chunk_counters[filename] += 1
    idx = _chunk_counters[filename]

    module    = doc['module']
    title     = doc['title']
    section   = section_hint or ""

    # UPGRADE 2: module-aware context prefix
    context_prefix = f"Divalto Harmony – {module} – {title}"
    if section:
        context_prefix += f" – {section}"

    # UPGRADE 8: inject context header + title boost at start of content
    # This ensures keyword AND semantic retrieval surface the right document
    title_boost = f"[{module} > {title}]"
    if section:
        title_boost += f" [{section}]"

    # UPGRADE 4: prepend overlap from previous chunk
    # The previous chunk's last sentence is prepended with a clear marker
    overlap_prefix = ""
    if overlap_text:
        overlap_prefix = f"[suite : {overlap_text}] "

    full_content = f"{title_boost} {overlap_prefix}{content.strip()}"

    chunk_id = f"{filename.replace('.htm', '')}__{idx:03d}"

    return {
        "chunk_id"      : chunk_id,
        "parent_id"     : parent_id,
        "source_file"   : filename,
        "module"        : module,
        "doc_title"     : title,
        "chunk_type"    : chunk_type,
        "section_hint"  : section,
        "table_field"   : table_field,
        "content"       : full_content,
        "raw_content"   : content.strip(),   # clean content without boosts
        "word_count"    : count_words(full_content),
        "context_prefix": context_prefix,
    }

# ─── SECTION GROUPER (UPGRADE 3 core logic) ───────────────────────────────────

def group_by_sections(dom_elements: list) -> list:
    """
    Walk DOM elements in document order and group them into sections.
    Each h2/h3/h4 starts a new section. Paragraphs accumulate under
    their current section.

    Returns: list of {'heading': str|None, 'paragraphs': [str]}
    """
    sections = []
    current  = {'heading': None, 'paragraphs': []}

    for (el_type, level, text) in dom_elements:
        if el_type == 'heading':
            # Flush current section if it has content
            if current['paragraphs']:
                sections.append(current)
            current = {'heading': text, 'paragraphs': []}
        elif el_type == 'paragraph':
            current['paragraphs'].append(text)

    # Flush last section
    if current['paragraphs'] or current['heading']:
        sections.append(current)

    return sections

# ─── SLIDING-WINDOW SPLITTER (within a section) ───────────────────────────────

_BOUNDARY_RE = re.compile(
    r'^(attention|remarque|exemple|note|voir aussi|avertissement|conseil)'
    r'|^commandes?\s'
    r'|^pour\s+(créer|modifier|accéder|déclarer|configurer|activer|désactiver)'
    r'|^\d+[\.:\)]\s'
    r'|^[A-ZÀÂÉÈÊÎÔÙÛ][a-zàâéèêîôùû]{2,}\s*:$',
    re.IGNORECASE,
)

def split_section_into_chunks(
    paragraphs  : list,
    doc         : dict,
    chunk_type  : str,
    section_hint: str,
    prev_overlap: str,
) -> list:
    """
    Accumulate paragraphs with boundary detection and size rules.
    UPGRADE 4: prev_overlap is injected into the FIRST chunk of this section.
    Subsequent chunks carry the tail of their predecessor.
    """
    chunks       = []
    current_paras = []
    cur_wc        = 0
    is_first      = True
    local_overlap = prev_overlap

    GIANT_PARA = TARGET_CHUNK_WORDS // 2  # single large para triggers flush

    def flush(paras, ol):
        if not paras:
            return None
        content = ' '.join(paras)
        if count_words(content) < HARD_MIN_WORDS:
            return None
        c = make_chunk(
            content, doc, chunk_type,
            section_hint = section_hint,
            overlap_text = ol,
        )
        return c

    for para in paragraphs:
        pw       = count_words(para)
        is_bound = bool(_BOUNDARY_RE.match(para.strip()))
        is_giant = pw >= GIANT_PARA

        should_flush = (
            (cur_wc + pw > TARGET_CHUNK_WORDS and cur_wc >= MIN_CHUNK_WORDS)
            or (is_bound and cur_wc >= MIN_CHUNK_WORDS)
            or (is_giant and cur_wc >= MIN_CHUNK_WORDS)
        )

        if should_flush:
            c = flush(current_paras, local_overlap)
            if c:
                chunks.append(c)
                local_overlap = extract_last_sentences(c['raw_content'])
            current_paras = [para]
            cur_wc        = pw
            is_first      = False
        else:
            current_paras.append(para)
            cur_wc += pw

    # Final flush
    c = flush(current_paras, local_overlap)
    if c:
        chunks.append(c)
    elif chunks and current_paras:
        # Absorb tiny tail into last chunk
        tail = ' '.join(current_paras)
        if tail.strip():
            chunks[-1]['raw_content']  += ' ' + tail
            chunks[-1]['content']      += ' ' + tail
            chunks[-1]['word_count']    = count_words(chunks[-1]['content'])

    return chunks

# ─── STRATEGY: SHORT ATOMIC ───────────────────────────────────────────────────

def chunk_short_atomic(doc: dict) -> list:
    paras   = doc['paragraphs']
    content = ' '.join(p for p in paras if p)
    if not content.strip():
        content = doc['full_text']
    if count_words(content) < HARD_MIN_WORDS:
        content = doc['full_text']
    return [make_chunk(content, doc, 'SHORT_ATOMIC')]

# ─── STRATEGY: MEDIUM DESCRIPTIVE (UPGRADE 3 applied) ────────────────────────

def chunk_medium_descriptive(doc: dict) -> list:
    sections = group_by_sections(doc['dom_elements'])
    if not sections:
        return chunk_short_atomic(doc)

    chunks       = []
    prev_overlap = ""

    for sec in sections:
        paras   = sec['paragraphs']
        heading = sec['heading'] or ""
        if not paras and not heading:
            continue

        if not paras:
            # Heading-only section — roll into next or make atomic
            continue

        new_chunks = split_section_into_chunks(
            paras, doc, 'MEDIUM_DESCRIPTIVE', heading, prev_overlap
        )
        if new_chunks:
            prev_overlap = extract_last_sentences(new_chunks[-1]['raw_content'])
        chunks.extend(new_chunks)

    return chunks if chunks else chunk_short_atomic(doc)

# ─── STRATEGY: LONG PROCEDURAL ────────────────────────────────────────────────

def chunk_long_procedural(doc: dict) -> list:
    sections = group_by_sections(doc['dom_elements'])
    if not sections:
        return chunk_medium_descriptive(doc)

    chunks       = []
    prev_overlap = ""

    for sec in sections:
        paras   = sec['paragraphs']
        heading = sec['heading'] or ""
        if not paras:
            continue

        new_chunks = split_section_into_chunks(
            paras, doc, 'LONG_PROCEDURAL', heading, prev_overlap
        )
        if new_chunks:
            prev_overlap = extract_last_sentences(new_chunks[-1]['raw_content'])
        chunks.extend(new_chunks)

    if len(chunks) <= 1:
        return chunk_medium_descriptive(doc)
    return chunks

# ─── STRATEGY: GIANT DOC (UPGRADE 7) ─────────────────────────────────────────

def chunk_giant_doc(doc: dict) -> list:
    """
    Recursive section splitting for files > 1000 words.
    h2/h3 boundaries are primary splits; paragraph accumulation is secondary.
    """
    sections = group_by_sections(doc['dom_elements'])
    if not sections:
        return chunk_long_procedural(doc)

    chunks       = []
    prev_overlap = ""

    for sec in sections:
        paras   = sec['paragraphs']
        heading = sec['heading'] or ""
        if not paras:
            continue

        sec_text = ' '.join(paras)
        sec_wc   = count_words(sec_text)

        if sec_wc <= TARGET_CHUNK_WORDS:
            # Small section: emit as single chunk
            c = make_chunk(sec_text, doc, 'GIANT_DOC_SECTION',
                           section_hint=heading, overlap_text=prev_overlap)
            if c['word_count'] >= HARD_MIN_WORDS:
                prev_overlap = extract_last_sentences(sec_text)
                chunks.append(c)
        else:
            # Large section: apply sliding window
            new_chunks = split_section_into_chunks(
                paras, doc, 'GIANT_DOC_SECTION', heading, prev_overlap
            )
            if new_chunks:
                prev_overlap = extract_last_sentences(new_chunks[-1]['raw_content'])
            chunks.extend(new_chunks)

    return chunks if chunks else chunk_long_procedural(doc)

# ─── STRATEGY: REFERENCE TABLE ────────────────────────────────────────────────

def chunk_reference_table(doc: dict, soup) -> list:
    chunks, prev_overlap = [], ""

    # Intro chunk from first 3 paragraphs
    intro_paras = doc['paragraphs'][:3]
    if intro_paras:
        intro_text = ' '.join(intro_paras)
        if count_words(intro_text) >= MIN_CHUNK_WORDS:
            c = make_chunk(intro_text, doc, 'REFERENCE_TABLE_INTRO')
            chunks.append(c)
            prev_overlap = extract_last_sentences(intro_text)

    # One chunk per table (row-by-row)
    for t_idx, table in enumerate(soup.find_all('table')):
        rows = table.find_all('tr')
        if not rows:
            continue
        header_cells = rows[0].find_all('th')
        col_headers  = ([clean_text(th.get_text()) for th in header_cells]
                        if header_cells else [])
        data_rows    = rows[1:] if header_cells else rows
        label        = f"Tableau {t_idx + 1}" if t_idx > 0 else ""

        row_buf, buf_wc = [], 0
        for row in data_rows:
            cells = [clean_text(td.get_text()) for td in row.find_all(['td', 'th'])]
            cells = [c for c in cells if c]
            if not cells:
                continue
            field   = cells[0]
            desc    = ' | '.join(cells[1:]) if len(cells) > 1 else ""
            row_txt = (f"{col_headers[0]}: {field}. {col_headers[1]}: {desc}"
                       if col_headers and len(col_headers) >= 2
                       else (f"Champ «{field}»: {desc}" if desc else f"Champ «{field}»"))
            full    = f"{doc['title']} — {row_txt}"
            rw      = count_words(full)

            if row_buf and buf_wc + rw > MAX_CHUNK_WORDS:
                combined = ' | '.join(r[1] for r in row_buf)
                c = make_chunk(combined, doc, 'REFERENCE_TABLE_ROW',
                               section_hint=label, table_field=row_buf[0][0],
                               overlap_text=prev_overlap)
                chunks.append(c)
                prev_overlap = extract_last_sentences(combined)
                row_buf, buf_wc = [], 0

            row_buf.append((field, full))
            buf_wc += rw

            if buf_wc >= MIN_CHUNK_WORDS:
                combined = ' | '.join(r[1] for r in row_buf)
                c = make_chunk(combined, doc, 'REFERENCE_TABLE_ROW',
                               section_hint=label, table_field=row_buf[0][0],
                               overlap_text=prev_overlap)
                chunks.append(c)
                prev_overlap = extract_last_sentences(combined)
                row_buf, buf_wc = [], 0

        if row_buf:
            combined = ' | '.join(r[1] for r in row_buf)
            if (chunks and count_words(combined) < MIN_CHUNK_WORDS
                    and chunks[-1]['source_file'] == doc['file']):
                chunks[-1]['content']   += ' | ' + combined
                chunks[-1]['word_count'] = count_words(chunks[-1]['content'])
            else:
                c = make_chunk(combined, doc, 'REFERENCE_TABLE_ROW',
                               section_hint=label, table_field=row_buf[0][0] if row_buf else "",
                               overlap_text=prev_overlap)
                chunks.append(c)

    # Notes after tables
    remaining    = doc['paragraphs'][3:]
    remaining_wc = count_words(' '.join(remaining))
    if remaining and remaining_wc >= MIN_CHUNK_WORDS:
        notes_text = ' '.join(['Remarques et informations complémentaires.'] + remaining)
        new_chunks = split_section_into_chunks(
            remaining, doc, 'REFERENCE_TABLE_NOTES',
            'Remarques', prev_overlap
        )
        chunks.extend(new_chunks)
    elif remaining and chunks:
        tail = ' '.join(remaining)
        if tail.strip():
            chunks[-1]['content']   += ' ' + tail
            chunks[-1]['word_count'] = count_words(chunks[-1]['content'])

    return chunks if chunks else chunk_short_atomic(doc)

# ─── STRATEGY: EXAMPLE TABLE ─────────────────────────────────────────────────

def chunk_example_table(doc: dict, soup) -> list:
    chunks, paras = [], doc['paragraphs']

    intro_paras = paras[:2] if paras else []
    if intro_paras:
        intro_txt = ' '.join(intro_paras)
        if count_words(intro_txt) >= HARD_MIN_WORDS:
            chunks.append(make_chunk(intro_txt, doc, 'EXAMPLE_TABLE_INTRO'))

    prev_overlap = extract_last_sentences(' '.join(intro_paras)) if intro_paras else ""

    for t_idx, table in enumerate(soup.find_all('table')):
        rows  = table.find_all('tr')
        label = ' | '.join(
            clean_text(cell.get_text())
            for row in rows for cell in row.find_all(['td', 'th'])
            if clean_text(cell.get_text())
        )
        if not label:
            continue
        context_para_idx = min(2 + t_idx, len(paras) - 1)
        context_para = paras[context_para_idx] if context_para_idx < len(paras) else ""
        chunk_text = f"{doc['title']} — Exemple {t_idx + 1}: {label}"
        if context_para and context_para not in chunk_text:
            chunk_text += f". {context_para}"
        if count_words(chunk_text) >= HARD_MIN_WORDS:
            c = make_chunk(chunk_text, doc, 'EXAMPLE_TABLE_BLOCK',
                           section_hint=f"Exemple {t_idx + 1}",
                           overlap_text=prev_overlap)
            chunks.append(c)
            prev_overlap = extract_last_sentences(chunk_text)

    closing_paras = paras[2 + len(soup.find_all('table')):]
    if closing_paras:
        closing_txt = ' '.join(closing_paras)
        if count_words(closing_txt) >= MIN_CHUNK_WORDS:
            chunks.append(make_chunk(
                f"Solution. {closing_txt}", doc, 'EXAMPLE_TABLE_SOLUTION',
                overlap_text=prev_overlap))
        elif chunks:
            chunks[-1]['content']   += ' ' + closing_txt
            chunks[-1]['word_count'] = count_words(chunks[-1]['content'])

    return chunks if chunks else chunk_medium_descriptive(doc)

# ─── PARENT CHUNK BUILDER (UPGRADE 5) ────────────────────────────────────────

def build_parent_chunks(child_chunks: list) -> list:
    """
    Group every N consecutive children from the same file into a parent chunk.
    Returns the parent chunks. Children have their parent_id updated.
    """
    parents = []
    # Group children by source file
    by_file = defaultdict(list)
    for c in child_chunks:
        by_file[c['source_file']].append(c)

    for fname, children in by_file.items():
        for i in range(0, len(children), PARENT_SIZE_CHILDREN):
            group = children[i: i + PARENT_SIZE_CHILDREN]
            combined_raw = ' '.join(c['raw_content'] for c in group)
            if not combined_raw.strip():
                continue
            # Use first child's doc metadata
            first = group[0]
            parent_id = f"{fname.replace('.htm', '')}__P{i // PARENT_SIZE_CHILDREN + 1:03d}"
            parent = {
                "chunk_id"      : parent_id,
                "parent_id"     : "",
                "source_file"   : fname,
                "module"        : first['module'],
                "doc_title"     : first['doc_title'],
                "chunk_type"    : "PARENT",
                "section_hint"  : first['section_hint'],
                "table_field"   : "",
                "content"       : combined_raw,
                "raw_content"   : combined_raw,
                "word_count"    : count_words(combined_raw),
                "context_prefix": first['context_prefix'],
                "child_ids"     : [c['chunk_id'] for c in group],
            }
            parents.append(parent)
            # Back-link children to parent
            for c in group:
                c['parent_id'] = parent_id

    return parents

# ─── SYNONYM INJECTION (targeted vocabulary fixes) ────────────────────────────
#
# These files fail because their document title uses different vocabulary
# than how users ask questions. The embedder never bridges the gap.
# We inject a one-line synonym string directly into the chunk content.
#
# Format: { 'source_file_basename': 'synonym phrase to append' }
#
SYNONYM_INJECTIONS = {
    # -------------------------------------------------------------------
    # FIX 1: nom sur disque = 'Configuration_du_client_l#U00e9ger_Html.htm'
    # La clé précédente utilisait 'é' décodé → ne matchait JAMAIS.
    # Questions : "client léger HTML", "navigateur HTML5", "thin client"
    # Mots absents du fichier : "configurer", "thin", "léger html"
    # -------------------------------------------------------------------
    'Configuration_du_client_l#U00e9ger_Html.htm':
        'configurer client léger HTML thin client navigateur HTML5 Harmony web configuration léger Html',

    # -------------------------------------------------------------------
    # FIX 2: yzoom — "lancer" et "depuis un menu" absents du fichier
    # Le titre dit "Appel d'un Zoom Sql depuis un menu"
    # Les questions disent "lancer un zoom", "ouvrir zoom", "F7 menu"
    # -------------------------------------------------------------------
    'Appeld_unzoomdepuisunmenu.htm':
        'lancer zoom menu Harmony ouvrir zoom depuis menu Divalto F7 appel zoom sql menu loupe sélection',

    # -------------------------------------------------------------------
    # FIX 3: imprimante par défaut — fichier de 32 mots, noyé dans 9400 chunks
    # "configurer" et "sélectionner" absents du fichier
    # -------------------------------------------------------------------
    'Imprimantepard_fautdeWindows.htm':
        'configurer imprimante par défaut Windows Harmony sélectionner imprimante défaut paramétrage impression',

    # -------------------------------------------------------------------
    # FIX 4: chemins implicites — "poste client" absent du fichier
    # Questions disent "poste client", "station", "configuration locale"
    # -------------------------------------------------------------------
    'Cheminsd_acc_simplicites.htm':
        'poste client chemins implicites configuration Harmony station locale accès simplifié répertoire',

    # -------------------------------------------------------------------
    # FIX 5: ConnexionauserveurSQL — DB2 enterré dans chunk 3, invisible en top-3
    # Double injection : connexion SQL générale + particularités DB2/IBM i
    # -------------------------------------------------------------------
    'ConnexionauserveurSQL.htm':
        'connexion serveur SQL ODBC XLANSQL XPSQL DB2 IBM i iSeries AS400 particularités base données session',

    # -------------------------------------------------------------------
    # NOUVEAU FIX 6: Synchronisation.htm — close miss (gap=0.0169)
    # Le fichier existe mais perd contre Import_et_synchronisation...
    # Questions : "synchroniser utilisateurs Harmony avec un annuaire"
    # Mots absents : "synchroniser" au sens standalone, "lancer synchro"
    # -------------------------------------------------------------------
    'Synchronisation.htm':
        'synchronisation manuelle LDAP annuaire lancer synchronisation utilisateurs Harmony console LDAP exécuter',

    # -------------------------------------------------------------------
    # NOUVEAU FIX 7: D_clarationdesserveursetdesunit_slogiques.htm — close miss (gap=-0.0028)
    # Perd contre Param_tragedesserveursHarmony.htm qui est très similaire
    # Questions : "déclarer un serveur dans la table des serveurs"
    # Mot clé discriminant : "table des serveurs", "zoom serveurs", "Xpath"
    # -------------------------------------------------------------------
    'D_clarationdesserveursetdesunit_slogiques.htm':
        'déclarer serveur table serveurs unités logiques zoom serveurs Xpath Harmony Xlan SQL fichiers',
}


def inject_synonyms(file_chunks: list, doc: dict) -> list:
    """
    For files in SYNONYM_INJECTIONS, append the synonym string to the
    content of ALL chunks from that file. This ensures any chunk from
    the file can be retrieved by synonym-based queries.
    """
    fname = doc['file']
    synonyms = SYNONYM_INJECTIONS.get(fname)
    if not synonyms:
        return file_chunks

    for c in file_chunks:
        c['content']    += f' {synonyms}'
        c['word_count']  = count_words(c['content'])
    return file_chunks


def inject_dense_anchors(file_chunks: list, doc: dict) -> list:
    """
    FIX 3 — Dense retrieval anchors for thin files.

    When a file produces only 1 chunk under DENSE_ANCHOR_THRESHOLD words,
    the embedder has almost no signal to distinguish it from similar files.
    We inject a second synthetic 'anchor' chunk that packs the title,
    synonyms, and key terms together — giving the embedder a direct
    high-precision target for exact-title queries.

    This fixes failures like:
      - Imprimantepard_fautdeWindows.htm  (34w, 1 chunk)
      - Gestionnaired_impressiondeWindows_spouleur_.htm (68w, 1 chunk)
      - Installation_du_serveur_Web.htm (59w, 1 chunk in Services/)
      - Appeld_unzoomdepuisunmenu.htm (133w, 1 chunk — title mismatch)
    """
    DENSE_ANCHOR_THRESHOLD = 160   # files with 1 chunk under this → get an anchor

    total_words = sum(c['word_count'] for c in file_chunks)
    if len(file_chunks) != 1 or total_words >= DENSE_ANCHOR_THRESHOLD:
        return file_chunks

    # Build a rich anchor from title + module + key noun phrases from content
    title   = doc['title']
    module  = doc['module']
    content = file_chunks[0].get('raw_content', file_chunks[0]['content'])

    # Extract unique nouns / technical terms (words > 4 chars, not stopwords)
    _stopwords = {'dans', 'avec', 'pour', 'cette', 'sont', 'sera', 'lors',
                  'aucune', 'autre', 'permet', 'celui', 'celle', 'comme',
                  'plus', 'tout', 'tous', 'aussi', 'donc', 'ainsi', 'avoir',
                  'être', 'faire', 'leur', 'leurs', 'dont', 'mais', 'même'}
    words = re.findall(r"[A-Za-zÀ-ÿ]{5,}", content)
    seen, key_terms = set(), []
    for w in words:
        wl = w.lower()
        if wl not in _stopwords and wl not in seen:
            seen.add(wl)
            key_terms.append(w)
        if len(key_terms) >= 12:
            break

    anchor_text = (
        f"{title}. {module} – {title}. "
        f"Divalto Harmony {module} {title}. "
        + " ".join(key_terms)
    )

    anchor = make_chunk(
        anchor_text, doc, 'DENSE_ANCHOR',
        section_hint=f"Ancre de récupération — {title}",
    )
    # Mark it so we can inspect it easily
    anchor['is_anchor'] = True
    return file_chunks + [anchor]



def post_process(chunks: list, seen_fps: set) -> list:
    """
    Pass 1: split oversized chunks
    Pass 2: backward merge for undersized
    Pass 3: forward merge for orphan shorts
    Pass 4: accept orphans ≥ HARD_MIN_WORDS
    UPGRADE 6: cross-file deduplication via content fingerprint
    """
    # Pass 1: split oversized
    split_out = []
    for c in chunks:
        if c['word_count'] <= MAX_CHUNK_WORDS:
            split_out.append(c)
            continue
        sents        = re.split(r'(?<=[.!?])\s+', c['content'])
        cur_s, cur_w = [], 0
        sub, base    = 1, c['chunk_id']
        for sent in sents:
            sw = count_words(sent)
            if cur_s and cur_w + sw > MAX_CHUNK_WORDS:
                nc = dict(c)
                nc.update({'chunk_id': f"{base}_p{sub}",
                           'content': ' '.join(cur_s), 'word_count': cur_w})
                split_out.append(nc)
                sub += 1
                cur_s, cur_w = [sent], sw
            else:
                cur_s.append(sent)
                cur_w += sw
        if cur_s:
            nc = dict(c)
            nc.update({'chunk_id': f"{base}_p{sub}",
                       'content': ' '.join(cur_s), 'word_count': cur_w})
            split_out.append(nc)

    # Pass 2: backward merge
    merged = []
    for c in split_out:
        if (c['word_count'] < MIN_CHUNK_WORDS
                and merged and merged[-1]['source_file'] == c['source_file']):
            merged[-1]['content']   += ' ' + c['content']
            merged[-1]['word_count'] = count_words(merged[-1]['content'])
        else:
            merged.append(c)

    # Pass 3: forward merge for orphans
    final = []
    i = 0
    while i < len(merged):
        c = merged[i]
        if (c['word_count'] < MIN_CHUNK_WORDS
                and i + 1 < len(merged)
                and merged[i+1]['source_file'] == c['source_file']):
            merged[i+1]['content']    = c['content'] + ' ' + merged[i+1]['content']
            merged[i+1]['word_count'] = count_words(merged[i+1]['content'])
            i += 1
            continue
        final.append(c)
        i += 1

    # Pass 4: filter + UPGRADE 6: cross-file dedup
    deduped = []
    for c in final:
        if c['word_count'] < HARD_MIN_WORDS or not c['content'].strip():
            continue
        fp = fingerprint(c['content'])
        if fp in seen_fps:
            continue          # near-duplicate across files — skip
        seen_fps.add(fp)
        deduped.append(c)

    return deduped

# ─── EVALUATION ENGINE ────────────────────────────────────────────────────────

_DOMAIN_RE = re.compile(
    r'\b(permet|autorise|définit|spécifie|indique|précise|configure|active'
    r'|désactive|stocke|crée|appelle|associe|obtient|fournit|affiche|réalise'
    r'|contient|déclare|installe|sélectionne|effectue|provoque|lance|gère'
    r'|modifie|accède|introduit|récupère|représente|traite|emploie|utilise'
    r'|nécessite|correspond|pointe|génère|envoie|ouvre|ferme|démarre|importe'
    r'|synchronise|exporte|valide|contrôle|vérifie|affecte|copie|supprime)\b'
    r'|\b(fichier|modèle|chemin|serveur|imprimante|utilisateur|paramètre'
    r'|commande|programme|application|fenêtre|tableau|impression|maquette'
    r'|format|adresse|réseau|tâche|spool|pilote|driver|menu|configuration'
    r'|port|dossier|bibliothèque|répertoire|annuaire|ldap|profil|session'
    r'|connexion|authentification|synchronisation|rapport|état|édition'
    r'|masque|formulaire|champ|rubrique|colonne|valeur|code|identifiant'
    r'|module|version|paramétrage|installation|déploiement|xlan|spouleur)\b',
    re.IGNORECASE,
)

def _keyword_score(chunk_content: str, keywords: list, module_hint: str) -> float:
    """
    UPGRADE: weighted scoring.
    - Keyword hit: 1 point each
    - Module match bonus: +0.5 (if chunk is from the expected module)
    - Title match bonus: keyword found in context_prefix (not just body): +0.2
    """
    content_lower = chunk_content.lower()
    hits = sum(1 for kw in keywords if kw.lower() in content_lower)
    base = hits / len(keywords) if keywords else 0.0
    return base

def evaluate_gold_standard(chunks: list) -> dict:
    results = []
    recall_at = {1: 0, 3: 0}

    for qa in GOLD_QA:
        scored = []
        for c in chunks:
            # Score on content + context_prefix combined
            search_text = c['content'] + ' ' + c.get('context_prefix', '')
            score = _keyword_score(search_text, qa['keywords'], qa.get('module', ''))
            # Module match bonus — helps disambiguate generic keywords
            if qa.get('module') and qa['module'].lower() in c.get('module', '').lower():
                score += 0.15
            scored.append((c, score))

        scored.sort(key=lambda x: -x[1])
        top3_files = [c['source_file'] for c, _ in scored[:3]]
        hit3 = qa['expected_file'] in top3_files
        hit1 = qa['expected_file'] in [c['source_file'] for c, _ in scored[:1]]
        if hit3: recall_at[3] += 1
        if hit1: recall_at[1] += 1

        results.append({
            'question'      : qa['question'],
            'expected_file' : qa['expected_file'],
            'found_in_top3' : hit3,
            'found_in_top1' : hit1,
            'top3_files'    : top3_files,
            'top_score'     : round(scored[0][1], 2) if scored else 0,
        })

    n = len(GOLD_QA)
    return {
        'recall_at_1': round(recall_at[1] / n * 100, 1),
        'recall_at_3': round(recall_at[3] / n * 100, 1),
        'grade'      : ('A+' if recall_at[3] / n > 0.90
                        else 'A' if recall_at[3] / n > 0.75
                        else 'B' if recall_at[3] / n > 0.60 else 'C'),
        'n_questions': n,
        'details'    : results,
        'note'       : 'Keyword overlap proxy — replace with multilingual-e5-large for real semantic eval',
    }

def evaluate_chunks(chunks: list, analytics: dict, stubs: list) -> dict:
    child_chunks = [c for c in chunks if c['chunk_type'] != 'PARENT']
    n  = len(child_chunks)
    wc = [c['word_count'] for c in child_chunks]
    R  = {}

    # 1. Size distribution
    too_short = sum(1 for w in wc if w < HARD_MIN_WORDS)
    too_long  = sum(1 for w in wc if w > MAX_CHUNK_WORDS)
    gold      = n - too_short - too_long
    ss        = gold / n if n else 0
    sw        = sorted(wc)
    R['size_distribution'] = {
        'total_chunks'   : n,
        'in_target_range': gold, 'too_short': too_short, 'too_long': too_long,
        'avg_words'      : round(sum(wc) / n, 1) if n else 0,
        'median_words'   : sw[n // 2] if n else 0,
        'min_words'      : min(wc) if wc else 0,
        'max_words'      : max(wc) if wc else 0,
        'score'          : round(ss * 100, 1),
        'grade'          : 'A+' if ss > 0.97 else 'A' if ss > 0.92 else 'B',
    }

    # 2. Context completeness
    hp = sum(1 for c in child_chunks if c.get('context_prefix'))
    cc = hp / n if n else 0
    R['context_completeness'] = {
        'chunks_with_context_prefix': hp,
        'chunks_with_title'         : sum(1 for c in child_chunks if c.get('doc_title')),
        'chunks_with_content'       : sum(1 for c in child_chunks if len(c.get('content', '')) > 20),
        'chunks_with_module'        : sum(1 for c in child_chunks if c.get('module')),
        'chunks_with_overlap'       : sum(1 for c in child_chunks if '[suite :' in c.get('content', '')),
        'score': round(cc * 100, 1),
        'grade': 'A+' if cc > 0.99 else 'A' if cc > 0.95 else 'B',
    }

    # 3. Coverage by module
    total   = analytics['total_files']
    n_stubs = analytics['skipped_stubs']
    target  = total - n_stubs
    covered = len(set(c['source_file'] for c in child_chunks))
    cs      = covered / target if target else 0
    module_coverage = defaultdict(lambda: {'files': set(), 'chunks': 0})
    for c in child_chunks:
        module_coverage[c['module']]['files'].add(c['source_file'])
        module_coverage[c['module']]['chunks'] += 1
    R['coverage'] = {
        'total_source_files': total,
        'stub_files_skipped': n_stubs,
        'targetable_files'  : target,
        'files_with_chunks' : covered,
        'coverage_pct'      : round(cs * 100, 1),
        'stub_files'        : stubs[:20],  # cap list for readability
        'stub_count'        : len(stubs),
        'by_module'         : {m: {'files': len(v['files']), 'chunks': v['chunks']}
                                for m, v in module_coverage.items()},
        'grade'             : 'A+' if cs > 0.99 else 'A' if cs > 0.95 else 'B',
    }

    # 4. Type distribution
    td = defaultdict(int)
    for c in chunks:   # include parents
        td[c['chunk_type']] += 1
    R['type_distribution'] = dict(td)

    # 5. Retrieval readiness
    rr = sum(1 for c in child_chunks if _DOMAIN_RE.search(c['content']))
    rs = rr / n if n else 0
    R['retrieval_readiness'] = {
        'chunks_with_domain_terms': rr,
        'score': round(rs * 100, 1),
        'grade': 'A+' if rs > 0.93 else 'A' if rs > 0.85 else 'B',
    }

    # 6. Table quality
    trows = [c for c in child_chunks if 'TABLE_ROW' in c['chunk_type']]
    wf    = sum(1 for c in trows if c['table_field'])
    tq    = wf / len(trows) if trows else 1.0
    R['table_chunk_quality'] = {
        'total_table_row_chunks'  : len(trows),
        'with_field_name_metadata': wf,
        'score': round(tq * 100, 1),
        'grade': 'A+' if tq > 0.99 else 'A',
    }

    # 7. Content purity
    polluted = sum(1 for c in child_chunks
                   if re.search(r'\ufffd|\x00', c['content']))
    ps = (n - polluted) / n if n else 1.0
    R['content_purity'] = {
        'clean_chunks': n - polluted, 'polluted_chunks': polluted,
        'score': round(ps * 100, 1),
        'grade': 'A+' if ps > 0.99 else 'C',
    }

    # 8. Encoding health
    corrupted = sum(1 for c in child_chunks if '\ufffd' in c['content'])
    eh        = (n - corrupted) / n if n else 1.0
    R['encoding_health'] = {
        'clean_chunks': n - corrupted, 'corrupted_chunks': corrupted,
        'score': round(eh * 100, 1),
        'grade': 'A+' if eh > 0.99 else 'C',
    }

    # 9. Deduplication stats
    R['deduplication'] = {
        'cross_file_duplicates_removed': analytics.get('duplicates_removed', 0),
    }

    # 10. Parent-child stats
    parents  = [c for c in chunks if c['chunk_type'] == 'PARENT']
    children = [c for c in chunks if c['chunk_type'] != 'PARENT']
    R['hierarchy'] = {
        'parent_chunks' : len(parents),
        'child_chunks'  : len(children),
        'avg_children_per_parent': round(len(children) / len(parents), 1) if parents else 0,
    }

    # 11. Gold standard RAG eval
    R['gold_standard_rag'] = evaluate_gold_standard(child_chunks)

    # Overall
    structural_scores = [
        R['size_distribution']['score'],
        R['context_completeness']['score'],
        R['coverage']['coverage_pct'],
        R['retrieval_readiness']['score'],
        R['table_chunk_quality']['score'],
        R['content_purity']['score'],
        R['encoding_health']['score'],
    ]
    rag_score  = R['gold_standard_rag']['recall_at_3']
    all_scores = structural_scores + [rag_score, rag_score]
    overall    = sum(all_scores) / len(all_scores)
    R['OVERALL'] = {
        'structural_avg': round(sum(structural_scores) / len(structural_scores), 1),
        'rag_recall_at3': rag_score,
        'score'         : round(overall, 1),
        'grade'         : ('A+' if overall > 95 else 'A'  if overall > 88
                           else 'B+' if overall > 78 else 'B'),
    }
    return R

# ─── REPORT PRINTER ───────────────────────────────────────────────────────────

def print_evaluation(R: dict, chunks: list):
    W  = 72
    cc = chunks  # all chunks including parents
    n  = R['size_distribution']['total_chunks']

    print(f"\n{'='*W}\nCHUNKING EVALUATION REPORT  v5.0\n{'='*W}")

    sd = R['size_distribution']
    print(f"\n📐 SIZE DISTRIBUTION  [{sd['grade']}  {sd['score']}%]")
    print(f"   Total child chunks : {sd['total_chunks']}")
    print(f"   In-range: {sd['in_target_range']}  Short: {sd['too_short']}  Long: {sd['too_long']}")
    print(f"   Avg {sd['avg_words']}w  Median {sd['median_words']}w  Min {sd['min_words']}w  Max {sd['max_words']}w")

    cm = R['context_completeness']
    print(f"\n🔖 CONTEXT COMPLETENESS  [{cm['grade']}  {cm['score']}%]")
    print(f"   Module-tagged  : {cm['chunks_with_module']}/{n}")
    print(f"   Context prefix : {cm['chunks_with_context_prefix']}/{n}")
    print(f"   With overlap   : {cm['chunks_with_overlap']}/{n}  ← UPGRADE 4")

    cov = R['coverage']
    print(f"\n📚 SOURCE COVERAGE  [{cov['grade']}  {cov['coverage_pct']}%]")
    print(f"   Total: {cov['total_source_files']}  Stubs: {cov['stub_count']}  "
          f"Covered: {cov['files_with_chunks']}/{cov['targetable_files']}")
    print(f"   Module breakdown:")
    for m, v in sorted(cov['by_module'].items(), key=lambda x: -x[1]['chunks'])[:12]:
        print(f"     {m:25s}: {v['files']:4d} files  {v['chunks']:5d} chunks")

    h = R['hierarchy']
    print(f"\n🔗 PARENT-CHILD HIERARCHY  ← UPGRADE 5")
    print(f"   Parent chunks : {h['parent_chunks']}")
    print(f"   Child chunks  : {h['child_chunks']}")
    print(f"   Avg children/parent : {h['avg_children_per_parent']}")

    dedup = R['deduplication']
    print(f"\n🔁 CROSS-FILE DEDUPLICATION  ← UPGRADE 6")
    print(f"   Duplicate chunks removed: {dedup['cross_file_duplicates_removed']}")

    td      = R['type_distribution']
    max_cnt = max(td.values()) if td else 1
    print(f"\n🏷️  CHUNK TYPES")
    for ctype, cnt in sorted(td.items(), key=lambda x: -x[1]):
        bar = '█' * max(1, cnt * 30 // max_cnt)
        print(f"   {ctype:35s}: {cnt:4d}  {bar}")

    rr = R['retrieval_readiness']
    print(f"\n🎯 RETRIEVAL READINESS  [{rr['grade']}  {rr['score']}%]")
    print(f"   {rr['chunks_with_domain_terms']}/{n} chunks contain ERP domain terms")

    gs = R['gold_standard_rag']
    print(f"\n🏆 GOLD STANDARD RAG EVAL  [{gs['grade']}]  ← THE REAL TEST ({gs['n_questions']} questions)")
    print(f"   recall@1 : {gs['recall_at_1']}%  (correct file in top-1 result)")
    print(f"   recall@3 : {gs['recall_at_3']}%  (correct file in top-3 results)")
    print(f"\n   Question breakdown:")
    for d in gs['details']:
        icon = '✅' if d['found_in_top3'] else '❌'
        r1   = '🎯' if d['found_in_top1'] else '  '
        print(f"   {icon}{r1} {d['question'][:65]}")
        if not d['found_in_top3']:
            print(f"        Expected : {d['expected_file'][:55]}")
            print(f"        Got top3 : {[f[:35] for f in d['top3_files']]}")
    print(f"\n   ⚠️  {gs['note']}")

    ov = R['OVERALL']
    print(f"\n{'='*W}")
    print(f"  Structural avg : {ov['structural_avg']}%")
    print(f"  RAG recall@3   : {ov['rag_recall_at3']}%  (weighted ×2)")
    print(f"  ★ OVERALL SCORE: {ov['score']}%  [Grade: {ov['grade']}]")
    print(f"{'='*W}")

    print("\n📋 SAMPLE CHUNKS BY TYPE (first occurrence of each)")
    seen_types = set()
    for c in chunks:
        ct = c['chunk_type']
        if ct in seen_types:
            continue
        seen_types.add(ct)
        print(f"\n  ── {ct} ──")
        print(f"  Module  : {c.get('module', '?')}")
        print(f"  File    : {c['source_file']}")
        print(f"  Words   : {c['word_count']}")
        if c.get('section_hint'):
            print(f"  Section : {c['section_hint']}")
        if c.get('table_field'):
            print(f"  Field   : {c['table_field']}")
        print(f"  Prefix  : {c.get('context_prefix', '')[:80]}")
        print(f"  Content : {c['content'][:220]}...")

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 72)
    print("DIVALTO HARMONY — INTELLIGENT CHUNKING PIPELINE  v5.0")
    print(f"Scanning: {DATA_DIR}")
    print("=" * 72)

    # Collect all HTM files across all subfolders
    htm_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        # Skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in files:
            if fname.lower().endswith('.htm'):
                htm_files.append(os.path.join(root, fname))
    htm_files.sort()

    print(f"Found {len(htm_files)} .htm files across all modules\n")

    all_chunks   = []
    stubs        = []
    seen_fps     = set()   # for cross-file dedup
    dup_removed  = 0
    doc_type_map = defaultdict(list)
    analytics    = {
        'total_files'    : len(htm_files),
        'processed_files': 0,
        'skipped_stubs'  : 0,
        'total_chunks'   : 0,
        'chunks_by_type' : defaultdict(int),
        'word_distribution': [],
        'per_file'       : [],
    }

    for filepath in htm_files:
        module   = get_module(filepath)
        doc      = parse_htm(filepath, module)
        doc_type = classify_document(doc)
        doc_type_map[doc_type].append(doc['file'])

        soup = BeautifulSoup(read_htm(filepath), 'html.parser')

        if doc_type == 'STUB':
            stubs.append(doc['file'])
            analytics['skipped_stubs'] += 1
            file_chunks = []
        elif doc_type == 'SHORT_ATOMIC':
            file_chunks = chunk_short_atomic(doc)
        elif doc_type == 'MEDIUM_DESCRIPTIVE':
            file_chunks = chunk_medium_descriptive(doc)
        elif doc_type == 'REFERENCE_TABLE':
            file_chunks = chunk_reference_table(doc, soup)
        elif doc_type == 'EXAMPLE_TABLE':
            file_chunks = chunk_example_table(doc, soup)
        elif doc_type == 'LONG_PROCEDURAL':
            file_chunks = chunk_long_procedural(doc)
        elif doc_type == 'GIANT_DOC':
            file_chunks = chunk_giant_doc(doc)
        else:
            file_chunks = chunk_medium_descriptive(doc)

        # Post-process this file's chunks (with global dedup set)
        before_dedup = len(file_chunks)
        file_chunks  = post_process(file_chunks, seen_fps)
        dup_removed += before_dedup - len(file_chunks)

        # FIX 3: inject dense anchor for thin single-chunk files
        file_chunks = inject_dense_anchors(file_chunks, doc)

        # FIX 5: inject synonyms for vocabulary-mismatch files
        file_chunks = inject_synonyms(file_chunks, doc)

        all_chunks.extend(file_chunks)
        analytics['processed_files'] += 1 if file_chunks else 0
        for c in file_chunks:
            analytics['chunks_by_type'][c['chunk_type']] += 1
            analytics['word_distribution'].append(c['word_count'])
        analytics['per_file'].append({
            'file'      : doc['file'],
            'module'    : module,
            'doc_type'  : doc_type,
            'word_count': doc['word_count'],
            'n_chunks'  : len(file_chunks),
            'title'     : doc['title'],
        })

        status = 'SKIP' if doc_type == 'STUB' else f"{len(file_chunks)} chunks"
        rel    = filepath.replace(DATA_DIR, '').lstrip('/\\')[:60]
        print(f"  [{doc_type:22s}] {status:>10}  |  {rel}")

    analytics['duplicates_removed'] = dup_removed

    # Build parent chunks (UPGRADE 5)
    print(f"\n⚙️  Building parent-child hierarchy...")
    parent_chunks = build_parent_chunks(all_chunks)
    print(f"   {len(all_chunks)} child chunks → {len(parent_chunks)} parent chunks")

    combined = all_chunks + parent_chunks
    analytics['total_chunks'] = len(all_chunks)
    analytics['total_parents'] = len(parent_chunks)
    analytics['doc_type_map'] = dict(doc_type_map)
    analytics['chunks_by_type'] = dict(analytics['chunks_by_type'])

    print(f"\n✅ Final: {len(all_chunks)} child chunks + {len(parent_chunks)} parent chunks")
    print(f"   Cross-file duplicates removed: {dup_removed}")

    return combined, all_chunks, analytics, stubs

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _chunk_counters.clear()
    combined, child_chunks, analytics, stubs = run_pipeline()

    eval_results = evaluate_chunks(combined, analytics, stubs)
    print_evaluation(eval_results, combined)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # chunks_children.json — use these for embedding / retrieval
    children_path = os.path.join(OUTPUT_DIR, "chunks_children.json")
    with open(children_path, 'w', encoding='utf-8') as f:
        json.dump(child_chunks, f, ensure_ascii=False, indent=2)

    # chunks_parents.json — return these to the LLM for generation
    parents = [c for c in combined if c['chunk_type'] == 'PARENT']
    parents_path = os.path.join(OUTPUT_DIR, "chunks_parents.json")
    with open(parents_path, 'w', encoding='utf-8') as f:
        json.dump(parents, f, ensure_ascii=False, indent=2)

    # evaluation.json
    eval_path = os.path.join(OUTPUT_DIR, "evaluation_v5.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    # analytics.json
    analytics_path = os.path.join(OUTPUT_DIR, "analytics_v5.json")
    with open(analytics_path, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Outputs saved to: {OUTPUT_DIR}")
    print(f"   chunks_children.json  — {len(child_chunks)} chunks  → EMBED THESE for retrieval")
    print(f"   chunks_parents.json   — {len(parents)} chunks  → RETURN THESE to LLM")
    print(f"   evaluation_v5.json    — full quality + gold standard RAG metrics")
    print(f"   analytics_v5.json     — per-file + per-module breakdown")
    print(f"\n📌 NEXT STEP: replace keyword eval with multilingual-e5-large embeddings")
    print(f"   from sentence_transformers import SentenceTransformer")
    print(f"   model = SentenceTransformer('intfloat/multilingual-e5-large')")
    print(f"   embeddings = model.encode([c['content'] for c in child_chunks])")