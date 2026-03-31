import os
import pickle
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Hugging Face Hub (Windows): éviter les timeouts trop agressifs + log plus propre.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None

# Configuration (fichier du graphe sauvegardé par ingest_docs.py)
DB_DIR = "networkx_graph.pkl"
# Modèles possibles (du + "ultra" au + léger). Le script choisit le meilleur qui passe.
# Astuce: tu peux forcer via l'env var LOCAL_MODEL_NAME.
MODEL_CANDIDATES = [
    # Bon compromis sur une machine 16 Go + GPU: bien meilleur que 0.5B, téléchargement plus raisonnable.
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

# Modèle "ultra" optionnel (à forcer via env var LOCAL_MODEL_NAME si tu veux vraiment).
ULTRA_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _load_llm(model_name: str):
    hf_token = os.getenv("HF_TOKEN")  # optionnel
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token) if hf_token else AutoTokenizer.from_pretrained(model_name)

    # Priorité: GPU + 4-bit si possible (tient mieux sur 16 Go).
    # Sur Windows, la quantification bitsandbytes peut ne pas être dispo selon l'install.
    quant_ok = BitsAndBytesConfig is not None
    use_cuda = torch is not None and torch.cuda.is_available()

    if use_cuda and quant_ok:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map="auto",
            quantization_config=qconf,
            torch_dtype="auto",
        )
    else:
        # Fallback: chargement standard (peut être plus lent / gourmand).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map="auto" if use_cuda else None,
            torch_dtype="auto" if use_cuda else None,
        )

    model.eval()
    return tokenizer, model

def query_vector_store(query):
    if not os.path.exists(DB_DIR):
        print(f"Erreur : Le graphe n'a pas été trouvé à {DB_DIR}. Veuillez exécuter ingest_docs.py d'abord.")
        return

    print("Chargement du graphe des connaissances...")
    try:
        with open(DB_DIR, "rb") as f:
            graph_data = pickle.load(f)
            # Reconstruire l'objet métier
            graph = NetworkxEntityGraph()
            graph._graph = graph_data
    except Exception as e:
        print(f"Erreur lors du chargement du graphe: {e}")
        return
        
    print(f"Graphe chargé avec {graph._graph.number_of_nodes()} noeuds.")

    print("\n--- Analyse du graphe ---")
    # 1. Extraction basique des entités de la question (mots clés simples)
    mots_cles = [word for word in query.split() if len(word) > 3]

    # 2. Recherche de correspondances dans le graphe (on ne garde QUE des extraits de texte)
    doc_snippets = []

    for noeud in graph._graph.nodes():
        for mot in mots_cles:
            if mot.lower() in str(noeud).lower():
                try:
                    # On regarde les documents liés à ce nœud (prédécesseurs)
                    voisins_docs = set(graph._graph.predecessors(noeud))
                    for v in voisins_docs:
                        attrs = graph._graph.nodes[v]
                        if "text" in attrs:
                            snippet = attrs["text"]
                            snippet_lower = snippet.lower()
                            # On ne garde que les extraits qui contiennent au moins un mot-clé de la requête
                            if any(m.lower() in snippet_lower for m in mots_cles):
                                doc_snippets.append(snippet[:250])
                except Exception:
                    pass

    # Contexte = concaténation de quelques extraits de documents pertinents
    if doc_snippets:
        texte_contexte = "\n\n--- EXTRAICTS DE DOCUMENTS ---\n\n" + "\n\n".join(doc_snippets[:3])
    else:
        texte_contexte = "Aucune information précise trouvée dans le graphe."
    print("Contexte extrait du Graphe RAG :")
    print(texte_contexte)

    print(f"\n--- Requête : '{query}' ---\n")

    # === Ancienne version OpenAI (désactivée pour solution 100% gratuite) ===
    # from langchain_core.prompts import PromptTemplate
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o-mini")
    # ...

    # === Version HuggingFace locale (Qwen Instruct) ===
    try:
        forced = os.getenv("LOCAL_MODEL_NAME")
        candidates = [forced] if forced else MODEL_CANDIDATES

        last_err = None
        for name in candidates:
            try:
                tokenizer, model = _load_llm(name)
                print(f"Modèle chargé: {name}")
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise RuntimeError(f"Impossible de charger un modèle parmi {candidates}. Dernière erreur: {last_err}")

        system_msg = (
            "Tu es un assistant RAG. Tu réponds en français.\n"
            "Règles STRICTES :\n"
            "- Utilise UNIQUEMENT les informations présentes dans le CONTEXTE.\n"
            "- Si le CONTEXTE ne suffit pas pour répondre, dis : \"Le contexte fourni ne permet pas de répondre précisément.\" puis ajoute au plus 1 phrase expliquant ce qui manque.\n"
            "- Réponds en 3 à 5 phrases complètes, sans liste, sans puces, sans numérotation.\n"
            "- Ne copie pas mot à mot des passages du CONTEXTE.\n"
        )
        user_msg = f"CONTEXTE:\n{texte_contexte}\n\nQUESTION:\n{query}\n"

        # Qwen Instruct suit beaucoup mieux le format chat
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,          # sortie plus stable, moins d'hallucinations
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # On ne décode que les nouveaux tokens générés (sans le prompt)
        generated_ids = outputs[0]
        prompt_length = inputs["input_ids"].shape[-1]
        generated_answer_ids = generated_ids[prompt_length:]
        answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()

        # Post-traitement léger: supprimer listes/numérotation et garder 3–5 phrases max.
        answer = re.sub(r"(?m)^\s*[\-\*\d]+\s*[\)\.\-]?\s*", "", answer).strip()
        answer = re.sub(r"\n{2,}", "\n", answer).replace("\n", " ").strip()
        sentences = re.split(r"(?<=[\.\!\?])\s+", answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 5:
            answer = " ".join(sentences[:5]).strip()

        # Garde-fou anti-hallucination: si la réponse ne recoupe pas le contexte, on bascule sur un fallback.
        ctx = texte_contexte.lower()
        ans = answer.lower()
        # On inclut aussi les mots clés issus de la question pour éviter de retomber sur un fallback "Harmony" hors sujet.
        query_tokens = [t.lower() for t in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", query) if len(t) >= 3]
        base_keywords = [
            "harmony",
            "windows",
            "imprim",
            "impression",
            "imprimante",
            "aperu",
            "aperçu",
            "etat",
            "état",
            "graphique",
            "caract",
            "fax",
            "messagerie",
            "utilisateur",
            "serveur",
            "chemin",
            "batch",
            "fichier pilote",
            "xlog",
            "xperf",
            "diva",
            "xrtdiva",
            "zoom",
        ]
        likely_keywords = list(dict.fromkeys(query_tokens + base_keywords))
        keyword_overlap = any((k in ans) and (k in ctx) for k in likely_keywords)
        contains_obvious_hallucination = any(bad in ans for bad in ["microsoft", "autodesk", "biblioth", "java", "python", "linux"])

        # Si la réponse contient des affirmations "définitionnelles" non supportées par le contexte, on force le fallback.
        suspicious_terms = [
            "développ",
            "developp",
            "éditeur",
            "editeur",
            "société",
            "societe",
            "autodesk",
            "microsoft",
            "bibliothèque",
            "bibliotheque",
            "langage",
        ]
        introduces_external_claim = any((t in ans) and (t not in ctx) for t in suspicious_terms)

        if (not keyword_overlap) or contains_obvious_hallucination or introduces_external_claim:
            facts = []
            # Fallback orienté "question": on ne parle que de ce qui apparaît réellement dans le contexte.
            if ("xrtdiva" in ctx) or ("diva" in ctx):
                facts.append("Le contexte mentionne le programme `xRtDiva.exe` (lié à l’iconisation d’une application Harmony), mais n’explique pas sa fonction exacte.")
            if "xlog" in ctx or "xlogf" in ctx:
                facts.append("Le contexte associe `Xlog`/`Xlogf` à une base utilisateurs et à l’identification préalable des utilisateurs.")
            if "xperf" in ctx:
                facts.append("Le contexte cite `Xperf`, mais ne détaille pas son rôle dans l’extrait fourni.")
            if "zoom" in ctx:
                facts.append("Le contexte mentionne un \"zoom\" de paramétrage (par exemple pour déclarer des serveurs), sans décrire davantage son fonctionnement.")
            if "fichier pilote" in ctx or "batch" in ctx:
                facts.append("Sous Harmony, un fichier \"batch\" peut être appelé \"fichier pilote\" pour enregistrer l’appel d’un programme (ou d’une séquence) avec ses paramètres.")
            if "imprim" in ctx or "imprimante" in ctx or "windows" in ctx:
                facts.append("Le contexte associe Harmony à des fonctions d’édition/impression sous Windows (imprimante par défaut, paramètres d’impression, aperçu avant impression).")
            if "utilisateur" in ctx:
                facts.append("Le contexte indique qu’Harmony gère des utilisateurs qui doivent s’identifier pour travailler dans l’environnement.")
            if "chemins harmony" in ctx or ("chemin" in ctx and "harmony" in ctx):
                facts.append("Le contexte mentionne des \"chemins Harmony\" servant à remplacer tout ou partie des chemins d’accès réels aux fichiers.")
            if not facts:
                facts.append("Le contexte ne contient pas d’éléments exploitables pour répondre à cette question.")

            # 3 à 5 phrases, sans liste.
            core = facts[:4]
            if len(core) < 2:
                core.append("Le contexte ne fournit pas suffisamment de détails sur le sujet demandé.")
            answer = " ".join(core[:4]).strip()
            answer += f" Le contexte fourni ne permet pas de répondre précisément à la question \"{query}\" au-delà de ces éléments."

        print("\n=== RÉPONSE ===")
        print(answer)
        print("===============\n")
        return answer
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête (modèle local) : {e}")
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        query_vector_store(user_query)
    else:
        # Interactive mode
        print("Enter your query (or 'quit' to exit):")
        while True:
            user_input = input("> ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input.strip():
                query_vector_store(user_input)
