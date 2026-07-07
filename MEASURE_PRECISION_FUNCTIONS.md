# Documentation des fonctions de `measure_precision.py`

Ce fichier explique le role de chaque fonction dans `measure_precision.py`.
Le script sert a evaluer la performance du pipeline RAG sur `test_questions.json`.

Il mesure deux familles de qualite :

- **Retrieval** : est-ce que le bon fichier/source est retrouve dans les resultats ?
- **Answer quality** : est-ce que la reponse generee ressemble a la reponse attendue et reste supportee par le contexte ?

Note session management :

- `measure_precision.py` evalue principalement des questions independantes ;
- il ne mesure pas encore un scenario multi-tours complet avec `session_id` ;
- les champs de session comme `standalone_question`, `history_used`, `reformulation_confidence` et `reformulation_focus` sont stockes par `query_docs.py` dans SQLite pendant l'usage interactif ;
- les tests conversationnels sont documentes dans `SESSION_MANAGEMENT_DOCUMENTATION.md` et `session_test_questions.json`.

## Vue Globale

Le flux principal est :

1. Charger les questions de test.
2. Filtrer eventuellement par module, difficulte ou limite.
3. Charger le pipeline `HybridRAG`.
4. Pour chaque question :
   - executer la recherche RAG ;
   - generer une reponse ;
   - comparer la reponse avec la reponse attendue ;
   - verifier si les sources attendues sont retrouvees ;
   - calculer les scores.
5. Regrouper les resultats par module, categorie et difficulte.
6. Sauvegarder un rapport JSON.

## `similarity(generated, expected)`

Calcule la similarite textuelle stricte entre la reponse generee et la reponse attendue.

Entrees :

- `generated` : reponse produite par le systeme.
- `expected` : reponse attendue dans `test_questions.json`.

Fonctionnement :

- normalise les deux textes avec `normalize_for_match()` ;
- compare les deux chaines avec `SequenceMatcher`.

Sortie :

- un score entre `0.0` et `1.0`.

Observation :

Cette metrique est stricte. Une reponse correcte mais formulee differemment peut avoir un score faible.

## `_normalize_source(value)`

Normalise un nom de fichier/source pour faciliter la comparaison.

Entree :

- `value` : chemin ou nom de fichier.

Fonctionnement :

- garde seulement le nom du fichier avec `os.path.basename()`;
- applique la normalisation texte de `rag_answer.normalize_for_match`.

Sortie :

- une chaine normalisee.

Exemple :

`data/Xlog/Xlog/Test.htm` devient une forme comparable comme `test htm`.

## `_expected_sources(test)`

Extrait les fichiers sources attendus depuis une question de test.

Entree :

- `test` : dictionnaire d'une question dans `test_questions.json`.

Champs lus :

- `source_files`
- ou `expected_source_contains`

Sortie :

- liste de sources normalisees.

Utilisation :

Cette liste sert ensuite a verifier si le RAG a retrouve le bon document.

## `_source_is_relevant(source, expected_sources)`

Verifie si une source retournee par le RAG correspond a une source attendue.

Entrees :

- `source` : fichier retourne par le pipeline.
- `expected_sources` : fichiers attendus.

Fonctionnement :

- normalise `source` ;
- compare par inclusion dans les deux sens :
  - `expected in source_norm`
  - ou `source_norm in expected`

Sortie :

- `True` si la source est jugee correcte ;
- `False` sinon.

## `_retrieval_metrics(results, expected_sources, k=5)`

Calcule les metriques de retrieval sur les `k` premiers resultats.

Entrees :

- `results` : resultats retournes par le RAG.
- `expected_sources` : sources attendues.
- `k` : profondeur d'evaluation, par defaut `5`.

Metriques calculees :

- `recall` / `recall_at_5` : proportion de sources attendues retrouvees dans le top-k.
- `reciprocal_rank` : inverse du rang du premier bon resultat.
- `ndcg_at_5` : qualite du classement des resultats pertinents.
- `first_relevant_rank` : position du premier bon resultat.

Sortie :

- dictionnaire contenant ces metriques.

Interpretation :

- `Recall@5 = 1.0` signifie que la source attendue est dans les 5 premiers resultats.
- `MRR` eleve signifie que le bon document arrive tres haut dans le classement.

## `_expects_not_found(test)`

Detecte si une question est censee ne pas avoir de reponse.

Entree :

- `test` : question de test.

Champs possibles :

- `expected_not_found`
- `should_find = False`
- `answerable = False`

Sortie :

- `True` si le systeme devrait repondre "information non trouvee".
- `False` sinon.

## `_keyword_overlap(generated, test)`

Mesure combien de mots-cles attendus apparaissent dans la reponse generee.

Entrees :

- `generated` : reponse produite.
- `test` : question de test.

Fonctionnement :

- si `expected_keywords` existe, il utilise ces mots-cles ;
- sinon, il extrait des termes importants depuis `expected_answer`.

Sortie :

- score entre `0.0` et `1.0`.

Interpretation :

- `1.0` signifie que tous les mots-cles attendus sont presents.
- `0.0` signifie qu'aucun mot-cle attendu n'est retrouve.

## `_quality_label(score)`

Transforme un score numerique en label lisible.

Regles :

- `>= 0.85` : `excellent`
- `>= 0.70` : `bon`
- `>= 0.55` : `moyen`
- sinon : `faible`

Utilisation :

Cette fonction est utilisee pour afficher une interpretation humaine du score.

## `_safe_avg(values)`

Calcule une moyenne sans erreur si la liste est vide.

Entree :

- `values` : liste de nombres.

Sortie :

- moyenne si la liste contient des valeurs ;
- `0.0` si la liste est vide.

## `_group_summary(rows, key)`

Regroupe les resultats par module, categorie ou difficulte.

Entrees :

- `rows` : liste des details de chaque question.
- `key` : champ de regroupement, par exemple :
  - `module`
  - `category`
  - `difficulty`

Sortie :

- dictionnaire de statistiques par groupe.

Statistiques produites :

- `total`
- `passed`
- `pass_rate`
- `average_similarity`
- `average_performance_score`
- `keyword_overlap`
- `source_match_rate`
- `supported_rate`
- `recall_at_5`
- `quality`

Utilisation :

Permet d'identifier les modules ou difficultes faibles.

## `_failure_reason(row)`

Attribue une cause principale d'echec a une question.

Entree :

- `row` : ligne de detail d'une question evaluee.

Priorite des causes :

1. `retrieval_source_miss` : la bonne source n'est pas dans les resultats.
2. `retrieval_recall_zero` : aucun rappel retrieval.
3. `unsupported_answer` : la reponse n'est pas supportee par le contexte.
4. `low_keyword_overlap` : peu de mots-cles attendus dans la reponse.
5. `low_similarity` : similarite stricte sous le seuil.
6. `ok` : aucun probleme majeur.

Observation :

La premiere cause trouvee est retenue. Par exemple, si la source est mauvaise, le script ne classe pas ensuite le cas en `low_similarity`.

## `_load_resume_details(output_path, resume)`

Charge les resultats deja calcules pour permettre la reprise.

Entrees :

- `output_path` : fichier JSON de sortie.
- `resume` : active ou non la reprise.

Fonctionnement :

- si `resume=False`, retourne une liste vide ;
- si le fichier existe, lit le champ `details` ;
- garde uniquement les lignes valides avec un `id`.

Sortie :

- liste de questions deja calculees.

Utilisation :

Permet a `--resume` de ne pas recalculer les questions deja terminees.

## `_save_resume_details(output_path, details, total)`

Sauvegarde un rapport partiel pendant l'execution.

Entrees :

- `output_path` : fichier JSON a ecrire.
- `details` : resultats deja calcules.
- `total` : nombre total de questions prevues.

Contenu sauvegarde :

- `partial: true`
- `timestamp`
- `completed`
- `total`
- `details`

Utilisation :

Apres chaque question, le script sauvegarde l'etat courant. Si l'execution est arretee, `--resume` peut reprendre.

## `evaluate_tests(...)`

Fonction principale d'evaluation.

Parametres :

- `test_file` : fichier de questions, par defaut `test_questions.json`.
- `use_rag` : active ou non le pipeline `HybridRAG`.
- `show_progress` : affiche les lignes `[i/total]`.
- `limit` : limite le nombre de questions.
- `module_filter` : teste un seul module.
- `difficulty_filter` : teste `facile`, `moyen`, `difficile` ou tout.
- `output_path` : fichier de rapport.
- `resume` : reprend depuis un rapport existant.

Etapes internes :

1. Charge les questions.
2. Applique les filtres.
3. Charge les details existants si `resume=True`.
4. Initialise `HybridRAG`.
5. Pour chaque question :
   - appelle `rag.query()`;
   - recupere les resultats de retrieval ;
   - recupere ou genere la reponse ;
   - calcule la similarite ;
   - calcule les metriques retrieval ;
   - verifie source, support, mots-cles ;
   - calcule le score de performance ;
   - sauvegarde progressivement.
6. Calcule les moyennes globales.
7. Retourne un rapport complet.

Formule du score de reponse :

```text
answer_quality_score =
    0.45 * similarity
  + 0.35 * keyword_overlap
  + 0.20 * answer_supported
```

Formule du score performance :

```text
performance_score =
    0.40 * retrieval_score
  + 0.45 * answer_quality_score
  + 0.15 * source_match
```

Interpretation :

Le score performance favorise :

- la presence du bon document ;
- une reponse proche de l'attendu ;
- des mots-cles corrects ;
- une reponse supportee par le contexte.

## `measure_curated(...)`

Point d'entree compatible avec les anciennes versions.

Role :

- appelle simplement `evaluate_tests(...)`.

Pourquoi elle existe :

- d'autres fichiers comme `run_precision.py` peuvent continuer a utiliser `measure_curated()` sans connaitre les details internes.

## `main()`

Point d'entree CLI du script.

Commande typique :

```powershell
python measure_precision.py --output precision_report.json --resume --no-fail
```

Options principales :

- `--test-file` : choisir un fichier de questions.
- `--no-rag` : tester sans pipeline RAG.
- `--output` : chemin du rapport JSON.
- `--no-fail` : ne pas retourner un code erreur si l'objectif n'est pas atteint.
- `--quiet` : cacher les lignes de progression.
- `--limit` : limiter le nombre de questions.
- `--module` : tester un seul module.
- `--difficulty` : tester une difficulte.
- `--facile`, `--moyen`, `--difficile`, `--all` : raccourcis de difficulte.
- `--resume` : reprendre depuis le fichier de sortie.

Sorties :

- affiche un rapport dans le terminal ;
- ecrit un rapport JSON si `--output` est fourni.

## Champs Importants Du Rapport JSON

### Scores globaux

- `average_performance_score` : score global RAG le plus utile.
- `average_similarity` : similarite stricte texte-a-texte.
- `success_rate_pct` : proportion de questions avec similarite >= seuil.
- `global_score` : moyenne combinee retrieval + reponse.

### Retrieval

- `recall_at_5`
- `recall_at_10`
- `mrr`
- `ndcg_at_5`
- `parent_hit_rate`
- `child_hit_rate`

### Reponse

- `supported_answer_rate`
- `keyword_overlap`
- `source_match`
- `hallucination_rate`
- `not_found_accuracy`

### Diagnostic

- `by_module`
- `by_category`
- `by_difficulty`
- `failure_reasons`
- `weakest_cases`
- `details`

## Comment Lire Les Causes D'echec

| Cause | Signification | Priorite d'action |
| --- | --- | --- |
| `retrieval_source_miss` | Le bon fichier n'est pas dans le top 5 | Ameliorer retrieval/reranking |
| `unsupported_answer` | La reponse n'est pas assez supportee | Ameliorer extraction/verifier |
| `low_keyword_overlap` | Les mots-cles attendus manquent | Ameliorer generation extractive |
| `low_similarity` | La formulation differe de l'attendu | Ameliorer formulation ou adapter metrique |
| `ok` | Cas juge correct | Rien d'urgent |

## Recommandation Pour Les Analyses

Pour juger le projet, privilegier :

1. `average_performance_score`
2. `recall_at_5`
3. `recall_at_10`
4. `source_match`
5. `supported_answer_rate`

La `average_similarity` est utile, mais elle est tres stricte lorsque les reponses attendues sont des extraits exacts.
