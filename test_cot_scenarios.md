# Scénarios de Test - Chain-of-Thought (CoT)

## Test Suite pour le Système RAG Hybride avec CoT

---

## **TEST 1: Installation TSE Multi-serveurs**

### Question
```
Je dois installer une architecture TSE multi-serveurs avec des clients légers Web. 
Quelles sont les étapes à suivre pour configurer les chemins Harmony, 
déclarer les serveurs de données, configurer les imprimantes, 
et mettre en place les fichiers d'aides ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Installation (357 docs)
- TSE
- Chemin Harmony
- Serveur Xlan
- Imprimante
- Aides

**Top 3 Sources Attendues (Idéales):**
1. `data/Installation/Installation_d'une_architecture_TSE.htm` (ou similaire)
2. `data/Chemins/Configuration_chemin_Harmony.htm` (ou similaire)
3. `data/Services/Configuration_serveur_donnees.htm` (ou similaire)

**Étapes CoT Attendues:**
- Identification des entités clés: TSE, Installation, Harmony
- Recherche graphe par entité Installation → documents TSE
- Recherche vectorielle par similarité sémantique
- Fusion des résultats (graphe + vectoriel)
- Synthèse en 3 extraits pertinents

**Méthodes de Retrieval Attendues:**
- Graphe: 60-70% (entités explicites)
- Vectoriel: 30-40% (termes similaires)

---

## **TEST 2: Diagnostic Accès aux Ressources**

### Question
```
Un utilisateur ne peut pas accéder aux fichiers via le chemin Harmony, 
les imprimantes ne fonctionnent pas, et les aides ne s'affichent pas. 
Quels fichiers de configuration dois-je vérifier et dans quel ordre 
pour diagnostiquer le problème ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Chemin Harmony (chemins)
- Imprimante
- Aides
- Configuration
- Diagnostic

**Top 3 Sources Attendues (Idéales):**
1. `data/Chemins/Diagnostic_chemin_Harmony.htm` (ou similaire)
2. `data/Services/Diagnostic_imprimantes.htm` (ou similaire)
3. `data/Aides/Configuration_affichage_aides.htm` (ou similaire)

**Étapes CoT Attendues:**
- Analyse des 3 problèmes: chemin, imprimantes, aides
- Recherche d'entités de diagnostic
- Correspondance avec documents de configuration
- Ordre logique: chemin → imprimantes → aides
- Fusion et classement par pertinence

**Indicateurs de Succès:**
- Tous les top 3 résultats du GRAPHE (pas du vectoriel)
- Entités détectées: Chemin, Imprimante, Aides
- Scores similaires (~0.50-0.60)

---

## **TEST 3: Configuration ODBC Oracle**

### Question
```
Comment configurer une connexion ODBC pour accéder à une base Oracle 
depuis Harmony ? Quels pilotes installer et quelles étapes suivre ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- ODBC (base données)
- Oracle (base données)
- Configuration
- Harmony
- Pilote

**Top 3 Sources Attendues (Idéales):**
1. `data/Odbc/Configuration_ODBC_Oracle.htm` (ou similaire)
2. `data/xlansql/Pilote_ODBC_Oracle.htm` (ou similaire)
3. `data/RecordSql/Configuration_base_Oracle.htm` (ou similaire)

**Étapes CoT Attendues:**
- Identification des termes: ODBC, Oracle, Pilote, Configuration
- Recherche graphe par ODBC + Oracle
- Recherche vectorielle par mots-clés techniques
- Fusion avec priorité aux documents ODBC-spécifiques
- Ordre d'étapes: installation pilote → configuration DSN → connexion

**Cas d'Attention:**
- Q3 actuellement retourne VECTOR (score 0.55)
- Devrait retourner GRAPH si entités ODBC/Oracle correctement détectées

---

## **TEST 4: Installation Client Léger**

### Question
```
Comment installer un client léger Web ? 
Quels sont les prérequis, les étapes d'installation, 
et comment vérifier que l'installation est réussie ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Installation
- Client léger
- Web
- Configuration

**Top 3 Sources Attendues:**
1. `data/Installation/Installation_client_leger.htm`
2. `data/Installation/Prerequisites_client_Web.htm`
3. `data/Installation/Verification_installation_client.htm`

**Étapes CoT Attendues:**
- Décomposition: prérequis → installation → vérification
- Recherche d'entité "Client léger"
- Fusion graphe+vectoriel
- Réponse ordonnée: 1) Prérequis, 2) Étapes, 3) Vérification

---

## **TEST 5: Gestion des Utilisateurs et Droits**

### Question
```
Comment gérer les utilisateurs et les droits d'accès dans Harmony ? 
Quels fichiers de configuration faut-il modifier ? 
Comment assigner les droits par rôle ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Administration (1687 docs)
- Utilisateur
- Droits d'accès
- Configuration
- Accès

**Top 3 Sources Attendues:**
1. `data/Administration/Gestion_utilisateurs_Harmony.htm`
2. `data/Administration/Configuration_droits_acces.htm`
3. `data/Administration/Roles_et_permissions.htm`

**Étapes CoT Attendues:**
- Identification: Administration > Utilisateurs > Droits
- Recherche graphe par "Administration" + "Accès"
- Localisation des fichiers config
- Synthèse: fichiers concernés et ordre de modification

---

## **TEST 6: Erreurs Courants - Résolution**

### Question
```
Quels sont les erreurs courantes lors de l'installation de Harmony ? 
Quels messages d'erreur devraient m'alerter ? 
Comment les diagnostiquer et les corriger ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Erreurs
- Installation
- Diagnostic
- Harmony

**Top 3 Sources Attendues:**
1. `data/Erreurs/Erreurs_installation_Harmony.htm`
2. `data/Erreurs/Messages_erreur_courants.htm`
3. `data/Erreurs/Diagnostic_et_resolution.htm`

**Étapes CoT Attendues:**
- Parcours de la section "Erreurs" du graphe
- Association erreur → message → résolution
- Synthèse des plus courants
- Tri par fréquence/sévérité

---

## **TEST 7: Réseau et Communication**

### Question
```
Comment configurer le réseau pour la communication entre serveurs ? 
Quels ports doivent être ouverts ? 
Comment diagnostiquer les problèmes de connectivité ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Réseau
- Serveur Xlan
- Configuration
- Diagnostic
- Port

**Top 3 Sources Attendues:**
1. `data/reseaux/Configuration_reseaux.htm`
2. `data/reseaux/Ports_communication.htm`
3. `data/reseaux/Diagnostic_connectivite.htm`

**Étapes CoT Attendues:**
- Identification des composants réseau
- Localisation config réseau
- Listes de ports requis
- Procédures diagnostic

---

## **TEST 8: Sauvegarde et Récupération**

### Question
```
Comment configurer la sauvegarde automatique de la base de données ? 
Quels fichiers sauvegarder ? 
Comment procéder à une récupération en cas de perte de données ?
```

### Réponses Attendues

**Entités Graphe Attendues:**
- Sauvegarde
- Configuration
- Base de données
- Récupération

**Top 3 Sources Attendues:**
1. `data/Miseenoeuvre/Configuration_sauvegarde.htm`
2. `data/RecordSql/Sauvegarde_base_donnees.htm`
3. `data/Miseenoeuvre/Procedure_recuperation.htm`

---

## **Test Execution Script**

Pour exécuter ces tests, vous pouvez utiliser:

```python
from query_docs import HybridRAG

test_questions = {
    "TEST_1_TSE": "Je dois installer une architecture TSE multi-serveurs...",
    "TEST_2_DIAGNOSTIC": "Un utilisateur ne peut pas accéder aux fichiers...",
    "TEST_3_ODBC": "Comment configurer une connexion ODBC...",
    "TEST_4_CLIENT_LEGER": "Comment installer un client léger Web...",
    "TEST_5_UTILISATEURS": "Comment gérer les utilisateurs et droits...",
    "TEST_6_ERREURS": "Quels sont les erreurs courantes...",
    "TEST_7_RESEAU": "Comment configurer le réseau...",
    "TEST_8_SAUVEGARDE": "Comment configurer la sauvegarde..."
}

rag = HybridRAG()
for test_id, question in test_questions.items():
    print(f"\n{'='*80}\n{test_id}\n{'='*80}")
    result = rag.query(question, top_k=3)
    
    print(f"Entités détectées: {result.get('entities', [])}")
    print(f"Graphe: {len([r for r in result.get('merged_results', []) if r.get('method') == 'graph'])} résultats")
    print(f"Vectoriel: {len([r for r in result.get('merged_results', []) if r.get('method') == 'vector'])} résultats")
    
    print("\nTOP 3 RÉSULTATS:")
    for rank, candidate in enumerate(result.get('merged_results', [])[:3], 1):
        print(f"  {rank}. [{candidate.get('method').upper()}] score={candidate.get('hybrid_score', 0):.2f}")
        print(f"     {candidate.get('source', 'N/A')}")
        print(f"     {candidate.get('text', '')[:100]}...")
    
    print("\nÉTAPES COT:")
    for step in result.get('cot_steps', []):
        print(f"  - {step}")
```

---

## **Critères de Succès**

### Pour chaque test:
- [ ] Graphe détecte les entités pertinentes
- [ ] Top 3 résultats incluent au moins 2 documents GRAPH
- [ ] Les sources correspondent aux domaines attendus
- [ ] Les étapes CoT sont logiques et traçables
- [ ] Scores graphe ≥ 0.50
- [ ] Aucune UnicodeEncodeError

### Succès Global du Système:
- ✅ 8/8 tests avec entités correctement détectées
- ✅ Graphe domine pour 6/8 tests (tests 1, 2, 4, 5, 6, 7)
- ✅ Vectoriel peut dominer pour 2/8 tests si pertinent (tests 3, 8)
- ✅ Fusions hybrides détectées (same doc dans graph et vector)
- ✅ CoT steps cohérentes et utiles

