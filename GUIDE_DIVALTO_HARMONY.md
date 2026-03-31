# 📘 Guide Complet — Divalto/Harmony 

## 🎯 Vue d'ensemble

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     📦 DIVALTO/HARMONY - ERP PROFESSIONNEL                       ║
║                                                                  ║
║     Plateforme intégrée pour gestion d'entreprise               ║
║     Architecture 3-tiers | Langage Diva | Multi-transport       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

Ce workspace contient une **extraction CHM (Compiled Help Module)** documentant le système **Divalto/Harmony**, une plateforme ERP (Enterprise Resource Planning) développée par Divalto. Les fichiers extraits constituent principalement la documentation officielle du produit et de ses modules d'administration.

---

## 🏗️ Qu'est-ce que Harmony ?

### Vue technique

```
┌─────────────────────────────────────────────────────────────┐
│                   HARMONY v7 - COMPOSANTS                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🎨 INTERFACE UTILISATEUR (WPF)                            │
│  ↑ Xwpf.exe — Client léger riche (Windows Presentation)   │
│  ├─ Support multi-langue                                   │
│  ├─ Profils utilisateur (stockent préférences)            │
│  └─ Modes de connexion flexibles                           │
│                                                             │
│  🔧 MOTEUR D'EXÉCUTION (DIVA)                              │
│  ↑ XrtDiva.exe — Interprète applications Diva             │
│  ├─ Processus métier                                       │
│  ├─ Gestion des tâches                                     │
│  └─ Logique applicative                                    │
│                                                             │
│  💾 STOCKAGE DONNÉES                                        │
│  ↑ Serveur Xlan + Connecteur SQL                           │
│  ├─ Fichiers séquentiels-indexés                           │
│  ├─ Bases SQL (MSSQL, Oracle, DB2)                         │
│  └─ ODBC pour accès universel                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Capacités principales

| Fonctionnalité | Description |
|---|---|
| 📋 **Gestion des Tâches** | Workflows et processus métier |
| 👥 **Gestion Utilisateurs** | Authentification, profils, droits |
| 📂 **Accès aux Fichiers** | Interface universelle données |
| 🖼️ **Administration UI** | Fenêtres, programmes, tableaux |
| 🖨️ **Impressions/Rapports** | Génération dynamique documents |
| 🌐 **Intégrations** | Lotus Notes, SQL, ODBC |

---

## 🗂️ Structure du Workspace — Modules

```
CHM_extrait/
├── 📘 Administration          ← Gestion & administration système
├── 🪟 AidesFenetrees          ← Documentation fenêtres UI
├── 📎 Annexes                 ← Documents annexes & références
├── 🛣️  Chemins                ← Gestion des chemins d'accès
├── 🔧 dlmt                    ← Module DLMT spécialisé
├── ⚠️  Erreurs                ← Codes erreur & diagnostiques
├── 🏛️  Harmony                ← Architecture générale Harmony
├── 📱 harmonya                ← Documentation alternative
├── 🔍 hql                     ← Harmony Query Language
├── 📥 Installation            ← Déploiement & installation
├── 🎨 InterfaceAccueil        ← Interface d'accueil utilisateur
├── 🪟 InterfaceWindows        ← Interfaces Windows spécifiques
├── 📮 LotusNotes              ← Intégration Lotus Notes
├── 🚀 Miseenoeuvre            ← Mise en œuvre & déploiement
├── 📋 modeles                 ← Modèles données & templates
├── 🔌 Odbc                    ← Connectivité ODBC multi-DB
├── ⚙️  Services               ← Services applicatifs
├── 📝 TextesRiches            ← Gestion textes enrichis
├── 🛠️  Xwin-*                 ← Utilitaires Windows
├── 📊 xlansql                 ← Serveur Xlan & SQL
├── 👁️  Xharview               ← Visualisation & consultation
├── 📋 Xlog*                   ← Gestion logs & événements
└── 🔨 Autres utilitaires      ← Xpath, xperf, Xrecup...
```

---

## 📊 Modes de Transport — Architecture Réseau

```
╔════════════════════════════════════════════════════════════════╗
║           HARMONY v7 — 3 MODES DE TRANSPORT                    ║
╚════════════════════════════════════════════════════════════════╝

MODE 1️⃣  — LOCAL (Même machine)
┌─────────────────────────────────────┐
│  Xwpf.exe (Client)                  │
│         ↕ IPC (Inter-Process)       │
│  XrtDiva.exe (Moteur)               │
│         ↕ Fichier local             │
│  Base de données (SQL/ODBC)         │
└─────────────────────────────────────┘
Cas d'usage: Installation standalone


MODE 2️⃣  — SOCKET (Réseau local/VPN)
┌──────────────────────┐      TCP/IP     ┌──────────────────────┐
│   POSTE CLIENT       │ ←─ Port 1246 ─→ │ SERVEUR D'APPLI.     │
│  Xwpf.exe (léger)    │                 │  XrtDiva.exe        │
└──────────────────────┘                 │  Services Diva       │
                                         └──────────────────────┘
Cas d'usage: Réseau d'entreprise, VPN


MODE 3️⃣  — SERVICE WEB (Internet)
┌──────────────────────┐      HTTPS/SOAP ┌──────────────────────┐
│   POSTE CLIENT       │ ←────────────→ │ SERVEUR WEB (IIS)   │
│  Xwpf.exe (léger)    │                 │  Services Diva       │
│  ou Navigateur       │                 │  + XrtDiva.exe      │
└──────────────────────┘                 └──────────────────────┘
Cas d'usage: Postes nomades, accès distant

```

---

## 📦 Types de Fichiers dans le Workspace

```
┌─────────────────────────────────────────────────────────────┐
│              FICHIERS DANS LE WORKSPACE CHM                 │
├─────────────────────────────────────────────────────────────┤

🌐 FICHIERS HTML/DOCUMENTATION
├─ .htm / .html
│  └─ Pages de documentation (générées Adobe RoboHelp 11)
│     • Charset: windows-1252
│     • Structurées avec CSS et JavaScript
│     └─ Consultables dans VS Code (F5 → Chrome)

📜 FICHIERS SCRIPT
├─ .brs (BrightScript)
│  └─ Scripts/extensions (Roku ou variantes)
│     • Modules d'intégration
│     • Extensions système

📋 MÉTADONNÉES CHM
├─ .glo (Glossaire)
│  └─ Définitions & termes clés
├─ .hhc (Table des matières/Contents)
│  └─ Hiérarchie du help système
├─ .hhk (Index/Keywords)
│  └─ Index complet par mots-clés
└─ .lng (Localisation/Language)
   └─ Fichiers de langue (RoboHHRE.lng)

🗂️  DOSSIERS SYSTÈME CHM
├─ #BSSC, #IDXHDR, #ITBITS
│  └─ Métadonnées compilation
├─ #STRINGS, #SYSTEM, #TOPICS
│  └─ Données compilées
└─ #URLSTR, #URLTBL, #WINDOWS
   └─ Index URL, tables URL, fenêtres

💾 DONNÉES COMPILÉES
├─ $FIftiMain
│  └─ Index full-text pour recherche
├─ $OBJINST
│  └─ Instances d'objets compilées
└─ $WWKeywordLinks, $WWAssociativeLinks
   └─ Liens contextuels pré-compilés

└─────────────────────────────────────────────────────────────┘
```

---

## 🔗 Architecture Complète — Data Flow

```
╔════════════════════════════════════════════════════════════════════╗
║          HARMONY v7 — ARCHITECTURE 3-TIERS COMPLÈTE               ║
╚════════════════════════════════════════════════════════════════════╝

                         🎨 COUCHE PRÉSENTATION
                        ┌─────────────────────┐
                        │   Xwpf.exe (WPF)    │
                        │  • Interface riche   │
                        │  • Multi-langue      │
                        │  • Profils user      │
                        └────────────┬─────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
    ┌──────▼─────┐         ┌─────────▼────────┐      ┌────────▼──────┐
    │  MODE 1    │         │    MODE 2        │      │    MODE 3     │
    │   LOCAL    │         │    SOCKET        │      │ SERVICE WEB   │
    │   (IPC)    │         │ (TCP/IP:1246)    │      │ (SOAP + IIS)  │
    └──────┬─────┘         └─────────┬────────┘      └────────┬──────┘
           │                         │                        │
           └─────────────────────────┼────────────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │  🔧 COUCHE MÉTIER      │
                        │  ┌──────────────────┐  │
                        │  │ XrtDiva.exe      │  │
                        │  │ • Moteur Diva    │  │
                        │  │ • Logique métier │  │
                        │  │ • Processus      │  │
                        │  └──────────────────┘  │
                        └────────────┬────────────┘
                                     │
           ┌─────────────────────────┴──────────────────────┐
           │                                                │
    ┌──────▼──────────┐                          ┌─────────▼────────┐
    │  SERVEUR XLAN   │                          │ CONNECTEUR SQL   │
    │  (ODBC Gateway) │                          │ (Direct SQL ORM) │
    │  • Fichiers SEQ │                          │ • RecordSQL      │
    │  • Index        │                          │ • Zooms SQL      │
    │  • Traduction   │                          │ • Requêtes       │
    │    ODBC         │                          └─────────┬────────┘
    └──────┬──────────┘                                    │
           │                                               │
           └────────────────────┬────────────────────────┘
                                │
                   💾 COUCHE BASE DE DONNÉES
                   ┌─────────────────────────┐
                   │  🗄️  BDDR SQL            │
                   │  ┌─ MSSQL 2008/2012     │
                   │  ├─ Oracle              │
                   │  ├─ IBM DB2             │
                   │  └─ Autres (via ODBC)   │
                   └─────────────────────────┘
```

### Flux de données clé

| Direction | Route | Usage |
|---|---|---|
| ➡️ **XLAN** | Fichiers séquentiels, ODBC multi-BD | Données métier, archives |
| ➡️ **RecordSQL** | SQL direct ORM | Zooms, requêtes optimisées |
| ⬅️ **Profils** | XML/INI client | Préférences, contexte connexion |
| ⬅️ **Logs** | Event logs système | Audit, diagnostiques |

---

## 📚 Documentation — Sujets Clés

```
┌─────────────────────────────────────────────────────────┐
│         DOCUMENTATION HARMONY — PARCOURS FORMATION      │
├─────────────────────────────────────────────────────────┤

🟢 DÉBUTANT — Concepts fondamentaux
├─ Harmony/Harmony/La_version_7_d_Harmony.htm
│  └─ Architecture 3-tiers, modes transport
├─ Administration/Introduction
│  └─ Tâches, Utilisateurs, Fichiers
└─ InterfaceAccueil/
   └─ Premiers pas, interface utilisateur

🟡 INTERMÉDIAIRE — Administration et gestion
├─ Administration/
│  ├─ Gestion des utilisateurs
│  ├─ Impressions et rapports
│  ├─ Fenêtres et programmes
│  └─ Gestion des tableaux
├─ Miseenoeuvre/
│  └─ Déploiement et configuration
└─ Services/
   └─ Services applicatifs

🔴 AVANCÉ — Intégrations et performance
├─ Odbc/
│  └─ Connectivité bases de données
├─ xlansql/
│  ├─ Serveur Xlan
│  ├─ RecordSQL
│  └─ Optimisation requêtes
├─ LotusNotes/
│  └─ Intégrations externes
└─ Installation/
   ├─ Serveur d'applications
   ├─ Client léger
   └─ Serveur données

⚡ UTILITAIRES & OUTILS
├─ xtools/ — Utilitaires généraux
├─ Xlog/ — Logs et événements
├─ xperf/ — Performance
├─ Xrecup/ — Récupération données
└─ Xtranslate/ — Gestion langues

⚠️  DIAGNOSTIC & SUPPORT
├─ Erreurs/ — Codes erreur
├─ hql/ — Harmony Query Language
└─ TextesRiches/ — Gestion contenus

└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Guide d'Utilisation

### 1️⃣ Consulter une page de documentation

```
┌─────────────────────────────────────────────┐
│     OUVRIR FICHIER HTML DANS VS CODE       │
├─────────────────────────────────────────────┤
│                                             │
│ 1. Naviguez vers le dossier souhaité       │
│    └─ Ex: Administration/Administration/   │
│                                             │
│ 2. Double-cliquez sur le fichier .htm      │
│    └─ Ex: Administration.htm                │
│                                             │
│ 3. Appuyez sur F5 (ou Run → Debug)        │
│                                             │
│ 4. Choisissez "Launch HTML file in Chrome" │
│                                             │
│ 5. Le fichier s'ouvre dans Chrome          │
│    └─ Vous pouvez maintenant :             │
│       • Lire le contenu HTML               │
│       • Utiliser F12 pour DevTools         │
│       • Poser des breakpoints              │
│                                             │
└─────────────────────────────────────────────┘
```

### 2️⃣ Chercher un sujet spécifique

```
Cas d'usage 1: Chercher "Installation serveur"
├─ 1. Allez dans Installation/
├─ 2. Ouvrez la table des matières (Installation.hhc)
└─ 3. Cherchez le topic "Installation du serveur d'applications"

Cas d'usage 2: Chercher code d'erreur
├─ 1. Allez dans Erreurs/
├─ 2. Ouvrez Erreurs.htm
└─ 3. Consultez l'index (Erreurs.hhk)

Cas d'usage 3: Comprendre architecture
├─ 1. Allez dans Harmony/Harmony/
├─ 2. Ouvrez La_version_7_d_Harmony.htm
└─ 3. Lisez les sections Architecture 3-tiers & Modes Transport
```

### 3️⃣ Naviguer par module

```
🔍 RECHERCHE RAPIDE PAR DOMAINE

Administration      → Administration/         (Gestion système)
Installation        → Installation/           (Déploiement)
Connectivité BD     → Odbc/ + xlansql/       (Bases de données)
Intégrations        → LotusNotes/            (Lotus Notes)
Utilitaires         → xtools/, Xlog/, etc.   (Maintenance)
Diagnostic          → Erreurs/               (Codes erreur)
Architecture        → Harmony/               (Concepts)
```

---

## 📖 Prochaines Étapes

```
╔════════════════════════════════════════════════════════════╗
║           PARCOURS RECOMMANDÉ D'APPRENTISSAGE              ║
╚════════════════════════════════════════════════════════════╝

📍 ÉTAPE 1 — Comprendre l'architecture (30 min)
   └─ Fichier: Harmony/Harmony/La_version_7_d_Harmony.htm
      • Lire "Architecture 3-tiers"
      • Comprendre XrtDiva.exe vs Xwpf.exe
      • Maîtriser les 3 modes de transport

📍 ÉTAPE 2 — Déploiement & installation (45 min)
   └─ Dossier: Installation/
      • Installation du serveur d'applications
      • Installation du client léger
      • Installation du serveur de données

📍 ÉTAPE 3 — Administration système (1 heure)
   └─ Dossier: Administration/
      • Gestion des utilisateurs
      • Gestion des fichiers
      • Gestion des impressions
      • Configuration tableaux et fenêtres

📍 ÉTAPE 4 — Connectivité données (30 min)
   └─ Dossiers: Odbc/ + xlansql/
      • Configuration ODBC
      • Serveur Xlan
      • RecordSQL et requêtes SQL

📍 ÉTAPE 5 — Utilitaires & optimisation (Selon besoin)
   └─ Dossiers: xtools/, Xlog/, xperf/
      • Maintenance système
      • Monitoring performances
      • Diagnostiques logs

📍 ÉTAPE 6 — Intégrations (Selon besoin)
   └─ Dossier: LotusNotes/
      • Intégrations externes
      • Connecteurs

```

---

## 🎓 Résumé Complet

```
╔════════════════════════════════════════════════════════════╗
║    DIVALTO/HARMONY — RÉSUMÉ EXÉCUTIF                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  🏢 PLATEFORME: ERP professionnel Divalto               ║
║  💻 VERSION: v7+ (compatible v6)                         ║
║                                                            ║
║  🏗️  ARCHITECTURE: 3-tiers découplée                      ║
║     • Couche Présentation: Xwpf.exe (WPF)               ║
║     • Couche Métier: XrtDiva.exe (Langage Diva)        ║
║     • Couche Données: ODBC + SQL (MSSQL/Oracle/DB2)    ║
║                                                            ║
║  🔌 TRANSPORTS: 3 modes                                  ║
║     • Local (IPC)                                        ║
║     • Socket (TCP/IP 1246)                              ║
║     • Web (SOAP + IIS)                                  ║
║                                                            ║
║  📊 GESTION: Utilisateurs, Tâches, Fichiers, Rapports   ║
║  🔒 SÉCURITÉ: Authentification, Profils, Droits         ║
║  🌍 INTÉGRATIONS: Lotus Notes, ODBC multi-BD            ║
║  ⚙️  UTILITAIRES: Logs, Performance, Recovery, Translate ║
║                                                            ║
║  📁 CONTENU: 50+ modules documentés en HTML             ║
║     Organized par: admin, install, integration, utils   ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

---

**Ce guide créé**: `GUIDE_DIVALTO_HARMONY.md` (à la racine du workspace)
**Déboguer pages HTML**: Appuyez sur `F5` dans VS Code → Choisissez "Launch HTML file in Chrome"
