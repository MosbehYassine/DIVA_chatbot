# Guide — Débogage des pages HTML dans ce workspace

But
- Ce workspace contient une extraction CHM (fichiers d'aide) composée principalement de pages HTML (`.htm`/`.html`) et de scripts BrightScript (`.brs`). Ce guide explique comment comprendre le contenu et lancer le débogage d'une page HTML depuis VS Code.

Structure importante
- Dossiers principaux : plusieurs sous-dossiers (Administration, AidesFenetrees, Annexes, etc.) contenant des fichiers `.htm` et des ressources associées.
- Fichiers script : `.brs` (BrightScript) — lié à des environnements spécifiques (Roku), pas ciblés pour le débogage HTML ici.
- Fichier d'exemple ouvert dans l'éditeur : Administration/Administration/Administrationd_Harmony__Introduction.htm

Fichier de configuration créé
- `.vscode/launch.json` : contient des configurations pour lancer une page HTML dans Chrome ou Edge depuis VS Code.

Que fait la configuration existante
- `Launch HTML file in Chrome` : ouvre le fichier actif (`${file}`) dans une instance Chrome contrôlée par le débogueur VS Code.
- `Launch specific HTML (Accueil)` : ouvre directement le fichier d'accueil fourni (chemin relatif vers `Administration/Administration/...Harmony.htm`).
- `Launch HTML in Edge` : option équivalente pour Microsoft Edge.

Quand utiliser un serveur local
- Si la page charge des ressources via XHR/fetch, effectue des inclusions relatives, ou si certains scripts s'attendent à être servis via HTTP, il est recommandé d'utiliser un petit serveur local plutôt que d'ouvrir le fichier en `file://`.

Commandes utiles (exécuter depuis la racine du workspace)

Python 3 (serveur simple) :
```bash
python -m http.server 8000
```

Node (http-server rapide) :
```bash
npx http-server -p 8000
```

Après avoir lancé un serveur, utilisez une configuration `launch.json` qui pointe sur l'URL. Exemple d'URL : `http://localhost:8000/Administration/Administration/Administrationd_Harmony__Introduction.htm`.

Étapes pour déboguer une page HTML
1. Ouvrez le fichier cible dans VS Code (ex. `Administration/Administration/Administrationd_Harmony__Introduction.htm`).
2. Si nécessaire, démarrez un serveur local (voir commandes ci-dessus).
3. Ouvrez la vue Run and Debug (icône Lecture/bug) ou appuyez sur `F5`.
4. Choisissez la configuration souhaitée : `Launch HTML file in Chrome` ou `Launch specific HTML (Accueil)`.
5. Dans le navigateur lancé par VS Code vous pouvez ouvrir les DevTools (F12) et poser des points d'arrêt dans l'éditeur VS Code ; les breakpoints côté JS/HTML seront pris en charge.

Conseils pratiques
- Si vous avez besoin d'accéder à des fichiers locaux via AJAX, utilisez un serveur local.
- Pour le debugging JS source-maps : assurez-vous que les chemins sources et les maps sont accessibles via HTTP.
- Si Chrome ne se lance pas, vérifiez que VS Code a l'extension `Debugger for Chrome` intégrée (depuis VS Code 1.45+ le debug pwa-chrome est intégré).

Prochaines étapes possibles
- Je peux :
  - ajouter une configuration `launch.json` qui lance automatiquement `http://localhost:8000/...` si vous préférez le mode serveur, ou
  - ajouter une tâche VS Code pour démarrer automatiquement `python -m http.server` avant le debug.

Fichier créé : `README_DEBUG_HTML.md` (à la racine du workspace).
