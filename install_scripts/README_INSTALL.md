Guide d'installation automatisée — Divalto/Harmony (Windows local)

BUT
Ce dossier contient des scripts PowerShell et SQL pour préparer une machine Windows locale en vue d'installer Divalto/Harmony (pré-requis, DSN ODBC, création de base SQL).

PRÉREQUIS
- Exécuter PowerShell en tant qu'Administrateur
- Avoir une connexion internet pour winget (optionnel)
- Avoir les installateurs Divalto/Harmony (fichiers fournis par l'éditeur). Les scripts ne contiennent pas les installateurs propriétaires.

FICHIERS
- install_prereqs.ps1  — active IIS (optionnel), tente d'installer SQL Express et ODBC via winget, ouvre firewall
- create_db.sql        — script SQL pour créer la base `HarmonyDB` et l'utilisateur `harmony_user`
- create_dsn.ps1       — crée un System DSN ODBC nommé `HarmonyXlan`

ÉTAPES (exécuter dans l'ordre)
1) Ouvrir PowerShell en tant qu'administrateur.
2) Lancer le script des prérequis :

```powershell
cd "C:\Users\DELL\Desktop\CHM_extrait\install_scripts"
.\\install_prereqs.ps1
```

3) Si vous avez installé SQL Server Express, exécutez la création de la base (adaptez le mot de passe SA) :

```powershell
sqlcmd -S localhost -U sa -P "YourSAPassword" -i .\install_scripts\create_db.sql
```

4) Créez le DSN ODBC système :

```powershell
.\create_dsn.ps1
```

5) Placez vos installateurs Divalto (ex: Harmony Power Foundation, Xlan, Xwpf) dans un dossier, par ex `C:\installers\Harmony`.
   - Exécutez l'installateur Harmony manuellement (suivre l'assistant). Si vous fournissez le chemin et les options silencieuses, je peux préparer un script pour lancer l'installation.

6) Après installation Harmony sur le serveur :
   - Vérifiez les services (DhsTerminalServer, XrtDiva, Xwpf selon composant).
   - Testez la connexion depuis le client avec les paramètres DSN/serveur.

NOTES IMPORTANTES
- Les installateurs Divalto sont propriétaires; je ne peux pas les télécharger ou les exécuter pour vous automatiquement sans les fichiers fournis.
- Si vous voulez une automatisation complète (installer Harmony silencieusement), fournissez le chemin des installateurs et les options de ligne de commande (ou autorisez l'exécution interactive) — je générerai un script d'installation silencieuse.

ASSISTANCE
Dites-moi :
- voulez-vous que je génère un script pour lancer automatiquement l'installateur Harmony (si vous me donnez le chemin des fichiers) ? (oui/non)
- préférez-vous que j'ajoute un script pour vérifier l'état des services après installation ?
