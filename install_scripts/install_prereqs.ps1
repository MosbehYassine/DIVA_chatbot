<#
install_prereqs.ps1
But: run as Administrator.
This script attempts to:
 - enable IIS (optional)
 - install ODBC Driver 17 for SQL Server via winget (if available)
 - open firewall port 1246 for Harmony socket mode
 - install SQL Server Express via winget (if available) or prompt manual download
 - prints next steps to run create_db.sql and create_dsn.ps1
#>

function Assert-Admin {
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Write-Error "Ce script doit être exécuté en tant qu'administrateur. Relancez PowerShell en tant qu'administrateur."
        exit 1
    }
}

Assert-Admin

Write-Host '=== Prerequis d installation pour Divalto/Harmony (Windows local) ===' -ForegroundColor Cyan

# 1) Installer IIS (optionnel) — utile si vous souhaitez le mode Service Web (SOAP/IIS)
$installIIS = Read-Host 'Voulez-vous activer IIS (mode Web)? (o/N)'
if ($installIIS -match '^[Oo]') {
    Write-Host 'Activation de IIS...' -ForegroundColor Yellow
    Install-WindowsFeature -Name Web-Server -IncludeManagementTools -ErrorAction Stop
    Write-Host 'IIS active.' -ForegroundColor Green
}

# 2) Vérifier winget
$winget = Get-Command winget -ErrorAction SilentlyContinue
if ($null -ne $winget) {
    Write-Host 'winget trouve. Utilisation de winget pour installer SQL Server Express et ODBC Driver si disponibles.' -ForegroundColor Yellow
    try {
        Write-Host 'Installation du pilote ODBC Driver 17 for SQL Server (via winget si disponible)...'
        winget install --id Microsoft.OdbcDriver17forSqlServer -e --silent
    } catch {
        Write-Warning 'Impossible d installer ODBC via winget - verifiez manuellement le pilote ODBC Driver 17 for SQL Server.'
    }

    try {
        Write-Host 'Installation de SQL Server Express (via winget si disponible)...'
        # Package id may vary; this is a best-effort. If not found, user will be prompted.
        winget install --id Microsoft.SQLServer.2019Express -e --silent
        Write-Host 'SQL Server Express installe (ou deja present).' -ForegroundColor Green
    } catch {
        Write-Warning 'winget n a pas pu installer SQL Server Express automatiquement. Telechargez SQL Server Express depuis Microsoft et installez-le manuellement: https://www.microsoft.com/en-us/sql-server/sql-server-downloads'
    }
} else {
    Write-Warning "winget non trouvé. Veuillez installer manuellement le pilote ODBC et SQL Server Express si nécessaire."
}

# 3) Ouvrir le port firewall 1246 (mode Socket)
Write-Host 'Ouverture du port firewall TCP 1246 (Harmony Socket)' -ForegroundColor Yellow
New-NetFirewallRule -DisplayName 'Harmony Socket 1246' -Direction Inbound -LocalPort 1246 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue
Write-Host 'Regle firewall creee (ou existante).' -ForegroundColor Green

# 4) Rappels et prochaine étape
Write-Host ''
Write-Host '=== Prochaines etapes ===' -ForegroundColor Cyan
Write-Host '1) Placer les installateurs Harmony (Power Foundation / Xlan / Xwpf) dans un dossier, par exemple C:\installers\Harmony' -ForegroundColor White
Write-Host '2) Si SQL Server a ete installe, executez le script create_db.sql via sqlcmd pour creer la base et l utilisateur.' -ForegroundColor White
Write-Host "   Exemple: sqlcmd -S localhost -U sa -P 'YourSAPassword' -i .\install_scripts\create_db.sql" -ForegroundColor Gray
Write-Host '3) Executez .\install_scripts\create_dsn.ps1 en administrateur pour creer le DSN ODBC systeme.' -ForegroundColor Gray
Write-Host '4) Lancez l installateur Harmony manuellement (ou fournissez le chemin si vous voulez un script d exécution silencieuse).' -ForegroundColor White

Write-Host 'Script termine.' -ForegroundColor Green
