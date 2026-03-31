<#
run_installers_interactive.ps1
But: run as Administrator.
Purpose: detect installers in a folder and launch them interactively (or silently if you provide switches).
Usage:
  - Place your installers in C:\installers\Harmony OR in the workspace folder ./installers/Harmony
  - Run PowerShell as Admin and execute this script.
#>

function Assert-Admin {
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Write-Error 'Ce script doit être exécuté en tant qu administrateur. Relancez PowerShell en tant qu administrateur.'
        exit 1
    }
}
Assert-Admin

# Default installer locations (modify if needed)
$defaultPaths = @(
    'C:\installers\Harmony',
    "$PSScriptRoot\\..\\..\\installers\\Harmony",
    "$PSScriptRoot\\installers\\Harmony"
)

$installerDir = $null
foreach ($p in $defaultPaths) {
    if (Test-Path $p) { $installerDir = (Resolve-Path $p).Path; break }
}

if (-not $installerDir) {
    $installerDir = Read-Host 'Chemin des installateurs (ex: C:\\installers\\Harmony)'
    if (-not (Test-Path $installerDir)) { Write-Error "Le chemin fourni n existe pas : $installerDir"; exit 1 }
}

Write-Host "Utilisation du dossier d installateurs : $installerDir" -ForegroundColor Cyan

# Find common installer filenames (customize patterns if your installers have other names)
$patterns = @('*Harmony*.exe','*PowerFoundation*.exe','*Xlan*.exe','*Xwpf*.exe','*.msi','*.zip')
$found = @()
foreach ($pat in $patterns) {
    $found += Get-ChildItem -Path $installerDir -Filter $pat -File -ErrorAction SilentlyContinue
}
$found = $found | Sort-Object Name -Unique

if ($found.Count -eq 0) {
    Write-Warning 'Aucun installateur trouvé avec les patterns par défaut.'
    Write-Host 'Liste des fichiers dans le dossier:'
    Get-ChildItem -Path $installerDir | ForEach-Object { Write-Host " - $($_.Name)" }
    exit 0
}

Write-Host "Installateurs detectes :" -ForegroundColor Green
$idx = 0
foreach ($f in $found) { $idx++; Write-Host "[$idx] $($f.Name)" }

# For each installer ask whether to run interactive or skip
foreach ($f in $found) {
    $choice = Read-Host "Executer $($f.Name) maintenant en mode interactif ? (o/N)"
    if ($choice -match '^[Oo]') {
        Write-Host "Lancement de $($f.FullName) ..." -ForegroundColor Yellow
        try {
            Start-Process -FilePath $f.FullName -WorkingDirectory $installerDir -Wait
            Write-Host "$($f.Name) termine." -ForegroundColor Green
        } catch {
            Write-Warning "Echec du lancement de $($f.Name): $_"
        }
    } else {
        Write-Host "Skip $($f.Name)" -ForegroundColor Gray
    }
}

Write-Host '---' -ForegroundColor Cyan
Write-Host 'Après installation :' -ForegroundColor Cyan
Write-Host " - Vérifiez les services DhsTerminalServer / XrtDiva / Xwpf selon composants installes." -ForegroundColor White
Write-Host " - Si vous préferez une installation silencieuse, fournissez les options de ligne de commande (/S /quiet /qn, etc.) pour chaque installateur et je génèrerai un script d installation silencieuse." -ForegroundColor White
Write-Host 'Script termine.' -ForegroundColor Green
