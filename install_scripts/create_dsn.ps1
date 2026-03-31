<#
create_dsn.ps1
Crée un System DSN ODBC pour SQL Server.
Exécuter en Administrateur.
Modifiez les variables en haut du script avant exécution.
#>

$DSNName = "HarmonyXlan"
$Driver = "ODBC Driver 17 for SQL Server"  # Assurez-vous que ce driver est installé
$Server = "localhost"
$Database = "HarmonyDB"

function Assert-Admin {
    if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Write-Error "Ce script doit être exécuté en tant qu'administrateur. Relancez PowerShell en tant qu'administrateur."
        exit 1
    }
}
Assert-Admin

Write-Host "Création du DSN système ODBC: $DSNName" -ForegroundColor Cyan

$odbcKey = "HKLM:\SOFTWARE\ODBC\ODBC.INI\$DSNName"
New-Item -Path $odbcKey -Force | Out-Null
New-ItemProperty -Path $odbcKey -Name "Driver" -Value $Driver -Force | Out-Null
New-ItemProperty -Path $odbcKey -Name "Server" -Value $Server -Force | Out-Null
New-ItemProperty -Path $odbcKey -Name "Database" -Value $Database -Force | Out-Null
New-Item -Path "HKLM:\SOFTWARE\ODBC\ODBC.INI\ODBC Data Sources" -Force | Out-Null
New-ItemProperty -Path "HKLM:\SOFTWARE\ODBC\ODBC.INI\ODBC Data Sources" -Name $DSNName -Value $Driver -Force | Out-Null

# If on 64-bit Windows and the application is 32-bit, also add to Wow6432Node
if (Test-Path 'HKLM:\SOFTWARE\Wow6432Node') {
    $odbcKey32 = "HKLM:\SOFTWARE\Wow6432Node\ODBC\ODBC.INI\$DSNName"
    New-Item -Path $odbcKey32 -Force | Out-Null
    New-ItemProperty -Path $odbcKey32 -Name "Driver" -Value $Driver -Force | Out-Null
    New-ItemProperty -Path $odbcKey32 -Name "Server" -Value $Server -Force | Out-Null
    New-ItemProperty -Path $odbcKey32 -Name "Database" -Value $Database -Force | Out-Null
    New-Item -Path "HKLM:\SOFTWARE\Wow6432Node\ODBC\ODBC.INI\ODBC Data Sources" -Force | Out-Null
    New-ItemProperty -Path "HKLM:\SOFTWARE\Wow6432Node\ODBC\ODBC.INI\ODBC Data Sources" -Name $DSNName -Value $Driver -Force | Out-Null
}

Write-Host "DSN '$DSNName' créé. Testez la connexion depuis le panneau ODBC ou via isql/sqlcmd." -ForegroundColor Green
