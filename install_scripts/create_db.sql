-- create_db.sql
-- Modifiez les variables si besoin avant d'exécuter via sqlcmd
-- Exemple d'exécution:
-- sqlcmd -S localhost -U sa -P "YourSAPassword" -i .\install_scripts\create_db.sql

CREATE DATABASE HarmonyDB;
GO
USE HarmonyDB;
GO
-- Création du login et de l'utilisateur pour l'application Harmony
IF NOT EXISTS (SELECT name FROM sys.sql_logins WHERE name = N'harmony_user')
BEGIN
    CREATE LOGIN harmony_user WITH PASSWORD = 'StrongP@ssw0rd!';
END
GO
IF NOT EXISTS (SELECT name FROM sys.database_principals WHERE name = N'harmony_user')
BEGIN
    CREATE USER harmony_user FOR LOGIN harmony_user;
    ALTER ROLE db_owner ADD MEMBER harmony_user;
END
GO
PRINT 'Base de données HarmonyDB et utilisateur harmony_user créés.'
