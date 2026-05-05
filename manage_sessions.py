#!/usr/bin/env python3
"""
CLI pour la gestion des sessions RAG
Outil en ligne de commande pour gérer les sessions Q/R persistantes.
"""

import argparse
import sys
import json
from datetime import datetime
from session_manager import SessionManager
from tabulate import tabulate


def format_timestamp(ts: str) -> str:
    """Formate un timestamp ISO pour l'affichage."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ts


def cmd_list(sm: SessionManager, args):
    """Liste toutes les sessions."""
    sessions = sm.list_sessions()
    if not sessions:
        print("❌ Aucune session trouvée")
        return

    table_data = []
    for session_id, info in sessions:
        is_active = "✓" if session_id == sm.current_session() else ""
        table_data.append([
            is_active,
            session_id,
            info["description"],
            format_timestamp(info["created"]),
            format_timestamp(info["modified"]),
            info["turns"],
        ])

    headers = ["", "Session ID", "Description", "Créée", "Modifiée", "Tours"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid") + "\n")
    print(f"Session active: {sm.current_session()}\n")


def cmd_create(sm: SessionManager, args):
    """Crée une nouvelle session."""
    session_id = args.name
    description = args.description or f"Session créée le {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    if sm.create_session(session_id, description):
        print(f"✅ Session '{session_id}' créée avec succès")
    else:
        print(f"❌ Impossible de créer la session '{session_id}'")


def cmd_switch(sm: SessionManager, args):
    """Bascule vers une autre session."""
    session_id = args.name
    if sm.switch_session(session_id):
        print(f"✅ Basculé vers la session '{session_id}'")
    else:
        print(f"❌ Impossible de basculer vers '{session_id}'")


def cmd_delete(sm: SessionManager, args):
    """Supprime une session."""
    session_id = args.name
    if args.confirm or input(f"Êtes-vous sûr de vouloir supprimer '{session_id}'? (y/n): ").lower() == "y":
        if sm.delete_session(session_id):
            print(f"✅ Session '{session_id}' supprimée")
        else:
            print(f"❌ Impossible de supprimer '{session_id}'")
    else:
        print("Suppression annulée")


def cmd_info(sm: SessionManager, args):
    """Affiche les informations d'une session."""
    session_id = args.name or sm.current_session()
    info = sm.get_session_info(session_id)

    if not info:
        print(f"❌ Session '{session_id}' non trouvée")
        return

    print(f"\n📋 Informations de la session '{session_id}':\n")
    print(f"  Description: {info['description']}")
    print(f"  Créée:       {format_timestamp(info['created'])}")
    print(f"  Modifiée:    {format_timestamp(info['modified'])}")
    print(f"  Tours:       {info['turns_count']}")
    print(f"  Active:      {'Oui' if info['is_active'] else 'Non'}\n")


def cmd_history(sm: SessionManager, args):
    """Affiche l'historique d'une session."""
    session_id = args.name or sm.current_session()
    limit = args.limit or 10

    history = sm.get_history(session_id, limit)

    if not history:
        print(f"❌ Aucun historique pour la session '{session_id}'")
        return

    print(f"\n📜 Historique de '{session_id}' ({len(history)} derniers tours):\n")

    for i, turn in enumerate(history, 1):
        ts = format_timestamp(turn["timestamp"])
        q = turn.get("question", "")[:60] + ("..." if len(turn.get("question", "")) > 60 else "")
        a = turn.get("answer", "")[:60] + ("..." if len(turn.get("answer", "")) > 60 else "")

        print(f"  [{i}] {ts}")
        print(f"      Q: {q}")
        print(f"      R: {a}\n")


def cmd_clear(sm: SessionManager, args):
    """Efface l'historique d'une session."""
    session_id = args.name or sm.current_session()

    if args.confirm or input(f"Êtes-vous sûr d'effacer tout l'historique de '{session_id}'? (y/n): ").lower() == "y":
        if sm.clear_session(session_id):
            print(f"✅ Session '{session_id}' effacée")
        else:
            print(f"❌ Impossible d'effacer '{session_id}'")
    else:
        print("Suppression annulée")


def cmd_search(sm: SessionManager, args):
    """Recherche dans l'historique."""
    session_id = args.name or sm.current_session()
    query = args.query

    results = sm.search_history(query, session_id)

    if not results:
        print(f"❌ Aucun résultat pour '{query}' dans '{session_id}'")
        return

    print(f"\n🔍 Résultats de recherche pour '{query}' ({len(results)} trouvés):\n")

    for i, turn in enumerate(results, 1):
        ts = format_timestamp(turn["timestamp"])
        print(f"  [{i}] {ts}")
        print(f"      Q: {turn.get('question', '')}")
        print(f"      R: {turn.get('answer', '')}\n")


def cmd_export(sm: SessionManager, args):
    """Exporte une session."""
    session_id = args.name or sm.current_session()
    format_type = args.format or "json"
    output_file = args.output

    content = sm.export_session(session_id, format_type)

    if not content:
        print(f"❌ Impossible d'exporter '{session_id}'")
        return

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Session exportée dans {output_file}")
        except Exception as e:
            print(f"❌ Erreur lors de l'export: {e}")
    else:
        print(content)


def cmd_current(sm: SessionManager, args):
    """Affiche la session active."""
    print(f"Session active: {sm.current_session()}")


def main():
    parser = argparse.ArgumentParser(
        description="Gestionnaire de sessions RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python manage_sessions.py list                    # Lister toutes les sessions
  python manage_sessions.py create -n "ma_session"  # Créer une session
  python manage_sessions.py switch -n "ma_session"  # Basculer vers une session
  python manage_sessions.py info                    # Infos de la session active
  python manage_sessions.py history --limit 5       # Voir les 5 derniers tours
  python manage_sessions.py search "harmony"        # Rechercher
  python manage_sessions.py export -f markdown -o session.md  # Exporter
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # list
    subparsers.add_parser("list", help="Lister toutes les sessions")

    # create
    create_parser = subparsers.add_parser("create", help="Créer une nouvelle session")
    create_parser.add_argument("-n", "--name", required=True, help="Nom de la session")
    create_parser.add_argument("-d", "--description", help="Description de la session")

    # switch
    switch_parser = subparsers.add_parser("switch", help="Basculer vers une session")
    switch_parser.add_argument("-n", "--name", required=True, help="Nom de la session")

    # delete
    delete_parser = subparsers.add_parser("delete", help="Supprimer une session")
    delete_parser.add_argument("-n", "--name", required=True, help="Nom de la session")
    delete_parser.add_argument("-y", "--confirm", action="store_true", help="Confirmer sans demander")

    # info
    info_parser = subparsers.add_parser("info", help="Infos sur une session")
    info_parser.add_argument("-n", "--name", help="Nom de la session (défaut: active)")

    # history
    history_parser = subparsers.add_parser("history", help="Afficher l'historique")
    history_parser.add_argument("-n", "--name", help="Nom de la session (défaut: active)")
    history_parser.add_argument("--limit", type=int, help="Nombre de tours (0 = tous)")

    # clear
    clear_parser = subparsers.add_parser("clear", help="Effacer une session")
    clear_parser.add_argument("-n", "--name", help="Nom de la session (défaut: active)")
    clear_parser.add_argument("-y", "--confirm", action="store_true", help="Confirmer sans demander")

    # search
    search_parser = subparsers.add_parser("search", help="Rechercher dans l'historique")
    search_parser.add_argument("query", help="Texte à rechercher")
    search_parser.add_argument("-n", "--name", help="Nom de la session (défaut: active)")

    # export
    export_parser = subparsers.add_parser("export", help="Exporter une session")
    export_parser.add_argument("-n", "--name", help="Nom de la session (défaut: active)")
    export_parser.add_argument("-f", "--format", choices=["json", "markdown", "txt"], default="json")
    export_parser.add_argument("-o", "--output", help="Fichier de sortie (défaut: stdout)")

    # current
    subparsers.add_parser("current", help="Afficher la session active")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    sm = SessionManager()

    commands = {
        "list": cmd_list,
        "create": cmd_create,
        "switch": cmd_switch,
        "delete": cmd_delete,
        "info": cmd_info,
        "history": cmd_history,
        "clear": cmd_clear,
        "search": cmd_search,
        "export": cmd_export,
        "current": cmd_current,
    }

    if args.command in commands:
        commands[args.command](sm, args)
    else:
        print(f"❌ Commande inconnue: {args.command}")


if __name__ == "__main__":
    main()
