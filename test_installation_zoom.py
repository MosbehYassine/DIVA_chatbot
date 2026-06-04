#!/usr/bin/env python3
"""
Script de test pour les fichiers situés entre Installation et zoom
Teste les répertoires: Installation, InterfaceAccueil, InterfaceWindows, 
LotusNotes, menus, Miseenoeuvre, modeles, Odbc, RecordSql, reseaux, 
Search, Services, TextesRiches, UnitesV24, utilitaires, et tous les répertoires X*
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

sys.path.append('.')

class DirectoryTester:
    """Teste la structure et le contenu des répertoires"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.results = {}
        
    def get_directories_between(self, start: str, end: str) -> List[str]:
        """Récupère les répertoires entre deux points"""
        all_dirs = []
        data_path = self.base_path / 'data'
        
        if data_path.exists():
            items = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
            
            try:
                start_idx = items.index(start)
                end_idx = items.index(end)
                return items[start_idx:end_idx+1]
            except ValueError:
                print(f"❌ Répertoire '{start}' ou '{end}' non trouvé")
                return []
        
        return []
    
    def analyze_directory(self, dir_name: str) -> Dict[str, Any]:
        """Analyse un répertoire"""
        dir_path = self.base_path / 'data' / dir_name
        
        if not dir_path.exists():
            return {'error': f'Répertoire non trouvé: {dir_name}'}
        
        try:
            # Compter les fichiers
            files = list(dir_path.glob('**/*'))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])
            
            # Analyser les types de fichiers
            file_types = {}
            total_size = 0
            
            for file in files:
                if file.is_file():
                    ext = file.suffix or 'no_extension'
                    file_types[ext] = file_types.get(ext, 0) + 1
                    total_size += file.stat().st_size
            
            # Chercher les fichiers principaux (*.brs, *.glo, *.hhc, *.hhk)
            main_files = {
                'brs': len(list(dir_path.glob('**/*.brs'))),
                'glo': len(list(dir_path.glob('**/*.glo'))),
                'hhc': len(list(dir_path.glob('**/*.hhc'))),
                'hhk': len(list(dir_path.glob('**/*.hhk'))),
                'htm': len(list(dir_path.glob('**/*.htm'))),
                'html': len(list(dir_path.glob('**/*.html'))),
                'jpg': len(list(dir_path.glob('**/*.jpg'))),
                'png': len(list(dir_path.glob('**/*.png'))),
                'gif': len(list(dir_path.glob('**/*.gif'))),
            }
            
            # Filtrer les zéros
            main_files = {k: v for k, v in main_files.items() if v > 0}
            
            return {
                'directory': dir_name,
                'file_count': file_count,
                'dir_count': dir_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024*1024), 2),
                'file_types': file_types,
                'main_files': main_files,
                'status': '✅ OK' if file_count > 0 else '⚠️ Vide'
            }
        except Exception as e:
            return {'error': str(e), 'directory': dir_name}
    
    def test_all_directories(self):
        """Teste tous les répertoires entre Installation et zoom"""
        print("=" * 80)
        print("🧪 TEST DES FICHIERS ENTRE INSTALLATION ET ZOOM")
        print("=" * 80)
        
        # Obtenir les répertoires
        directories = self.get_directories_between('Installation', 'zoom')
        
        if not directories:
            print("❌ Aucun répertoire trouvé!")
            return
        
        print(f"\n📂 Répertoires à tester ({len(directories)}):")
        print("-" * 80)
        for d in directories:
            print(f"  • {d}")
        
        print("\n" + "=" * 80)
        print("🔍 ANALYSE DÉTAILLÉE")
        print("=" * 80)
        
        # Analyser chaque répertoire
        total_files = 0
        total_size = 0
        
        for dir_name in directories:
            result = self.analyze_directory(dir_name)
            self.results[dir_name] = result
            
            if 'error' not in result:
                print(f"\n✨ {result['directory']}")
                print(f"   └─ Fichiers: {result['file_count']}")
                print(f"   └─ Sous-répertoires: {result['dir_count']}")
                print(f"   └─ Taille: {result['total_size_mb']} MB")
                print(f"   └─ Types: {', '.join(result['file_types'].keys())}")
                print(f"   └─ Fichiers principaux: {result['main_files']}")
                print(f"   └─ {result['status']}")
                
                total_files += result['file_count']
                total_size += result['total_size_bytes']
            else:
                print(f"\n❌ {result['directory']}: {result['error']}")
        
        # Résumé
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ")
        print("=" * 80)
        print(f"Total de répertoires: {len(directories)}")
        print(f"Total de fichiers: {total_files}")
        print(f"Taille totale: {round(total_size / (1024*1024), 2)} MB")
        print(f"Taille totale: {round(total_size / (1024*1024*1024), 2)} GB")
        
        # Sauvegarder les résultats
        self.save_results()
    
    def save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        output_file = self.base_path / 'test_results_installation_zoom.json'
        
        # Calculer les statistiques
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        stats = {
            'timestamp': str(Path.cwd()),
            'test_scope': 'Installation to zoom',
            'total_directories': len(self.results),
            'total_files': sum(r.get('file_count', 0) for r in valid_results.values()),
            'total_size_bytes': sum(r.get('total_size_bytes', 0) for r in valid_results.values()),
            'results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Résultats sauvegardés: {output_file}")

def test_chm_integrity():
    """Vérifie l'intégrité des fichiers CHM"""
    print("\n" + "=" * 80)
    print("🔐 VÉRIFICATION DE L'INTÉGRITÉ CHM")
    print("=" * 80)
    
    base_path = Path('c:\\Users\\DELL\\Desktop\\CHM_extrait')
    data_path = base_path / 'data'
    
    # Chercher les fichiers .hhp (project files)
    hhp_files = list(data_path.glob('*/*.hhp')) + list(data_path.glob('*/*.HHP'))
    
    if hhp_files:
        print(f"\n📁 Fichiers projet trouvés: {len(hhp_files)}")
        for f in hhp_files[:10]:  # Afficher les 10 premiers
            print(f"   • {f.relative_to(data_path)}")
    
    # Chercher les fichiers d'index
    index_files = list(data_path.glob('**/#BSSC')) + list(data_path.glob('**/#IDXHDR'))
    
    if index_files:
        print(f"\n📇 Fichiers d'index trouvés: {len(index_files)}")

def test_document_structure():
    """Teste la structure des documents"""
    print("\n" + "=" * 80)
    print("📚 ANALYSE DE LA STRUCTURE DES DOCUMENTS")
    print("=" * 80)
    
    base_path = Path('c:\\Users\\DELL\\Desktop\\CHM_extrait')
    data_path = base_path / 'data'
    
    # Analyser les répertoires avec les fichiers CHM
    chm_content_dirs = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            # Chercher les fichiers .brs, .glo, .hhc, .hhk
            brs = list(item.glob('*.brs'))
            glo = list(item.glob('*.glo'))
            hhc = list(item.glob('*.hhc'))
            hhk = list(item.glob('*.hhk'))
            
            if brs or glo or hhc or hhk:
                chm_content_dirs.append({
                    'name': item.name,
                    'brs': len(brs),
                    'glo': len(glo),
                    'hhc': len(hhc),
                    'hhk': len(hhk)
                })
    
    print(f"\n📖 Répertoires avec contenus CHM: {len(chm_content_dirs)}")
    for dir_info in sorted(chm_content_dirs, key=lambda x: x['name'])[:20]:
        print(f"   {dir_info['name']:25} - brs:{dir_info['brs']}, glo:{dir_info['glo']}, hhc:{dir_info['hhc']}, hhk:{dir_info['hhk']}")

if __name__ == '__main__':
    base_path = Path('c:\\Users\\DELL\\Desktop\\CHM_extrait')
    
    # Exécuter les tests
    tester = DirectoryTester(str(base_path))
    tester.test_all_directories()
    
    # Tests supplémentaires
    test_chm_integrity()
    test_document_structure()
    
    print("\n" + "=" * 80)
    print("✅ TESTS TERMINÉS")
    print("=" * 80)
