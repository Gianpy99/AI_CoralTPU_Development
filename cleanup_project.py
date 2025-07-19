#!/usr/bin/env python3
"""
CLEANUP UTILITY - Pulizia File Obsoleti
Identifica e rimuove file non necessari dal progetto
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

class ProjectCleanup:
    """Utility per pulizia progetto"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.obsolete_files = []
        self.obsolete_dirs = []
        self.duplicate_files = []
        self.large_files = []
        
        # File da rimuovere (obsoleti/duplicati)
        self.files_to_remove = [
            # File di test obsoleti
            'simple_test.py',
            'quick_crypto_test.py',
            'test_universal_simple.py',
            'quick_camera_test.py',
            
            # Demo obsoleti
            'demo.py',
            'demo_universal.py',
            'quick_camera_ai.py',
            
            # File setup ridondanti
            'setup_system.py',
            'setup_universal.py',
            
            # App obsolete
            'universal_app.py',
            'main.py',  # se non usato
            
            # File recognition obsoleti
            'simple_person_recognition.py',
            'personal_face_recognition.py',
            'coral_camera_real.py',
            'real_coral_camera.py',
            'windows_coral_camera.py',
            'integrated_coral_recognition.py',
            
            # File reports temporanei
            'database_comparison_radar.png',
            'database_comparison_report.json',
            'database_comparison_report.txt',
            'database_upgrade_summary.json',
            
            # Esempi generati
            'example_face_recognition.py',
            'example_faiss.py', 
            'example_mediapipe.py',
            
            # File CSV test
            'crypto_test_data.csv',
            
            # Config obsoleti
            'custom_objects_config.json',
            'inference_config.json',
            'recognition_config.json',
            'unified_system_config.json'
        ]
        
        # Directory da rimuovere se vuote
        self.dirs_to_check = [
            'data',
            'scripts', 
            'src',
            'notebooks',
            'logs',
            '__pycache__'
        ]
        
        # File da mantenere (importanti)
        self.keep_files = [
            'unified_ai_system.py',           # Sistema principale
            'improved_coral_detection.py',   # Core TPU
            'improved_face_recognition_system.py',  # Face recognition
            'tpu_performance_monitor.py',    # Monitoring
            'advanced_database_system.py',   # Database avanzato
            'custom_objects_system.py',      # Oggetti custom
            'complete_recognition_system.py', # Sistema completo
            'database_upgrade_guide.py',     # Guida upgrade
            'install_advanced_dependencies.py', # Installer
            'install_better_database.py',    # Installer semplice
            'migration_guide.py',            # Guida migrazione
            'database_comparison_analysis.py', # Analisi comparativa
            
            # Documenti
            'README.md',
            'GUIDA_RAPIDA.md',
            'QUICKSTART.md',
            'README_UNIVERSAL.md',
            'CORAL_TPU_SETUP.md',
            'SUCCESS_REPORT.md',
            'TROUBLESHOOTING.md',
            'LICENSE',
            
            # Config necessari
            'requirements.txt',
            'requirements_universal.txt',
            '.gitignore',
            '.gitattributes',
            '.env.template',
            'docker-compose.yml',
            'Dockerfile',
            'avvia_sistema.bat'
        ]
    
    def analyze_project(self):
        """Analizza progetto per file obsoleti"""
        print("🔍 ANALISI PROGETTO PER CLEANUP")
        print("=" * 50)
        
        # Conta file totali
        all_files = list(self.project_root.rglob('*'))
        total_files = len([f for f in all_files if f.is_file()])
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        
        print(f"📊 File totali: {total_files}")
        print(f"📦 Dimensione totale: {total_size / (1024*1024):.1f} MB")
        
        # Identifica file obsoleti
        self.find_obsolete_files()
        
        # Trova duplicati
        self.find_duplicate_files()
        
        # File grandi
        self.find_large_files()
        
        # Directory vuote
        self.find_empty_directories()
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024*1024),
            'obsolete_files': len(self.obsolete_files),
            'duplicate_files': len(self.duplicate_files),
            'large_files': len(self.large_files),
            'empty_dirs': len(self.obsolete_dirs)
        }
    
    def find_obsolete_files(self):
        """Trova file obsoleti"""
        print("\n🗑️ RICERCA FILE OBSOLETI")
        
        for filename in self.files_to_remove:
            file_path = self.project_root / filename
            if file_path.exists():
                size = file_path.stat().st_size
                self.obsolete_files.append({
                    'path': file_path,
                    'size': size,
                    'reason': 'obsoleto/duplicato'
                })
                print(f"  🗑️ {filename} ({size / 1024:.1f} KB)")
        
        print(f"✅ Trovati {len(self.obsolete_files)} file obsoleti")
    
    def find_duplicate_files(self):
        """Trova possibili duplicati"""
        print("\n🔍 RICERCA DUPLICATI")
        
        # Patterns per possibili duplicati
        duplicate_patterns = [
            ('requirements.txt', 'requirements_universal.txt'),
            ('README.md', 'README_UNIVERSAL.md'),
            ('person_templates.pkl', 'persons_templates.pkl'),
        ]
        
        for file1, file2 in duplicate_patterns:
            path1 = self.project_root / file1
            path2 = self.project_root / file2
            
            if path1.exists() and path2.exists():
                size1 = path1.stat().st_size
                size2 = path2.stat().st_size
                
                # Se dimensioni simili, potrebbero essere duplicati
                if abs(size1 - size2) < 1024:  # Differenza < 1KB
                    self.duplicate_files.append({
                        'files': [path1, path2],
                        'sizes': [size1, size2],
                        'reason': 'possibile duplicato'
                    })
                    print(f"  🔄 {file1} ↔ {file2}")
        
        print(f"✅ Trovati {len(self.duplicate_files)} possibili duplicati")
    
    def find_large_files(self):
        """Trova file grandi"""
        print("\n📦 RICERCA FILE GRANDI")
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                
                # File > 10MB
                if size > 10 * 1024 * 1024:
                    self.large_files.append({
                        'path': file_path,
                        'size': size,
                        'size_mb': size / (1024*1024)
                    })
                    print(f"  📦 {file_path.name} ({size / (1024*1024):.1f} MB)")
        
        print(f"✅ Trovati {len(self.large_files)} file grandi")
    
    def find_empty_directories(self):
        """Trova directory vuote"""
        print("\n📁 RICERCA DIRECTORY VUOTE")
        
        for dir_name in self.dirs_to_check:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # Controlla se vuota (solo file nascosti permessi)
                    contents = list(dir_path.rglob('*'))
                    visible_contents = [f for f in contents if not f.name.startswith('.')]
                    
                    if len(visible_contents) == 0:
                        self.obsolete_dirs.append({
                            'path': dir_path,
                            'reason': 'directory vuota'
                        })
                        print(f"  📁 {dir_name}/ (vuota)")
                except PermissionError:
                    print(f"  ⚠️ {dir_name}/ (accesso negato)")
        
        print(f"✅ Trovate {len(self.obsolete_dirs)} directory vuote")
    
    def create_backup(self):
        """Crea backup prima della pulizia"""
        backup_dir = self.project_root / 'cleanup_backup'
        backup_dir.mkdir(exist_ok=True)
        
        backup_info = {
            'timestamp': datetime.now().isoformat(),
            'files_backed_up': [],
            'reason': 'Pre-cleanup backup'
        }
        
        print(f"\n💾 CREAZIONE BACKUP in {backup_dir}")
        
        # Backup file obsoleti
        for file_info in self.obsolete_files:
            source = file_info['path']
            dest = backup_dir / source.name
            
            try:
                shutil.copy2(source, dest)
                backup_info['files_backed_up'].append(str(source))
                print(f"  💾 {source.name}")
            except Exception as e:
                print(f"  ❌ Errore backup {source.name}: {e}")
        
        # Salva info backup
        with open(backup_dir / 'backup_info.json', 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        print(f"✅ Backup completato: {len(backup_info['files_backed_up'])} file")
        return backup_dir
    
    def perform_cleanup(self, create_backup=True):
        """Esegui pulizia"""
        print("\n🧹 AVVIO PULIZIA PROGETTO")
        print("=" * 40)
        
        if create_backup:
            backup_dir = self.create_backup()
        
        removed_files = 0
        removed_size = 0
        removed_dirs = 0
        
        # Rimuovi file obsoleti
        print("\n🗑️ RIMOZIONE FILE OBSOLETI")
        for file_info in self.obsolete_files:
            file_path = file_info['path']
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                removed_files += 1
                removed_size += size
                print(f"  ✅ Rimosso: {file_path.name}")
            except Exception as e:
                print(f"  ❌ Errore rimozione {file_path.name}: {e}")
        
        # Rimuovi directory vuote
        print("\n📁 RIMOZIONE DIRECTORY VUOTE")
        for dir_info in self.obsolete_dirs:
            dir_path = dir_info['path']
            try:
                shutil.rmtree(dir_path)
                removed_dirs += 1
                print(f"  ✅ Rimossa: {dir_path.name}/")
            except Exception as e:
                print(f"  ❌ Errore rimozione {dir_path.name}: {e}")
        
        # Pulisci __pycache__
        print("\n🧹 PULIZIA CACHE PYTHON")
        cache_removed = 0
        for cache_dir in self.project_root.rglob('__pycache__'):
            try:
                shutil.rmtree(cache_dir)
                cache_removed += 1
                print(f"  ✅ Cache rimossa: {cache_dir.parent.name}/__pycache__")
            except Exception as e:
                print(f"  ⚠️ Errore cache: {e}")
        
        # Report finale
        print(f"\n📊 PULIZIA COMPLETATA")
        print("=" * 30)
        print(f"🗑️ File rimossi: {removed_files}")
        print(f"📦 Spazio liberato: {removed_size / (1024*1024):.1f} MB")
        print(f"📁 Directory rimosse: {removed_dirs}")
        print(f"🧹 Cache pulite: {cache_removed}")
        
        if create_backup:
            print(f"💾 Backup disponibile in: {backup_dir}")
        
        return {
            'removed_files': removed_files,
            'freed_space_mb': removed_size / (1024*1024),
            'removed_dirs': removed_dirs,
            'cache_cleaned': cache_removed
        }
    
    def generate_keep_list(self):
        """Genera lista file da mantenere"""
        print("\n✅ FILE DA MANTENERE (CORE SISTEMA)")
        print("=" * 50)
        
        for category, files in {
            "🚀 Sistema Principale": [
                'unified_ai_system.py',
                'improved_coral_detection.py',
                'improved_face_recognition_system.py'
            ],
            "🧠 Sistemi Avanzati": [
                'advanced_database_system.py',
                'custom_objects_system.py',
                'complete_recognition_system.py'
            ],
            "🌡️ Monitoring": [
                'tpu_performance_monitor.py'
            ],
            "📚 Documentazione": [
                'README.md',
                'GUIDA_RAPIDA.md',
                'CORAL_TPU_SETUP.md'
            ],
            "🔧 Setup": [
                'install_advanced_dependencies.py',
                'install_better_database.py',
                'requirements.txt'
            ]
        }.items():
            print(f"\n{category}:")
            for file in files:
                status = "✅" if (self.project_root / file).exists() else "❌"
                print(f"  {status} {file}")

def main():
    """Esegui cleanup progetto"""
    print("🧹 CLEANUP UTILITY - PULIZIA PROGETTO")
    print("=" * 60)
    
    cleanup = ProjectCleanup()
    
    # Analisi
    stats = cleanup.analyze_project()
    
    # Mostra cosa mantenere
    cleanup.generate_keep_list()
    
    # Riepilogo
    print(f"\n📊 RIEPILOGO ANALISI")
    print("=" * 30)
    print(f"📁 File totali: {stats['total_files']}")
    print(f"📦 Dimensione: {stats['total_size_mb']:.1f} MB")
    print(f"🗑️ File obsoleti: {stats['obsolete_files']}")
    print(f"🔄 Duplicati: {stats['duplicate_files']}")
    print(f"📦 File grandi: {stats['large_files']}")
    print(f"📁 Dir vuote: {stats['empty_dirs']}")
    
    # Chiedi conferma
    if stats['obsolete_files'] > 0 or stats['empty_dirs'] > 0:
        print(f"\n⚠️ ATTENZIONE: Questa operazione rimuoverà {stats['obsolete_files']} file!")
        print("Un backup sarà creato automaticamente.")
        
        response = input("\nProcedere con la pulizia? (y/n): ").lower()
        
        if response == 'y':
            result = cleanup.perform_cleanup()
            print(f"\n🎉 PULIZIA COMPLETATA!")
            print(f"Spazio liberato: {result['freed_space_mb']:.1f} MB")
        else:
            print("❌ Pulizia annullata")
    else:
        print("\n✅ Progetto già pulito! Nessun file da rimuovere.")

if __name__ == "__main__":
    main()
