#!/usr/bin/env python3
"""
INSTALLER - Database Avanzato Dependencies
Installa tutte le librerie per il sistema di riconoscimento avanzato
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Esegui comando con gestione errori"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completato")
            return True
        else:
            print(f"❌ {description} fallito: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Errore {description}: {e}")
        return False

def install_basic_packages():
    """Installa pacchetti base"""
    packages = [
        "opencv-python",
        "numpy", 
        "Pillow",
        "scikit-learn",
        "scipy"
    ]
    
    for package in packages:
        success = run_command(f"pip install {package}", f"Installazione {package}")
        if not success:
            print(f"⚠️ {package} fallito, continuando...")

def install_face_recognition():
    """Installa face-recognition con dipendenze"""
    print(f"\n🧠 INSTALLAZIONE FACE RECOGNITION")
    
    # Face recognition richiede dlib
    print("📦 Installazione dlib (può richiedere tempo)...")
    
    # Prova diverse strategie per dlib
    dlib_commands = [
        "pip install dlib",
        "pip install --upgrade dlib",
        "conda install -c conda-forge dlib",
    ]
    
    dlib_installed = False
    for cmd in dlib_commands:
        if run_command(cmd, "Installazione dlib"):
            dlib_installed = True
            break
    
    if not dlib_installed:
        print("⚠️ dlib fallito, provo con wheel precompilato...")
        run_command("pip install https://files.pythonhosted.org/packages/dlib-19.24.1-cp39-cp39-win_amd64.whl", "dlib wheel")
    
    # Face recognition
    run_command("pip install face-recognition", "Installazione face-recognition")

def install_mediapipe():
    """Installa MediaPipe"""
    print(f"\n📱 INSTALLAZIONE MEDIAPIPE")
    run_command("pip install mediapipe", "Installazione MediaPipe")

def install_faiss():
    """Installa FAISS per similarity search"""
    print(f"\n🚀 INSTALLAZIONE FAISS")
    
    # FAISS ha versioni diverse per CPU/GPU
    faiss_commands = [
        "pip install faiss-cpu",  # Versione CPU
        "pip install faiss-gpu",  # Versione GPU (se disponibile)
        "conda install -c conda-forge faiss-cpu"
    ]
    
    for cmd in faiss_commands:
        if run_command(cmd, "Installazione FAISS"):
            break

def install_advanced_ml():
    """Installa librerie ML avanzate"""
    print(f"\n🤖 INSTALLAZIONE LIBRERIE ML AVANZATE")
    
    packages = [
        "sentence-transformers",  # Per embeddings semantici
        "transformers",           # Hugging Face transformers
        "torch",                  # PyTorch
        "torchvision",           # Computer vision per PyTorch
        "insightface",           # Face recognition avanzato
        "mtcnn",                 # Face detection
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installazione {package}")

def install_database_tools():
    """Installa strumenti database"""
    print(f"\n💾 INSTALLAZIONE DATABASE TOOLS")
    
    packages = [
        "sqlalchemy",     # ORM database
        "alembic",        # Database migrations  
        "redis",          # Cache in-memory
        "pymongo",        # MongoDB driver
        "psycopg2-binary" # PostgreSQL driver
    ]
    
    for package in packages:
        run_command(f"pip install {package}", f"Installazione {package}")

def download_models():
    """Scarica modelli pre-addestrati"""
    print(f"\n📥 DOWNLOAD MODELLI PRE-ADDESTRATI")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # dlib face landmarks
    landmarks_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    landmarks_file = models_dir / "shape_predictor_68_face_landmarks.dat"
    
    if not landmarks_file.exists():
        print("📥 Downloading dlib face landmarks...")
        run_command(f"curl -L {landmarks_url} -o {models_dir}/landmarks.dat.bz2", "Download landmarks")
        run_command(f"cd {models_dir} && bzip2 -d landmarks.dat.bz2", "Estrazione landmarks")
        run_command(f"cd {models_dir} && move landmarks.dat shape_predictor_68_face_landmarks.dat", "Rename landmarks")
    
    # Face recognition model
    face_rec_url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    face_rec_file = models_dir / "dlib_face_recognition_resnet_model_v1.dat"
    
    if not face_rec_file.exists():
        print("📥 Downloading face recognition model...")
        run_command(f"curl -L {face_rec_url} -o {models_dir}/face_rec.dat.bz2", "Download face recognition")
        run_command(f"cd {models_dir} && bzip2 -d face_rec.dat.bz2", "Estrazione face recognition")
        run_command(f"cd {models_dir} && move face_rec.dat dlib_face_recognition_resnet_model_v1.dat", "Rename face recognition")

def create_requirements_txt():
    """Crea requirements.txt per futuro uso"""
    requirements = """# Database Avanzato Requirements
opencv-python>=4.0.0
numpy>=1.19.0
Pillow>=8.0.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Face Recognition
dlib>=19.24.0
face-recognition>=1.3.0

# MediaPipe
mediapipe>=0.8.0

# Vector Search
faiss-cpu>=1.7.0

# Advanced ML
sentence-transformers>=2.0.0
transformers>=4.20.0
torch>=1.12.0
torchvision>=0.13.0
insightface>=0.7.0
mtcnn>=0.1.0

# Database
sqlalchemy>=1.4.0
alembic>=1.8.0
redis>=4.0.0
pymongo>=4.0.0
psycopg2-binary>=2.9.0

# Utilities
tqdm>=4.60.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
    
    with open("requirements_advanced.txt", "w") as f:
        f.write(requirements)
    
    print("✅ requirements_advanced.txt creato")

def test_installations():
    """Testa installazioni"""
    print(f"\n🧪 TEST INSTALLAZIONI")
    
    tests = [
        ("import cv2; print('OpenCV:', cv2.__version__)", "OpenCV"),
        ("import numpy; print('NumPy:', numpy.__version__)", "NumPy"),
        ("import face_recognition; print('Face Recognition: OK')", "Face Recognition"),
        ("import mediapipe; print('MediaPipe:', mediapipe.__version__)", "MediaPipe"),
        ("import faiss; print('FAISS: OK')", "FAISS"),
        ("import dlib; print('dlib: OK')", "dlib"),
        ("from sentence_transformers import SentenceTransformer; print('Sentence Transformers: OK')", "Sentence Transformers"),
    ]
    
    results = {}
    
    for test_code, name in tests:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
                results[name] = True
            else:
                print(f"❌ {name}: {result.stderr.strip()}")
                results[name] = False
        except Exception as e:
            print(f"❌ {name}: {e}")
            results[name] = False
    
    # Summary
    total = len(results)
    passed = sum(results.values())
    print(f"\n📊 RISULTATI TEST: {passed}/{total} pacchetti funzionanti")
    
    if passed == total:
        print("🎉 Tutte le installazioni completate con successo!")
    else:
        print("⚠️ Alcune installazioni fallite, ma il sistema base dovrebbe funzionare")
    
    return results

def main():
    """Installer principale"""
    print("🚀 INSTALLER DATABASE AVANZATO")
    print("=" * 50)
    print("Questo script installerà tutte le dipendenze per il sistema di riconoscimento avanzato")
    print("⚠️ Alcune installazioni possono richiedere tempo (dlib, PyTorch, etc.)")
    
    response = input("\nContinuare con l'installazione? (y/n): ").lower()
    if response != 'y':
        print("❌ Installazione annullata")
        return
    
    print(f"\n🔧 Aggiornamento pip...")
    run_command("python -m pip install --upgrade pip", "Aggiornamento pip")
    
    # Installazioni step by step
    install_basic_packages()
    install_face_recognition() 
    install_mediapipe()
    install_faiss()
    install_advanced_ml()
    install_database_tools()
    
    # Download modelli
    download_models()
    
    # Crea requirements
    create_requirements_txt()
    
    # Test finale
    results = test_installations()
    
    print(f"\n🎯 INSTALLAZIONE COMPLETATA!")
    print(f"📁 File creati:")
    print(f"  - requirements_advanced.txt")
    print(f"  - models/ (con modelli dlib)")
    print(f"\n🚀 Ora puoi usare il sistema database avanzato!")
    
    if results.get('Face Recognition', False) and results.get('MediaPipe', False):
        print(f"✅ Sistema completamente funzionale con AI avanzata")
    elif results.get('OpenCV', False):
        print(f"⚠️ Sistema base funzionale (solo OpenCV)")
    else:
        print(f"❌ Problemi rilevati, controlla i messaggi di errore")

if __name__ == "__main__":
    main()
