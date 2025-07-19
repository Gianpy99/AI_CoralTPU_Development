#!/usr/bin/env python3
"""
INSTALLER SEMPLIFICATO - Database Migliorato
Installa le tecnologie raccomandate per migliorare OpenCV
"""

import subprocess
import sys

def install_package(package):
    """Installa pacchetto con gestione errori"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installato")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package} fallito")
        return False

def main():
    print("🚀 INSTALLER DATABASE MIGLIORATO")
    print("=" * 40)
    
    # Pacchetti essenziali
    essential_packages = [
        "face-recognition",
        "mediapipe", 
        "Pillow",
        "numpy"
    ]
    
    # Pacchetti opzionali
    optional_packages = [
        "faiss-cpu",
        "sqlalchemy",
        "matplotlib",
        "seaborn"
    ]
    
    print("📦 Installazione pacchetti essenziali...")
    for package in essential_packages:
        install_package(package)
    
    print("\n📦 Installazione pacchetti opzionali...")
    for package in optional_packages:
        install_package(package)
    
    print("\n✅ Installazione completata!")
    print("🧪 Test le installazioni:")
    
    # Test imports
    tests = [
        ("import face_recognition", "Face Recognition"),
        ("import mediapipe", "MediaPipe"),
        ("import cv2", "OpenCV")
    ]
    
    for test, name in tests:
        try:
            exec(test)
            print(f"✅ {name}: OK")
        except ImportError:
            print(f"❌ {name}: Fallito")

if __name__ == "__main__":
    main()
