#!/usr/bin/env python3
"""Quick crypto prediction test"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from universal_app import CoralTPUApp

def main():
    app = CoralTPUApp()
    if app.initialize():
        print("Running crypto predictions...")
        app.crypto_mode()
    else:
        print("Failed to initialize app")

if __name__ == "__main__":
    main()
