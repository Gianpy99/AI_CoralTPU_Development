#!/usr/bin/env python3
"""Quick camera test with AI"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from universal_app import CoralTPUApp

def main():
    app = CoralTPUApp()
    if app.initialize():
        print("Taking photo with AI analysis...")
        app.photo_mode()
    else:
        print("Failed to initialize app")

if __name__ == "__main__":
    main()
