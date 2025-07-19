"""
Coral TPU Crypto AI Trading System
==================================

A comprehensive system that leverages Google Coral TPU for real-time cryptocurrency
trading decisions using AI models optimized for edge computing.

Features:
- Real-time crypto data collection from multiple exchanges
- Technical analysis and feature engineering
- AI models optimized for Coral TPU (Edge TPU)
- Decision making system for trading signals
- Portfolio management and risk assessment
- Real-time monitoring and visualization

Requirements:
- Google Coral TPU (USB Accelerator or Dev Board)
- Python 3.8+
- TensorFlow Lite runtime
- PyCoral library

Setup:
1. Install Coral TPU runtime and PyCoral
2. Install Python dependencies: pip install -r requirements.txt
3. Configure API keys in .env file
4. Run the system: python main.py

Project Structure:
├── src/
│   ├── data/              # Data collection and preprocessing
│   ├── models/            # AI models and TPU optimization
│   ├── trading/           # Trading logic and decision making
│   ├── utils/             # Utilities and helpers
│   └── config/            # Configuration management
├── models/                # Trained model files (.tflite)
├── data/                  # Data storage and cache
├── logs/                  # Application logs
├── tests/                 # Unit tests
└── notebooks/             # Jupyter notebooks for analysis

Author: Your Name
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
